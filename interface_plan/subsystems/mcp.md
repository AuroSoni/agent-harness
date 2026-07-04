# Subsystem: External MCP servers (`mcp`)

> Ledger: `AMENDMENTS.md` "External MCP servers (MC)" — LANDED with the implementation cut
> (2026-07-03, MC-D1..D14); on conflict, AMENDMENTS wins. Specs: `tests/interface/mcp/` (72, same
> cut). Design discussion decisions (2026-07-03) are recorded in §10. Status: **IMPLEMENTED**
> (`agent_base/mcp/`, `agent-base[mcp]` extra).

Lets an agent consume tools from **external MCP servers** — local subprocesses (stdio) and remote
services (Streamable HTTP / SSE) — as first-class registry tools. agent-base plays the role the
Claude Code CLI plays in Anthropic's Agent SDK: **it owns the MCP client sessions itself** (via the
official `mcp` Python SDK's client half). There is no equivalent of the SDK's "SDK MCP server" /
control-protocol reverse bridge — in-process tools are already `@tool` / `ConfigurableToolBase`.

**Prime directive:** MCP tools enter through the existing `Toolish`/`ToolRegistry` seam and are
indistinguishable from native tools downstream. The subsystem's whole job is (a) connection
lifecycle + auth, and (b) compiling discovered remote tools into registry-ready callables.
Everything else (parallel execution, cancellation, hooks, budgeting, relay confirmation, envelopes)
is inherited, not reimplemented — the integration expectations in §8 make that contractual.

```
agent_base/mcp/
├── __init__.py      # public re-exports; lazy `mcp` import with actionable ImportError
├── spec.py          # McpStdioSpec / McpHttpSpec / McpSseSpec / McpServerSpec / McpReconnectPolicy
├── auth.py          # McpAuthProvider protocol + StaticHeadersAuth / BearerTokenAuth / SessionHeadersAuth /
│                    # ClientCredentialsOAuth
├── oauth.py         # split-phase browser-OAuth mechanics: discover / register_client / build_authorize_url /
│                    # exchange_code / refresh + TokenSet / TokenStore / OAuthTokenAuth (§4, MC-D9)
├── source.py        # McpToolSource (per-agent), McpServerHandle (per-server), status types; module-level
│                    # probe(); mcp_status tool + change-notice rendering (MC-D13)
└── convert.py       # MCP result content -> ToolResultEnvelope (text/media/resource projection)
```

The `mcp` package is an **optional extra** (`agent-base[mcp]`), imported lazily; constructing an
agent with `mcp_servers=` without it raises `ImportError("pip install agent-base[mcp]")` at
construction, not at first call.

## 1. Config surface — `mcp/spec.py`

Dataclasses (repo convention; the Agent SDK uses TypedDicts only because it serializes to a CLI
flag — we never serialize these). Transport says *how to reach* the server; `McpServerSpec` says
*how this agent uses it*.

```python
@dataclass
class McpStdioSpec:
    """Spawn a local MCP server subprocess; newline-delimited JSON-RPC over stdio."""
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)   # secrets live here — never persisted/logged
    cwd: str | None = None

@dataclass
class McpHttpSpec:
    """Remote server over Streamable HTTP (the modern transport — default choice for remote)."""
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    auth: McpAuthProvider | None = None                 # §4; wins over static headers on overlap

@dataclass
class McpSseSpec:
    """Legacy SSE transport — back-compat with older servers only."""
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    auth: McpAuthProvider | None = None

McpTransportSpec = McpStdioSpec | McpHttpSpec | McpSseSpec

@dataclass
class McpReconnectPolicy:
    """Exponential backoff with full jitter. Defaults are the library contract."""
    max_attempts: int = 5          # per outage, not per session lifetime
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0

@dataclass
class McpServerSpec:
    transport: McpTransportSpec
    include_tools: list[str] | None = None      # allowlist of REMOTE names; None = all
    exclude_tools: list[str] = field(default_factory=list)
    confirm_destructive: bool = False           # destructiveHint -> needs_confirmation (§5)
    tool_timeout_s: float = 60.0                # per tools/call round-trip
    required: bool = False                      # True: connect failure fails initialize()
    reconnect: McpReconnectPolicy = field(default_factory=McpReconnectPolicy)
```

Server **keys** (the dict key in `mcp_servers={...}`) must match `^[a-zA-Z0-9_-]+$` and must not
contain `__` (it is the name separator, §5). Validated at construction — `ValueError` immediately,
not at connect time. The same key rules apply to dynamically added servers (§3). `required=` is an
**init-time-only** contract — it governs `initialize()` failure and is ignored by `add_server`.

## 2. Constructor + lifecycle call sites

```python
agent = AnthropicAgent(
    tools=[...],                                   # unchanged
    mcp_servers={                                  # NEW ctor kwarg (also on the LiteLLM agent)
        "github": McpServerSpec(
            transport=McpHttpSpec(url="https://api.githubcopilot.com/mcp/",
                                  auth=BearerTokenAuth(token_cb=fetch_github_token)),
            exclude_tools=["delete_repository"],
            confirm_destructive=True,
        ),
        "local_tools": McpServerSpec(
            transport=McpStdioSpec(command="uv", args=["run", "tools-server"]),
            required=True,
        ),
    },
)
```

- **Constructor**: builds `self._mcp = McpToolSource(mcp_servers, principal=...)` — pure object
  construction, **no I/O** (mirrors the deferred sandbox). `None`/empty → `self._mcp = None`, zero
  overhead on the non-MCP path.
- **`initialize()`** (eager connect — decided): `await self._mcp.start()` connects all servers
  concurrently, runs the MCP handshake + `tools/list`, then
  `self.tool_registry.register_tools(self._mcp.compile_tools())`. Per-server failure isolation: a
  failed `required=False` server lands in status `failed` and the agent boots without its tools; a
  failed `required=True` server raises out of `initialize()`.
- **Teardown**: `await self._mcp.aclose()` from agent close and the `SessionManager` actor
  teardown path. Closing cancels in-flight reconnect tasks and terminates stdio children —
  **no leaked subprocesses past the session actor** (E6).
- **Boot-profile interplay**: like `tools=` (CM-G3d), a profile's declarative tool surface may name
  MCP servers; the ctor kwarg stays the override. Profile *switches* re-register from the already-
  connected source — never reconnect (E10).

## 3. Connection lifecycle & reconnect — first-class (v1)

Per-server state machine, owned by `McpServerHandle`:

```
pending ──connect──> connected ──transport drop──> reconnecting ──success──> connected
   │                     │                              │ attempts exhausted
   │ failure             │ 401/unauthorized             ▼
   ▼                     ▼ (after failed refresh)     failed ──reconnect()/call──> reconnecting
 failed               needs_auth ──reconnect()──> pending
                         any state ──set_enabled(False)──> disabled ──set_enabled(True)──> pending
```

- **Auto-reconnect** on transport drop (stdio child death, HTTP/SSE stream break, streamable-HTTP
  session expiry — a stale `Mcp-Session-Id` gets HTTP 404 and the client must re-`initialize` per
  spec rev 2025-11-25): background
  single-flight task per server, exponential backoff with full jitter per `McpReconnectPolicy`.
  Attempts exhausted → `failed` (terminal until poked).
- **Calls while down**: a tool call arriving while `reconnecting`/`failed` triggers (or joins) one
  single-flight connect attempt bounded by `tool_timeout_s`; if still down, the call returns
  `ToolResultEnvelope.error(...)` — **the turn never crashes on a dead server** (E7). Calls while
  `needs_auth` or `disabled` fail fast with a state-explaining error envelope (no upstream traffic).
- **Manual verbs** (on `McpToolSource`): `reconnect(name)` (any state → fresh connect; re-runs
  discovery and reconciles the registry via the §5 diff rules), `set_enabled(name, enabled)`
  (disable disconnects + unregisters that server's tools; enable reconnects + re-registers).
- **Dynamic registration** (§7 `add_server`/`remove_server` — MC-D8): consumer-driven add/remove on a
  **live** agent, the "user connects a new MCP on the go" surface. `add_server(name, spec)` validates
  the key (ctor rules, §1; duplicate key → `ValueError`), then connects + handshakes + discovers
  immediately — out-of-band I/O that never blocks the turn loop. The registry mutation follows the §5
  boundary discipline: applied immediately when no run is active, queued to the next turn boundary
  otherwise. A 401 during the add parks the handle in `needs_auth` (with `auth_challenge` populated,
  §7) — **the registration itself still succeeds**: the server is visible in `statuses()` and
  recoverable via `reconnect(name)` once the consumer completes authorization. `remove_server(name)`
  disconnects, unregisters that server's tools under the same boundary discipline, and drops the
  handle. Nothing is persisted — a cold boot re-adds servers via the constructor spec (E11).
  `reconcile(desired)` (MC-D14) is the declarative composition of the two: it diffs the desired
  map against current handles and drives `add_server`/`remove_server` itself — same boundary
  discipline, one meta frame per change. **Keys are the identity**: an existing key whose spec
  changed is left untouched (credential changes flow through the per-request auth provider without
  reconnecting; transport changes need explicit `remove_server`+`add_server` or `reconnect`). A
  no-op diff emits nothing — no meta frames, no change notice.
- **Re-handshake discipline**: every (re)connect is a full MCP `initialize` handshake + fresh
  `tools/list`; we never assume tool stability across reconnects.

## 4. Authentication — first-class (v1) — `mcp/auth.py` + `mcp/oauth.py`

```python
class McpAuthProvider(Protocol):
    async def headers(self) -> dict[str, str]:
        """Called on every (re)connect and merged over the transport's static headers."""
    async def on_unauthorized(self) -> bool:
        """Called on 401/unauthorized. True = credentials refreshed, retry once.
        False (or raise) = mark server needs_auth."""
```

Built-ins shipped in v1:

- `StaticHeadersAuth(headers)` — bearer/API-key, the 90% case; `on_unauthorized` returns False.
- `BearerTokenAuth(token_cb: Callable[[], Awaitable[str]])` — consumer-owned token fetch/refresh;
  `on_unauthorized` re-invokes `token_cb` and returns True once per outage (guards refresh loops).
- `SessionHeadersAuth(headers_cb: Callable[[], Awaitable[dict[str, str]]])` — `BearerTokenAuth`
  generalized from a token string to a whole header set: cookie-session logins, rotating API keys,
  HMAC-signed headers. `headers()` serves the cached set (invoking `headers_cb` on first use);
  `on_unauthorized()` re-invokes it once per outage. Persistence lives **inside** the consumer's
  callback — no store protocol (MC-D11).
- `ClientCredentialsOAuth(token_url, client_id, client_secret, scopes=...)` — headless OAuth2
  client-credentials flow with expiry-aware caching + refresh (machine-to-machine remotes).
- `OAuthTokenAuth(store: TokenStore, info: AuthServerInfo, creds: ClientCreds)` (`mcp/oauth.py`) —
  browser-OAuth token sets: serves the bearer from the consumer's `TokenStore`, proactively
  refreshing when expired; on 401 runs the refresh-token grant and persists the new `TokenSet` via
  `store.set_tokens()`; when no refresh is possible → `needs_auth`, and the consumer re-runs the
  interactive leg (below). Implemented by composing this module's `refresh()` directly (the
  earlier wrap-`OAuthClientProvider(handlers=None)` sketch was superseded by the `refresh_lock`
  requirement — the SDK provider offers no serialization hook around its internal
  read→grant→persist leg; see the spec-review deltas).

The **401 contract** (uniform across connect-time and call-time): unauthorized → invoke
`on_unauthorized()` → `True` = retry the operation exactly once with fresh headers → still
unauthorized (or `False`) = state `needs_auth`. Call-time retry is safe by construction: an auth
rejection means the remote tool did **not** execute, so the single re-issued `tools/call` cannot
double a side effect. For HTTP transports the contract runs at **two layers** (spike-verified,
mcp 1.28.1): the handle bridges the provider into an `httpx.Auth` (`async_auth_flow`) on the
client it hands to `streamable_http_client(url, http_client=...)`, so the primary path — serve
headers per request, catch the 401, single-flight `on_unauthorized()`, re-issue the HTTP request
once — happens *inside* httpx without tearing the transport down. A 401 that escapes the bridge
(refresh impossible/failed) kills the whole transport by SDK design: it surfaces as
`ExceptionGroup[httpx.HTTPStatusError]` (`.response` intact — status + `WWW-Authenticate`
readable; `exceptiongroup` backport on py3.10) at the transport context, which the handle's
supervisor catches, classifies, and maps to `needs_auth` with the challenge captured; recovery is
then the §3 reconnect machinery (full re-handshake), all bounded by `tool_timeout_s`. `needs_auth`
is quiet: no retry storm, no upstream traffic until `reconnect(name)` (the consumer's "user
re-authorized" signal).

**Single-flight refresh (handle contract, not provider politeness).** When parallel in-flight
calls to the same server hit 401 together (e.g. the model fired three calls just as the token
expired), the `McpServerHandle` invokes `on_unauthorized()` **exactly once per outage**; the other
calls await that one outcome and then act in unison — refresh succeeded = each retries its own
call exactly once; refresh failed = all resolve to error envelopes and the server transitions to
`needs_auth` once (one state transition, one meta frame). A custom `McpAuthProvider` must never
observe N concurrent `on_unauthorized()` invocations for one outage — serialization is the
handle's job, so providers don't need their own guards (though `BearerTokenAuth` keeps its
once-per-outage guard as defense in depth). Browser-interactive
OAuth (authorization-code + redirect) is **split** (MC-D9, amending MC-D6): the protocol mechanics
live in the library (`mcp/oauth.py`, next), while the user-facing flow stays consumer-side — the
consumer owns the redirect UX, the callback endpoint, and token persistence, and hands the library an
`OAuthTokenAuth`; the library's runtime surface remains `needs_auth` + `reconnect()`. Stdio "auth" is
env injection via `McpStdioSpec.env` — no provider protocol involved.

**Cookie-session servers (username / password / session cookie — MC-D11).** A session cookie is
just a header, so the whole form-login pattern rides the protocol unchanged via
`SessionHeadersAuth`: the consumer's callback reads its stored username/password, POSTs the
server's login endpoint (not MCP — the library never learns the login protocol, same stance as the
OAuth AS), persists the fresh cookie however it likes, and returns `{"Cookie": ...}`; session
expiry hits the standard 401 contract → one single-flight re-login → per-call retry-once
(rate-limited/captcha'd login endpoints are exactly why single-flight matters here). Three boundary
rules: (1) **unauthorized = HTTP 401 only** — a server signaling expiry via 302-to-login or
200+HTML-login-page is out of contract in v1 (ordinary transport/protocol error envelope);
(2) **the transport's cookie jar is not part of the contract** — provider-supplied headers are
authoritative, and the handle must not let httpx's implicit jar become a second, invisible
credential store. Spike-verified: an explicit client-level `Cookie` header natively **wins over
the jar** (httpx never overrides a header already present), so provider cookies are authoritative
by construction; the handle additionally clears the jar via a response hook so server `Set-Cookie`
cannot ride alongside `Authorization`-only providers (`Set-Cookie` rotations are picked up at the
next 401-triggered re-login); (3) HTTP
**Basic** is already `StaticHeadersAuth` (`Authorization: Basic ...`) — no session state involved.
MFA/captcha-gated logins can't refresh headlessly and degrade to the interactive path:
`needs_auth` → consumer UI → `reconnect()` — same shape as browser OAuth, zero new mechanism.

**OAuth protocol helpers — `mcp/oauth.py` (MC-D9).** The standards plumbing of the MCP authorization
spec — RFC 9728 protected-resource metadata → RFC 8414 AS metadata discovery → RFC 7591 dynamic
client registration → PKCE authorization-code flow → token refresh — is implemented once, as
**split-phase** helpers usable from a web backend where authorization completes out-of-band across
HTTP requests. Spike-verified reuse surface (mcp 1.28.1): `mcp.client.auth.utils` ships the
protocol legs as importable functions (WWW-Authenticate parsing, RFC 9728/8414 discovery-URL
builders + response handlers, RFC 7591 registration request/response), `mcp.shared.auth` ships the
pydantic models (`OAuthToken`, `OAuthMetadata`, `ProtectedResourceMetadata`,
`OAuthClientInformationFull`) that back `TokenSet`/`AuthServerInfo`/`ClientCreds`, and
`PKCEParameters.generate()` covers PKCE — `oauth.py` is composition, not reimplementation. The
SDK's `OAuthClientProvider` models the *interactive* leg as inline redirect/callback callables
(`redirect_handler(url)` then `await callback_handler()` — a coroutine parked across the whole
browser dance; wrong shape for out-of-band web flows). `OAuthTokenAuth` therefore implements the
provider contract directly on top of this module's `refresh()` — required so the `refresh_lock`
can wrap the whole read→grant→persist leg (the SDK provider has no such hook) — and the escape to
`needs_auth` is the typed `McpAuthRequiredError`/`McpOAuthError` classification.

```python
async def discover(url_or_challenge) -> AuthServerInfo   # RFC 9728 resource metadata -> RFC 8414 AS metadata
async def register_client(info, redirect_uri, ...) -> ClientCreds  # RFC 7591 (or bring static creds)
def build_authorize_url(info, creds, redirect_uri, scopes) -> tuple[str, PendingAuth]  # PKCE
async def exchange_code(info, creds, pending, code) -> TokenSet
async def refresh(info, creds, tokens) -> TokenSet

@dataclass
class TokenSet:
    access_token: str
    refresh_token: str | None
    expires_at: float | None
    scope: str | None
    token_type: str = "Bearer"

class TokenStore(Protocol):
    """Consumer-implemented persistence (encrypted, tenant-scoped). The library never persists.
    Shape-aligned with the mcp SDK's TokenStorage protocol so OAuthTokenAuth can wrap
    OAuthClientProvider directly (spike-verified, mcp 1.28.1)."""
    async def get_tokens(self) -> TokenSet | None
    async def set_tokens(self, tokens: TokenSet) -> None
    async def get_client_info(self) -> ClientCreds | None
    async def set_client_info(self, info: ClientCreds) -> None
```

`PendingAuth` (PKCE state + verifier + redirect_uri) is **serializable by design** — the consumer
stashes it in its own pending store across the browser redirect round-trip and hands it back to
`exchange_code`. `discover()` accepts either a bare server URL or the `McpAuthChallenge` off a
`needs_auth` status (§7), so a 401 flows directly into discovery with no re-probing.

**Spec-review deltas (2026-07-03, read against MCP spec rev 2025-11-25 — the current revision).**

1. **Client ID Metadata Documents (CIMD) outrank DCR.** The 2025-11-25 revision demotes RFC 7591
   dynamic client registration to a backwards-compatibility fallback and introduces CIMD at
   SHOULD-level: the `client_id` is an HTTPS URL pointing to a static JSON metadata document the
   consumer hosts (client_name + redirect_uris). Mandated client priority: pre-registered creds →
   CIMD (when AS metadata advertises `client_id_metadata_document_supported`) → DCR → prompt.
   `oauth.py` therefore accepts a URL-`client_id` `ClientCreds` alongside `register_client()`;
   whether mcp 1.28.1 implements CIMD natively is an implementation-time check (§11) — if not it
   is a small extension (the client_id is a URL string in the authorize/token requests).
2. **Scope challenges / step-up authorization.** A server MAY reject a *scoped* token at runtime
   with `403` + `WWW-Authenticate: Bearer error="insufficient_scope", scope="..."`. Classification
   treats this as an auth challenge, not a tool failure: captured into `McpAuthChallenge` (which
   carries `scope`, §7) → `needs_auth`; the consumer re-runs the authorize leg with the upgraded
   scopes. The *automatic* step-up loop is deferred — browser-OAuth servers need user consent for
   new scopes anyway. This refines the "unauthorized = 401 only" stance: 401 remains the only
   *unauthorized* signal (MC-D11 unchanged), and 403-`insufficient_scope` is the one additional,
   OAuth-specific challenge classification.
3. **PKCE support must be verified before proceeding**: `build_authorize_url` refuses (typed
   error) when the discovered AS metadata lacks `code_challenge_methods_supported` — the spec
   forbids falling back to plain authorization-code.
4. **Refresh serialization across agents sharing a store.** The spec MANDATES refresh-token
   rotation for public clients, and OAuth 2.1 recommends ASes treat rotated-refresh-token *reuse*
   as a breach signal that revokes the whole grant chain — so two live agents sharing one
   `TokenStore` (two chats of one member) that refresh concurrently can destroy each other's
   grants, and the handle's single-flight (per-agent) cannot prevent it. `OAuthTokenAuth` accepts
   an optional consumer-supplied `refresh_lock` (async context manager) entered around the
   read-store → grant → persist leg, with a **re-read-after-acquire skip** (a sibling already
   refreshed → adopt its tokens, skip the grant). Consumers with one store per live agent may omit
   it; nova supplies a pg advisory lock keyed by the credential row.

Also spec-validating, no change needed: a backend-resident host running MCP clients "on behalf of
resource owners" is the spec's explicitly supported model (confidential and public clients both
first-class), and streamable-HTTP session expiry (server 404s a stale `Mcp-Session-Id` → client
re-initializes) confirms connections are disposable by protocol design (§3 reconnect trigger).

**Secrecy invariant (E8):** `env`, `headers`, and anything a provider returns are never persisted
(no `AgentConfig`/checkpoint capture — §8 E11) and never logged (structlog redaction at the
`mcp.*` logger boundary). `TokenSet`s live in library memory only — durable storage happens solely
through the consumer's `TokenStore` implementation.

## 5. Tool compilation & registry integration — `source.compile_tools()`

Each discovered remote tool compiles into a `@tool`-shaped callable (schema attributes attached),
registered through the normal `ToolRegistry.register_tools()` path:

- **Naming (decided):** `mcp__{server_key}__{remote_name}`. Unique by construction across servers;
  hook matchers can scope per-server (`mcp__github__*`). Remote names that violate the provider
  tool-name charset (`^[a-zA-Z0-9_-]{1,64}$` for Anthropic) are sanitized (`[^a-zA-Z0-9_-]` → `_`,
  truncate, `_2` suffix on residual collision); the **original remote name** is kept on the compiled
  callable and always used on the wire in `tools/call`.
- **Schema pass-through:** the remote `inputSchema` is already JSON Schema — passed through
  verbatim into `ToolSchema.input_schema`; description from the remote description. No re-modeling.
- **Filtering = permissioning:** `include_tools`/`exclude_tools` filter at compile time (matching
  **remote** names). A filtered tool is never registered — invisible to the model, stronger than a
  runtime permission check.
- **Execution mode:** always `executor="backend"` (an MCP server is a backend resource; frontend
  relay for MCP is a non-goal). `confirm_destructive=True` maps the MCP annotation
  `destructiveHint: true` → `__tool_needs_confirmation__ = True`, which routes through the existing
  `classify_tool_calls` confirmation plane — user approval of destructive MCP calls reuses the
  relay/await machinery with **zero new mechanism**.
- **Timeout:** the compiled callable wraps `handle.call(...)` in `tool_timeout_s`; timeout → error
  envelope (never a hang past the loop's control).
- **Discovery diff on `reconnect`/`refresh`:** re-`tools/list` reconciles the registry — new tools
  registered, vanished tools unregistered, changed schemas re-registered. Never mid-turn: diffs are
  applied at turn boundaries (the source queues the diff if a run is active). `listChanged`
  subscription is deferred (§10 MC-D4) — `refresh(name)` is the manual v1 verb.
- **Reconciliation mechanism = registry swap (MC-D12).** `ToolRegistry` stays append-only — there is
  no in-place unregister. Applying a diff means rebuilding: ONE canonical **recompose function**
  (owned by the agent, shared with profile switches) constructs a fresh registry from declared
  sources — ctor tools/frontend tools/sub-agent tool + the current `compile_tools()` surface — then
  swaps `self.tool_registry`, re-attaches the sandbox, re-injects agent UUIDs, and refreshes
  `agent_config.tool_schemas`/`tool_names`. The swap is transactional by construction: a running
  turn keeps servicing the old registry object; the next turn reads the new one. Compiled MCP
  callables carry a **`__mcp_server__ = server_key`** marker attribute so composition filters
  deterministically (never copy-from-old-registry heuristics). Because profile switches use the
  same recompose, a profile rebuild can never silently drop the MCP surface (E10).
- **Model-visible surface changes (MC-D13 — always on, no flag).** Schemas ride every provider
  request, so *added* tools are immediately callable — but the transcript preserves stale capability
  claims after *removals* (and stale denials after additions). Every applied diff therefore leaves
  an explicit context footprint: the `McpToolDiff` is rendered into a compact system-note text
  block spliced into the **next model-bound user content** (never a standalone transcript message —
  provider alternation rules), e.g. `<system-note>MCP servers changed since your last turn:
  + github connected — 12 tools (mcp__github__*); − slack disconnected — do not claim access to its
  tools.</system-note>`. No notice on initial `initialize()` registration (that is the boot
  surface, not a change). Restored transcripts (fork/reset, E11) may contain past notices for
  servers the restored session no longer has — they read as history; the first post-restore
  reconciliation emits a fresh notice if the surface differs.

## 6. Result conversion — `mcp/convert.py`

MCP `CallToolResult` → `ToolResultEnvelope`, honoring existing library conventions:

| MCP content | Envelope treatment |
|---|---|
| `text` | `GenericTextEnvelope`; oversized output goes through **`ctx.emit_capped`** (tools.md §2.4) — spill to sandbox `.tool_results/`, never flood context |
| `image` / `audio` | media helpers → **`media_backend`** (R16) + `media_registry` entry; block projected per provider formatter |
| `resource_link` / `embedded_resource` | v1: text projection (name + uri + description); native resource fetch deferred |
| `isError: true` | `ToolResultEnvelope.error(...)` — a *returned* tool error (no `raised_error`), model-visible |
| transport/timeout/protocol failure | `ToolResultEnvelope.error(...)` **with** `raised_error` set (CM-G4) so `on_tool_error` can distinguish and synthesize recovery |

`structuredContent` (when a server returns it) is appended as a fenced JSON block in v1.

## 7. Runtime management & observability

```python
# On McpToolSource (and re-exported as thin agent methods):
def statuses() -> list[McpServerStatus]          # pull surface — THE status contract
async def reconnect(name) -> McpServerStatus
async def set_enabled(name, enabled) -> None
async def refresh(name) -> McpToolDiff           # re-discovery without reconnect
async def add_server(name, spec) -> McpServerStatus   # dynamic registration on a live agent (§3, MC-D8)
async def remove_server(name) -> None                 # disconnect + unregister + drop handle (§3)
async def reconcile(desired: dict[str, McpServerSpec]) -> list[McpServerStatus]  # diff-to-set (MC-D14)

# Module-level, agent-free (mcp/source.py) — for a consumer's "Add server" UI:
async def probe(transport: McpTransportSpec, *, auth=None, timeout_s=10.0) -> McpProbeResult

@dataclass
class McpServerStatus:
    name: str
    state: Literal["pending", "connected", "reconnecting", "failed", "needs_auth", "disabled"]
    server_info: tuple[str, str] | None          # (name, version) from the handshake
    error: str | None                            # last failure, when failed/needs_auth
    tool_names: list[str]                        # REGISTERED (post-filter, prefixed) names
    auth_challenge: McpAuthChallenge | None      # set on the 401 that landed needs_auth

@dataclass
class McpAuthChallenge:
    www_authenticate: str | None                 # raw header from the 401/403
    resource_metadata_url: str | None            # parsed RFC 9728 pointer — feeds oauth.discover() (§4)
    scope: str | None                            # RFC 6750 scope param when present (401 guidance or
                                                 # 403 insufficient_scope step-up — §4 spec-review deltas)

@dataclass
class McpProbeResult:
    ok: bool
    state: Literal["connected", "needs_auth", "failed"]
    server_info: tuple[str, str] | None
    tools: list[tuple[str, str | None, dict]]    # (remote name, description, annotations) — preview only
    auth_challenge: McpAuthChallenge | None
    error: str | None
```

`probe()` validates a transport, previews the tool list, and detects the auth requirement in **one
bounded, agent-free call** (MC-D10) — connect + handshake + `tools/list`, then clean close. It reuses
the handle machinery internally but registers nothing and persists nothing; a 401 yields
`state="needs_auth"` with the challenge attached, ready for `oauth.discover()`.

**The `mcp_status` tool (MC-D13 — always on, no flag).** Whenever the agent has an `McpToolSource`,
a native library tool named `mcp_status` is auto-registered (no `mcp__` prefix — it is not a remote
tool; the name is `__`-free so it can never collide with compiled names). It takes an optional
`server` argument and answers from `statuses()` — server states, `server_info`, and registered tool
names — so when a user asks "are you connected to X?" the model answers from **ground truth**
instead of introspecting schema names or trusting stale transcript narrative. It is part of the
canonical recompose composition (MC-D12), so it survives registry swaps and profile switches;
executor `"backend"`, no provider cost (E12), and its output contains no credentials (E8 — states
and names only).

- **Push:** on transitions into `failed` / `needs_auth` / back to `connected`, and on dynamic
  `add_server` / `remove_server` (§3), emit a `MetaEnvelope` frame (`mcp_server_state`) on the live
  stream so a frontend can render "GitHub tools unavailable" without polling. Standard stream
  semantics apply — no consumer attached, frame dropped by design.
- **Logging:** structured per-call log (`get_logger("agent_base.mcp")`) with server key, remote
  tool, duration_ms, ok/error — args/results at debug only, credentials never (§4).

## 8. Integration expectations — the contract

The requirements the integration MUST satisfy. Each maps to interface specs in
`tests/interface/mcp/`; "identical to native tools" is always the bar, because MCP tools are
ordinary `RegisteredTool`s after §5.

- **E1 — Hooks.** `before_tool` / `after_tool` / `on_tool_error` fire for MCP tools exactly as for
  native tools; matchers can target exact names and per-server prefixes (`mcp__github__*`); input
  enrichment via `ctx.call.with_input(...)` reaches the wire `tools/call` arguments; a `before_tool`
  deny short-circuits without any upstream traffic.
- **E2 — Naming.** `mcp__{server_key}__{remote_name}`; key charset validated at ctor; sanitization
  + collision suffix per §5; original remote name preserved and used on the wire; two servers
  exposing the same remote tool name coexist.
- **E3 — Result conversion.** The §6 table, verbatim: text/media/resource/isError/transport-failure
  each produce the specified envelope shape; oversized text spills via `ctx.emit_capped`; media
  lands in `media_backend` + `media_registry`; `raised_error` set only for raised (not returned)
  failures.
- **E4 — Permissioning.** `include_tools`/`exclude_tools` are compile-time (filtered = never
  registered = invisible to the model); `confirm_destructive` routes destructive-annotated tools
  through the existing confirmation relay plane (pause → `ToolReply` resume); executor is always
  `"backend"`.
- **E5 — Execution semantics.** MCP calls ride `execute_tools` unchanged: bounded parallelism
  (semaphore), abort via cancellation event → `TOOL_ABORT_TEXT` envelopes, per-call
  `tool_timeout_s`, `duration_ms` stamped. A hung server can never hang the turn loop.
- **E6 — Lifecycle.** Eager connect in `initialize()`; concurrent per-server connects with failure
  isolation; `required=True` failure raises out of `initialize()`; `aclose()` on agent close AND
  session-actor teardown covers ctor-declared **and dynamically added** servers identically; no stdio
  child outlives the actor; ctor with `mcp_servers=` and the extra missing raises actionable
  `ImportError` at construction.
- **E7 — Reconnect.** The §3 state machine: auto-reconnect with backoff+jitter, single-flight;
  calls while down attempt one bounded reconnect then return an error envelope (turn survives);
  `reconnect(name)` recovers from `failed`/`needs_auth`/`disabled`+enable; every reconnect re-runs
  discovery and reconciles the registry at a turn boundary.
- **E8 — Auth.** The §4 401 contract (refresh → retry once → `needs_auth`); refresh is
  **single-flight at the handle**: N parallel calls hitting 401 in one outage produce exactly one
  `on_unauthorized()` invocation, the remaining calls join its outcome (all retry once on success;
  all resolve to error envelopes + ONE `needs_auth` transition and ONE meta frame on failure);
  `needs_auth` is quiet (no upstream traffic until manual `reconnect`); the unauthorized
  classification is **HTTP 401 only** (non-401 expiry signals — 302-to-login, 200+HTML — are
  ordinary transport errors, §4); provider-supplied headers are authoritative over any transport
  cookie jar; credentials (env/headers/provider output) are never persisted to any storage adapter,
  never checkpointed, never logged.
- **E9 — Sub-agents.** `McpToolSource` is a runtime resource: `SubAgentSpec`'s field-aware deepcopy
  keeps it **by reference** (like `memory_store`/tools); children share live connections; a child
  spec can narrow visibility with its own `include_tools` view without affecting the parent; servers
  added dynamically (E14) become visible to children through the shared source.
- **E10 — Profiles.** Profile switches re-register tool surfaces from the connected source without
  reconnecting; a profile that doesn't include an MCP server simply doesn't register its tools
  (connections may stay warm). Switches go through the same canonical recompose function as MCP
  reconciliation (MC-D12) — a profile rebuild can never silently drop the MCP surface.
- **E11 — Checkpoint / fork / reset.** MCP state is **never captured**: no `AgentConfig` fields, no
  checkpoint columns, no schema bump (`LIBRARY_SCHEMA_VERSION` unchanged). After reset/fork, the
  restored/new session reconnects from its constructor `mcp_servers=` spec exactly as a cold boot.
  Tool-result envelopes already in the transcript restore fine (they're plain envelopes).
- **E12 — Cost & settlement.** MCP calls incur no provider cost and never touch
  `TurnSettlement`/pricing; they appear in structured logs (§7) with duration for consumer-side
  metering.
- **E13 — Streaming.** `tool_use`/`tool_result` stream frames for MCP tools are byte-shape
  identical to native tools; the only MCP-specific frame is the §7 `mcp_server_state` meta frame,
  subject to standard drop-if-unconsumed semantics.
- **E14 — Dynamic registration.** `add_server`/`remove_server` work on a live agent: ctor-identical
  key validation, duplicate key → `ValueError`; connect/discovery I/O never blocks the turn loop;
  the registry mutation applies immediately when the agent is idle and queues to the next turn
  boundary when a run is active (§5 discipline — never mid-turn); a 401 during add parks the handle
  in `needs_auth` with `auth_challenge` populated while the registration itself succeeds;
  `remove_server` unregisters that server's tools and drops the handle; each add/remove/state change
  emits the §7 meta frame; dynamic servers are torn down by `aclose()` like ctor-declared ones (E6)
  and are never persisted (E11 holds — a cold boot re-adds via the constructor spec).
  `reconcile(desired)` (MC-D14) composes these verbs declaratively: keys are the identity
  (unchanged-key spec changes are no-ops), a no-op diff emits nothing, and a reconcile during an
  active run queues the whole diff to the boundary.
- **E15 — Model-visible surface changes.** Both MC-D13 mechanisms are always on, no configuration:
  (a) every applied `McpToolDiff` (dynamic add/remove, reconnect/refresh discovery drift,
  enable/disable) renders a system-note context block spliced into the next model-bound user
  content — no notice on the initial `initialize()` surface, and no standalone transcript message;
  (b) `mcp_status` is auto-registered whenever an `McpToolSource` exists, answers from `statuses()`
  ground truth, survives registry swaps and profile switches via the canonical recompose, and never
  exposes credentials. Registry reconciliation itself is a transactional whole-registry swap
  (MC-D12) — `ToolRegistry` stays append-only, and an in-flight turn keeps servicing the old
  registry object while the swap lands for the next turn.

## 9. Cross-subsystem dependencies

| Subsystem | Dependency |
|---|---|
| `tools` | `Toolish` registration seam, `ToolSchema`, `ToolResultEnvelope`, `ctx.emit_capped`, executor/confirmation attrs — consumed verbatim, no changes |
| `agent-loop-hooks` | hook firing for compiled tools (E1) — no new hook kinds (catalog stays LOCKED) |
| `relay-await` | confirmation plane for `confirm_destructive` (E4) — no new mechanism |
| `media-backend` | image/audio result landing (E3, R16) |
| `streaming-and-meta` | `mcp_server_state` MetaEnvelope (§7) — one new meta body type |
| `session-control` | `aclose()` on actor teardown (E6) |
| `core` (profiles) | boot-profile seeding; switch re-registration via the shared canonical recompose (E10, MC-D12) |
| `fork-reset` / `storage` | explicit NON-dependency: nothing persisted (E11) |

## 10. Decisions (2026-07-03 design discussion — to be ledgered as MC-*)

- **MC-D1 — Connect timing: eager**, in `initialize()`, concurrent, failure-isolated, `required=`
  escape hatch. Lazy connect rejected: schemas must exist at first render.
- **MC-D2 — Naming: `mcp__{server_key}__{remote_name}`** (ecosystem-standard). Clean names with
  collision detection rejected: breaks per-server hook scoping and risks collisions.
- **MC-D3 — Dedicated `mcp_servers=` ctor kwarg** (not a `Toolish`/bundle spelling): bundles expand
  synchronously at registration; MCP discovery is async. An async-bundle notion would change the
  tools contract for one consumer — rejected.
- **MC-D4 — `listChanged` notifications deferred.** v1 does not subscribe; `refresh(name)` +
  reconnect-time re-discovery cover drift, with registry reconciliation only at turn boundaries.
  Mid-run registry mutation interacts with checkpoint/fork semantics — revisit post-v1.
- **MC-D5 — Connect, reconnect, and auth are first-class v1 mechanisms** (§3/§4): backoff policy,
  state machine, single-flight, 401-refresh-retry-once, `needs_auth` quiescence, and the §4
  built-in auth providers ship in the first cut — not deferred.
- **MC-D6 — Browser-interactive OAuth stays consumer-side** behind `McpAuthProvider` +
  `needs_auth`/`reconnect()`; no token persistence in the library (no schema change). Revisit only
  if multiple consumers need it. **Amended by MC-D9** (the revisit trigger fired: nova's end users
  connect OAuth-guarded servers on the go): protocol mechanics moved into the library; the
  user-facing flow and token persistence remain consumer-side.
- **MC-D7 — Optional dependency**: `agent-base[mcp]` extra, lazy import, actionable ctor
  `ImportError`.
- **MC-D8 — Dynamic `add_server`/`remove_server` are first-class v1** (§3/§7). Driven by the
  application-layer requirement that end users connect new MCP servers mid-session. The alternative —
  evict + rebuild the resident session to pick up a new ctor spec — was rejected: it kills
  resident-session UX and is absurdly heavy for "user clicked Connect". Registry reconciliation
  reuses the §5 turn-boundary diff discipline; no second mechanism.
- **MC-D9 — OAuth protocol mechanics live in the library** (`mcp/oauth.py`, amends MC-D6):
  split-phase helpers (`discover` / `register_client` / `build_authorize_url` / `exchange_code` /
  `refresh`) + `TokenSet`/`TokenStore` + the `OAuthTokenAuth` provider. The consumer keeps redirect
  UX, the callback endpoint, and encrypted token persistence. Wrapping the `mcp` SDK's
  `OAuthClientProvider` directly was rejected — its inline redirect/callback callables model a CLI,
  not a web backend completing authorization out-of-band across HTTP requests (its primitives are
  reused internally where they fit). Still zero library persistence and zero schema change.
  **Refined by the 2026-07-03 SDK spike (mcp 1.28.1):** the interactive handlers are optional and
  their absence raises a typed `OAuthFlowError`, so `OAuthTokenAuth` is implemented by wrapping
  `OAuthClientProvider(handlers=None)` for the non-interactive legs; `TokenStore` is shape-aligned
  with the SDK's `TokenStorage` (4 methods); the split-phase helpers compose
  `mcp.client.auth.utils` + `mcp.shared.auth` models + `PKCEParameters` rather than reimplementing
  the RFCs. The split-phase *split* itself stands unchanged. **Further refined by the 2026-07-03
  spec review (rev 2025-11-25):** CIMD URL-client_ids ahead of DCR, 403 `insufficient_scope`
  scope-challenge classification, the PKCE-support refusal check, and the optional `refresh_lock`
  on `OAuthTokenAuth` — see the §4 spec-review deltas.
- **MC-D10 — `probe()` ships v1** (§7): agent-free validate/preview/auth-detect for consumer
  "Add server" UIs — one bounded call, registers nothing, persists nothing.
- **MC-D11 — Cookie/header-session auth via `SessionHeadersAuth`** (callback-based, v1): a header
  set with consumer-owned refresh subsumes cookie-session logins (username/password stored
  consumer-side, encrypted), rotating keys, and signed headers. A store-based spelling (a
  `TokenStore` analog for header sets) was rejected — persistence belongs inside the consumer
  callback; no new protocol. Boundary stances (§4): unauthorized = HTTP 401 only; the transport
  cookie jar is out of contract — provider headers are authoritative.
- **MC-D12 — Registry reconciliation = whole-registry swap** through ONE canonical recompose
  function owned by the agent and shared by profile switches and MCP diffs; `ToolRegistry` stays
  append-only (no changes — §9 holds). Adding in-place `unregister_tools()` was rejected: it puts
  mutation semantics (unknown-name, in-flight-call, partial-failure states) into the most-consumed
  subsystem's contract forever, while the swap is transactional by construction and matches the
  existing `reconfigure()` idiom. Compiled MCP callables carry `__mcp_server__ = server_key` so
  composition filters deterministically instead of copying from the old registry by heuristic.
- **MC-D13 — Model-visible surface changes, always on, no flags**: (a) a boundary **change notice**
  — every applied `McpToolDiff` renders a system-note block spliced into the next model-bound user
  content; (b) the **`mcp_status`** native tool, auto-registered whenever an `McpToolSource`
  exists, answering from `statuses()`. Rationale: schemas-per-request make additions visible, but
  the transcript preserves stale capability claims after removals (and stale denials after
  additions) — the model needs an explicit context footprint for both directions, plus a
  ground-truth introspection verb for direct user questions. Flags were considered and dropped:
  the mechanisms are cheap, and a silent surface change is never the right default.
- **MC-D14 — `reconcile(desired)` ships v1** (§3/§7): the declarative diff-to-set verb, composed
  entirely from `add_server`/`remove_server`, for consumers whose frontends declare the complete
  attachment set per request (nova's `/run` field). Keys are the identity — existing keys with a
  changed spec are untouched. Keeping the diff loop consumer-side was rejected: every consumer
  would reimplement the same edge cases (duplicate adds, remove-while-connecting, no-op diffs)
  that belong in one canonical, specced place.

## 11. Testing & migration note

Interface specs (`tests/interface/mcp/`) run against an **in-process fake MCP server over memory
streams** (the `mcp` SDK supports paired memory transports) — no subprocesses, no network, no test
flakiness; stdio spawn/kill gets one integration-marked test. Reconnect/backoff specs drive the
state machine with injected clocks. Dynamic-registration specs (E14) exercise `add_server`/
`remove_server` against the fake server, including the queue-to-turn-boundary case (add during an
active run → tools appear only at the next boundary) and needs_auth-on-add. `reconcile()`
(MC-D14) gets diff-matrix specs: add-only, remove-only, mixed, no-op (zero meta frames, zero
change notice), unchanged-key-with-changed-spec is a no-op, and reconcile-during-active-run
queues the whole diff to the boundary. `mcp/oauth.py` helpers
are unit-tested against a **fake authorization server** (httpx `MockTransport`) covering discovery,
dynamic client registration, PKCE exchange, and refresh; `probe()` gets its own spec (connected /
needs_auth-with-challenge / failed). `SessionHeadersAuth` gets a cookie-expiry spec: the fake
server invalidates the session mid-run → exactly one `headers_cb` re-invocation under N parallel
calls, then per-call retry-once (the §4 single-flight contract on the cookie path). MC-D12/D13
specs: profile switch after connect preserves the MCP surface (the E10 recompose regression);
diff application splices exactly one change notice into the next turn's user content (and none on
boot or on a no-op diff); `mcp_status` is present iff `mcp_servers` is configured, reports
`statuses()` truthfully after a disconnect, and its output is credential-free.

**Spike-resolved facts (2026-07-03, mcp 1.28.1 — pin `mcp>=1.28,<2` on the extra):** the fixture
entry point is `mcp.shared.memory.create_connected_server_and_client_session` (live-verified:
`destructiveHint` annotations, JSON-Schema pass-through, `structuredContent`, `isError`);
HTTP-level specs (401s, headers, cookies) run against `httpx.MockTransport` injected through
`streamable_http_client(url, http_client=...)` — `streamablehttp_client` is deprecated in 1.28; a
401 surfaces as `ExceptionGroup[httpx.HTTPStatusError]` with `.response` (status +
`WWW-Authenticate`) intact, and a call-time 401 tears down the whole transport — hence the §4
two-layer contract (py3.10 needs the `exceptiongroup` backport, already shipped via anyio); stdio
child termination verified clean on win32 (spawn → serve → context exit → child gone).
No DB columns, no `LIBRARY_SCHEMA_VERSION` bump, no migration.
nova_backend consumes via the editable source — its suite runs in the same cut (Living-Spec
Discipline); the new ctor kwarg is additive, so no breaking surface for existing consumers.

**Spec-review checks (rev 2025-11-25 — RESOLVED at implementation, 2026-07-03):** mcp 1.28.1
ships CIMD natively (`mcp.client.auth.utils.should_use_client_metadata_url` /
`create_client_info_from_metadata_url`; `client_metadata_url` ctor param) — `register_client`
composes it. A 404'd `Mcp-Session-Id` tears the transport down like any other death — the runner
supervisor's reconnect classification covers it with no special case. The listed specs landed in
`tests/interface/mcp/` (403 `insufficient_scope` challenge, PKCE refusal, `refresh_lock`
one-grant collapse).

## 12. Application-layer pattern (consumer cookbook)

How a consumer drives this subsystem when **end users** connect servers on the go and authenticate
before prompting. nova_backend is the reference consumer — its concrete design (tables, endpoints,
factory wiring) lives in `nova_backend/refactor_plans/mcp_integrations_design.md`; this section fixes
the division of labor and the canonical flows so both sides evolve against the same contract.

| Concern | library | consumer |
|---|---|---|
| Connection lifecycle, reconnect, 401 single-flight | §3–§4 | — |
| Add/remove server on a live agent | `add_server` / `remove_server` | invokes via its control plane |
| Detect auth requirement | `auth_challenge` on status; `probe()` | renders "Connect"/"Reconnect" UI from it |
| OAuth protocol mechanics | `mcp/oauth.py` (discovery, DCR, PKCE, exchange, refresh) | — |
| Redirect UX, callback endpoint, pending-auth state | — | owns (PendingAuth is serializable for this) |
| Token persistence | never (E8) | `TokenStore` impl, encrypted + tenant-scoped |
| Server registration persistence | never (E11) | its own registration store, specs sans secrets |
| Status push to frontend | `mcp_server_state` meta frame (§7) | SSE pass-through + pull via `statuses()` |

**Canonical flows** (consumer verbs in *italics*):

- **A — add, no OAuth** (open server, or a pasted API key → `StaticHeadersAuth`): *register* →
  `probe()` → *persist spec* → `add_server()` — or the key appears in the next `reconcile(desired)`
  for declarative consumers (MC-D14) → tools live at the next turn boundary → `mcp_server_state`
  frame.
- **B — add + OAuth before prompting**: *register* → `probe()` returns `needs_auth` +
  `auth_challenge` → `oauth.discover()` (+ `register_client()`) → `build_authorize_url()` → *stash
  PendingAuth, open browser popup* → user consents → *callback endpoint* → `exchange_code()` →
  *encrypt + store TokenSet* → `add_server()`/`reconcile()` with `OAuthTokenAuth` → connected.
- **C — token expiry mid-session**: 401 on `tools/call` → single-flight `on_unauthorized()` →
  `OAuthTokenAuth` refresh grant via the consumer's `TokenStore` — success is an invisible
  retry-once; failure is ONE `needs_auth` transition + ONE meta frame → *frontend shows "Reconnect"*
  → flow B's authorize leg → *consumer calls* `reconnect(name)`.
- **D — cold session boot**: *factory loads the user's registrations* → builds `mcp_servers=` ctor
  kwarg (authorized servers only; un-authorized registrations drive UI state instead) → eager
  connect in `initialize()` as normal (§2).

**Model visibility is free for the consumer (MC-D13):** app-layer connects/disconnects become
visible to the model automatically — the library splices the change notice into the next turn and
ships `mcp_status`, so "am I connected to X?" is answered from ground truth with zero consumer
prompt engineering.

**Declarative consumers (nova's ratified pattern, 2026-07-03):** the frontend declares the
complete desired server set on each run request; the consumer's handler resolves keys against its
registrations, builds specs, calls `reconcile(desired)` (MC-D14) before the turn, and persists
the set as its own session config. Under this pattern integration *deletion* needs no
cross-session fan-out: the next declaration simply no longer contains the key, a stale
declaration of a deleted key is skipped at consumer spec-build time, and cold boot's
attachments ∩ authorized intersection completes the cleanup.
