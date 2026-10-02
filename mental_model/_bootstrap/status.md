# Bootstrap status

Read this first when resuming. Working state for the mental-model bootstrap of this repo; this folder is deleted before the PR is marked ready. The procedure is `HANDOFF.md` in the mental-model kit, which is kept outside the repo and never committed.

## Where things stand

| | |
|---|---|
| Phase | 3: **waiting on Auro at Checkpoint 2** ([2-plot.md](2-plot.md)) |
| Branch | `AuroSoni/mental-model` (the Conductor workspace branch; the handoff's default name `mental-model-bootstrap` is not used) |
| PR | [#13](https://github.com/AuroSoni/agent-harness/pull/13), draft, base `dev` |
| Surveyed at | `71ecf49` (= `origin/dev`), 2 Oct 2026 |
| Session run by | Auro himself, so his chat answers count as sources |

## Product-level facts

- Product: **Nova**. Homes: `nova_backend` and `nova_excel_addin`; both carry the product map (Auro, Checkpoint 1, Q1). This repo is a provider only and has no `product/` folder.
- The settled product map is in the backend run: `nova_backend`, branch `AuroSoni/mental-model`, `mental_model/_bootstrap/1-characters-and-map.md` section 1 (draft PR `Project-Hedge/nova_backend#69`). The add-in run is on branch `mental-model-bootstrap` (draft PR `Project-Hedge/nova_excel_addin#87`).
- This repo provides two of the map's five contracts: the **`agent-base` package** (to the backend) and the **wire protocol** (to the add-in, through the backend).
- This repo is **public**; the other two are private. The model names Nova as the consumer and describes contracts only.

## Waiting on Auro

Checkpoint 2, in [2-plot.md](2-plot.md):

- **Part A1** (S1 to S12) and **Q1 to Q4** unblock the pilot.
- **Part A2** (S13 to S40) and **Q5 to Q13** are needed before the remaining files are written.

## Answers received

- **Checkpoint 1, 2 Oct 2026, in chat:** all of Q1 to Q15. Recorded verbatim in [1-characters-and-map.md](1-characters-and-map.md) section 5; what they settle is section 6; his statements are rows A1 to A11 of [sources.md](sources.md).
- **Checkpoint 2:** none yet. Copy chat answers into `2-plot.md` under an "Answers" heading, mark each S row in `sources.md` section B, and add his answers to the questions as new A rows.

## Next

1. Apply the Checkpoint 2 answers.
2. Pilot (Checkpoint 3): write `subsystems/session-actor.md` and `features/pause-and-resume.md` in full, following the writing rules in the kit's `templates/mental_model.CLAUDE.md`, using only confirmed whys. Ask Auro to review depth, structure, accuracy, names and missing whys.
3. Phases 5 to 8 as in the handoff. Templates are installed only in phase 5.

## Notes for whoever resumes

- Use Auro's names: a **run** is the agent's work on one prompt (`run_id`); a **turn** is a user turn or an agent turn; a **session** is one main agent id; a **step** is one provider call. The library class `Conversation` is one run's record. See section 6 of the Checkpoint 1 file.
- The turn loop is in `AnthropicAgent._resume_loop`. `AgentRuntime.run` raises `NotImplementedError`. Several docstrings and `CLAUDE.md` say otherwise; describe what is built.
- Left out of the model on Auro's instruction: planned work (`providers/any_llm/`, the workflows plan, the cost ledger plan) and the demos.
- Docs (phase 6): the model replaces `interface_plan/subsystems/`; `tests/interface/` stays; `AMENDMENTS.md` can be removed once the as-built model captures it. Propose each change and get a go-ahead before deleting anything.
- No review comments exist in this repo, so they are not a source.
- Nothing in the repo's code changes in this PR. Bugs and dead code found so far are listed at the end of `2-plot.md` and in `survey.md` section 5.
- Cross-repo follow-up, not edited from here: the backend run's draft map says the home is `nova_backend` alone, while Auro told this run and the add-in run that both `nova_backend` and `nova_excel_addin` are homes.

## Files in this folder

| File | What it is |
|---|---|
| [survey.md](survey.md) | Phase 1 findings, condensed |
| [sources.md](sources.md) | Why-to-source table: Auro's statements (A), candidates sent at Checkpoint 2 (B), dropped (C) |
| [1-characters-and-map.md](1-characters-and-map.md) | Checkpoint 1, **answered** |
| [2-plot.md](2-plot.md) | Checkpoint 2, **open**: 40 reasons to strike or confirm, 13 questions |
