#!/usr/bin/env bash
# Conductor setup script. Prepares a new workspace from .claude/worktree-setup.json,
# the same file the pwsh worktree setup reads, so there is one list to keep up to date.
#
#   symlinkFromSource   files linked from the main checkout (CONDUCTOR_ROOT_PATH)
#   copyFromSource      files copied from the main checkout
#   junctionFromSource  directories linked from the main checkout (a symlink on macOS)
#   setup               commands run in the workspace, in order
#
# The "shell" key is ignored: commands run in bash here, so keep them shell-neutral.
#
# Arguments are the Homebrew formulae those commands need (e.g. uv, node@22). Each one is
# installed if missing and upgraded if present, before the commands run.
set -euo pipefail

root="${CONDUCTOR_ROOT_PATH:?not set; run this as a Conductor setup script}"
workspace="${CONDUCTOR_WORKSPACE_PATH:-$PWD}"
config=".claude/worktree-setup.json"

# Conductor reuses PATH from a login shell, which can miss tools if that shell is slow to start.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

cd "$workspace"
jq empty "$config"

ensure_formulae() {
  (( $# )) || return 0
  if ! command -v brew >/dev/null; then
    echo "error: Homebrew is needed to install $*; see https://brew.sh" >&2
    exit 1
  fi
  export HOMEBREW_NO_ENV_HINTS=1
  local prefix formula
  prefix="$(brew --prefix)"
  for formula in "$@"; do
    if brew list --formula --versions "$formula" >/dev/null 2>&1; then
      echo "==> brew upgrade $formula"
      # Another workspace's setup may hold Homebrew's lock; the installed version still works.
      brew upgrade "$formula" || echo "  warning: could not upgrade $formula, using the installed version" >&2
    else
      echo "==> brew install $formula"
      brew install "$formula"
    fi
    # Put the formula's own bin first: versioned ones such as node@22 are keg-only, and a newer
    # unversioned formula could own the links in Homebrew's bin.
    export PATH="$prefix/opt/$formula/bin:$PATH"
  done
}

entries() {
  jq -r --arg key "$1" '.[$key] // [] | .[]' "$config"
}

# Gitignore patterns like "tmp/" match only real directories, not symlinks to them.
ensure_ignored() {
  git check-ignore -q "$1" && return 0
  echo "/$1" >> "$(git rev-parse --git-common-dir)/info/exclude"
  echo "    added /$1 to .git/info/exclude"
}

# bring_in <link|copy|link-dir> <path>
bring_in() {
  local mode=$1 path=$2
  local source="$root/$path"

  if [[ -e "$path" && ! -L "$path" ]]; then
    echo "  skip $path: already exists in the workspace"
    return 0
  fi
  if [[ ! -e "$source" ]]; then
    if [[ $mode != link-dir ]]; then
      echo "  warning: $source does not exist, $path not set up" >&2
      return 0
    fi
    mkdir -p "$source"
  fi

  [[ -L "$path" ]] && rm "$path"
  mkdir -p "$(dirname "$path")"
  if [[ $mode == copy ]]; then
    cp -R "$source" "$path"
  else
    ln -s "$source" "$path"
  fi
  echo "  $mode $path"
  ensure_ignored "$path"
}

if [[ "$(cd "$root" && pwd -P)" == "$(pwd -P)" ]]; then
  echo "Workspace is the main checkout; skipping links and copies."
else
  echo "Bringing in local files from $root"
  while IFS= read -r path; do bring_in link "$path"; done < <(entries symlinkFromSource)
  while IFS= read -r path; do bring_in copy "$path"; done < <(entries copyFromSource)
  while IFS= read -r path; do bring_in link-dir "$path"; done < <(entries junctionFromSource)
fi

ensure_formulae "$@"

while IFS= read -r command; do
  echo "==> $command"
  bash -c "$command" </dev/null
done < <(entries setup)

echo "Workspace ready."
