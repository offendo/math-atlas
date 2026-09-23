#!/usr/bin/env bash
#
# Create Lake projects for the A1 agent arms, pinned to the same Lean/Mathlib
# release that `blv` verifies against, so the agent's own compile checks agree
# with the scorer.
#
# One base project holds the Mathlib build; each arm gets its own project whose
# `.lake/packages` is a symlink to the base's, so arms never see each other's
# item modules (the A1 runner lets agents import earlier items *within* a run)
# but share the ~6 GB Mathlib build.
#
# Usage:
#   scripts/setup_agent_projects.sh [ROOT] [ARM ...]
#   scripts/setup_agent_projects.sh ~/src/ma-hard-iclr none dep
#
set -euo pipefail

ROOT="${1:-$HOME/src/ma-hard-iclr}"; shift || true
ARMS=("$@")
LEAN_VERSION="${LEAN_VERSION:-v4.28.0}"      # must match ghcr.io/offendo/blv:<tag>
LIB="${LIB:-MaHard}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

write_project() {  # write_project <dir>
  local dir="$1"
  mkdir -p "$dir/$LIB"
  echo "leanprover/lean4:$LEAN_VERSION" > "$dir/lean-toolchain"
  cat > "$dir/lakefile.toml" <<EOF
name = "mahard"
version = "0.1.0"
defaultTargets = ["$LIB"]

[[require]]
name = "mathlib"
scope = "leanprover-community"
rev = "$LEAN_VERSION"

[[lean_lib]]
name = "$LIB"
EOF
  [[ -f "$dir/$LIB.lean" ]] || printf 'import Mathlib\n' > "$dir/$LIB.lean"
}

BASE="${BASE:-$ROOT/base}"          # reuse an existing Mathlib build if given
if [[ ! -f "$BASE/.lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean" ]]; then
  log "Building base project at $BASE (Lean/Mathlib $LEAN_VERSION)"
  write_project "$BASE"
  (cd "$BASE" && lake update && lake exe cache get && lake build)
else
  log "Base project already built: $BASE"
fi

for arm in "${ARMS[@]}"; do
  dir="$ROOT/$arm"
  log "Arm project: $dir"
  write_project "$dir"
  cp "$BASE/lake-manifest.json" "$dir/lake-manifest.json"
  mkdir -p "$dir/.lake"
  if [[ ! -e "$dir/.lake/packages" ]]; then ln -s "$BASE/.lake/packages" "$dir/.lake/packages"; fi
  (cd "$dir" && lake build)
done
log "Done."
