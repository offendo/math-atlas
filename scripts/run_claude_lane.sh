#!/usr/bin/env bash
#
# The Claude lane of run_iclr_experiments.sh as a standalone, restartable script.
#
# Exists because the first main run's Claude lane hit the subscription session
# limit and returned empty generations (fixed in generators.py / run_claude_code.py:
# limits are now waited out). This finishes the lane with the fixed code:
#   * Sonnet single-pass arms: run if missing; if present but with empty rows
#     (rejected calls), regenerate just those rows (--fill-empty-from)
#   * the A1 agent arms, then Sonnet compile-repair K=5
# and hands its outputs to the main run's judging phase:
#   MAIN_LOG_DIR=<main run's LOG_DIR>  -> appends to generated.txt, touches lanes-done/claude
#
#   MAIN_LOG_DIR=~/src/math-atlas/logs/iclr-main nohup scripts/run_claude_lane.sh > lane.log 2>&1 &
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PY:-/home/npatel37/src/math-atlas/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/npatel37/src/math-atlas}"
OUT_DIR="${OUT_DIR:-$DATA_ROOT/outputs/iclr}"
MAIN_LOG_DIR="${MAIN_LOG_DIR:?set MAIN_LOG_DIR to the main run LOG_DIR}"
LOG_DIR="${LOG_DIR:-$MAIN_LOG_DIR/claude-lane}"
DATASET="${DATASET:-offendo/math-atlas-official}"; SPLIT="${SPLIT:-hard}"; SEED="${SEED:-1337}"
CLAUDE_MODEL="${CLAUDE_MODEL:-sonnet}"
CLAUDE_CONCURRENCY="${CLAUDE_CONCURRENCY:-16}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
AGENT_ROOT="${AGENT_ROOT:-$HOME/src/ma-hard-iclr}"
MATHATLAS_PROJECT="${MATHATLAS_PROJECT:-$HOME/src/mathatlas-formalization}"
AGENT_CONCURRENCY="${AGENT_CONCURRENCY:-24}"
MAX_BUDGET_USD="${MAX_BUDGET_USD:-0.75}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-900}"
AGENT_ARMS="${AGENT_ARMS:-dep none opt}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$LOG_DIR" "$OUT_DIR/iterative" "$OUT_DIR/agentic"

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[%s] WARN %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
handoff() { echo "$1" >> "$MAIN_LOG_DIR/generated.txt"; }

n_empty() {  # rows with an empty generation in any round
  "$PY" - "$1" <<'EOF'
import json, sys
rows = json.load(open(sys.argv[1]))
print(sum(1 for r in rows if not r.get("rounds") or any(not str(x.get("raw_output") or "").strip() for x in r["rounds"])))
EOF
}

sonnet() {  # sonnet <tag> [run_iterative args...]
  local tag="$1"; shift
  local out="$OUT_DIR/iterative/$tag.json" fill=()
  if [[ -s "$out" ]]; then
    local e; e=$(n_empty "$out")
    if [[ "$e" == 0 ]]; then log "skip $tag (complete)"; handoff "$out"; return 0; fi
    log "$tag: $e empty rows -> regenerating them"
    cp "$out" "$out.with-empties.bak"
    fill=(--fill-empty-from "$out.with-empties.bak")
  fi
  log "run $tag"
  if "$PY" benchmarks/iterative/run_iterative.py --model "$CLAUDE_MODEL" --backend claude-cli \
        --concurrency "$CLAUDE_CONCURRENCY" --dataset "$DATASET" --split "$SPLIT" --seed "$SEED" \
        --max-tokens "$MAX_TOKENS" --skip-judge --output "$out" "${fill[@]}" "$@" \
        > "$LOG_DIR/$tag.log" 2>&1 && [[ -s "$out" ]]; then
    log "done $tag (empty rows now: $(n_empty "$out"))"; handoff "$out"
  else
    warn "$tag FAILED (see $LOG_DIR/$tag.log)"
  fi
}

agent() {  # agent <arm>
  local arm="$1" out="$OUT_DIR/agentic/claude-code-$CLAUDE_MODEL.$1.json" extra=()
  if [[ -s "$out" ]]; then log "skip agent $arm (exists)"; handoff "$out"; return 0; fi
  case "$arm" in
    dep)  extra=(--mathatlas-project "$MATHATLAS_PROJECT" --require-dependencies
                 --prompt-file benchmarks/agentic/prompts/agent_task_dependency_aware.txt) ;;
    opt)  extra=(--mathatlas-project "$MATHATLAS_PROJECT") ;;
    none) extra=(--allowed-tools "Read,Write,Edit,Glob,Grep,Bash,mcp__lean-lsp") ;;
  esac
  log "agent $arm"
  if "$PY" benchmarks/agentic/run_claude_code.py --dataset "$DATASET" --split "$SPLIT" --seed "$SEED" \
        --project "$AGENT_ROOT/$arm" --no-build-project --model "$CLAUDE_MODEL" \
        --max-budget-usd "$MAX_BUDGET_USD" --timeout "$AGENT_TIMEOUT" --concurrency "$AGENT_CONCURRENCY" \
        --resume --skip-judge --output "$out" "${extra[@]}" > "$LOG_DIR/agent-$arm.log" 2>&1 && [[ -s "$out" ]]; then
    log "done agent $arm"; handoff "$out"
  else
    warn "agent $arm FAILED (see $LOG_DIR/agent-$arm.log)"
  fi
}

for mode in none both random; do
  sonnet "sonnet.sp.dep-$mode" --max-rounds 1 --dependency-context "$mode" --temperature 0.0
done
for arm in $AGENT_ARMS; do
  [[ "$arm" == opt ]] && sonnet "sonnet.k5.dep-none" --max-rounds 5 --temperature 0.0 --retry-temperature 0.7
  agent "$arm"
done
[[ " $AGENT_ARMS " == *" opt "* ]] || sonnet "sonnet.k5.dep-none" --max-rounds 5 --temperature 0.0 --retry-temperature 0.7
touch "$MAIN_LOG_DIR/lanes-done/claude"
log "Claude lane finished; handed off to the main run's judging phase."
