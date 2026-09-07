#!/usr/bin/env bash
#
# Run the MA-Hard baselines end to end and write BASELINE-RESULTS.md.
#
# Schedule: the two GPU-hosted generators run one after the other on the H200
# pair, while the API model and the Claude Code agent (neither needs a GPU) run
# in parallel alongside them. Judging happens last, once the GPUs are free for
# CriticLean.
#
#   GPUs:    [ gpt-oss-120b ][ Qwen3-30B-A3B ]          [ CriticLean judge ]
#   no GPU:  [ gpt-5-mini ........ ][ claude code ..... ]
#
# Usage:
#   N_EXAMPLES=5 scripts/run_ma_hard_baselines.sh          # smoke test first
#   scripts/run_ma_hard_baselines.sh                       # full run
#   SKIP_AGENT=1 scripts/run_ma_hard_baselines.sh          # iterative only
#   ASSUME_YES=1 nohup scripts/run_ma_hard_baselines.sh &  # unattended
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------- #
# Configuration (override with environment variables)
# --------------------------------------------------------------------------- #
read -r -a PYTHON <<< "${PYTHON:-uv run python}"
read -r -a VLLM   <<< "${VLLM:-uv run vllm}"

# --- item selection: SUBSET_FILE now; FILTER once MA-Hard is a column/split
DATASET="${DATASET:-offendo/math-atlas}"
SPLIT="${SPLIT:-train}"
SUBSET_FILE="${SUBSET_FILE:-ma_hard_uuids.json}"
FILTER="${FILTER:-}"                       # e.g. FILTER="split=hard"
ITEM_TYPES="${ITEM_TYPES:-all}"            # space separated, or "all"
N_EXAMPLES="${N_EXAMPLES:-}"               # set for a smoke test
SEED="${SEED:-1337}"

# --- iterative (B1)
MAX_ROUNDS="${MAX_ROUNDS:-5}"
TEMPERATURE="${TEMPERATURE:-0.0}"
RETRY_TEMPERATURE="${RETRY_TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
RUN_CONTROL="${RUN_CONTROL:-1}"            # also run --max-rounds 1 single-pass control

GPT_OSS="${GPT_OSS:-openai/gpt-oss-120b}"
QWEN_MOE="${QWEN_MOE:-Qwen/Qwen3-30B-A3B-Thinking-2507}"
API_MODEL="${API_MODEL:-gpt-5-mini}"
API_URL="${API_URL:-https://api.openai.com/v1}"
API_CONCURRENCY="${API_CONCURRENCY:-10}"

# --- GPU serving
GPUS="${GPUS:-1,2}"
TP_SIZE="${TP_SIZE:-2}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
SERVER_BOOT_TIMEOUT="${SERVER_BOOT_TIMEOUT:-2400}"   # the 120b takes a while to load

# --- agentic (A1)
LEAN_PROJECT="${LEAN_PROJECT:-$HOME/src/mathatlas-formalization}"
CLAUDE_MODEL="${CLAUDE_MODEL:-sonnet}"
MAX_BUDGET_USD="${MAX_BUDGET_USD:-0.75}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-900}"
AGENT_CONCURRENCY="${AGENT_CONCURRENCY:-1}"
MATH_ATLAS_MCP="${MATH_ATLAS_MCP:-}"       # your MathAtlas MCP config, when ready
BUILD_PROJECT="${BUILD_PROJECT:-1}"

# --- judging
JUDGE_MODEL="${JUDGE_MODEL:-criticleangpt-qwen3-32b-rl}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-}"   # local path / HF id of the CriticLean checkpoint
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-20}"

# --- phases
SKIP_GPT_OSS="${SKIP_GPT_OSS:-0}"
SKIP_QWEN="${SKIP_QWEN:-0}"
SKIP_API="${SKIP_API:-0}"
SKIP_AGENT="${SKIP_AGENT:-0}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"

# --- output
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-outputs}"
LOG_DIR="${LOG_DIR:-logs/ma-hard-$STAMP}"
RESULTS_MD="${RESULTS_MD:-BASELINE-RESULTS.md}"

mkdir -p "$OUT_DIR/iterative" "$OUT_DIR/agentic" "$LOG_DIR"
# Lanes run in subshells, so completed runs are recorded in a file rather than
# an array -- that is what the judging phase iterates over.
MANIFEST="$LOG_DIR/manifest.txt"
: > "$MANIFEST"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[%s] WARN\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '\033[1;31m[%s] FATAL\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

SERVER_PID=""
BG_PIDS=()

stop_server() {
  [[ -z "$SERVER_PID" ]] && return 0
  log "Stopping server (pid $SERVER_PID)"
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  sleep 10   # let the GPUs drain before the next model loads
  return 0
}

cleanup() {
  local rc=$?
  stop_server || true
  if (( ${#BG_PIDS[@]} )); then
    for pid in "${BG_PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        warn "Killing background lane $pid"
        pkill -P "$pid" 2>/dev/null || true
        kill -TERM "$pid" 2>/dev/null || true
      fi
    done
  fi
  exit $rc
}
trap cleanup EXIT INT TERM

# Dataset selection flags, shared by every runner.
selection_args() {
  local args=(--dataset "$DATASET" --split "$SPLIT" --seed "$SEED")
  local t f
  for t in $ITEM_TYPES; do args+=(--item-type "$t"); done
  if [[ -n "$FILTER" ]]; then
    for f in $FILTER; do args+=(--filter "$f"); done
  fi
  if [[ -n "$SUBSET_FILE" && -f "$SUBSET_FILE" ]]; then args+=(--subset-file "$SUBSET_FILE"); fi
  if [[ -n "$N_EXAMPLES" ]]; then args+=(--n-examples "$N_EXAMPLES"); fi
  printf '%s\n' "${args[@]}"
}

start_server() {  # start_server <model_path> <served_name>
  local model="$1" served="$2" waited=0
  log "Serving $served on GPUs $GPUS (port $PORT)"
  CUDA_VISIBLE_DEVICES="$GPUS" setsid "${VLLM[@]}" serve "$model" \
      --served-model-name "$served" \
      --tensor-parallel-size "$TP_SIZE" \
      --max-model-len "$MAX_MODEL_LEN" \
      --port "$PORT" >"$LOG_DIR/server.$(basename "$served").log" 2>&1 &
  SERVER_PID=$!
  until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -30 "$LOG_DIR/server.$(basename "$served").log" >&2
      die "Server for $served died during startup."
    fi
    sleep 5; waited=$((waited + 5))
    if (( waited % 120 == 0 )); then log "  ...still loading $served (${waited}s)"; fi
    if (( waited >= SERVER_BOOT_TIMEOUT )); then die "Server for $served did not come up in ${SERVER_BOOT_TIMEOUT}s."; fi
  done
  log "$served is up after ${waited}s"
}

run_iterative() {  # run_iterative <model> <url|""> <tag> <rounds> [extra args...]
  local model="$1" url="$2" tag="$3" rounds="$4"; shift 4
  local out="$OUT_DIR/iterative/${tag}.json"
  if [[ -s "$out" ]]; then
    log "Skipping $tag (output exists: $out)"
    echo "$out" >> "$MANIFEST"
    return 0
  fi

  local args=(); mapfile -t args < <(selection_args)
  local url_args=(); if [[ -n "$url" ]]; then url_args=(--model-url "$url"); fi

  log "B1: $tag (max-rounds=$rounds)"
  if "${PYTHON[@]}" benchmarks/iterative/run_iterative.py \
        --model "$model" "${url_args[@]}" "${args[@]}" \
        --max-rounds "$rounds" \
        --temperature "$TEMPERATURE" \
        --retry-temperature "$RETRY_TEMPERATURE" \
        --top-p "$TOP_P" \
        --max-tokens "$MAX_TOKENS" \
        --skip-judge \
        --output "$out" \
        "$@" >"$LOG_DIR/$tag.log" 2>&1; then
    log "B1: $tag done -> $out"
    echo "$out" >> "$MANIFEST"
  else
    warn "B1 run '$tag' FAILED; see $LOG_DIR/$tag.log"
    tail -20 "$LOG_DIR/$tag.log" >&2 || true
  fi
  return 0
}

# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
log "Preflight checks"

if [[ -n "$SUBSET_FILE" && ! -f "$SUBSET_FILE" && -z "$FILTER" && -z "$N_EXAMPLES" ]]; then
  warn "SUBSET_FILE '$SUBSET_FILE' not found and no FILTER set -- this would run the WHOLE dataset."
  die  "Point SUBSET_FILE at your MA-Hard uuid list, set FILTER=split=hard, or set N_EXAMPLES."
fi

"${PYTHON[@]}" - <<'EOF' || die "blv is not reachable. Is the blv docker stack up?"
import sys
sys.path.insert(0, "benchmarks")
import common
out = common.verify_batch(["theorem preflight (n : Nat) : n = n := by sorry"], timeout=120)
assert out[0].get("verified"), out[0]
print("blv OK")
EOF

if [[ "$SKIP_AGENT" != "1" ]]; then
  command -v claude >/dev/null || die "claude CLI not found on PATH."
  command -v uvx    >/dev/null || warn "uvx not found; the lean-lsp MCP will fail to start."
  if [[ ! -d "$LEAN_PROJECT" ]]; then warn "LEAN_PROJECT '$LEAN_PROJECT' missing; it will be created with 'lake init'."; fi
  if [[ -z "$MATH_ATLAS_MCP" ]]; then warn "MATH_ATLAS_MCP unset: the agent gets lean-lsp only, no dependency/context tools."; fi
fi

if [[ "$SKIP_API" != "1" && -z "${OPENAI_API_KEY:-}" ]]; then
  warn "OPENAI_API_KEY unset -- skipping the $API_MODEL lane."
  SKIP_API=1
fi

if [[ "$SKIP_JUDGE" != "1" && -z "$JUDGE_MODEL_PATH" ]]; then
  warn "JUDGE_MODEL_PATH unset -- generation will run but nothing will be judged."
  warn "Set it and re-run (finished runs are skipped), or judge later with benchmarks/judge_results.py."
  SKIP_JUDGE=1
fi

# Spend confirmation: the agent lane is the expensive one.
if [[ "$SKIP_AGENT" != "1" ]]; then
  n_items="${N_EXAMPLES:-700}"
  worst_case="$(awk -v n="$n_items" -v c="$MAX_BUDGET_USD" 'BEGIN {printf "%.2f", n * c}')"
  log "Agent lane: ${N_EXAMPLES:-all} items, \$$MAX_BUDGET_USD/item cap => worst case ~\$$worst_case"
  if [[ "${ASSUME_YES:-0}" != "1" ]]; then
    if [[ -t 0 ]]; then
      read -r -p "Proceed? [y/N] " reply
      [[ "$reply" =~ ^[Yy]$ ]] || die "Aborted."
    else
      die "Non-interactive: set ASSUME_YES=1 to confirm agent spend, or SKIP_AGENT=1."
    fi
  fi
fi

log "Logs: $LOG_DIR"

# --------------------------------------------------------------------------- #
# Lanes that need no GPU -- launched first, run alongside the GPU phases
# --------------------------------------------------------------------------- #
if [[ "$SKIP_API" != "1" ]]; then
  log "Launching $API_MODEL lane in the background"
  (
    run_iterative "$API_MODEL" "$API_URL" "$API_MODEL.ma-hard" "$MAX_ROUNDS" --concurrency "$API_CONCURRENCY"
    if [[ "$RUN_CONTROL" == "1" ]]; then
      run_iterative "$API_MODEL" "$API_URL" "$API_MODEL.control" 1 --concurrency "$API_CONCURRENCY"
    fi
  ) &
  BG_PIDS+=($!)
fi

if [[ "$SKIP_AGENT" != "1" ]]; then
  log "Launching Claude Code lane in the background"
  AGENT_OUT="$OUT_DIR/agentic/claude-code-$CLAUDE_MODEL.ma-hard.json"
  (
    args=(); mapfile -t args < <(selection_args)
    extra=()
    if [[ -n "$MATH_ATLAS_MCP" ]]; then extra+=(--mcp-config "$MATH_ATLAS_MCP"); fi
    if [[ "$BUILD_PROJECT" != "1" ]]; then extra+=(--no-build-project); fi
    if "${PYTHON[@]}" benchmarks/agentic/run_claude_code.py \
          "${args[@]}" \
          --project "$LEAN_PROJECT" \
          --model "$CLAUDE_MODEL" \
          --max-budget-usd "$MAX_BUDGET_USD" \
          --timeout "$AGENT_TIMEOUT" \
          --concurrency "$AGENT_CONCURRENCY" \
          --resume --skip-judge \
          --output "$AGENT_OUT" \
          "${extra[@]}" >"$LOG_DIR/claude-code.log" 2>&1; then
      echo "$AGENT_OUT" >> "$MANIFEST"
    else
      echo "Claude Code lane FAILED; see $LOG_DIR/claude-code.log" >&2
      tail -20 "$LOG_DIR/claude-code.log" >&2 || true
    fi
  ) &
  BG_PIDS+=($!)
fi

# --------------------------------------------------------------------------- #
# GPU phase 1: gpt-oss-120b
# --------------------------------------------------------------------------- #
if [[ "$SKIP_GPT_OSS" != "1" ]]; then
  start_server "$GPT_OSS" "$GPT_OSS"
  run_iterative "$GPT_OSS" "http://localhost:$PORT/v1" "gpt-oss-120b.ma-hard" "$MAX_ROUNDS"
  if [[ "$RUN_CONTROL" == "1" ]]; then
    run_iterative "$GPT_OSS" "http://localhost:$PORT/v1" "gpt-oss-120b.control" 1
  fi
  stop_server
fi

# --------------------------------------------------------------------------- #
# GPU phase 2: Qwen3-30B-A3B (MoE)
# --------------------------------------------------------------------------- #
if [[ "$SKIP_QWEN" != "1" ]]; then
  start_server "$QWEN_MOE" "$QWEN_MOE"
  run_iterative "$QWEN_MOE" "http://localhost:$PORT/v1" "qwen3-30b-a3b.ma-hard" "$MAX_ROUNDS"
  if [[ "$RUN_CONTROL" == "1" ]]; then
    run_iterative "$QWEN_MOE" "http://localhost:$PORT/v1" "qwen3-30b-a3b.control" 1
  fi
  stop_server
fi

# --------------------------------------------------------------------------- #
# Wait for the non-GPU lanes before handing the GPUs to the judge
# --------------------------------------------------------------------------- #
if (( ${#BG_PIDS[@]} )); then
  log "Waiting for background lanes (${BG_PIDS[*]})"
  for pid in "${BG_PIDS[@]}"; do
    wait "$pid" || warn "Background lane $pid exited non-zero"
  done
  BG_PIDS=()
  log "Background lanes finished"
fi

# --------------------------------------------------------------------------- #
# GPU phase 3: judge everything that was produced
# --------------------------------------------------------------------------- #
if [[ "$SKIP_JUDGE" != "1" ]]; then
  mapfile -t TO_JUDGE < <(sort -u "$MANIFEST")
  if (( ${#TO_JUDGE[@]} == 0 )); then
    warn "Nothing was produced; skipping the judge."
  else
    start_server "$JUDGE_MODEL_PATH" "$JUDGE_MODEL"
    for out in "${TO_JUDGE[@]}"; do
      if [[ ! -s "$out" ]]; then warn "No output at $out; nothing to judge."; continue; fi
      tag="$(basename "$out" .json)"
      log "Judging $tag"
      if ! "${PYTHON[@]}" benchmarks/judge_results.py \
            --input "$out" \
            --judge-model "$JUDGE_MODEL" \
            --judge-model-url "http://localhost:$PORT/v1" \
            --judge-concurrency "$JUDGE_CONCURRENCY" >"$LOG_DIR/judge.$tag.log" 2>&1; then
        warn "Judging '$tag' FAILED; see $LOG_DIR/judge.$tag.log"
        tail -20 "$LOG_DIR/judge.$tag.log" >&2 || true
      fi
    done
    stop_server
  fi
fi

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
log "Writing $RESULTS_MD"
"${PYTHON[@]}" benchmarks/summarize_results.py "$OUT_DIR/iterative" "$OUT_DIR/agentic" --output "$RESULTS_MD"
log "Done. Logs in $LOG_DIR"
