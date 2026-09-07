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

# --- item selection: FILTER once MA-Hard is a column/split
DATASET="${DATASET:-offendo/math-atlas-official}"
SPLIT="${SPLIT:-hard}"
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
QWEN_MOE="${QWEN_MOE:-Qwen/Qwen3.6-35B-A3B}"
API_MODEL="${API_MODEL:-gpt-5-mini}"
API_URL="${API_URL:-https://api.openai.com/v1}"
API_CONCURRENCY="${API_CONCURRENCY:-10}"

# --- GPU serving (vLLM runs from the official docker image)
DOCKER="${DOCKER:-docker}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
VLLM_CONTAINER="${VLLM_CONTAINER:-ma-hard-vllm}"
VLLM_DOCKER_ARGS="${VLLM_DOCKER_ARGS:-}"             # extra docker args, e.g. --shm-size=32g
HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
GPUS="${GPUS:-1,2}"
TP_SIZE="${TP_SIZE:-2}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
SERVER_BOOT_TIMEOUT="${SERVER_BOOT_TIMEOUT:-2400}"   # the 120b takes a while to load

# --- agentic (A1)
LEAN_PROJECT="${LEAN_PROJECT:-$HOME/src/MathProjectTemplate}"
CLAUDE_MODEL="${CLAUDE_MODEL:-sonnet}"
MAX_BUDGET_USD="${MAX_BUDGET_USD:-0.50}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-900}"
AGENT_CONCURRENCY="${AGENT_CONCURRENCY:-10}"
MATHATLAS_PROJECT="${MATHATLAS_PROJECT:-$HOME/src/mathatlas-formalization}"   # provides the MathAtlas MCP
MATHATLAS_DATA="${MATHATLAS_DATA:-}"       # dataset JSON; defaults to the project's data/
MATHATLAS_TEXTBOOKS="${MATHATLAS_TEXTBOOKS:-}"   # .mmd dir; defaults to the project's data/
MATH_ATLAS_MCP="${MATH_ATLAS_MCP:-}"       # any *extra* MCP config JSON to merge in
BUILD_PROJECT="${BUILD_PROJECT:-1}"

# --- judging
JUDGE_MODEL="${JUDGE_MODEL:-m-a-p/CriticLeanGPT-Qwen3-32B-RL}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-$JUDGE_MODEL}"   # local path, if you have the weights on disk
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

SERVER_LOG=""
SERVER_UP=0
BG_PIDS=()

stop_server() {
  [[ "$SERVER_UP" == "0" ]] && return 0
  log "Stopping $VLLM_CONTAINER"
  # Capture the container log before --rm takes it away.
  [[ -n "$SERVER_LOG" ]] && "$DOCKER" logs "$VLLM_CONTAINER" >"$SERVER_LOG" 2>&1 || true
  "$DOCKER" stop -t 30 "$VLLM_CONTAINER" >/dev/null 2>&1 || true
  "$DOCKER" rm -f "$VLLM_CONTAINER" >/dev/null 2>&1 || true
  SERVER_UP=0
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
  if [[ -n "$N_EXAMPLES" ]]; then args+=(--n-examples "$N_EXAMPLES"); fi
  printf '%s\n' "${args[@]}"
}

start_server() {  # start_server <model_path> <served_name>
  local model="$1" served="$2" waited=0
  SERVER_LOG="$LOG_DIR/server.$(basename "$served").log"
  "$DOCKER" rm -f "$VLLM_CONTAINER" >/dev/null 2>&1 || true

  # `--gpus '"device=1,2"'` is the documented form: docker needs the inner
  # quotes so the comma is not read as an option separator.
  local docker_args=(
    run -d --rm --name "$VLLM_CONTAINER"
    --gpus "\"device=$GPUS\""
    --ipc=host                                  # TP needs a real /dev/shm
    -p "$PORT:$PORT"
    -v "$HF_CACHE:/root/.cache/huggingface"
    -e "HF_HOME=/root/.cache/huggingface"
    --entrypoint vllm                           # works whether or not the image already entrypoints `vllm serve`
  )
  if [[ -n "${HF_TOKEN:-}" ]]; then docker_args+=(-e "HF_TOKEN=$HF_TOKEN"); fi
  if [[ -n "$VLLM_DOCKER_ARGS" ]]; then
    local extra=(); read -r -a extra <<< "$VLLM_DOCKER_ARGS"; docker_args+=("${extra[@]}")
  fi
  docker_args+=(
    "$VLLM_IMAGE" serve "$model"
    --served-model-name "$served"
    --tensor-parallel-size "$TP_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --port "$PORT"
  )

  log "Serving $served in $VLLM_IMAGE on GPUs $GPUS (port $PORT)"
  "$DOCKER" "${docker_args[@]}" >/dev/null || die "Could not start $VLLM_CONTAINER."
  SERVER_UP=1

  until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
    if ! "$DOCKER" ps --filter "name=^${VLLM_CONTAINER}$" --format '{{.Names}}' | grep -q .; then
      "$DOCKER" logs "$VLLM_CONTAINER" >"$SERVER_LOG" 2>&1 || true
      tail -30 "$SERVER_LOG" >&2 || true
      SERVER_UP=0
      die "Container for $served exited during startup (see $SERVER_LOG)."
    fi
    sleep 5; waited=$((waited + 5))
    if (( waited % 120 == 0 )); then log "  ...still loading $served (${waited}s)"; fi
    if (( waited >= SERVER_BOOT_TIMEOUT )); then die "Server for $served did not come up in ${SERVER_BOOT_TIMEOUT}s."; fi
  done
  log "$served is up after ${waited}s"
}

model_tag() {  # model_tag <model> -> filesystem-friendly run name
  local name="${1##*/}"          # drop the HF org prefix
  name="${name,,}"
  name="${name//[^a-z0-9._-]/-}"
  printf '%s' "$name"
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

"${PYTHON[@]}" - <<'EOF' || die "blv is not reachable. Is the blv docker stack up?"
import sys
sys.path.insert(0, "benchmarks")
import common
out = common.verify_batch(["theorem preflight (n : Nat) : n = n := by sorry"], timeout=120)
assert out[0].get("verified"), out[0]
print("blv OK")
EOF

if [[ "$SKIP_GPT_OSS" != "1" || "$SKIP_QWEN" != "1" || "$SKIP_JUDGE" != "1" ]]; then
  command -v "$DOCKER" >/dev/null || die "'$DOCKER' not found on PATH; the vLLM lanes need it."
  "$DOCKER" info >/dev/null 2>&1 || die "Cannot talk to the docker daemon."
  if ! "$DOCKER" image inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
    warn "$VLLM_IMAGE is not present locally; docker will pull it on first use."
  fi
  if [[ ! -d "$HF_CACHE" ]]; then warn "HF_CACHE '$HF_CACHE' does not exist; weights will download into it."; fi
fi

if [[ "$SKIP_AGENT" != "1" ]]; then
  command -v claude >/dev/null || die "claude CLI not found on PATH."
  command -v uvx    >/dev/null || warn "uvx not found; the lean-lsp MCP will fail to start."
  if [[ ! -d "$LEAN_PROJECT" ]]; then warn "LEAN_PROJECT '$LEAN_PROJECT' missing; it will be created with 'lake init'."; fi
  if [[ -z "$MATHATLAS_PROJECT" ]]; then
    warn "MATHATLAS_PROJECT unset: the agent gets lean-lsp only, no dependency/context tools."
  elif [[ ! -d "$MATHATLAS_PROJECT" ]]; then
    die "MATHATLAS_PROJECT '$MATHATLAS_PROJECT' does not exist; set it or clear it to run without the MathAtlas MCP."
  fi
fi

if [[ "$SKIP_API" != "1" && -z "${OPENAI_API_KEY:-}" ]]; then
  warn "OPENAI_API_KEY unset -- skipping the $API_MODEL lane."
  SKIP_API=1
fi

if [[ "$SKIP_JUDGE" == "1" ]]; then
  warn "SKIP_JUDGE=1 -- generation will run but nothing will be judged."
  warn "Judge later by re-running (finished runs are skipped), or with benchmarks/judge_results.py."
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
  API_TAG="$(model_tag "$API_MODEL")"
  (
    run_iterative "$API_MODEL" "$API_URL" "$API_TAG.ma-hard" "$MAX_ROUNDS" --concurrency "$API_CONCURRENCY"
    if [[ "$RUN_CONTROL" == "1" ]]; then
      run_iterative "$API_MODEL" "$API_URL" "$API_TAG.control" 1 --concurrency "$API_CONCURRENCY"
    fi
  ) &
  BG_PIDS+=($!)
fi

if [[ "$SKIP_AGENT" != "1" ]]; then
  log "Launching Claude Code lane in the background"
  AGENT_OUT="$OUT_DIR/agentic/claude-code-$(model_tag "$CLAUDE_MODEL").ma-hard.json"
  (
    args=(); mapfile -t args < <(selection_args)
    extra=()
    if [[ -n "$MATHATLAS_PROJECT" ]]; then extra+=(--mathatlas-project "$MATHATLAS_PROJECT"); fi
    if [[ -n "$MATHATLAS_DATA" ]]; then extra+=(--mathatlas-data "$MATHATLAS_DATA"); fi
    if [[ -n "$MATHATLAS_TEXTBOOKS" ]]; then extra+=(--mathatlas-textbooks "$MATHATLAS_TEXTBOOKS"); fi
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
  GPT_OSS_TAG="$(model_tag "$GPT_OSS")"
  start_server "$GPT_OSS" "$GPT_OSS"
  run_iterative "$GPT_OSS" "http://localhost:$PORT/v1" "$GPT_OSS_TAG.ma-hard" "$MAX_ROUNDS"
  if [[ "$RUN_CONTROL" == "1" ]]; then
    run_iterative "$GPT_OSS" "http://localhost:$PORT/v1" "$GPT_OSS_TAG.control" 1
  fi
  stop_server
fi

# --------------------------------------------------------------------------- #
# GPU phase 2: Qwen3-30B-A3B (MoE)
# --------------------------------------------------------------------------- #
if [[ "$SKIP_QWEN" != "1" ]]; then
  QWEN_TAG="$(model_tag "$QWEN_MOE")"
  start_server "$QWEN_MOE" "$QWEN_MOE"
  run_iterative "$QWEN_MOE" "http://localhost:$PORT/v1" "$QWEN_TAG.ma-hard" "$MAX_ROUNDS"
  if [[ "$RUN_CONTROL" == "1" ]]; then
    run_iterative "$QWEN_MOE" "http://localhost:$PORT/v1" "$QWEN_TAG.control" 1
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
