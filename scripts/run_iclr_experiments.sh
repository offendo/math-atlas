#!/usr/bin/env bash
#
# ICLR revision experiments on MA-Hard (see ICLR-EXPERIMENTS.md).
#
# Three lanes run concurrently; each is sequential inside:
#
#   GPU lane    [gpt-oss-120b: E2a dep arms, E2a x K=5, E6 prompt cells]
#               [CriticLean-32B: E0 validation, judge every output (E1a/E1b/E2/E6)]
#               [gpt-oss-120b as 2nd judge: E0 validation + E3b re-judging]
#   Claude lane [Sonnet single-pass: dep none/both/random][A1-dep][A1-none]
#               [Sonnet compile-repair K=5][A1-opt]
#   API lane    [E1a slice full-set runs][gpt-5.2 single-pass none/both]
#               [gpt-5.2: E0 validation]  ... then, after judging: [gpt-5.2 E3b re-judge]
#
# Everything is resumable: a finished output is skipped on re-run.
#
#   ASSUME_YES=1 nohup scripts/run_iclr_experiments.sh > run.log 2>&1 &
#   N_EXAMPLES=4 OUT_DIR=/tmp/iclr-smoke ASSUME_YES=1 scripts/run_iclr_experiments.sh   # smoke test
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PY="${PY:-/home/npatel37/src/math-atlas/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/npatel37/src/math-atlas}"           # where the old full-set outputs live
OUT_DIR="${OUT_DIR:-$DATA_ROOT/outputs/iclr}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${LOG_DIR:-$DATA_ROOT/logs/iclr-$STAMP}"
DATASET="${DATASET:-offendo/math-atlas-official}"
SPLIT="${SPLIT:-hard}"
N_EXAMPLES="${N_EXAMPLES:-}"
SEED="${SEED:-1337}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

GPT_OSS="${GPT_OSS:-openai/gpt-oss-120b}"
JUDGE_MODEL="${JUDGE_MODEL:-m-a-p/CriticLeanGPT-Qwen3-32B-RL}"
CLAUDE_MODEL="${CLAUDE_MODEL:-sonnet}"
API_MODEL="${API_MODEL:-gpt-5.2}"
API_URL="${API_URL:-https://api.openai.com/v1}"

MAX_TOKENS="${MAX_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-0.0}"                 # matches the Sep-6 B1 runs
RETRY_TEMPERATURE="${RETRY_TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
LOCAL_CONCURRENCY="${LOCAL_CONCURRENCY:-64}"
CLAUDE_CONCURRENCY="${CLAUDE_CONCURRENCY:-16}"
API_CONCURRENCY="${API_CONCURRENCY:-32}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-64}"
JV_N="${JV_N:-}"                                   # subsample judge-validation sets (smoke tests)

# vLLM (docker)
DOCKER="${DOCKER:-docker}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
VLLM_CONTAINER="${VLLM_CONTAINER:-iclr-vllm}"
HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
GPUS="${GPUS:-1,2}"
TP_SIZE="${TP_SIZE:-2}"
PORT="${PORT:-8000}"
GEN_MODEL_LEN="${GEN_MODEL_LEN:-65536}"
JUDGE_MODEL_LEN="${JUDGE_MODEL_LEN:-40960}"
SERVER_BOOT_TIMEOUT="${SERVER_BOOT_TIMEOUT:-2400}"

# A1 agent arms
AGENT_ROOT="${AGENT_ROOT:-$HOME/src/ma-hard-iclr}"   # scripts/setup_agent_projects.sh <root> dep none opt
MATHATLAS_PROJECT="${MATHATLAS_PROJECT:-$HOME/src/mathatlas-formalization}"
AGENT_CONCURRENCY="${AGENT_CONCURRENCY:-24}"
MAX_BUDGET_USD="${MAX_BUDGET_USD:-0.75}"   # Sep-6 used 0.50; the dependency protocol hit that cap on 2/3 smoke items
AGENT_TIMEOUT="${AGENT_TIMEOUT:-900}"
AGENT_ARMS="${AGENT_ARMS:-dep none opt}"

SKIP_GPU="${SKIP_GPU:-0}"; SKIP_CLAUDE="${SKIP_CLAUDE:-0}"; SKIP_API="${SKIP_API:-0}"; SKIP_SLICE="${SKIP_SLICE:-0}"

mkdir -p "$OUT_DIR"/{iterative,agentic,sliced,judge-validation,rejudge} "$LOG_DIR"
GEN_MANIFEST="$LOG_DIR/generated.txt"; : > "$GEN_MANIFEST"
LANE_DONE="$LOG_DIR/lanes-done"; mkdir -p "$LANE_DONE"

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[%s] WARN %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '[%s] FATAL %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

selection() {
  local a=(--dataset "$DATASET" --split "$SPLIT" --seed "$SEED")
  [[ -n "$N_EXAMPLES" ]] && a+=(--n-examples "$N_EXAMPLES")
  printf '%s\n' "${a[@]}"
}

# --------------------------------------------------------------------------- #
# vLLM server
# --------------------------------------------------------------------------- #
SERVER_UP=0
start_server() {  # start_server <model> <max_model_len>
  local model="$1" mml="$2" waited=0
  "$DOCKER" rm -f "$VLLM_CONTAINER" >/dev/null 2>&1 || true
  log "GPU: serving $model (TP=$TP_SIZE, GPUs $GPUS, max_len $mml)"
  "$DOCKER" run -d --rm --name "$VLLM_CONTAINER" --gpus "\"device=$GPUS\"" --ipc=host \
      -p "$PORT:$PORT" -v "$HF_CACHE:/root/.cache/huggingface" -e HF_HOME=/root/.cache/huggingface \
      -e HF_HUB_OFFLINE=1 --entrypoint vllm "$VLLM_IMAGE" serve "$model" \
      --served-model-name "$model" --tensor-parallel-size "$TP_SIZE" --max-model-len "$mml" --port "$PORT" \
      >/dev/null || die "could not start vLLM for $model"
  SERVER_UP=1
  until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
    if ! "$DOCKER" ps --filter "name=^${VLLM_CONTAINER}$" --format '{{.Names}}' | grep -q .; then
      die "vLLM container for $model exited during startup"
    fi
    sleep 5; waited=$((waited + 5))
    (( waited >= SERVER_BOOT_TIMEOUT )) && die "vLLM for $model not up after ${waited}s"
  done
  log "GPU: $model up after ${waited}s"
}
stop_server() {
  [[ "$SERVER_UP" == 1 ]] || return 0
  "$DOCKER" logs "$VLLM_CONTAINER" > "$LOG_DIR/server.$(date +%s).log" 2>&1 || true
  "$DOCKER" stop -t 30 "$VLLM_CONTAINER" >/dev/null 2>&1 || true
  "$DOCKER" rm -f "$VLLM_CONTAINER" >/dev/null 2>&1 || true
  SERVER_UP=0; sleep 10
}

BG=()
cleanup() {
  local rc=$?
  stop_server || true
  for pid in "${BG[@]:-}"; do [[ -n "$pid" ]] && { pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; }; done
  exit $rc
}
trap cleanup EXIT INT TERM

# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
# run_iter <tag> <model> <url|""> [extra args...]  -> $OUT_DIR/iterative/<tag>.json (unjudged)
run_iter() {
  local tag="$1" model="$2" url="$3"; shift 3
  local out="$OUT_DIR/iterative/$tag.json"
  if [[ -s "$out" ]]; then log "skip $tag (exists)"; echo "$out" >> "$GEN_MANIFEST"; return 0; fi
  local sel=(); mapfile -t sel < <(selection)
  local url_args=(); [[ -n "$url" ]] && url_args=(--model-url "$url")
  log "run $tag"
  if "$PY" benchmarks/iterative/run_iterative.py --model "$model" "${url_args[@]}" "${sel[@]}" \
        --temperature "$TEMPERATURE" --retry-temperature "$RETRY_TEMPERATURE" --top-p "$TOP_P" \
        --max-tokens "$MAX_TOKENS" --skip-judge --output "$out" "$@" > "$LOG_DIR/$tag.log" 2>&1 && [[ -s "$out" ]]; then
    log "done $tag"; echo "$out" >> "$GEN_MANIFEST"
  else
    warn "$tag FAILED (see $LOG_DIR/$tag.log)"; tail -5 "$LOG_DIR/$tag.log" >&2
  fi
}

# run_agent <arm>  (dep | none | opt) -> $OUT_DIR/agentic/claude-code-<model>.<arm>.json
run_agent() {
  local arm="$1" out="$OUT_DIR/agentic/claude-code-$CLAUDE_MODEL.$1.json"
  if [[ -s "$out" ]]; then log "skip agent $arm (exists)"; echo "$out" >> "$GEN_MANIFEST"; return 0; fi
  local sel=(); mapfile -t sel < <(selection)
  local extra=()
  case "$arm" in
    dep)  extra=(--mathatlas-project "$MATHATLAS_PROJECT" --prompt-file benchmarks/agentic/prompts/agent_task_dependency_aware.txt
                 --require-dependencies) ;;
    opt)  extra=(--mathatlas-project "$MATHATLAS_PROJECT") ;;
    none) extra=(--allowed-tools "Read,Write,Edit,Glob,Grep,Bash,mcp__lean-lsp") ;;
    *) die "unknown agent arm $arm" ;;
  esac
  [[ -d "$AGENT_ROOT/$arm" ]] || die "agent project $AGENT_ROOT/$arm missing; run scripts/setup_agent_projects.sh $AGENT_ROOT $arm"
  log "agent $arm (concurrency $AGENT_CONCURRENCY, \$$MAX_BUDGET_USD/item, ${AGENT_TIMEOUT}s)"
  if "$PY" benchmarks/agentic/run_claude_code.py "${sel[@]}" --project "$AGENT_ROOT/$arm" --no-build-project \
        --model "$CLAUDE_MODEL" --max-budget-usd "$MAX_BUDGET_USD" --timeout "$AGENT_TIMEOUT" \
        --concurrency "$AGENT_CONCURRENCY" --resume --skip-judge --output "$out" "${extra[@]}" \
        > "$LOG_DIR/agent-$arm.log" 2>&1 && [[ -s "$out" ]]; then
    log "done agent $arm"; echo "$out" >> "$GEN_MANIFEST"
  else
    warn "agent $arm FAILED (see $LOG_DIR/agent-$arm.log)"; tail -5 "$LOG_DIR/agent-$arm.log" >&2
  fi
}

# judge_one <in> <out> <judge-model> <url|""> [api-key]
judge_one() {
  local in="$1" out="$2" jm="$3" url="$4" key="${5:-EMPTY}"
  [[ -s "$in" ]] || return 0
  if [[ "$in" != "$out" && -s "$out" ]]; then return 0; fi
  if [[ "$in" == "$out" ]] && "$PY" - "$in" <<'EOF'
import json, sys, pathlib
m = pathlib.Path(sys.argv[1]).with_suffix(".metrics.json")
sys.exit(0 if m.exists() and json.loads(m.read_text())["metrics"].get("joint_rate") is not None else 1)
EOF
  then return 0; fi
  mkdir -p "$(dirname "$out")"
  local url_args=(); [[ -n "$url" ]] && url_args=(--judge-model-url "$url")
  "$PY" benchmarks/judge_results.py --input "$in" --output "$out" --judge-model "$jm" "${url_args[@]}" \
      --judge-api-key "$key" --judge-concurrency "$JUDGE_CONCURRENCY" \
      > "$LOG_DIR/judge.$(basename "$out" .json).$(basename "$jm").log" 2>&1 \
    || warn "judging $in with $jm failed"
}

validate_judge() {  # validate_judge <tag> <model> <url|""> [api-key]
  local tag="$1" jm="$2" url="$3" key="${4:-EMPTY}"
  [[ -s "$OUT_DIR/judge-validation/$tag.metrics.json" ]] && { log "skip validation $tag"; return 0; }
  local url_args=(); [[ -n "$url" ]] && url_args=(--judge-model-url "$url")
  local n_args=(); [[ -n "$JV_N" ]] && n_args=(--n-examples "$JV_N")
  log "E0: validating judge $tag"
  "$PY" benchmarks/judge_validation.py --judge-model "$jm" "${url_args[@]}" --judge-api-key "$key" \
      --tag "$tag" --output-dir "$OUT_DIR/judge-validation" "${n_args[@]}" \
      > "$LOG_DIR/validate.$tag.log" 2>&1 || warn "validation $tag failed"
}

# All MA-Hard outputs: this campaign's plus the Sep-6 runs (for multi-judge re-scoring).
all_outputs() {
  ls "$OUT_DIR"/iterative/*.json "$OUT_DIR"/agentic/*.json "$OUT_DIR"/sliced/*.json 2>/dev/null \
    | grep -vE '\.metrics\.json$|coverage\.json$|pairwise_identity\.json$|\.partial\.' || true
  if [[ -z "$N_EXAMPLES" ]]; then
    ls "$DATA_ROOT"/outputs/iterative/*.json "$DATA_ROOT"/outputs/agentic/*.json 2>/dev/null | grep -vE '\.metrics\.json$' || true
  fi
}

# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
log "Out: $OUT_DIR   Logs: $LOG_DIR"
"$PY" - <<'EOF' || die "blv not reachable (docker compose -f ~/src/blv/compose.yaml up -d)"
import sys; sys.path.insert(0, "benchmarks"); import common
assert common.verify_batch(["theorem t (n : Nat) : n = n := by sorry"], timeout=120)[0].get("verified")
print("blv OK")
EOF
[[ "$SKIP_CLAUDE" == 1 ]] || command -v claude >/dev/null || die "claude CLI missing"
if [[ "$SKIP_API" != 1 && -z "${OPENAI_API_KEY:-}" ]]; then warn "OPENAI_API_KEY unset: skipping API lane"; SKIP_API=1; fi
if [[ "$SKIP_CLAUDE" != 1 && "${ASSUME_YES:-0}" != 1 ]]; then
  die "the Claude lane runs ${AGENT_ARMS} agent arms (~\$$MAX_BUDGET_USD/item cap each); set ASSUME_YES=1 to confirm"
fi

# --------------------------------------------------------------------------- #
# Claude lane
# --------------------------------------------------------------------------- #
claude_lane() {
  local cc=(--backend claude-cli --concurrency "$CLAUDE_CONCURRENCY")
  for mode in none both random; do
    run_iter "sonnet.sp.dep-$mode" "$CLAUDE_MODEL" "" --max-rounds 1 --dependency-context "$mode" "${cc[@]}"
  done
  for arm in $AGENT_ARMS; do
    [[ "$arm" == opt ]] && run_iter "sonnet.k5.dep-none" "$CLAUDE_MODEL" "" --max-rounds 5 "${cc[@]}"
    run_agent "$arm"
  done
  [[ " $AGENT_ARMS " == *" opt "* ]] || run_iter "sonnet.k5.dep-none" "$CLAUDE_MODEL" "" --max-rounds 5 "${cc[@]}"
  touch "$LANE_DONE/claude"
}

# --------------------------------------------------------------------------- #
# API / CPU lane
# --------------------------------------------------------------------------- #
api_lane() {
  if [[ "$SKIP_SLICE" != 1 && -z "$N_EXAMPLES" ]]; then
    log "E1a: slicing full-set runs to MA-Hard"
    local D="$DATA_ROOT"
    "$PY" benchmarks/analysis/slice_full_runs.py --output-dir "$OUT_DIR/sliced" \
      --run reform-8b="$D/outputs/reform.statements.aligned.json" \
      --run goedel-8b="$D/criticlean/goedel-lm.goedel-formalizer-v2-8b.statements.aligned.json" \
      --run goedel-32b="$D/criticlean/goedel-lm.goedel-formalizer-v2-32b.statements.aligned.json" \
      --run kimina-7b="$D/criticlean/ai-mo.kimina-autoformalizer-7b.statements.verified.json" \
      --run herald-7b="$D/criticlean/frenzymath.herald_translator.statements.aligned.json" \
      --run atlas-8b="$D/outputs/xiaoyangliu-sjtu.atlas_translator_q.statements.json" \
      --run gpt-oss-120b.stmts.default="$D/criticlean/openai.gpt-oss-120b.statements.json" \
      --run gpt-oss-120b.stmts.zero-shot="$D/criticlean/openai.gpt-oss-120b.statements.prompt=theorem_zero_shot.aligned.json" \
      --run gpt-oss-120b.stmts.tuned-prompt="$D/criticlean/openai.gpt-oss-120b.statements.prompt=theorem_zero_shot_tuned_prompt.aligned.json" \
      --run gpt-oss-20b.stmts.default="$D/criticlean/openai.gpt-oss-20b.statements.aligned.json" \
      --run gpt-oss-120b.defs.default="$D/criticlean/openai.gpt-oss-120b.definitions.aligned.json" \
      --run gpt-oss-120b.defs.zero-shot="$D/criticlean/openai.gpt-oss-120b.definitions.prompt=definition_zero_shot.json" \
      --run gpt-oss-120b.defs.tuned-prompt="$D/criticlean/openai.gpt-oss-120b.definitions.prompt=definition_zero_shot_tuned_prompt.json" \
      --run gpt-oss-120b.defs.tuned-examples="$D/criticlean/openai.gpt-oss-120b.definitions.prompt=definition_few_shot_tuned_examples.json" \
      --run gpt-oss-120b.defs.context-300="$D/criticlean/openai.gpt-oss-120b.definitions.context=300.aligned.json" \
      --run gpt-oss-20b.defs.default="$D/criticlean/openai.gpt-oss-20b.definitions.aligned.json" \
      > "$LOG_DIR/slice.log" 2>&1 || warn "E1a slicing failed (see $LOG_DIR/slice.log)"
    for f in "$OUT_DIR"/sliced/*.json; do
      [[ "$f" == *.metrics.json || "$f" == */coverage.json || "$f" == */pairwise_identity.json ]] || echo "$f" >> "$GEN_MANIFEST"
    done
  fi
  if [[ "$SKIP_API" != 1 ]]; then
    local api=(--api-style chat --concurrency "$API_CONCURRENCY" --api-key "$OPENAI_API_KEY")
    run_iter "gpt-5.2.sp.dep-none" "$API_MODEL" "$API_URL" --max-rounds 1 --dependency-context none "${api[@]}"
    run_iter "gpt-5.2.sp.dep-both" "$API_MODEL" "$API_URL" --max-rounds 1 --dependency-context both "${api[@]}"
    validate_judge "$API_MODEL" "$API_MODEL" "$API_URL" "$OPENAI_API_KEY"
  fi
  touch "$LANE_DONE/api"
}

# --------------------------------------------------------------------------- #
# GPU lane
# --------------------------------------------------------------------------- #
judge_manifest() {  # judge everything generated so far with CriticLean (in place)
  local f
  for f in $(sort -u "$GEN_MANIFEST"); do judge_one "$f" "$f" "$JUDGE_MODEL" "http://localhost:$PORT/v1"; done
}

gpu_lane() {
  local url="http://localhost:$PORT/v1" local_args=(--concurrency "$LOCAL_CONCURRENCY")
  local SP=benchmarks/single-pass/prompts
  local THM=(--item-type theorem --item-type example --item-type exercise)
  start_server "$GPT_OSS" "$GEN_MODEL_LEN"
  # E2a: dependency-context arms, single pass
  for mode in none informal mathlib both random; do
    run_iter "gpt-oss-120b.sp.dep-$mode" "$GPT_OSS" "$url" --max-rounds 1 --dependency-context "$mode" "${local_args[@]}"
  done
  # E2a x feedback: does the graph still help once compiler errors are fed back?
  for mode in none both; do
    run_iter "gpt-oss-120b.k5.dep-$mode" "$GPT_OSS" "$url" --max-rounds 5 --dependency-context "$mode" "${local_args[@]}"
  done
  # E6: prompt (base|tuned) x examples (none|LeanWorkbook|graduate), single pass, paper prompts
  local sys=(--system-prompt-file "$SP/system_single_pass.txt")
  local t
  for t in theorem_zero_shot theorem_few_shot theorem_zero_shot_tuned_prompt \
           theorem_few_shot_tuned_prompt_tuned_examples theorem_few_shot_base_prompt_tuned_examples \
           theorem_few_shot_tuned_prompt_lw_examples; do
    run_iter "gpt-oss-120b.e6.$t" "$GPT_OSS" "$url" --max-rounds 1 "${sys[@]}" "${THM[@]}" \
      --theorem-prompt-file "$SP/$t.txt" "${local_args[@]}"
  done
  for t in definition_zero_shot definition_zero_shot_tuned_prompt definition_few_shot_tuned_examples \
           definition_few_shot_tuned_prompt_tuned_examples; do
    run_iter "gpt-oss-120b.e6.$t" "$GPT_OSS" "$url" --max-rounds 1 "${sys[@]}" --item-type definition \
      --definition-prompt-file "$SP/$t.txt" "${local_args[@]}"
  done
  stop_server

  # CriticLean-32B: E0 validation, then judge what exists, then wait for the other lanes.
  start_server "$JUDGE_MODEL" "$JUDGE_MODEL_LEN"
  validate_judge criticlean-32b "$JUDGE_MODEL" "$url"
  judge_manifest
  while [[ ! -e "$LANE_DONE/claude" || ! -e "$LANE_DONE/api" ]]; do
    sleep 120; judge_manifest
  done
  judge_manifest
  # the Sep-6 runs are already CriticLean-judged; nothing to do for them here
  stop_server

  # gpt-oss-120b as a second judge (E0 + E3b), into $OUT_DIR/rejudge/gpt-oss-120b/
  start_server "$GPT_OSS" "$GEN_MODEL_LEN"
  validate_judge gpt-oss-120b "$GPT_OSS" "$url"
  local f
  for f in $(all_outputs); do judge_one "$f" "$OUT_DIR/rejudge/gpt-oss-120b/$(basename "$f")" "$GPT_OSS" "$url"; done
  stop_server
  touch "$LANE_DONE/gpu"
}

# --------------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------------- #
if [[ "$SKIP_CLAUDE" != 1 ]]; then claude_lane > "$LOG_DIR/lane-claude.log" 2>&1 & BG+=($!); else touch "$LANE_DONE/claude"; fi
api_lane > "$LOG_DIR/lane-api.log" 2>&1 & BG+=($!)
if [[ "$SKIP_GPU" != 1 ]]; then gpu_lane > "$LOG_DIR/lane-gpu.log" 2>&1; else touch "$LANE_DONE/gpu"; fi
for pid in "${BG[@]}"; do wait "$pid"; done
BG=()

# gpt-5.2 as a third judge (E3b), once every output is final.
api_has_quota() {  # a 1-token call; false on insufficient_quota / any failure
  "$PY" - "$API_MODEL" "$API_URL" <<'PYEOF' > /dev/null 2>&1
import sys
from openai import OpenAI
OpenAI(base_url=sys.argv[2]).chat.completions.create(
    model=sys.argv[1], messages=[{"role": "user", "content": "ok"}], max_completion_tokens=16)
PYEOF
}
if [[ "$SKIP_API" != 1 ]] && ! api_has_quota; then
  warn "E3b: $API_MODEL unreachable or out of credits; skipping gpt-5.2 re-judging (re-run later to resume)"
  SKIP_API=1
fi
if [[ "$SKIP_API" != 1 ]]; then
  log "E3b: gpt-5.2 re-judging"
  for f in $(all_outputs); do
    judge_one "$f" "$OUT_DIR/rejudge/$API_MODEL/$(basename "$f")" "$API_MODEL" "$API_URL" "$OPENAI_API_KEY"
  done
fi
log "All lanes finished; running analyses"
REPS="${REPS:-100}" scripts/run_iclr_analysis.sh "$OUT_DIR" > "$LOG_DIR/analysis.log" 2>&1 || warn "analysis failed (see $LOG_DIR/analysis.log)"
log "Done. Report: $OUT_DIR/analysis/report.md"
