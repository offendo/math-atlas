#!/usr/bin/env bash
#
# Judge every MA-Hard output (this campaign's + the Sep-6 runs) with one judge,
# writing to a separate directory so the raw generations are never modified.
#
#   scripts/judge_all.sh <judge-model> <judge-url> <tag> [extra judge_results.py args...]
#   scripts/judge_all.sh m-a-p/CriticLeanGPT-Qwen3-32B-RL http://localhost:8000/v1 criticlean-32b-v2 \
#       --judge-prompt-file prompts/my_statement_prompt.txt --judge-definition-prompt-file prompts/my_def_prompt.txt
#
# Output: $OUT_DIR/judged/<tag>/<run>.json (+ .metrics.json). Finished files are skipped, so
# re-running resumes. Then point the analysis at it:
#   python benchmarks/analysis/report.py --runs "$OUT_DIR/judged/<tag>/*.json" ...
# Validate the same judge/prompts first with benchmarks/judge_validation.py --tag <tag>.
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
JUDGE_MODEL="${1:?judge model}"; JUDGE_URL="${2:?judge url}"; TAG="${3:?tag}"; shift 3
PY="${PY:-/home/npatel37/src/math-atlas/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/npatel37/src/math-atlas}"
OUT_DIR="${OUT_DIR:-$DATA_ROOT/outputs/iclr}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-64}"
DEST="$OUT_DIR/judged/$TAG"
mkdir -p "$DEST"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

# Fail fast if the judge endpoint is not serving: a dead endpoint turns every
# verdict into "misaligned" (this happened once; see ICLR-EXPERIMENTS.md).
curl -sf "${JUDGE_URL%/}/models" > /dev/null || { echo "judge endpoint $JUDGE_URL not reachable" >&2; exit 1; }

inputs=$(ls "$OUT_DIR"/iterative/*.json "$OUT_DIR"/agentic/*.json "$OUT_DIR"/sliced/*.json \
            "$DATA_ROOT"/outputs/iterative/*.json "$DATA_ROOT"/outputs/agentic/*.json 2>/dev/null \
         | grep -vE '\.metrics\.json$|coverage\.json$|pairwise_identity\.json$|\.partial\.|\.bak$')
n=0
for f in $inputs; do
  out="$DEST/$(basename "$f")"
  if [[ -s "$out" ]]; then echo "skip $(basename "$f")"; continue; fi
  echo "judging $(basename "$f")"
  if ! "$PY" benchmarks/judge_results.py --input "$f" --output "$out" --judge-model "$JUDGE_MODEL" \
        --judge-model-url "$JUDGE_URL" --judge-concurrency "$JUDGE_CONCURRENCY" "$@" \
        > "$DEST/$(basename "$f" .json).log" 2>&1; then
    echo "FAILED $(basename "$f") (see $DEST/$(basename "$f" .json).log)" >&2; rm -f "$out"; continue
  fi
  # a judged file whose every call errored is not a result
  if grep -q "Judge call failed" "$DEST/$(basename "$f" .json).log"; then
    echo "WARN $(basename "$f"): some judge calls failed (see log)" >&2
  fi
  n=$((n + 1))
done
echo "judged $n files into $DEST"
