#!/usr/bin/env bash
#
# Run every analysis over the ICLR experiment outputs (CPU only; blv not needed).
#
#   scripts/run_iclr_analysis.sh [OUT_DIR]
#
# Writes into $OUT_DIR/analysis: report.md (E0/E1/E2/E3/E6 tables with CIs,
# McNemar contrasts, multi-judge agreement), error_taxonomy.md (E5a), agent tool
# usage + leak audit (E2b), confounds/ (E4a), and the depth robustness files in
# $OUT_DIR/depth (E4b). Annotation sheets (E3a/E5b) go to $OUT_DIR/annotation.
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PY:-/home/npatel37/src/math-atlas/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/npatel37/src/math-atlas}"
OUT="${1:-$DATA_ROOT/outputs/iclr}"
A="$OUT/analysis"
REPS="${REPS:-100}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$A"

echo "== E4b depth (3 noise models, $REPS reps each)"
for noise in drop rewire-backward rewire; do
  [[ -s "$OUT/depth/depth_summary.$noise.json" ]] || \
    "$PY" scripts/graph_depth.py --reps "$REPS" --noise "$noise" --output-dir "$OUT/depth" > "$A/depth.$noise.log" 2>&1
done

# The judge is Qwen3.8-27B (open weights; no proprietary judge), run by scripts/judge_all.sh
# into $OUT/judged/$JUDGE_TAG. JUDGE_VALIDATION_TAG picks the judge_validation.py run whose
# MA-Align sensitivity/specificity drive the judge-adjusted column.
JUDGE_TAG="${JUDGE_TAG:-qwen38-27b}"
JUDGE_VALIDATION_TAG="${JUDGE_VALIDATION_TAG:-qwen38-27b-medium}"
echo "== E0/E1/E2/E3/E6 report (judge: $JUDGE_TAG)"
"$PY" benchmarks/analysis/report.py \
  --runs "$OUT/judged/$JUDGE_TAG/*.json" \
  --validation-dir "$OUT/judge-validation" --judge-tag "$JUDGE_VALIDATION_TAG" \
  --rejudge-dir "$OUT/rejudge" --output-dir "$A" > "$A/report.log" 2>&1

echo "== E5a error taxonomy"
"$PY" benchmarks/analysis/error_taxonomy.py --runs "$OUT/judged/$JUDGE_TAG/*.json" \
  --output-dir "$A" > "$A/error_taxonomy.log" 2>&1

echo "== E2b agent tool usage + leak audit"
agent_files=$(ls "$OUT"/agentic/*.json "$DATA_ROOT"/outputs/agentic/*.json 2>/dev/null | grep -v metrics)
# shellcheck disable=SC2086
"$PY" benchmarks/analysis/agent_tool_usage.py $agent_files --output "$A/agent_tool_usage.json" > "$A/agent_tool_usage.txt" 2>&1

echo "== E4a confounds (the paper's Fig. 3/4 systems, full set)"
"$PY" benchmarks/analysis/confounds.py \
  --system "reform-8b:statements=$DATA_ROOT/outputs/reform.statements.aligned.json" \
  --system "goedel-8b:statements=$DATA_ROOT/criticlean/goedel-lm.goedel-formalizer-v2-8b.statements.aligned.json" \
  --system "gpt-oss-120b-tuned-exs:definitions=$DATA_ROOT/criticlean/openai.gpt-oss-120b.definitions.prompt=definition_few_shot_tuned_examples.json" \
  --system "gpt-oss-120b-default:definitions=$DATA_ROOT/criticlean/openai.gpt-oss-120b.definitions.aligned.json" \
  --depth "$OUT/depth/depth.rewire-backward.csv.gz" --output-dir "$A/confounds" > "$A/confounds.log" 2>&1

echo "== Open-split table (R3.6), recomputed from per-item outputs"
D="$DATA_ROOT"
"$PY" benchmarks/analysis/open_split_table.py \
  --system "ReForm 8B=$D/outputs/reform.statements.aligned.json" \
  --system "Goedel 8B=$D/criticlean/goedel-lm.goedel-formalizer-v2-8b.statements.aligned.json" \
  --system "Goedel 32B=$D/criticlean/goedel-lm.goedel-formalizer-v2-32b.statements.aligned.json" \
  --system "Kimina 7B=$D/criticlean/ai-mo.kimina-autoformalizer-7b.statements.verified.json" \
  --system "Herald 7B=$D/criticlean/frenzymath.herald_translator.statements.aligned.json" \
  --system "gpt-oss-120b stmts (default)=$D/criticlean/openai.gpt-oss-120b.statements.json" \
  --system "gpt-oss-120b stmts zero-shot=$D/criticlean/openai.gpt-oss-120b.statements.prompt=theorem_zero_shot.aligned.json" \
  --system "gpt-oss-20b stmts (default)=$D/criticlean/openai.gpt-oss-20b.statements.aligned.json" \
  --system "gpt-oss-120b defs (default)=$D/criticlean/openai.gpt-oss-120b.definitions.aligned.json" \
  --system "gpt-oss-120b defs tuned exs=$D/criticlean/openai.gpt-oss-120b.definitions.prompt=definition_few_shot_tuned_examples.json" \
  --system "gpt-oss-20b defs (default)=$D/criticlean/openai.gpt-oss-20b.definitions.aligned.json" \
  --output "$A/open_split_table.md" > "$A/open_split_table.log" 2>&1

echo "== E3a/E5b annotation sheets"
ann=()
for spec in "claude-code-sonnet.dep:agentic:50:25" "claude-code-sonnet.none:agentic:25:10" \
            "sonnet.sp.dep-both:iterative:25:10" "gpt-oss-120b.sp.dep-none:iterative:25:10"; do
  IFS=: read -r name sub nf nu <<< "$spec"
  f="$OUT/judged/$JUDGE_TAG/$name.json"
  [[ -s "$f" ]] && ann+=(--run "$name=$f:$nf:$nu")
done
if (( ${#ann[@]} )) && [[ ! -s "$OUT/annotation/blind.csv" ]]; then
  "$PY" benchmarks/analysis/make_annotation_set.py "${ann[@]}" --output-dir "$OUT/annotation" > "$A/annotation.log" 2>&1
fi
echo "Done: $A/report.md"
