#!/usr/bin/env python3
"""E0 -- validate an alignment judge exactly as the benchmark uses it.

Scores a judge on MA-Align (definitions, statements) and the two external
faithfulness benchmarks (ConsistencyCheck, CriticLeanBench) through the *same*
code path the baselines use (`common.judge_alignment`: same prompts per item
type, same structured-output schema, same parser). The previous MA-Align numbers
came from `scripts/run_alignment_benchmark.py`, whose parser did not strip
```json fences: every CriticLean-32B verdict failed to parse and defaulted to
"misaligned" (100/100, 100/100, 500/500).

Reports what an imbalanced test set needs, next to plain accuracy:
balanced accuracy, MCC, Cohen's kappa, sensitivity (aligned recall),
specificity (misaligned recall), the majority-class baseline, and the number of
unparseable verdicts. Sensitivity/specificity per item type feed the
judge-error correction in analysis/report.py.

    python benchmarks/judge_validation.py \
        --judge-model m-a-p/CriticLeanGPT-Qwen3-32B-RL --judge-model-url http://localhost:8000/v1 \
        --tag criticlean-32b --output-dir outputs/iclr/judge-validation
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

app = typer.Typer(pretty_exceptions_show_locals=False)
logger = logging.getLogger("judge_validation")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
logger.setLevel(logging.INFO)

# name -> (HF dataset, split, item kind that picks the production prompt)
BENCHMARKS = {
    "ma-align-defs": ("offendo/ma-alignment-defs", "train", "definition"),
    "ma-align-stmts": ("offendo/ma-alignment-stmts", "train", "statement"),
    "consistency-check": ("offendo/consistency-check", "train", "statement"),
    "criticleanbench": ("offendo/criticleanbench", "test", "statement"),
}
POSITIVE = {"aligned", "true", "correct", "1"}


def to_bool(label) -> bool:
    return str(label).strip().lower() in POSITIVE


def binary_metrics(gold: np.ndarray, pred: np.ndarray) -> dict:
    gold, pred = gold.astype(bool), pred.astype(bool)
    tp = int((gold & pred).sum()); tn = int((~gold & ~pred).sum())
    fp = int((~gold & pred).sum()); fn = int((gold & ~pred).sum())
    n = tp + tn + fp + fn
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    acc = (tp + tn) / n if n else float("nan")
    p_yes = ((tp + fp) / n) * ((tp + fn) / n) if n else 0.0
    p_no = ((tn + fn) / n) * ((tn + fp) / n) if n else 0.0
    pe = p_yes + p_no
    kappa = (acc - pe) / (1 - pe) if pe < 1 else float("nan")
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {
        "n": n,
        "n_aligned_gold": tp + fn,
        "accuracy": acc,
        "balanced_accuracy": (sens + spec) / 2,
        "sensitivity": sens,
        "specificity": spec,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "mcc": float(mcc),
        "cohen_kappa": kappa,
        "majority_baseline": max(tp + fn, tn + fp) / n if n else float("nan"),
        "predicted_aligned_rate": (tp + fp) / n if n else float("nan"),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


@app.command()
def run(
    judge_model: str = typer.Option(..., help="Judge model name as served."),
    judge_model_url: str | None = typer.Option(None, help="OpenAI-compatible base URL (omit for api.openai.com)."),
    judge_api_key: str = typer.Option("EMPTY", envvar="JUDGE_API_KEY", help="API key for the judge endpoint."),
    tag: str = typer.Option(..., help="Short name for this judge, used in file names."),
    output_dir: Path = typer.Option(Path("outputs/iclr/judge-validation"), help="Where to write results."),
    benchmark: list[str] = typer.Option(list(BENCHMARKS), help="Benchmarks to run; repeatable."),
    judge_prompt_file: Path = typer.Option(common.REPO_ROOT / "prompts" / "critic_lean_prompt.txt"),
    judge_definition_prompt_file: Path = typer.Option(common.REPO_ROOT / "prompts" / "definition_alignment.txt"),
    judge_max_tokens: int = typer.Option(8192),
    judge_concurrency: int = typer.Option(32),
    judge_structured: bool = typer.Option(True, help="JSON-schema structured output (production setting)."),
    judge_reasoning_effort: str | None = typer.Option(None, help="Reasoning effort for the judge"),
    n_examples: int | None = typer.Option(None, help="Subsample per benchmark (debugging)."),
):
    from datasets import load_dataset

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in benchmark:
        ds_name, split, kind = BENCHMARKS[name]
        df = load_dataset(ds_name, split=split).to_pandas()
        if n_examples:
            df = df.sample(min(n_examples, len(df)), random_state=0)
        prompt = judge_definition_prompt_file if kind == "definition" else judge_prompt_file
        logger.info("%s: %d items, prompt=%s", name, len(df), prompt.name)
        judgements = common.judge_alignment(
            list(zip(df["informal"].astype(str), df["formal"].astype(str))),
            model=judge_model,
            model_url=judge_model_url,
            prompt_file=prompt,
            max_tokens=judge_max_tokens,
            concurrency=judge_concurrency,
            api_key=judge_api_key,
            structured=judge_structured,
            temperature=1.0,
            reasoning_effort=judge_reasoning_effort,
        )
        gold = df["label"].map(to_bool).to_numpy()
        pred = np.array([j["result"] == "aligned" for j in judgements])
        metrics = binary_metrics(gold, pred)
        metrics["parse_errors"] = int(sum(j.get("error") is not None for j in judgements))
        metrics["judge_call_errors"] = int(sum(str(j.get("reasoning", "")).startswith("JUDGE_ERROR") for j in judgements))
        summary[name] = metrics

        keep = [c for c in ("uuid", "name", "id") if c in df.columns]
        per_item = df[keep].copy()
        per_item["gold_aligned"] = gold
        per_item["pred_aligned"] = pred
        per_item["judge_output"] = [json.dumps({k: str(v) for k, v in j.items()}) for j in judgements]
        per_item.to_json(output_dir / f"{tag}.{name}.json", orient="records", indent=1)
        logger.info("%s: acc=%.3f bal_acc=%.3f kappa=%.3f parse_errors=%d", name, metrics["accuracy"],
                    metrics["balanced_accuracy"], metrics["cohen_kappa"], metrics["parse_errors"])

    out = {"judge_model": judge_model, "tag": tag, "structured": judge_structured,
           "statement_prompt": str(judge_prompt_file), "definition_prompt": str(judge_definition_prompt_file),
           "benchmarks": summary}
    (output_dir / f"{tag}.metrics.json").write_text(json.dumps(out, indent=2))

    rows = [{"benchmark": k, **{m: v for m, v in s.items() if m != "confusion"}} for k, s in summary.items()]
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    app()
