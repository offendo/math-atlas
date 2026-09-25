#!/usr/bin/env python3
"""Re-score existing MA-Align judge outputs against re-verified gold labels.

`benchmarks/labels/ma-align-relabel.tsv` holds a second, item-by-item review of
all 200 MA-Align pairs (100 definitions, 100 statements): the original label,
the re-verified label, a confidence, and the reasoning. The re-verification
treats a formalization as aligned when it states the same mathematics; unlike
the original annotation it does not reject a definition merely for reusing a
matching Mathlib definition, or for generalizing harmlessly (e.g. dropping
`a ≠ 0` from gcd, `Ring` instead of `Field`), and it rejects statements that
are false as written (junk `deriv`, non-commutative `Ring`, empty types, ...).

No judge is re-run. Predictions come from files already on disk:
  * `outputs/iclr/judge-validation/<tag>.ma-align-{defs,stmts}.json`
    (judge_validation.py; joined on uuid), and
  * the paper's Table 1 runs `outputs/*-ma-*.json` (run_alignment_benchmark.py;
    row order = dataset order, checked against the stored `label` column).

Writes `<output-dir>/<tag>.metrics.json` in judge_validation.py's schema (MA-Align
benchmarks only, so report.py's `--validation-dir` can point at it) plus
`summary.md` / `summary.csv` comparing original-label and relabelled metrics.

    python benchmarks/analysis/rescore_ma_align.py --outputs-dir outputs \
        --output-dir outputs/iclr/judge-validation-relabel
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from judge_validation import binary_metrics, to_bool  # noqa: E402

app = typer.Typer(pretty_exceptions_show_locals=False)
REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("ma-align-defs", "ma-align-stmts")
# Paper Table 1 runs (run_alignment_benchmark.py): tag -> {benchmark: file under outputs/}.
LEGACY = {
    "legacy/criticleangpt-qwen3-14b-rl": {"ma-align-defs": "criticleangpt-qwen3-14b-rl-ma-defs.json",
                                          "ma-align-stmts": "criticleangpt-qwen3-14b-rl-ma-stmts.json"},
    "legacy/criticleangpt-qwen3-32b-rl": {"ma-align-defs": "criticleangpt-qwen3-32b-rl-ma-defs.json",
                                          "ma-align-stmts": "criticleangpt-qwen3-32b-rl-ma-stmts.json"},
    "legacy/gpt-oss-120b": {"ma-align-stmts": "gpt-120b-ma-align-stmts.json"},
    "legacy/qwen3-14b": {"ma-align-defs": "qwen3-14b-ma-defs.json", "ma-align-stmts": "qwen3-14b-ma-stmts.json"},
    "legacy/qwen3-32b": {"ma-align-defs": "qwen3-32b-ma-defs.json", "ma-align-stmts": "qwen3-32b-ma-stmts.json"},
    "legacy/reform-prompt-gpt-oss-120b": {"ma-align-defs": "reform-prompt-gpt-120b-ma-defs.json",
                                          "ma-align-stmts": "reform-prompt-gpt-120b-ma-stmts.json"},
    "legacy/reform-qwen3-14b": {"ma-align-stmts": "reform-qwen3-14b-ma-stmts.json"},
    "legacy/reform-qwen3-32b": {"ma-align-defs": "reform-qwen3-32b-ma-defs.json",
                                "ma-align-stmts": "reform-qwen3-32b-ma-stmts.json"},
}


def load_labels(path: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path, sep="\t")
    return {b: g.sort_values("index").reset_index(drop=True) for b, g in df.groupby("benchmark")}


def judge_validation_preds(path: Path, labels: pd.DataFrame) -> tuple[np.ndarray, int, int]:
    """Predictions in label order, joined on uuid; also checks the stored gold matches the original label."""
    df = pd.DataFrame(json.loads(path.read_text()))
    m = labels[["uuid", "original_label"]].merge(df, on="uuid", how="left", validate="one_to_one")
    if m["pred_aligned"].isna().any():
        raise ValueError(f"{path}: {int(m['pred_aligned'].isna().sum())} items missing")
    if not (m["gold_aligned"].astype(bool) == m["original_label"].map(to_bool)).all():
        raise ValueError(f"{path}: stored gold labels disagree with the dataset")
    outs = [json.loads(j) for j in m["judge_output"]]
    parse_errors = sum(o.get("error") not in (None, "None") for o in outs)
    call_errors = sum(str(o.get("reasoning", "")).startswith("JUDGE_ERROR") for o in outs)
    return m["pred_aligned"].astype(bool).to_numpy(), parse_errors, call_errors


def legacy_preds(path: Path, labels: pd.DataFrame) -> tuple[np.ndarray, int, int]:
    """Predictions from run_alignment_benchmark.py output (column-oriented; rows in dataset order)."""
    df = pd.read_json(path)
    df.index = df.index.astype(int)
    df = df.sort_index()
    if len(df) != len(labels) or not (df["label"].map(to_bool).to_numpy()
                                        == labels["original_label"].map(to_bool).to_numpy()).all():
        raise ValueError(f"{path}: row order does not match the dataset's label sequence")
    # Same rule as run_alignment_benchmark.py.
    pred = df["result"].apply(lambda x: x in {"Correct", "aligned", True}).to_numpy()
    errors = int(df["error"].notna().sum()) if "error" in df.columns else 0
    return pred, errors, 0


def score(pred: np.ndarray, labels: pd.DataFrame) -> dict:
    orig = binary_metrics(labels["original_label"].map(to_bool).to_numpy(), pred)
    new = binary_metrics(labels["relabel"].map(to_bool).to_numpy(), pred)
    return {"original": orig, "relabel": new}


@app.command()
def run(
    labels_file: Path = typer.Option(REPO_ROOT / "benchmarks" / "labels" / "ma-align-relabel.tsv"),
    outputs_dir: Path = typer.Option(REPO_ROOT / "outputs", help="Directory holding the judge outputs."),
    output_dir: Path = typer.Option(Path("outputs/iclr/judge-validation-relabel")),
):
    labels = load_labels(labels_file)
    sources: dict[str, dict[str, tuple[np.ndarray, int, int]]] = {}
    vdir = outputs_dir / "iclr" / "judge-validation"
    for f in sorted(vdir.glob("*.ma-align-*.json")):
        tag, bench = f.name[: -len(".json")].rsplit(".", 1)
        sources.setdefault(tag, {})[bench] = judge_validation_preds(f, labels[bench])
    for tag, files in LEGACY.items():
        for bench, name in files.items():
            if (outputs_dir / name).exists():
                sources.setdefault(tag, {})[bench] = legacy_preds(outputs_dir / name, labels[bench])

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, benches in sources.items():
        summary = {}
        for bench in BENCHMARKS:
            if bench not in benches:
                continue
            pred, parse_errors, call_errors = benches[bench]
            s = score(pred, labels[bench])
            summary[bench] = {**s["relabel"], "parse_errors": parse_errors, "judge_call_errors": call_errors,
                              "original_labels": s["original"]}
            for which, m in s.items():
                rows.append({"judge": tag, "benchmark": bench, "labels": which, "parse_errors": parse_errors,
                             **{k: v for k, v in m.items() if k != "confusion"}, **m["confusion"]})
        out = {"tag": tag, "label_source": str(labels_file.name), "benchmarks": summary}
        (output_dir / f"{tag.replace('/', '__')}.metrics.json").write_text(json.dumps(out, indent=2))

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "summary.csv", index=False)

    lines = ["# MA-Align judges re-scored with re-verified labels", ""]
    for bench in BENCHMARKS:
        lab = labels[bench]
        n_flip = int((lab["original_label"] != lab["relabel"]).sum())
        lines += [f"## {bench}",
                  f"Gold aligned: original {int(lab['original_label'].map(to_bool).sum())}/{len(lab)}, "
                  f"relabel {int(lab['relabel'].map(to_bool).sum())}/{len(lab)}; {n_flip} labels changed.", "",
                  "| judge | acc (orig → new) | bal. acc (orig → new) | sens (orig → new) | spec (orig → new) "
                  "| κ (orig → new) | maj. (new) | pred. aligned | parse err |",
                  "|---|---|---|---|---|---|---|---|---|"]
        sub = df[df["benchmark"] == bench]
        for tag in sources:
            o = sub[(sub["judge"] == tag) & (sub["labels"] == "original")]
            n = sub[(sub["judge"] == tag) & (sub["labels"] == "relabel")]
            if o.empty:
                continue
            o, n = o.iloc[0], n.iloc[0]

            def arrow(k, pctfmt=True):
                f = (lambda x: f"{100 * x:.1f}") if pctfmt else (lambda x: f"{x:.2f}")
                return f"{f(o[k])} → {f(n[k])}"

            lines.append(f"| {tag} | {arrow('accuracy')} | {arrow('balanced_accuracy')} | {arrow('sensitivity')} | "
                         f"{arrow('specificity')} | {arrow('cohen_kappa', False)} | {100 * n['majority_baseline']:.1f} | "
                         f"{100 * n['predicted_aligned_rate']:.0f}% | {int(n['parse_errors'])} |")
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    app()
