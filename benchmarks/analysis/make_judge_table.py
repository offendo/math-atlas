#!/usr/bin/env python3
"""Judge-quality table for the paper, scored against the re-verified MA-Align labels.

Rows are the judges run through the production judging path (judge_validation.py);
MA-Align columns are recomputed per item against benchmarks/labels/ma-align-relabel.tsv:
balanced accuracy with a 95% bootstrap CI, Cohen's kappa (defs), sensitivity /
specificity (stmts), and an exact McNemar p-value on per-item correctness vs the
primary judge. ConsistencyCheck / CriticLeanBench balanced accuracy comes from the
stored metrics (their labels are unchanged); runs with judge call errors print "--".

    python benchmarks/analysis/make_judge_table.py --outputs-dir outputs \
        --output outputs/iclr/judge-validation-relabel/judge_table.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from judge_validation import binary_metrics, to_bool  # noqa: E402

app = typer.Typer(pretty_exceptions_show_locals=False)
REPO_ROOT = Path(__file__).resolve().parents[2]
PRIMARY = "qwen38-27b-medium"
JUDGES = [  # (validation tag, row label)
    (PRIMARY, r"Qwen3.8-27B (ours)"),
    ("criticlean-32b", "CriticLean-32B"),
    ("gpt-5.2", "gpt-5.2 (ref.)"),
    ("sonnet", "Claude Sonnet (ref.)"),
]


def bootstrap_bal_acc(gold: np.ndarray, pred: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(gold), np.flatnonzero(~gold)
    vals = []
    for _ in range(n_boot):  # stratified, so every replicate has both classes
        p, n = rng.choice(pos, len(pos)), rng.choice(neg, len(neg))
        vals.append((pred[p].mean() + (~pred[n]).mean()) / 2)
    return np.percentile(vals, [2.5, 97.5])


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    b, c = int((correct_a & ~correct_b).sum()), int((~correct_a & correct_b).sum())
    return 1.0 if b + c == 0 else float(stats.binomtest(b, b + c, 0.5).pvalue)


def load_preds(vdir: Path, tag: str, bench: str, labels: pd.DataFrame) -> np.ndarray:
    df = pd.DataFrame(json.loads((vdir / f"{tag}.{bench}.json").read_text()))
    m = labels[["uuid"]].merge(df, on="uuid", how="left", validate="one_to_one")
    return m["pred_aligned"].astype(bool).to_numpy()


def fmt_p(p: float) -> str:
    return "<.001" if p < 0.001 else f"{p:.2f}".lstrip("0")


@app.command()
def run(
    labels_file: Path = typer.Option(REPO_ROOT / "benchmarks" / "labels" / "ma-align-relabel.tsv"),
    outputs_dir: Path = typer.Option(REPO_ROOT / "outputs"),
    output: Path = typer.Option(Path("outputs/iclr/judge-validation-relabel/judge_table.tex")),
):
    vdir = outputs_dir / "iclr" / "judge-validation"
    lab = pd.read_csv(labels_file, sep="\t")
    labels = {b: g.sort_values("index").reset_index(drop=True) for b, g in lab.groupby("benchmark")}
    gold = {b: labels[b]["relabel"].map(to_bool).to_numpy() for b in labels}
    preds = {(t, b): load_preds(vdir, t, b, labels[b]) for t, _ in JUDGES for b in labels}

    body = []
    for tag, name in JUDGES:
        cells = []
        for bench in ("ma-align-defs", "ma-align-stmts"):
            g, p = gold[bench], preds[(tag, bench)]
            m = binary_metrics(g, p)
            lo, hi = bootstrap_bal_acc(g, p)
            cell = f"{100 * m['balanced_accuracy']:.1f} \\ci{{{100 * lo:.1f}}}{{{100 * hi:.1f}}}"
            if tag != PRIMARY:
                pv = mcnemar(p == g, preds[(PRIMARY, bench)] == g)
                cell += f"$^{{p={fmt_p(pv)}}}$" if pv >= 0.05 else f"$^{{p={fmt_p(pv)}*}}$"
            cells.append(cell)
            cells += ([f"{m['cohen_kappa']:.2f}"] if bench == "ma-align-defs"
                      else [f"{100 * m['sensitivity']:.1f}", f"{100 * m['specificity']:.1f}"])
        ext = json.loads((vdir / f"{tag}.metrics.json").read_text())["benchmarks"]
        for bench in ("consistency-check", "criticleanbench"):
            m = ext.get(bench)
            cells.append(f"{100 * m['balanced_accuracy']:.1f}" if m and m.get("judge_call_errors", 0) == 0 else "--")
        row = " & ".join([name, *cells])
        body.append(("    \\rowcolor{blue!8}\n" if tag == PRIMARY else "") + f"    {row} \\\\")

    maj = [f"{100 * max(g.mean(), 1 - g.mean()):.1f}" for g in (gold["ma-align-defs"], gold["ma-align-stmts"])]
    tex = "\n".join([
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Faithfulness judges on \alignmentname{} (gold labels re-verified item by item) and on two",
        r"  external benchmarks, all run through the judging path used to score every system (same prompts,",
        r"  structured output, parser). Bal.\ = balanced accuracy with 95\% bootstrap CI; superscripts are exact",
        r"  McNemar $p$-values on per-item correctness against Qwen3.8-27B (* $p<0.05$). Sens./Spec.\ = recall",
        r"  on aligned/misaligned items; statement specificity is what the judge-adjusted MA-Hard scores",
        r"  correct for. A judge that always answers ``misaligned'' scores 50.0 balanced accuracy",
        f"  (plain accuracy {maj[0]}\\% on definitions, {maj[1]}\\% on statements). gpt-5.2 and Claude Sonnet are",
        r"  proprietary and shown for reference only.}",
        r"  \label{tab:semantic-faithfulness-results}",
        r"  \vspace{0.5em}",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{@{}l cc ccc cc@{}}",
        r"    \toprule",
        r"    & \multicolumn{2}{c}{\alignmentname~Defs.} & \multicolumn{3}{c}{\alignmentname~Stmts.}"
        r" & \multicolumn{2}{c}{External (Bal.)} \\",
        r"    \cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(l){7-8}",
        r"    Judge & Bal. & $\kappa$ & Bal. & Sens. & Spec. & ConsistencyCheck & CriticLeanBench \\",
        r"    \midrule",
        *body,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex)
    print(tex)


if __name__ == "__main__":
    app()
