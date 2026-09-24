#!/usr/bin/env python3
"""Regenerate the judge-dependent LaTeX tables in paper/experiments.tex.

Rebuilds two tables in place (matched by their \\label), leaving the others
(full-set recomputation, depth) untouched:
  * tab:iclr-judge-validation -- from outputs/iclr/judge-validation/<tag>.metrics.json
  * tab:iclr-mahard           -- from outputs/iclr/analysis/report.json (report.py)

    python benchmarks/analysis/make_latex_tables.py \
        --report outputs/iclr/analysis/report.json --validation-dir outputs/iclr/judge-validation \
        --tex paper/experiments.tex
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from pathlib import Path

import typer

app = typer.Typer(pretty_exceptions_show_locals=False)

# (section title, [(run stem, label)]) -- a missing run renders as "not run".
MAHARD_SECTIONS = [
    ("Dependency context, single pass (gpt-oss-120b)", [
        ("gpt-oss-120b.sp.dep-none", "None"),
        ("gpt-oss-120b.sp.dep-informal", "+ informal deps."),
        ("gpt-oss-120b.sp.dep-mathlib", "+ Mathlib names"),
        ("gpt-oss-120b.sp.dep-both", "+ both"),
        ("gpt-oss-120b.sp.dep-random", "+ random (control)"),
    ]),
    ("Compiler feedback, K5 (gpt-oss-120b)", [
        ("gpt-oss-120b.k5.dep-none", "None"),
        ("gpt-oss-120b.k5.dep-both", "+ both"),
    ]),
    ("Frontier models, single pass", [
        ("gpt-5.2.sp.dep-none", "gpt-5.2"),
        ("gpt-5.2.sp.dep-both", r"\hspace{0.5em}+ both"),
        ("sonnet.sp.dep-none", "Sonnet"),
        ("sonnet.sp.dep-both", r"\hspace{0.5em}+ both"),
        ("sonnet.sp.dep-random", r"\hspace{0.5em}+ random (control)"),
        ("sonnet.k5.dep-none", "Sonnet, K5"),
    ]),
    (r"Agentic: Claude Code (Sonnet), \$0.75/item cap", [
        ("claude-code-sonnet.none", "No MathAtlas tools"),
        ("claude-code-sonnet.opt", "MathAtlas tools optional"),
        ("claude-code-sonnet.dep", "Dependency-aware (enforced)"),
    ]),
    (r"Earlier runs (Sep.~6; older Mathlib, \$0.50/item cap)", [
        ("gpt-oss-120b.control", "gpt-oss-120b"),
        ("gpt-oss-120b.ma-hard", r"\hspace{0.5em}+ K5"),
        ("gpt-5-mini.control", "gpt-5-mini"),
        ("gpt-5-mini.ma-hard", r"\hspace{0.5em}+ K5"),
        ("claude-code-sonnet.ma-hard", "Claude Code (tools optional)"),
    ]),
    (r"Paper systems re-scored on \hardname{} (statements only, $n=622$)", [
        ("reform-8b", "ReForm 8B"),
        ("goedel-8b", "Goedel 8B"),
        ("goedel-32b", "Goedel 32B"),
        ("kimina-7b", "Kimina 7B"),
        ("herald-7b", "Herald 7B"),
        ("gpt-oss-120b.stmts.zero-shot", "gpt-oss-120b (zs)"),
        ("gpt-oss-120b.stmts.tuned-prompt", "gpt-oss-120b (tuned prompt)"),
        ("gpt-oss-120b.stmts.default", "gpt-oss-120b (default)"),
    ]),
]
# Prompt ablation: theorem cell for the first columns, definition cell (if any) for Defs.
E6_ROWS = [
    ("Base prompt", "theorem_zero_shot", "definition_zero_shot"),
    (r"\hspace{0.5em}+ LeanWorkbook exs.", "theorem_few_shot", None),
    (r"\hspace{0.5em}+ graduate exs.", "theorem_few_shot_base_prompt_tuned_examples", "definition_few_shot_tuned_examples"),
    ("Tuned prompt", "theorem_zero_shot_tuned_prompt", "definition_zero_shot_tuned_prompt"),
    (r"\hspace{0.5em}+ LeanWorkbook exs.", "theorem_few_shot_tuned_prompt_lw_examples", None),
    (r"\hspace{0.5em}+ graduate exs.", "theorem_few_shot_tuned_prompt_tuned_examples", "definition_few_shot_tuned_prompt_tuned_examples"),
]
JUDGES = [  # (validation tag, section title, paper accuracies or None)
    ("qwen38-27b-medium", "Qwen3.8-27B (primary judge; reasoning effort medium)", None),
    ("criticlean-32b", "CriticLean (32B)", {"ma-align-defs": 80.0, "ma-align-stmts": 75.0,
                                            "consistency-check": 82.6, "criticleanbench": 86.4}),
    ("gpt-5.2", "gpt-5.2 (Our Prompt; reference only, proprietary)", {"ma-align-defs": 86.0, "ma-align-stmts": 80.0}),
]
BENCH_LABEL = {"ma-align-defs": r"\alignmentname{} Defs.", "ma-align-stmts": r"\alignmentname{} Stmts.",
               "consistency-check": "ConsistencyCheck", "criticleanbench": "CriticLeanBench"}


def pct(x) -> str:
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}\\%"


def pct_ci(x, ci) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{100 * x:.1f}\\% \\ci{{{100 * ci[0]:.1f}}}{{{100 * ci[1]:.1f}}}" if ci else f"{100 * x:.1f}\\%"


def replace_table(tex: str, label: str, new_block: str) -> str:
    pat = re.compile(r"\\begin\{table\*?\}(?:(?!\\end\{table\*?\}).)*?\\label\{" + re.escape(label)
                     + r"\}.*?\\end\{table\*?\}", re.S)
    if not pat.search(tex):
        raise ValueError(f"table with label {label} not found")
    return pat.sub(lambda _: new_block, tex, count=1)


def judge_table(vdir: Path) -> str:
    rows = []
    for tag, title, paper in JUDGES:
        path = vdir / f"{tag}.metrics.json"
        if not path.exists():
            continue
        bench = json.loads(path.read_text())["benchmarks"]
        body = []
        for key, label in BENCH_LABEL.items():
            m = bench.get(key)
            if not m or m.get("judge_call_errors", 0) > 0:  # failed API runs are not results
                continue
            p = f"{paper[key]:.1f}" if paper and key in paper else "--"
            body.append(f"    {label} & {100*m['accuracy']:.1f} & {p} & {100*m['balanced_accuracy']:.1f} & "
                        f"{100*m['sensitivity']:.1f} & {100*m['specificity']:.1f} & {m['cohen_kappa']:.2f} & "
                        f"{100*m['majority_baseline']:.1f} \\\\")
        if body:
            rows += ["    \\midrule" if rows else "", "    \\rowcolor{gray!20}",
                     f"    \\multicolumn{{8}}{{c}}{{\\textbf{{{title}}}}} \\\\", *body]
    rows = [r for r in rows if r]
    return "\n".join([
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Faithfulness judges on \alignmentname{} and prior benchmarks, run through the exact",
        r"  judging path used to score all systems (same prompts, structured output, parser). Bal.\ = balanced",
        r"  accuracy; Sens./Spec.\ = recall on aligned/misaligned items; Maj.\ = always-majority baseline.",
        r"  \alignmentname{} statements are 25\% aligned. The primary judge is open-weights (Qwen3.8-27B);",
        r"  gpt-5.2 is shown for reference only.}",
        r"  \label{tab:iclr-judge-validation}",
        r"  \vspace{0.5em}",
        r"  \small",
        r"  \begin{tabular}{@{}llllllll@{}}",
        r"    \toprule",
        r"    Benchmark & Acc. & Paper acc. & Bal. & Sens. & Spec. & $\kappa$ & Maj. \\",
        r"    \midrule",
        *rows,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])


def mahard_table(report: dict, judge_name: str, stamp: str) -> str:
    runs = {r["run"]: r for r in report["runs"]}

    def row(stem: str, label: str) -> str:
        r = runs.get(stem)
        if r is None:
            return f"    {label} & \\multicolumn{{5}}{{c}}{{\\pend{{not run}}}} \\\\"
        return (f"    {label} & {pct_ci(r['compile'], r.get('compile_ci'))} & {pct(r.get('faithful_of_compiling'))} & "
                f"{pct_ci(r.get('joint'), r.get('joint_ci'))} & {pct(r.get('joint_corrected'))} & {pct(r.get('joint_def'))} \\\\")

    lines = []
    for i, (title, items) in enumerate(MAHARD_SECTIONS):
        lines += (["    \\midrule"] if i else []) + ["    \\rowcolor{gray!20}",
                                                     f"    \\multicolumn{{6}}{{c}}{{\\textbf{{{title}}}}} \\\\"]
        lines += [row(stem, label) for stem, label in items]
    lines += ["    \\midrule", "    \\rowcolor{gray!20}",
              r"    \multicolumn{6}{c}{\textbf{Prompt ablation, single pass (gpt-oss-120b): first four columns theorems ($n=622$), Defs.\ definitions ($n=76$)}} \\"]
    for label, thm, dfn in E6_ROWS:
        t = runs.get(f"gpt-oss-120b.e6.{thm}")
        d = runs.get(f"gpt-oss-120b.e6.{dfn}") if dfn else None
        if t is None:
            lines.append(f"    {label} & \\multicolumn{{5}}{{c}}{{\\pend{{not run}}}} \\\\")
            continue
        lines.append(f"    {label} & {pct(t['compile'])} & {pct(t.get('faithful_of_compiling'))} & {pct(t.get('joint'))} & "
                     f"{pct(t.get('joint_corrected'))} & {pct(d.get('joint')) if d else '--'} \\\\")
    return "\n".join([
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Results on \hardname{} (698 items: 622 statements, 76 definitions). ``Correct'' requires",
        f"  compiling and a faithful judgement by {judge_name}; ``Faithful'' is among compiling outputs; ``Adj.''",
        r"  corrects correctness for the judge's measured error rates (Table~\ref{tab:iclr-judge-validation}); ``Defs.''",
        r"  is correctness on the 76 definitions. Dependency context: none, informal text of the item's direct",
        r"  prerequisites, their Mathlib names, both, or a matched random-definition control. K5 = up to five",
        f"  rounds of compiler feedback. Generated {stamp}.}}",
        r"  \label{tab:iclr-mahard}",
        r"  \vspace{0.5em}",
        r"  \small",
        r"  \begin{tabular}{@{}llllll@{}}",
        r"    \toprule",
        r"    Model / setting & Compiles & Faithful & Correct & Adj. & Defs. \\",
        r"    \midrule",
        *lines,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table*}",
    ])


@app.command()
def run(
    report: Path = typer.Option(Path("outputs/iclr/analysis/report.json")),
    validation_dir: Path = typer.Option(Path("outputs/iclr/judge-validation")),
    tex: Path = typer.Option(Path("paper/experiments.tex")),
    judge_name: str = typer.Option("Qwen3.8-27B"),
):
    stamp = dt.datetime.now().strftime("%b.~%-d, %H:%M")
    text = tex.read_text()
    text = replace_table(text, "tab:iclr-judge-validation", judge_table(validation_dir))
    text = replace_table(text, "tab:iclr-mahard", mahard_table(json.loads(report.read_text()), judge_name, stamp))
    text = re.sub(r"^% ICLR revision experiments -- .*$",
                  f"% ICLR revision experiments -- regenerated {dt.datetime.now():%Y-%m-%d %H:%M} (judge: {judge_name}).",
                  text, count=1, flags=re.M)
    text = text.replace('% "Correct" = compiles AND judged faithful by CriticLean-32B;',
                        f'% "Correct" = compiles AND judged faithful by {judge_name};')
    tex.write_text(text)
    print(f"rewrote judge-validation and MA-Hard tables in {tex}")


if __name__ == "__main__":
    app()
