#!/usr/bin/env python3
"""E3a / E5b -- build the human-verification and failure-labelling sheets.

Samples, per run, outputs the judge called faithful (to estimate the judge's
precision on real system outputs -- the agent's 29% headline rests on it) and
outputs that compile but were judged unfaithful (NPV + failure categories).

Writes a *blind* sheet (no judge verdict, rows shuffled, runs anonymized) for
annotators, a key to join back, and the guidelines. Score with --score after
annotation: judge precision/NPV per run with Wilson CIs, human-corrected joint.

    python benchmarks/analysis/make_annotation_set.py \
        --run claude-code-sonnet.dep=outputs/iclr/agentic/claude-code-sonnet.dep.json:50:25 \
        --run gpt-oss-120b.sp.dep-none=outputs/iclr/iterative/gpt-oss-120b.sp.dep-none.json:25:10 \
        --output-dir outputs/iclr/annotation
    # after two annotators fill `human_faithful` in copies of blind.csv:
    python benchmarks/analysis/make_annotation_set.py --score annotator1.csv --score annotator2.csv \
        --output-dir outputs/iclr/annotation
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(pretty_exceptions_show_locals=False)

GUIDELINES = """# Annotation guidelines (E3a judge verification, E5b failure categories)

Each row is an informal textbook item and a Lean 4 formalization that **compiles**.
Judge only faithfulness; ignore style.

`human_faithful`:
- `Y` -- same mathematical content: every hypothesis, quantifier, object and conclusion
  of the informal text is present with the intended meaning. Reasonable Mathlib
  encodings of the textbook's notions are fine; auxiliary definitions are fine if faithful.
- `N` -- any of: dropped/added hypothesis, wrong generality, wrong object, conclusion
  changed, vacuous/tautological statement (`P → P`, `: True`, hypotheses restating the
  conclusion), a concept replaced by an unrelated or placeholder definition.
- `U` -- cannot decide (say why in `notes`).

`failure_category` (only when `N`; pick the main one):
- `vacuous` -- tautology/`True`/conclusion assumed as hypothesis
- `placeholder-def` -- key concept defined as an opaque/trivial/wrong stand-in
- `dropped-hypothesis` / `added-hypothesis`
- `wrong-generality` -- over/under-specialized (e.g. fields for rings, finite for arbitrary)
- `wrong-object` -- a different mathematical object/notion than the text's
- `wrong-conclusion`
- `partial` -- only part of a multi-part item formalized
- `other`

Work independently; do not discuss rows until both sheets are done (we report Cohen's kappa).
"""


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


@app.command()
def run(
    run: list[str] = typer.Option([], help="`name=path.json:n_faithful:n_unfaithful`; repeatable."),
    output_dir: Path = typer.Option(Path("outputs/iclr/annotation")),
    seed: int = typer.Option(0),
    score: list[Path] = typer.Option([], help="Filled blind sheets to score (one per annotator)."),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    if score:
        return score_sheets(score, output_dir)
    rows = []
    for spec in run:
        name, _, rest = spec.partition("=")
        path, n_f, n_u = rest.rsplit(":", 2)
        df = pd.read_json(path)
        comp = df[df["verified"].fillna(False).astype(bool)]
        al = comp["aligned"].fillna(False).astype(bool)
        for verdict, pool, n in [("faithful", comp[al], int(n_f)), ("unfaithful", comp[~al], int(n_u))]:
            take = pool.sample(min(n, len(pool)), random_state=seed)
            for _, r in take.iterrows():
                rows.append({"run": name, "uuid": r["uuid"], "type": r["type"], "textbook": r.get("file_id"),
                             "informal": r["text"], "lean": r["code"], "judge_verdict": verdict,
                             "judge_reasoning": str((r.get("alignment_output") or {}).get("reasoning", ""))[:2000]})
    key = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    key.insert(0, "row_id", [f"r{i:04d}" for i in range(len(key))])
    blind = key[["row_id", "type", "informal", "lean"]].assign(human_faithful="", failure_category="", notes="")
    key.to_json(output_dir / "key.json", orient="records", indent=1)
    blind.to_csv(output_dir / "blind.csv", index=False)
    (output_dir / "GUIDELINES.md").write_text(GUIDELINES)
    print(key.groupby(["run", "judge_verdict"]).size().to_string())
    print(f"\nWrote {len(blind)} rows to {output_dir}/blind.csv (+ key.json, GUIDELINES.md)")


def score_sheets(paths: list[Path], output_dir: Path) -> None:
    from sklearn.metrics import cohen_kappa_score

    key = pd.read_json(output_dir / "key.json")
    sheets = [pd.read_csv(p)[["row_id", "human_faithful", "failure_category"]] for p in paths]
    out = {}
    if len(sheets) >= 2:
        m = sheets[0].merge(sheets[1], on="row_id")
        m = m[m["human_faithful_x"].isin(["Y", "N"]) & m["human_faithful_y"].isin(["Y", "N"])]
        out["cohen_kappa_annotators"] = float(cohen_kappa_score(m["human_faithful_x"], m["human_faithful_y"]))
        out["n_both_decided"] = len(m)
    # adjudicated label: agreement, else first annotator (disagreements listed for discussion)
    lab = sheets[0].rename(columns={"human_faithful": "h"})
    df = key.merge(lab[["row_id", "h", "failure_category"]], on="row_id")
    df = df[df["h"].isin(["Y", "N"])]
    per_run = []
    for (name, verdict), g in df.groupby(["run", "judge_verdict"]):
        k = int((g["h"] == ("Y" if verdict == "faithful" else "N")).sum())
        per_run.append({"run": name, "judge_verdict": verdict, "n": len(g),
                        "human_agrees": k / len(g), "ci95": wilson(k, len(g))})
    out["per_run"] = per_run
    out["failure_categories"] = df[df["h"] == "N"]["failure_category"].value_counts().to_dict()
    (output_dir / "annotation_scores.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    app()
