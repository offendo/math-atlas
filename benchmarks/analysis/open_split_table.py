#!/usr/bin/env python3
"""Regenerate the Open-Split table (paper Table 5) from per-item outputs (R3.6).

Several rows of the paper's Table 5 are digit-for-digit identical to Table 2 rows,
which is implausible for a 70% subset. This recomputes both from the same per-item
files: "full" = every row in the file, "open" = rows whose uuid is in the public
release (offendo/math-atlas-official test + hard). Correct = compiled AND aligned,
as stored by the original pipeline (no re-judging), with Wilson CIs.

    python benchmarks/analysis/open_split_table.py \
        --system "ReForm 8B=/path/reform.statements.aligned.json" ... \
        --output outputs/iclr/analysis/open_split_table.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confounds import load_correctness, wilson  # noqa: E402

app = typer.Typer(pretty_exceptions_show_locals=False)


def cell(df: pd.DataFrame) -> dict:
    n = len(df)
    kc, kj = int(df["verified"].sum()), int(df["correct"].sum())
    lo, hi = wilson(kj, n)
    return {"n": n, "compile %": round(100 * kc / n, 1) if n else None,
            "faithful|compile %": round(100 * kj / kc, 1) if kc else None,
            "correct %": round(100 * kj / n, 1) if n else None,
            "correct 95% CI": f"[{100 * lo:.1f}, {100 * hi:.1f}]" if n else "–"}


@app.command()
def run(system: list[str] = typer.Option(..., help="`Label=path`; repeatable."),
        output: Path = typer.Option(Path("outputs/iclr/analysis/open_split_table.md"))):
    from datasets import load_dataset

    ds = load_dataset("offendo/math-atlas-official")
    open_uuids = set(ds["test"]["uuid"]) | set(ds["hard"]["uuid"])
    rows = []
    for spec in system:
        label, _, path = spec.partition("=")
        df = load_correctness(path)
        for split, sub in [("full", df), ("open", df[df["uuid"].isin(open_uuids)])]:
            rows.append({"system": label, "split": split, **cell(sub)})
    t = pd.DataFrame(rows)
    md = ("# Open split vs full set, recomputed from per-item outputs\n\n"
          "Correct = compiled AND aligned as stored by the original pipeline.\n\n" + t.to_markdown(index=False) + "\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md)
    print(md)


if __name__ == "__main__":
    app()
