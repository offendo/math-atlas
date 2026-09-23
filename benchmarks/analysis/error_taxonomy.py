#!/usr/bin/env python3
"""E5a -- why do MA-Hard formalizations fail?

Buckets the first Lean error of every non-compiling output, per run, and reports
the compiled-but-unfaithful and degenerate shares. Comparing runs shows what each
intervention fixes: feedback (K=5) should shrink syntax/type errors; dependency
context should shrink unknown-identifier errors (missing/misnamed prerequisites).

    python benchmarks/analysis/error_taxonomy.py --runs 'outputs/iclr/iterative/*.json' \
        --runs 'outputs/iclr/agentic/*.json' --output-dir outputs/iclr/analysis
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(pretty_exceptions_show_locals=False)

# Order matters: first match wins.
CATEGORIES = [
    ("empty/no code", r"empty generation"),
    ("unknown identifier/constant", r"unknown (identifier|constant|namespace)|unknown universe|does not contain"),
    ("typeclass synthesis", r"failed to synthesize"),
    ("type mismatch", r"type mismatch|has type .* but is expected|application type mismatch"),
    ("invalid field/notation", r"invalid field|invalid dotted|ambiguous|overloaded|invalid binder|invalid pattern"),
    ("function expected/elaboration", r"function expected|elaboration|cannot synthesize placeholder|don't know how to synthesize"),
    ("syntax/parse", r"unexpected token|expected|unterminated|unknown command|invalid 'import'"),
    ("timeout/resources", r"timeout|heartbeat|maximum recursion|deterministic"),
    ("verifier/infra", r"job failed|did not return|repl|No such file"),
]


def first_error(co) -> str:
    if not isinstance(co, dict):
        return ""
    errs = co.get("errors") or []
    for e in errs:
        data = e.get("data", "") if isinstance(e, dict) else str(e)
        if data:
            return data
    return ""


def bucket(msg: str) -> str:
    for name, pat in CATEGORIES:
        if re.search(pat, msg, flags=re.IGNORECASE | re.DOTALL):
            return name
    return "other"


@app.command()
def run(runs: list[str] = typer.Option(...), output_dir: Path = typer.Option(Path("outputs/iclr/analysis"))):
    rows, examples = [], []
    for pattern in runs:
        for path in sorted(glob.glob(pattern)):
            if path.endswith((".metrics.json", "coverage.json", "pairwise_identity.json")):
                continue
            try:
                df = pd.read_json(path)
            except ValueError:
                continue
            if "verified" not in df.columns or "compiler_output" not in df.columns:
                continue
            name = Path(path).stem
            ok = df["verified"].fillna(False).astype(bool)
            fails = df[~ok]
            cats = fails["compiler_output"].apply(lambda co: bucket(first_error(co)))
            row = {"run": name, "n": len(df), "not compiling %": round(100 * (~ok).mean(), 1)}
            if "aligned" in df.columns:
                al = df["aligned"].fillna(False).astype(bool)
                row["compiled but unfaithful %"] = round(100 * (ok & ~al).mean(), 1)
            if "degenerate" in df.columns:
                row["degenerate %"] = round(100 * df["degenerate"].fillna(False).astype(bool).mean(), 1)
            shares = cats.value_counts(normalize=True)
            for c, _ in CATEGORIES + [("other", "")]:
                row[f"{c} (% of failures)"] = round(100 * shares.get(c, 0.0), 1)
            rows.append(row)
            for (_, r), c in list(zip(fails.iterrows(), cats))[:3]:
                examples.append({"run": name, "uuid": r["uuid"], "category": c, "error": first_error(r["compiler_output"])[:300]})
    output_dir.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows)
    md = "# E5a -- Lean failure taxonomy (first error per non-compiling output)\n\n" + t.to_markdown(index=False) + "\n"
    (output_dir / "error_taxonomy.md").write_text(md)
    t.to_csv(output_dir / "error_taxonomy.csv", index=False)
    pd.DataFrame(examples).to_csv(output_dir / "error_taxonomy_examples.csv", index=False)
    print(md)


if __name__ == "__main__":
    app()
