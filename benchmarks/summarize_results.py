#!/usr/bin/env python3
"""Collate baseline `*.metrics.json` files into one table.

Prints a markdown table to stdout and optionally writes it to a file.

    python benchmarks/summarize_results.py outputs/iterative outputs/agentic \
        --output BASELINE-RESULTS.md
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer

app = typer.Typer(pretty_exceptions_show_locals=False)

COLUMNS = [
    ("baseline", "Baseline"),
    ("run", "Run"),
    ("model", "Model"),
    ("n_items", "N"),
    ("compile_rate", "Compile %"),
    ("aligned_rate_of_compiling", "Aligned % (of compiling)"),
    ("joint_rate", "**Joint %**"),
    ("degenerate_rate", "Degenerate %"),
    ("rounds_or_turns", "Rounds/Turns"),
    ("cost", "Cost"),
    ("wall", "Wall clock"),
]


def collect(paths: list[Path]) -> list[Path]:
    """Expand directories into the metrics files they contain."""
    found: list[Path] = []
    for p in paths:
        if p.is_dir():
            found.extend(sorted(p.rglob("*.metrics.json")))
        elif p.name.endswith(".metrics.json"):
            found.append(p)
        elif p.suffix == ".json":  # a results file; take its sibling metrics
            sibling = p.with_suffix(".metrics.json")
            if sibling.exists():
                found.append(sibling)
    return found


def fmt_pct(v) -> str:
    return "—" if v is None else f"{100 * float(v):.1f}"


def row_for(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    cfg, m = payload.get("config", {}), payload.get("metrics", {})

    if "mean_rounds" in m:
        rounds = f"{m['mean_rounds']:.2f} (max {cfg.get('max_rounds', '?')})"
    elif "mean_turns" in m:
        rounds = f"{m['mean_turns']:.1f} turns"
    else:
        rounds = "—"

    cost = f"${m['total_cost_usd']:.2f}" if m.get("total_cost_usd") else "—"
    wall = f"{m['total_duration_s'] / 3600:.2f} h" if m.get("total_duration_s") else "—"
    judged = cfg.get("judge_model")

    return {
        "baseline": cfg.get("baseline", "?"),
        # The file stem is what separates e.g. `.ma-hard` from the `.control` run.
        "run": path.name.replace(".metrics.json", ""),
        "model": f"`{cfg.get('model', '?')}`" + ("" if judged else " ⚠️ unjudged"),
        "n_items": str(m.get("n_items", "?")),
        "compile_rate": fmt_pct(m.get("compile_rate")),
        "aligned_rate_of_compiling": fmt_pct(m.get("aligned_rate_of_compiling")),
        "joint_rate": fmt_pct(m.get("joint_rate")),
        "degenerate_rate": fmt_pct(m.get("degenerate_rate")),
        "rounds_or_turns": rounds,
        "cost": cost,
        "wall": wall,
        "_file": str(path),
        "_compile_at_k": m.get("compile_at_k"),
        "_sort": -(m.get("joint_rate") if m.get("joint_rate") is not None else m.get("compile_rate", 0) - 1),
    }


def render(rows: list[dict], sources: list[Path]) -> str:
    lines = [
        "# MA-Hard baseline results",
        "",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from {len(sources)} run(s).",
        "",
        "Headline is **Joint %** (compiles *and* judged aligned). Compile rate alone is",
        "inflated by degenerate statements — see the Degenerate % column.",
        "",
        "| " + " | ".join(h for _, h in COLUMNS) + " |",
        "|" + "|".join(["---"] * len(COLUMNS)) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r[k] for k, _ in COLUMNS) + " |")

    curves = [r for r in rows if r.get("_compile_at_k")]
    if curves:
        lines += ["", "## Compile@k (iterative saturation curve)", ""]
        ks = sorted({int(k) for r in curves for k in r["_compile_at_k"]})
        lines.append("| Run | " + " | ".join(f"k={k}" for k in ks) + " |")
        lines.append("|" + "|".join(["---"] * (len(ks) + 1)) + "|")
        for r in curves:
            cells = [fmt_pct(r["_compile_at_k"].get(str(k))) for k in ks]
            lines.append(f"| `{r['run']}` | " + " | ".join(cells) + " |")

    lines += ["", "## Sources", ""]
    lines += [f"- `{r['_file']}`" for r in rows]
    lines.append("")
    return "\n".join(lines)


@app.command()
def run(
    paths: list[Path] = typer.Argument(..., help="Metrics files, results files, or directories to scan."),
    output: Path | None = typer.Option(None, help="Also write the markdown table here."),
):
    """Summarize every baseline run found under `paths`."""
    sources = collect(paths)
    if not sources:
        typer.echo("No *.metrics.json files found.", err=True)
        raise typer.Exit(1)

    rows = []
    for path in sources:
        try:
            rows.append(row_for(path))
        except Exception as e:  # a half-written run shouldn't sink the summary
            typer.echo(f"Skipping `{path}`: {e}", err=True)
    rows.sort(key=lambda r: r["_sort"])

    table = render(rows, sources)
    print(table)
    if output:
        Path(output).write_text(table)
        typer.echo(f"\nWrote {output}", err=True)


if __name__ == "__main__":
    app()
