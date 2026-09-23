#!/usr/bin/env python3
"""Audit A1 agent transcripts: tool usage per arm, and a leakage check.

For every item in an A1 output file, reads the Claude Code session transcript
(`~/.claude/projects/<encoded project path>/<session_id>.jsonl`) and records:

  * per-item tool-call counts, and whether any MathAtlas MCP tool was used
    (how "dependency-aware" the arm actually was, not just what it was offered)
  * **leak hits**: tool calls whose input touches the raw dataset or result files
    (the agent has Bash; the MCP redacts the item's Mathlib grounding, but the raw
    NDJSON on disk does not)

    python benchmarks/analysis/agent_tool_usage.py \
        outputs/iclr/agentic/claude-code-sonnet.dep.json [more.json ...] \
        --output outputs/iclr/analysis/agent_tool_usage.json
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import typer

app = typer.Typer(pretty_exceptions_show_locals=False)

LEAK_MARKERS = (
    "mathatlas-formalization/data",
    "MathAtlas.json",
    "math-atlas.json",
    "math-atlas/outputs",
    "math-atlas-official",
    "huggingface/datasets",
    "huggingface/hub/datasets",
)


def transcript_dir(project: str) -> Path:
    """Claude Code stores sessions under the project path with non-alphanumerics -> '-'."""
    return Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", project)


def audit_file(path: Path) -> dict:
    rows = json.loads(Path(path).read_text())
    cfg_path = Path(path).with_suffix(".metrics.json")
    project = json.loads(cfg_path.read_text())["config"]["project"] if cfg_path.exists() else None
    tdir = transcript_dir(project) if project else None

    per_item, totals, leaks, missing = [], collections.Counter(), [], 0
    for r in rows:
        sid = r.get("session_id")
        tfile = tdir / f"{sid}.jsonl" if (tdir and sid) else None
        counts: collections.Counter = collections.Counter()
        if tfile is None or not tfile.exists():
            missing += 1
        else:
            for line in tfile.read_text().splitlines():
                try:
                    content = (json.loads(line).get("message") or {}).get("content")
                except json.JSONDecodeError:
                    continue
                if not isinstance(content, list):
                    continue
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        name = block.get("name", "?")
                        counts[name] += 1
                        blob = json.dumps(block.get("input"))
                        if any(m in blob for m in LEAK_MARKERS):
                            leaks.append({"uuid": r["uuid"], "tool": name, "input": blob[:300]})
        totals.update(counts)
        ma = {k: v for k, v in counts.items() if k.startswith("mcp__mathatlas")}
        per_item.append(
            {
                "uuid": r["uuid"],
                "n_tool_calls": sum(counts.values()),
                "mathatlas_calls": sum(ma.values()),
                "used_get_dependencies": counts.get("mcp__mathatlas__mathatlas_get_dependencies", 0) > 0,
                "used_mathatlas": bool(ma),
            }
        )
    n = len(rows)
    return {
        "file": str(path),
        "project": project,
        "n_items": n,
        "transcripts_missing": missing,
        "frac_items_using_mathatlas": sum(p["used_mathatlas"] for p in per_item) / n if n else 0.0,
        "frac_items_using_get_dependencies": sum(p["used_get_dependencies"] for p in per_item) / n if n else 0.0,
        "mean_tool_calls": sum(p["n_tool_calls"] for p in per_item) / n if n else 0.0,
        "tool_totals": dict(totals.most_common()),
        "leak_hits": leaks,
        "per_item": per_item,
    }


@app.command()
def run(inputs: list[Path], output: Path = typer.Option(None, help="Write the full audit JSON here.")):
    audits = [audit_file(p) for p in inputs]
    for a in audits:
        print(f"== {a['file']}")
        print(f"   items={a['n_items']} missing_transcripts={a['transcripts_missing']}")
        print(f"   used any MathAtlas tool: {100 * a['frac_items_using_mathatlas']:.1f}%  "
              f"get_dependencies: {100 * a['frac_items_using_get_dependencies']:.1f}%  "
              f"mean tool calls: {a['mean_tool_calls']:.1f}")
        print(f"   LEAK HITS: {len(a['leak_hits'])}")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(audits, indent=2))


if __name__ == "__main__":
    app()
