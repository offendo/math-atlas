#!/usr/bin/env python3
"""Token-usage analysis for the MA-Hard runs.

Claude Code arms: exact per-item token counts summed from the session transcripts
(main + subagent files, deduped by message id).

Sonnet iterative arms: run with `--no-session-persistence`, so no token counts were
recorded; only `config.generation_cost_usd` and the raw outputs exist. We (a) fit
per-category $/token prices by regressing Claude Code item costs on their exact token
categories, (b) estimate output tokens from `rounds[].raw_output` length, and
(c) back out an input-token equivalent from the remaining cost. Those rows are ESTIMATES.

    python benchmarks/analysis/token_usage.py --root outputs/iclr \
        --output outputs/iclr/analysis/token_usage.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(pretty_exceptions_show_locals=False)
CC = {
    "Claude Code (optional deps)": "agentic/claude-code-sonnet.opt.json",
    "Claude Code (enforced deps)": "agentic/claude-code-sonnet.dep.json",
}
SONNET = {
    "Sonnet single-pass": "iterative/sonnet.sp.dep-none.json",
    "Sonnet iterative (K=5)": "iterative/sonnet.k5.dep-none.json",
    "Sonnet + deps in context": "iterative/sonnet.sp.dep-both.json",
}
CHARS_PER_TOKEN = 3.0  # Lean/LaTeX-heavy text; rough


def transcript_dir(project: str) -> Path:
    return Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", project)


def item_usage(tdir: Path, sid: str) -> dict | None:
    f = tdir / f"{sid}.jsonl"
    if not f.exists():
        return None
    files = [f, *sorted((f.parent / f.stem / "subagents").glob("*.jsonl"))]
    seen, tot = set(), dict(inp=0, out=0, cw=0, cr=0, think=0, calls=0)
    for fp in files:
        for line in fp.read_text().splitlines():
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = o.get("message") or {}
            u = m.get("usage")
            if o.get("type") != "assistant" or not u:
                continue
            key = m.get("id") or o.get("requestId") or o.get("uuid")
            if key in seen:
                continue
            seen.add(key)
            tot["inp"] += u.get("input_tokens", 0) or 0
            tot["out"] += u.get("output_tokens", 0) or 0
            tot["cw"] += u.get("cache_creation_input_tokens", 0) or 0
            tot["cr"] += u.get("cache_read_input_tokens", 0) or 0
            tot["think"] += (u.get("output_tokens_details") or {}).get("thinking_tokens", 0) or 0
            tot["calls"] += 1
    return tot


def stats(v):
    a = np.array(v, float)
    return dict(mean=a.mean(), median=float(np.median(a)), p90=float(np.percentile(a, 90)), total=a.sum())


@app.command()
def main(root: Path = Path("outputs/iclr"), output: Path = Path("outputs/iclr/analysis/token_usage.json")):
    result, X, y = {}, [], []
    for name, rel in CC.items():
        rows = json.loads((root / rel).read_text())
        cfg = json.loads((root / rel).with_suffix(".metrics.json").read_text())["config"]
        tdir = transcript_dir(cfg["project"])
        per, missing = [], 0
        for r in rows:
            u = item_usage(tdir, r.get("session_id") or "")
            if u is None:
                missing += 1
                continue
            u["cost"] = r.get("cost_usd") or 0.0
            u["turns"] = r.get("num_turns") or 0
            per.append(u)
            X.append([u["inp"], u["out"], u["cw"], u["cr"]])
            y.append(u["cost"])
        res = {"n_items": len(rows), "n_missing_transcripts": missing, "exact": True}
        for k in ("inp", "out", "cw", "cr", "think", "calls", "cost", "turns"):
            res[k] = stats([p[k] for p in per])
        res["total_tokens_per_item"] = stats([p["inp"] + p["out"] + p["cw"] + p["cr"] for p in per])
        res["fresh_tokens_per_item"] = stats([p["inp"] + p["out"] + p["cw"] for p in per])
        result[name] = res

    Xa, ya = np.array(X, float), np.array(y)
    price, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
    r2 = 1 - ((ya - Xa @ price) ** 2).sum() / ((ya - ya.mean()) ** 2).sum()
    p_in, p_out = price[0], price[1]
    result["_fit"] = {
        "usd_per_mtok": dict(zip(["input", "output", "cache_write", "cache_read"], (price * 1e6).tolist())),
        "r2": r2,
    }

    for name, rel in SONNET.items():
        rows = json.loads((root / rel).read_text())
        cfg = json.loads((root / rel).with_suffix(".metrics.json").read_text())["config"]
        cost = cfg["generation_cost_usd"]
        out_chars = [sum(len(x.get("raw_output") or "") for x in r.get("rounds") or []) for r in rows]
        out_tok = np.array(out_chars) / CHARS_PER_TOKEN
        calls = [len(r.get("rounds") or []) for r in rows]
        in_tok_total = max(cost - out_tok.sum() * p_out, 0) / p_in
        result[name] = {
            "n_items": len(rows), "exact": False, "cost_total": cost, "cost_per_item": cost / len(rows),
            "calls_per_item": stats(calls), "est_output_tokens_per_item": stats(out_tok),
            "est_input_equiv_tokens_per_item_mean": in_tok_total / len(rows),
            "est_total_tokens_per_item_mean": (in_tok_total + out_tok.sum()) / len(rows),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=float))
    for k, v in result.items():
        print(k, json.dumps(v, default=float)[:900])


if __name__ == "__main__":
    app()
