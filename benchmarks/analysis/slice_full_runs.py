#!/usr/bin/env python3
"""E1a -- put the paper's full-set single-pass systems on MA-Hard, scored uniformly.

The paper evaluates fine-tuned formalizers (ReForm, Goedel, Kimina, Herald, ATLAS)
on the full benchmark but only prompted gpt-oss on MA-Hard. Their full-set
generations already cover MA-Hard, so no new generation is needed: slice each
output file to the MA-Hard uuids, extract the Lean the same way for every file,
(use the *processed* `.verified`/`.aligned` files: raw generation files still hold
unfilled placeholders such as Kimina's `theorem {thm_example}` that the original
pipeline fixed before compiling),
**re-verify with the current blv**, and write the standard baseline layout so
`judge_results.py` scores everything with the same CriticLean judge.

Also writes a pairwise identity check between the sliced runs (fraction of
items with byte-identical extracted code). Two different systems should almost
never agree exactly; a high value means a file holds another system's output.

    python benchmarks/analysis/slice_full_runs.py \
        --run reform-8b=/path/reform.statements.aligned.json --run herald-7b=... \
        --output-dir outputs/iclr/sliced
"""

from __future__ import annotations

import itertools
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common  # noqa: E402

app = typer.Typer(pretty_exceptions_show_locals=False)
logger = logging.getLogger("slice_full_runs")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
logger.setLevel(logging.INFO)


def stored_verified(rec: dict):
    """The original pipeline's verdict, when the source file carries one."""
    co = rec.get("compiler_output")
    return bool(co.get("verified")) if isinstance(co, dict) and "verified" in co else None


def raw_code(rec: dict) -> str:
    """First non-empty Lean-bearing field, in order of how processed it is."""
    po = rec.get("parsed_output")
    if isinstance(po, dict) and po.get("text"):
        return po["text"]
    for key in ("code", "formal"):
        if isinstance(rec.get(key), str) and rec[key].strip():
            return rec[key]
    fs = rec.get("formal_statements")
    if isinstance(fs, list) and fs and isinstance(fs[0], str):
        return fs[0]
    ro = rec.get("raw_output")
    if isinstance(ro, dict):
        ro = ro.get("text")
    return ro if isinstance(ro, str) else ""


@app.command()
def run(
    run: list[str] = typer.Option(..., help="`name=path/to/full-set-output.json`; repeatable."),
    output_dir: Path = typer.Option(Path("outputs/iclr/sliced")),
    dataset: str = typer.Option("offendo/math-atlas-official"),
    split: str = typer.Option("hard"),
    verify_timeout: int = typer.Option(60),
    skip_existing: bool = typer.Option(True, help="Skip runs whose sliced output already exists."),
):
    hard = common.load_items(dataset, split)
    meta = hard.set_index("uuid")
    output_dir.mkdir(parents=True, exist_ok=True)
    codes_by_run: dict[str, dict[str, str]] = {}
    coverage = {}

    for spec in run:
        name, _, path = spec.partition("=")
        out_path = output_dir / f"{name}.json"
        if skip_existing and out_path.exists():
            prev = pd.read_json(out_path)
            codes_by_run[name] = dict(zip(prev["uuid"], prev["code"]))
            logger.info("%s: exists, skipping (%d rows)", name, len(prev))
            continue
        logger.info("%s: loading %s", name, path)
        with open(path) as f:
            records = json.load(f)
        if isinstance(records, dict):  # pandas column-oriented dump: {col: {row: value}}
            records = pd.DataFrame(records).to_dict("records")
        rows, seen = [], set()
        for rec in records:
            uid = rec.get("uuid")
            if uid not in meta.index or uid in seen:
                continue
            seen.add(uid)
            code = common.strip_imports(common.extract_lean(raw_code(rec)))
            m = meta.loc[uid]
            rows.append({"uuid": uid, "file_id": m["file_id"], "type": m["type"], "text": m["text"],
                         "code": code, "parsed_output": {"text": code}, "stored_verified": stored_verified(rec)})
        del records
        df = pd.DataFrame(rows)
        if df.empty:
            logger.warning("%s: no MA-Hard items in %s", name, path)
            continue
        types = set(df["type"])
        eligible = hard[hard["type"].isin(types)] if types <= common.DEFINITION_TYPES else hard[~hard["type"].isin(common.DEFINITION_TYPES)]
        coverage[name] = {"source": path, "n_sliced": len(df), "n_eligible_in_ma_hard": len(eligible)}
        logger.info("%s: %d MA-Hard items (of %d eligible); verifying", name, len(df), len(eligible))

        results = common.verify_batch(df["code"].tolist(), timeout=verify_timeout)
        df["compiler_output"] = results
        df["verified"] = [bool(r.get("verified")) for r in results]
        df["degenerate"] = [common.is_degenerate(c) for c in df["code"]]
        metrics = common.summarize(df)
        if df["stored_verified"].notna().any():
            metrics["stored_compile_rate"] = float(df["stored_verified"].fillna(False).astype(bool).mean())
        common.save_results(df, metrics, out_path, {"baseline": "E1a-sliced-single-pass", "system": name,
                                                    "source": path, "dataset": dataset, "split": split})
        codes_by_run[name] = dict(zip(df["uuid"], df["code"]))

    identity = []
    for a, b in itertools.combinations(sorted(codes_by_run), 2):
        shared = set(codes_by_run[a]) & set(codes_by_run[b])
        if not shared:
            continue
        same = sum(codes_by_run[a][u] == codes_by_run[b][u] and bool(codes_by_run[a][u].strip()) for u in shared)
        identity.append({"run_a": a, "run_b": b, "n_shared": len(shared), "frac_identical_code": same / len(shared)})
    identity.sort(key=lambda r: -r["frac_identical_code"])
    (output_dir / "coverage.json").write_text(json.dumps(coverage, indent=2))
    (output_dir / "pairwise_identity.json").write_text(json.dumps(identity, indent=2))
    print(pd.DataFrame(identity).head(10).to_string(index=False))
    flagged = [r for r in identity if r["frac_identical_code"] > 0.2]
    if flagged:
        logger.warning("Suspicious duplicate outputs across systems: %s", flagged)


if __name__ == "__main__":
    app()
