#!/usr/bin/env python3
"""Run the Aria autoformalizer (github.com/frenzymath/Aria) over MA-Hard.

Aria lives in its own repo and venv, so generation is delegated to
`aria_driver.py` running under Aria's interpreter; this script selects items,
re-verifies Aria's output with `blv` (the same verifier every other baseline
uses, so Aria's self-reported compile verdict is not trusted), and writes the
usual output/metrics pair. Judge afterwards with `benchmarks/judge_results.py`.

Services Aria needs while generating:
  * an OpenAI-compatible LLM endpoint      (Aria configs/config.yaml, `llm.*`)
  * the Lean verifier on :8001/verify      (Aria-autoformalizer/verify-server/run.sh)
  * LeanSearch, only with --rag            (Aria configs/leansearch.yaml)

    python benchmarks/aria/run_aria.py --split hard \
        --output outputs/iclr/aria/aria-opus.ma-hard.json --resume
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import typer
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402

app = typer.Typer(pretty_exceptions_show_locals=False)
logger = logging.getLogger("run_aria")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
logger.setLevel(logging.INFO)

ARIA_REPO = Path.home() / "src" / "Aria-autoformalizer"
DRIVER = Path(__file__).resolve().parent / "aria_driver.py"


def read_partial(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


@app.command()
def run(
    # --- item selection -----------------------------------------------------
    dataset: str = typer.Option("offendo/math-atlas-official", help="HF dataset."),
    split: str = typer.Option("hard", help="Dataset split."),
    item_type: list[str] = typer.Option(["all"], help="Item types; repeatable."),
    filter: list[str] = typer.Option([], help="col=value filters; repeatable."),
    subset_file: Path | None = typer.Option(None, help="uuid list restricting the items."),
    n_examples: int | None = typer.Option(None, help="Subsample (smoke tests)."),
    seed: int = typer.Option(1337),
    # --- aria ---------------------------------------------------------------
    aria_root: Path = typer.Option(ARIA_REPO / "Aria-autoformalizer", help="Aria project dir."),
    aria_python: Path = typer.Option(
        ARIA_REPO / "Aria-autoformalizer" / ".venv" / "bin" / "python", help="Interpreter with Aria's deps."
    ),
    model: str | None = typer.Option(None, help="Override Aria's LLM (both roles); default keeps its config."),
    base_url: str | None = typer.Option(None, help="Override Aria's LLM base URL."),
    verify_url: str | None = typer.Option(None, help="Aria's Lean verifier (ARIA_VERIFY_URL)."),
    rag: bool = typer.Option(False, help="Enable Aria's LeanSearch grounding."),
    concurrency: int = typer.Option(16, help="Items formalized in parallel."),
    timeout: int = typer.Option(3600, help="Per-item wall clock (seconds)."),
    skip_generate: bool = typer.Option(False, help="Only verify/collect what the checkpoint already holds."),
    # --- verification -------------------------------------------------------
    verify_timeout: int = typer.Option(60, help="Per-theorem REPL timeout (seconds)."),
    redis_host: str = typer.Option("localhost"),
    redis_port: int = typer.Option(6379),
    redis_db: int = typer.Option(0),
    # --- output -------------------------------------------------------------
    output: Path = typer.Option(..., dir_okay=False, help="Output JSON path."),
    resume: bool = typer.Option(False, help="Keep the checkpoint and skip items already in it."),
):
    """Generate with Aria, re-verify with blv, save results + metrics."""
    df = common.load_items(dataset, split, item_type, filter, subset_file, n_examples, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = str(output).removesuffix(".json")
    partial_path = Path(stem + ".partial.jsonl")
    if not resume and not skip_generate and partial_path.exists():
        partial_path.unlink()

    if not skip_generate:
        input_path = Path(stem + ".input.jsonl")
        with open(input_path, "w") as f:
            for _, row in df.iterrows():
                f.write(json.dumps({"uuid": row["uuid"], "text": row["text"]}, ensure_ascii=False) + "\n")
        cmd = [
            str(aria_python), str(DRIVER),
            "--aria-root", str(aria_root),
            "--input", str(input_path.resolve()),
            "--partial", str(partial_path.resolve()),
            "--log", str(Path(stem + ".aria.log").resolve()),
            "--concurrency", str(concurrency),
            "--timeout", str(timeout),
            "--rag" if rag else "--no-rag",
        ]
        for flag, val in (("--model", model), ("--base-url", base_url), ("--verify-url", verify_url)):
            if val:
                cmd += [flag, val]
        logger.info("Running Aria: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    records = {r["uuid"]: r for r in read_partial(partial_path)}  # last write wins
    df = df[df["uuid"].isin(records)].reset_index(drop=True)
    missing = len(records) - len(df)
    if missing:
        logger.warning("%d checkpointed uuids are outside the current selection; ignoring them.", missing)
    logger.info("Collected %d Aria outputs", len(df))

    gen = pd.DataFrame([records[u] for u in df["uuid"]])
    out = df[["uuid", "file_id", "type", "text"]].copy()
    out["code"] = [common.strip_imports(c or "") for c in gen["code"]]
    out["parsed_output"] = [{"text": c} for c in out["code"]]
    for col in ("aria_success", "aria_error", "generated_context", "stats", "duration_s", "error"):
        out[col] = gen[col].tolist()

    results = common.verify_batch(
        out["code"].tolist(),
        timeout=verify_timeout,
        force_header=common.DEFAULT_HEADER,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=redis_db,
    )
    out["compiler_output"] = results
    out["verified"] = [bool(r.get("verified")) for r in results]
    out["degenerate"] = [common.is_degenerate(c) for c in out["code"]]
    out["alignment_output"] = None
    out["aligned"] = False

    metrics = common.summarize(out)
    common.print_metrics(metrics)
    aria_llm = yaml.safe_load((aria_root / "configs" / "config.yaml").read_text())["llm"]["main_model"]
    config = {
        "baseline": "aria",
        "model": model or aria_llm["model"],
        "base_url": base_url or aria_llm["base_url"],
        "rag": rag,
        "concurrency": concurrency,
        "timeout_s": timeout,
        "dataset": dataset,
        "split": split,
        "item_type": list(item_type),
        "filter": list(filter),
        "subset_file": str(subset_file) if subset_file else None,
        "n_examples": n_examples,
        "judge_model": None,
        "seed": seed,
    }
    common.save_results(out, metrics, output, config)


if __name__ == "__main__":
    app()
