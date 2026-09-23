#!/usr/bin/env python3
"""Judge (or re-judge) an existing baseline output file.

Both runners judge inline, but with one GPU pair you serve the generator and the
CriticLean judge at different times. So: run generation with `--skip-judge`,
then point this at the output files once the judge is up.

    python benchmarks/judge_results.py \
        --input outputs/iterative/gpt-oss-120b.ma-hard.json \
        --judge-model m-a-p/CriticLeanGPT-Qwen3-32B-RL \
        --judge-model-url http://localhost:8000/v1
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

app = typer.Typer(pretty_exceptions_show_locals=False)
logger = logging.getLogger("judge_results")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
logger.setLevel(logging.INFO)


@app.command()
def run(
    input: Path = typer.Option(..., dir_okay=False, help="Baseline output JSON (from B1 or A1)."),
    output: Path | None = typer.Option(None, dir_okay=False, help="Where to write; defaults to --input in place."),
    judge_model: str = typer.Option("m-a-p/CriticLeanGPT-Qwen3-32B-RL", help="Alignment judge model name."),
    judge_model_url: str | None = typer.Option(None, help="OpenAI-compatible base URL for the judge."),
    judge_api_key: str = typer.Option("EMPTY", help="API key for the judge endpoint."),
    judge_prompt_file: Path = typer.Option(
        common.REPO_ROOT / "prompts" / "critic_lean_prompt.txt", help="Judge prompt for statements."
    ),
    judge_definition_prompt_file: Path = typer.Option(
        common.REPO_ROOT / "prompts" / "definition_alignment.txt", help="Judge prompt for definitions."
    ),
    judge_max_tokens: int = typer.Option(8192, help="Max judge output tokens."),
    judge_concurrency: int = typer.Option(20, help="Concurrent judge requests."),
    judge_structured: bool = typer.Option(True, help="Request JSON-schema structured judge output."),
    judge_reasoning_effort: str | None = typer.Option(None, help="Reasoning effort for the judge (e.g. medium)."),
):
    """Attach alignment judgements to a finished run and recompute the metrics."""
    df = pd.read_json(input)
    logger.info("Loaded %d rows from `%s`", len(df), input)
    if "code" not in df.columns:  # tolerate outputs that only carry parsed_output
        df["code"] = df["parsed_output"].apply(lambda x: (x or {}).get("text", ""))

    common.judge_and_attach(
        df,
        judge_model=judge_model,
        judge_model_url=judge_model_url,
        judge_prompt_file=judge_prompt_file,
        judge_definition_prompt_file=judge_definition_prompt_file,
        judge_max_tokens=judge_max_tokens,
        judge_concurrency=judge_concurrency,
        api_key=judge_api_key,
        structured=judge_structured,
        reasoning_effort=judge_reasoning_effort,
    )

    out_path = output or input
    metrics = common.summarize(df)
    common.print_metrics(metrics)

    # Preserve the generation config recorded by the runner, if it is still there.
    config = {}
    prev_metrics = Path(input).with_suffix(".metrics.json")
    if prev_metrics.exists():
        import json

        config = json.loads(prev_metrics.read_text()).get("config", {})
    config["judge_model"] = judge_model
    config["judge_settings"] = {"max_tokens": judge_max_tokens, "reasoning_effort": judge_reasoning_effort,
                                "structured": judge_structured, "statement_prompt": str(judge_prompt_file),
                                "definition_prompt": str(judge_definition_prompt_file)}
    common.save_results(df, metrics, out_path, config)


if __name__ == "__main__":
    app()
