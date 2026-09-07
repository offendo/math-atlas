#!/usr/bin/env python3
"""B1 -- iterative compile-repair baseline on MA-Hard.

Generate a Lean 4 statement, compile it with `blv`, feed the compiler errors back
verbatim, repeat up to K rounds. Everything that compiles is then scored by the
CriticLean judge, so the headline number is joint (compiles AND aligned).

Example (vLLM server):

    python benchmarks/iterative/run_iterative.py \
        --model openai/gpt-oss-120b --model-url http://localhost:8000/v1 \
        --dataset offendo/math-atlas --filter split=hard \
        --max-rounds 5 --temperature 0.0 --retry-temperature 0.7 \
        --judge-model criticleangpt-qwen3-32b-rl --judge-model-url http://localhost:8001/v1 \
        --output outputs/iterative/gpt-oss-120b.ma-hard.json
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common  # noqa: E402
from generators import build_generator  # noqa: E402

app = typer.Typer(pretty_exceptions_show_locals=False)
logger = logging.getLogger("run_iterative")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
logger.setLevel(logging.INFO)

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def build_initial_conversation(row, system_prompt: str, theorem_tmpl: str, definition_tmpl: str) -> list[dict[str, str]]:
    tmpl = definition_tmpl if row["type"] in common.DEFINITION_TYPES else theorem_tmpl
    user = tmpl.format(text=row["text"], item_type=row["type"])
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def trim_history(conversation: list[dict[str, str]], max_history_rounds: int) -> list[dict[str, str]]:
    """Keep system + original task + the most recent N (assistant, repair) pairs.

    Long error transcripts blow past context on 32k-window local models, and the
    early rounds carry little signal once a few repairs have happened.
    """
    if max_history_rounds <= 0:
        return conversation
    head, tail = conversation[:2], conversation[2:]
    keep = 2 * max_history_rounds
    return head + tail[-keep:] if len(tail) > keep else conversation


@app.command()
def run(
    # --- data selection -----------------------------------------------------
    dataset: str = typer.Option("offendo/math-atlas", help="HuggingFace dataset name/path."),
    split: str = typer.Option("train", help="Dataset split (use this if MA-Hard is its own split)."),
    item_type: list[str] = typer.Option(["all"], help="Item type(s) to run; repeatable or 'all'."),
    filter: list[str] = typer.Option([], help="Column filter `col=value`; repeatable (e.g. --filter split=hard)."),
    subset_file: Path | None = typer.Option(None, help="File of MA-Hard uuids (JSON list / JSONL / plain text)."),
    n_examples: int | None = typer.Option(None, help="Subsample this many items (for debugging)."),
    seed: int = typer.Option(1337, help="Seed for subsampling and generation."),
    # --- generation ---------------------------------------------------------
    model: str = typer.Option(..., help="Model name (HF path for offline vLLM, or served model name)."),
    model_url: str | None = typer.Option(None, help="OpenAI-compatible base URL. Omit to run vLLM in-process."),
    api_key: str = typer.Option("EMPTY", envvar="OPENAI_API_KEY", help="API key for --model-url."),
    api_style: str = typer.Option("chat", help="`chat` (completions) or `responses`."),
    reasoning_effort: str | None = typer.Option(None, help="Reasoning effort, if the endpoint supports it."),
    max_tokens: int = typer.Option(8192, help="Max output tokens per round."),
    temperature: float = typer.Option(0.0, help="Temperature for the first attempt."),
    retry_temperature: float = typer.Option(0.7, help="Temperature for repair rounds (0 tends to loop on one error)."),
    top_p: float = typer.Option(0.95, help="Top-p sampling."),
    concurrency: int = typer.Option(20, help="Concurrent requests against --model-url."),
    tensor_parallel_size: int = typer.Option(1, help="vLLM tensor parallel size (offline mode)."),
    max_model_len: int = typer.Option(32768, help="vLLM max model length (offline mode)."),
    gpu_memory_utilization: float = typer.Option(0.90, help="vLLM GPU memory fraction (offline mode)."),
    # --- loop ---------------------------------------------------------------
    max_rounds: int = typer.Option(5, help="Max attempts per item (1 == single-pass control)."),
    max_history_rounds: int = typer.Option(2, help="Repair turns kept in context; 0 = keep everything."),
    max_error_chars: int = typer.Option(4000, help="Truncate compiler feedback to this many characters."),
    # --- prompts ------------------------------------------------------------
    system_prompt_file: Path = typer.Option(PROMPT_DIR / "system.txt", help="System prompt."),
    theorem_prompt_file: Path = typer.Option(PROMPT_DIR / "theorem.txt", help="Prompt for theorem/example/exercise."),
    definition_prompt_file: Path = typer.Option(PROMPT_DIR / "definition.txt", help="Prompt for definitions."),
    repair_prompt_file: Path = typer.Option(PROMPT_DIR / "repair.txt", help="Repair prompt, formatted with {errors}."),
    # --- verification -------------------------------------------------------
    verify_timeout: int = typer.Option(60, help="Per-theorem REPL timeout (seconds)."),
    force_header: bool = typer.Option(True, help="Force `import Mathlib` / `import Aesop` and drop model imports."),
    redis_host: str = typer.Option("localhost", help="Redis host for blv workers."),
    redis_port: int = typer.Option(6379, help="Redis port for blv workers."),
    redis_db: int = typer.Option(0, help="Redis DB for blv workers."),
    # --- judging ------------------------------------------------------------
    skip_judge: bool = typer.Option(False, help="Skip alignment judging (compile rate only)."),
    judge_model: str = typer.Option("criticleangpt-qwen3-32b-rl", help="Alignment judge model name."),
    judge_model_url: str | None = typer.Option(None, help="OpenAI-compatible base URL for the judge."),
    judge_prompt_file: Path = typer.Option(
        common.REPO_ROOT / "prompts" / "critic_lean_prompt.txt", help="Judge prompt for statements."
    ),
    judge_definition_prompt_file: Path = typer.Option(
        common.REPO_ROOT / "prompts" / "definition_alignment.txt", help="Judge prompt for definitions."
    ),
    judge_max_tokens: int = typer.Option(8192, help="Max judge output tokens."),
    judge_concurrency: int = typer.Option(20, help="Concurrent judge requests."),
    judge_api_key: str = typer.Option("EMPTY", help="API key for the judge endpoint."),
    judge_structured: bool = typer.Option(True, help="Request JSON-schema structured judge output."),
    # --- output -------------------------------------------------------------
    output: Path = typer.Option(..., dir_okay=False, help="Output JSON path."),
):
    """Run the compile-repair loop over MA-Hard."""
    df = common.load_items(dataset, split, item_type, filter, subset_file, n_examples, seed)
    system_prompt = system_prompt_file.read_text().strip()
    theorem_tmpl = theorem_prompt_file.read_text()
    definition_tmpl = definition_prompt_file.read_text()
    repair_tmpl = repair_prompt_file.read_text()

    generator = build_generator(
        model=model,
        model_url=model_url,
        api_key=api_key,
        api_style=api_style,
        max_tokens=max_tokens,
        top_p=top_p,
        concurrency=concurrency,
        reasoning_effort=reasoning_effort,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    conversations = [build_initial_conversation(row, system_prompt, theorem_tmpl, definition_tmpl) for _, row in df.iterrows()]
    history: list[list[dict]] = [[] for _ in range(len(df))]
    final_code: list[str] = ["" for _ in range(len(df))]
    final_compiler: list[dict | None] = [None for _ in range(len(df))]
    n_rounds = [0 for _ in range(len(df))]
    pending = list(range(len(df)))

    for rnd in range(1, max_rounds + 1):
        if not pending:
            break
        logger.info("Round %d/%d: %d item(s) pending", rnd, max_rounds, len(pending))
        temp = temperature if rnd == 1 else retry_temperature
        raw_outputs = generator.chat([conversations[i] for i in pending], temperature=temp, seed=seed)

        codes = [common.extract_lean(o) for o in raw_outputs]
        codes = [common.strip_imports(c) if force_header else c for c in codes]
        results = common.verify_batch(
            codes,
            timeout=verify_timeout,
            force_header=common.DEFAULT_HEADER if force_header else None,
            redis_host=redis_host,
            redis_port=redis_port,
            redis_db=redis_db,
        )

        still_pending: list[int] = []
        for idx, raw, code, res in zip(pending, raw_outputs, codes, results):
            verified = bool(res.get("verified"))
            errors = common.error_text(res, max_error_chars)
            history[idx].append(
                {"round": rnd, "raw_output": raw, "code": code, "verified": verified, "errors": None if verified else errors}
            )
            final_code[idx], final_compiler[idx], n_rounds[idx] = code, res, rnd
            if verified:
                continue
            conversations[idx] = trim_history(
                conversations[idx]
                + [{"role": "assistant", "content": raw}, {"role": "user", "content": repair_tmpl.format(errors=errors)}],
                max_history_rounds,
            )
            still_pending.append(idx)

        solved = len(pending) - len(still_pending)
        logger.info("Round %d: %d/%d newly compiled", rnd, solved, len(pending))
        pending = still_pending

    generator.close()

    out = pd.DataFrame(
        {
            "uuid": df["uuid"],
            "file_id": df["file_id"],
            "type": df["type"],
            "text": df["text"],
            "parsed_output": [{"text": c} for c in final_code],
            "code": final_code,
            "compiler_output": final_compiler,
            "rounds": history,
            "n_rounds": n_rounds,
            "verified": [bool((c or {}).get("verified")) for c in final_compiler],
            "degenerate": [common.is_degenerate(c) for c in final_code],
        }
    )

    if not skip_judge:
        common.judge_and_attach(
            out,
            judge_model=judge_model,
            judge_model_url=judge_model_url,
            judge_prompt_file=judge_prompt_file,
            judge_definition_prompt_file=judge_definition_prompt_file,
            judge_max_tokens=judge_max_tokens,
            judge_concurrency=judge_concurrency,
            api_key=judge_api_key,
            structured=judge_structured,
        )

    metrics = common.summarize(out, max_rounds=max_rounds)
    common.print_metrics(metrics)
    config = {
        "baseline": "B1-iterative-compile-repair",
        "model": model,
        "model_url": model_url,
        "dataset": dataset,
        "split": split,
        "item_type": list(item_type),
        "filter": list(filter),
        "subset_file": str(subset_file) if subset_file else None,
        "max_rounds": max_rounds,
        "temperature": temperature,
        "retry_temperature": retry_temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "max_history_rounds": max_history_rounds,
        "force_header": force_header,
        "judge_model": None if skip_judge else judge_model,
        "seed": seed,
    }
    common.save_results(out, metrics, output, config)


if __name__ == "__main__":
    app()
