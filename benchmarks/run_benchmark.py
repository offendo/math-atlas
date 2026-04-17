"""Run model benchmarks with vLLM and verify outputs."""

import asyncio
import logging
from pathlib import Path
from typing import Any
import json

import blv
import pandas as pd
import typer
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer

from benchmarks.formatters import get_formatter
from benchmarks.formatters.base import Output, BaseFormatter

app = typer.Typer(pretty_exceptions_show_locals=False)
logger = logging.getLogger("run_benchmark")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
logger.setLevel(logging.INFO)


def score_alignment(informals: list[str], formals: list[str]) -> list[dict[str, Any]]:
    """Score alignment for generated outputs.

    TODO: Implement and return a list of dicts with keys: aligned (bool), reason (str).
    """
    raise NotImplementedError("score_alignment is not implemented yet.")


def generate(
    prompts: list[list[dict[str, str]]],
    model: str,
    formatter: BaseFormatter,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None = None,
    model_url: str | None = None,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    dtype: str | None = None,
) -> tuple[list, list]:
    if model_url is None:
        from vllm import LLM, SamplingParams  # type:ignore

        llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
        )
        logger.info("Loaded model `%s`", model)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            skip_special_tokens=False,
        )
        if isinstance(prompts[0], list):
            llm_outputs = llm.chat(prompts, sampling_params, use_tqdm=True)
        else:
            llm_outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        raw_outputs = []
        outputs = []
        for out in llm_outputs:
            full = out.outputs[0].text
            output = formatter.parse_output(full)
            outputs.append(output)
            raw_outputs.append(full)
    else:
        client = AsyncOpenAI(base_url=model_url)
        semaphore = asyncio.Semaphore(30)

        async def complete(prompt):
            async with semaphore:
                response = await client.responses.create(
                    model=model,
                    input=prompt,
                    reasoning={"effort": "low"},
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    top_p=top_p,
                )
            return response

        async def _run(prompts):
            # TODO input the other sampling parameters here
            tasks = []
            for prompt in prompts:
                task = complete(prompt)
                tasks.append(task)

            results = await tqdm.gather(*tasks)
            return results

        llm_outputs: list = asyncio.run(_run(prompts))
        raw_outputs = []
        outputs = []
        for out in llm_outputs:
            output = formatter.parse_output(out.output_text)
            outputs.append(output)
            raw_outputs.append(out.output_text)

    return raw_outputs, outputs


@app.command()
def run(
    item_type: list[str] = typer.Argument(..., help="Entity type(s) to autoformalize; repeatable or 'all'."),
    model: str = typer.Option(..., help="Model name or path for vLLM."),
    model_url: str | None = typer.Option(default=None, help="vLLM url"),
    dataset: str = typer.Option(..., exists=False, dir_okay=False, help="Input huggingface dataset name/path."),
    output: Path = typer.Option(..., dir_okay=False, help="Output JSON path."),
    max_tokens: int = typer.Option(4096, help="Max tokens to generate."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    top_p: float = typer.Option(1.0, help="Top-p sampling."),
    seed: int | None = typer.Option(None, help="Random seed."),
    tensor_parallel_size: int = typer.Option(1, help="Tensor parallel size."),
    data_parallel_size: int = typer.Option(1, help="Data parallel size."),
    skip_verification: bool = typer.Option(False, help="Skip verification"),
    n_context_tokens: int | None = typer.Option(None, help="Number of tokens of prior context to include"),
):
    """Run vLLM on a dataset, verify outputs, and save results."""

    ds = load_dataset(dataset, split="train")
    logger.info("Loaded dataset `%s`", dataset)
    # handle multiple requested item types
    if "all" not in item_type:
        original_len = len(ds)
        ds = ds.filter(lambda ex, types=item_type: ex["type"] in types)
        logger.info("Filtered dataset to %s (%i -> %i items)", item_type, original_len, len(ds))

    id2tokens = None
    if n_context_tokens is not None:
        context_ds = load_dataset("offendo/math-atlas-documents", split='train')
        id2text = {item['file_id']: item['content'] for item in context_ds.to_list()}
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenized_content = tokenizer(id2text.values(), truncation=False)
        id2tokens = {file_id: tokens for file_id, tokens in zip(id2text.keys(), tokenized_content.input_ids)}


    model_formatter = get_formatter(model)
    ds = ds.map(lambda batch: model_formatter.format_batch(batch, id2tokens=id2tokens, n_tokens=n_context_tokens), batched=True)

    raw_outputs, parsed_outputs = generate(
        ds["prompt"],
        model,
        model_formatter,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        model_url=model_url,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
    )
    logger.info("Finished generation!")

    # Save generations in case verification goes wrong
    df = pd.DataFrame(
        {
            "uuid": ds["uuid"],
            "file_id": ds["file_id"],
            "raw_output": raw_outputs,
            "parsed_output": parsed_outputs,
        }
    )
    answers = [out.text for out in parsed_outputs]
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output)

    if not skip_verification:
        compiler_output = blv.verify(
            answers,
            force_header=("import Mathlib", "import Aesop"),
        )
        logger.info("Finished compile rate check.")
        try:
            alignment_output = score_alignment(ds["text"], answers)
            logger.info("Finished alignment rate check.")
        except Exception:
            logger.warning("Alignment check not implemented. Skipping for now.")
            alignment_output = [dict(aligned=False) for _ in answers]

        verified_rate = sum([out["verified"] if out else 0 for out in compiler_output]) / len(compiler_output)
        aligned_rate = sum([out["aligned"] for out in alignment_output]) / len(alignment_output)

        print(f"Verified: {100 * verified_rate:.2f}%")
        print(f"Aligned: {100 * aligned_rate:.2f}%")

        df = pd.DataFrame(
            {
                "uuid": ds["uuid"],
                "file_id": ds["file_id"],
                "raw_output": raw_outputs,
                "parsed_output": parsed_outputs,
                "compiler_output": compiler_output,
                "alignment_output": alignment_output,
            }
        )
        df.to_json(output)
        metric_path = output.with_suffix(".metrics.json")
        with open(metric_path, "w") as f:
            metrics = {
                "verified_rate": verified_rate,
                "aligned_rate": aligned_rate,
            }
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved results to {output} and metrics to {metric_path}")



if __name__ == "__main__":
    app()
