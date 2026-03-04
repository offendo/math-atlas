"""Run model benchmarks with vLLM and verify outputs."""

import asyncio
import logging
from pathlib import Path
from typing import Any

import blv
import pandas as pd
import typer
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer
from dataclasses import dataclass, asdict

from benchmarks.formatters import get_formatter, get_output_parser


@dataclass
class Output:
    thinking: str
    text: str

app = typer.Typer(pretty_exceptions_show_locals=False)
logger = logging.getLogger("run_benchmark")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
logger.setLevel(logging.INFO)


def format_fn(batch, fn) -> dict[str, list[str]]:
    prompts = [fn(text, names) for text, names in zip(batch["text"], batch["names"])]
    return {"prompt": prompts}


def score_alignment(outputs: list[str], item_type: str) -> list[dict[str, Any]]:
    """Score alignment for generated outputs.

    TODO: Implement and return a list of dicts with keys: aligned (bool), reason (str).
    """
    raise NotImplementedError("score_alignment is not implemented yet.")


def generate(
    prompts: list[list[dict[str, str]]],
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None = None,
    model_url: str | None = None,
    tensor_parallel_size: int = 1,
    dtype: str | None = None,
) -> list[Output]:
    parse_output = get_output_parser(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    if model_url is None:
        from vllm import LLM, SamplingParams

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
        )
        llm_outputs = llm.chat(prompts, sampling_params, use_tqdm=True)
        outputs = []
        for out in llm_outputs:
            if len(out.outputs) > 1:
                thinking = out.outputs[0].content[0].text
                code = out.outputs[1].content[0].text
                outputs.append(Output(thinking=thinking, text=code))
            else:
                full = tokenizer.decode(out.outputs[0].token_ids)
                thinking, code = parse_output(full)
                if thinking is None:
                    outputs.append(Output(thinking=None, text=out.outputs[0].text))
                else:
                    outputs.append(Output(thinking=thinking, text=code))
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

        raw_outputs: list[Response] = asyncio.run(_run(prompts))
        outputs = []
        for out in raw_outputs:
            thinking, code = parse_output(out.output_text)
            outputs.append(Output(thinking=thinking, text=code))

    return outputs


@app.command()
def run(
    model: str = typer.Option(..., help="Model name or path for vLLM."),
    model_url: str | None = typer.Option(default=None, help="vLLM url"),
    dataset: str = typer.Option(..., exists=False, dir_okay=False, help="Input huggingface dataset name/path."),
    output: Path = typer.Option(..., dir_okay=False, help="Output JSON path."),
    item_type: str = typer.Option("theorem", help="Entity type to autoformalize. "),
    max_tokens: int = typer.Option(4096, help="Max tokens to generate."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    top_p: float = typer.Option(1.0, help="Top-p sampling."),
    seed: int | None = typer.Option(None, help="Random seed."),
    tensor_parallel_size: int = typer.Option(1, help="Tensor parallel size."),
    dtype: str | None = typer.Option(None, help="vLLM dtype, e.g. bfloat16, float16."),
    precomputed_generations: Path = typer.Option(None, help="path to precomputed generations (json), if exists"),
):
    """Run vLLM on a dataset, verify outputs, and save results."""

    ds = load_dataset(dataset, split="train")
    logger.info("Loaded dataset `%s`", dataset)
    if item_type != "all":
        original_len = len(ds)
        ds = ds.filter(lambda ex: ex["type"] == item_type)
        logger.info("Filtered dataset to %s (%i -> %i items)", item_type, original_len, len(ds))

    model_formatter = get_formatter(model)
    ds = ds.map(lambda batch: format_fn(batch, model_formatter), batched=True)
    prompts = ds["prompt"]

    if precomputed_generations:
        df = pd.read_json(precomputed_generations)
        raw_outputs = list(df["raw_output"])
        model_outputs = list(df["parsed_output"])
    else:
        raw_outputs = generate(
            prompts,
            model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            model_url=model_url,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
        )
        model_outputs = [out.text for out in raw_outputs]
    logger.info("Finished generation!")

    compiler_output = blv.verify_theorems(
        model_outputs,
        force_header=("import Mathlib", "import Aesop"),
    )
    logger.info("Finished compile rate check.")
    try:
        alignment_output = score_alignment(model_outputs, item_type)
        logger.info("Finished alignment rate check.")
    except Exception:
        logger.warning("Alignment check not implemented. Skipping for now.")
        alignment_output = [dict(aligned=False) for _ in model_outputs]

    verified_rate = sum([out["verified"] for out in compiler_output]) / len(compiler_output)
    aligned_rate = sum([out["aligned"] for out in alignment_output]) / len(alignment_output)

    print(f"Verified: {verified_rate:.2f}%")
    print(f"Aligned: {aligned_rate:.2f}%")

    df = pd.DataFrame(
        {
            "uuid": ds["uuid"],
            "file_id": ds["file_id"],
            "raw_output": raw_outputs,
            "parsed_output": model_outputs,
            "compiler_output": compiler_output,
            "alignment_output": alignment_output,
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output)
    logger.info(f"Saved results to {output}")


if __name__ == "__main__":
    app()
