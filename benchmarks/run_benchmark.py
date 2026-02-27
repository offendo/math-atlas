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
from openai.types.responses import Response

from vllm import LLM, SamplingParams

app = typer.Typer(pretty_exceptions_show_locals=False)
logger = logging.getLogger("run_benchmark")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.WARNING
)
logger.setLevel(logging.INFO)


def format_fn(batch, template) -> dict[str, list[str]]:
    prompts = [template.format(text=text) for text in batch["text"]]
    return {"prompt": prompts}


def score_alignment(outputs: list[str], item_type: str) -> list[dict[str, Any]]:
    """Score alignment for generated outputs.

    TODO: Implement and return a list of dicts with keys: aligned (bool), reason (str).
    """
    raise NotImplementedError("score_alignment is not implemented yet.")


def generate(
    prompts: list[str],
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None = None,
    model_url: str | None = None,
    tensor_parallel_size: int = 1,
    dtype: str | None = None,
) -> list:
    if model_url is None:
        llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            dtype=dtype,
        )
        logger.info("Loaded model `%s`", model)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        raw_outputs: list[Response] = llm.generate(prompts, sampling_params)
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

            results = await asyncio.gather(*tasks)
            return results

        raw_outputs: list[Response] = asyncio.run(_run(prompts))

    return raw_outputs


@app.command()
def run(
    model: str = typer.Option(..., help="Model name or path for vLLM."),
    model_url: str | None = typer.Option(default=None, help="vLLM url"),
    dataset: str = typer.Option(
        ..., exists=False, dir_okay=False, help="Input huggingface dataset name/path."
    ),
    output: Path = typer.Option(..., dir_okay=False, help="Output JSON path."),
    item_type: str = typer.Option("theorem", help="Entity type to autoformalize. "),
    template_path: Path = typer.Option(
        None, dir_okay=False, help="Path to prompt template."
    ),
    max_tokens: int = typer.Option(4096, help="Max tokens to generate."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    top_p: float = typer.Option(1.0, help="Top-p sampling."),
    seed: int | None = typer.Option(None, help="Random seed."),
    tensor_parallel_size: int = typer.Option(1, help="Tensor parallel size."),
    dtype: str | None = typer.Option(None, help="vLLM dtype, e.g. bfloat16, float16."),
):
    """Run vLLM on a dataset, verify outputs, and save results."""

    ds = load_dataset(dataset, split="train")
    logger.info("Loaded dataset `%s`", dataset)
    with open(template_path, "r") as f:
        template = f.read()

    if item_type != "all":
        original_len = len(ds)
        ds = ds.filter(lambda ex: ex["type"] == item_type)
        logger.info(
            "Filtered dataset to %s (%i -> %i items)", item_type, original_len, len(ds)
        )

    ds = ds.map(lambda batch: format_fn(batch, template), batched=True)
    prompts = ds["prompt"]
    logger.info("Applied template from `%s`", template_path)

    logger.info("Finished generation!")
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
    model_outputs = [out.outputs[0].text for out in raw_outputs]

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

    verified_rate = sum([out["verified"] for out in compiler_output]) / len(
        compiler_output
    )
    aligned_rate = sum([out["aligned"] for out in alignment_output]) / len(
        alignment_output
    )

    print(f"Verified: {verified_rate:.2f}%")
    print(f"Aligned: {aligned_rate:.2f}%")

    df = pd.DataFrame(
        {
            "uuid": ds["uuid"],
            "file_id": ds["file_id"],
            "raw_output": [out.json() for out in raw_outputs],
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
