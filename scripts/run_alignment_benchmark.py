import asyncio
import json
import logging
import pandas as pd

import typer

from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from sklearn.metrics import classification_report
from datasets import load_dataset

app = typer.Typer(pretty_exceptions_show_locals=False)
logger = logging.getLogger("run_alignment_benchmark")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
logger.setLevel(logging.INFO)


SCHEMA = {
    "name": "reasoning_alignment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "The explanation or justification for the result."},
            "result": {
                "type": "string",
                "description": "Whether the object is aligned or misaligned.",
                "enum": ["aligned", "misaligned"],
            },
        },
        "required": ["reasoning", "result"],
        "additionalProperties": False,
    },
}


def load_prompt(path: Path):
    with open(path, "r") as f:
        return f.read()


def make_prompt(informal, formal, prompt):
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Informal:\n{informal}\n\nFormal:\n{formal}"},
    ]


def try_json_loads(s):
    try:
        out = json.loads(s)
        out.update({'error': None})
        return out
    except Exception as e:
        return {'error': e, 'result': "misaligned", "reasoning": s}

async def async_run(model_url, prompts, **kwargs):
    client = AsyncOpenAI(base_url=model_url)
    semaphore = asyncio.Semaphore(20)

    async def complete(prompt):
        async with semaphore:
            response = await client.responses.create(input=prompt, **kwargs)
        return response

    async def _run(prompts):
        tasks = []
        for prompt in prompts:
            task = complete(prompt)
            tasks.append(task)

        results = await tqdm.gather(*tasks)
        return results

    return await _run(prompts)


@app.command()
def run(
    model: str = typer.Option(..., help="Model name or path for vLLM."),
    model_url: str | None = typer.Option(default=None, help="vLLM url"),
    dataset: str = typer.Option(..., help="Input huggingface dataset name/path."),
    prompt_file: str = typer.Option(..., dir_okay=False, help="Prompt path."),
    output: Path = typer.Option(..., dir_okay=False, help="Path to save metrics to."),
    max_tokens: float = typer.Option(10000, help="Max output tokens"),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    top_p: float = typer.Option(1.0, help="Top-p sampling."),
    seed: int | None = typer.Option(None, help="Random seed."),
    n_examples: int | None = typer.Option(None, help="Number of samples to run (for debugging)"),
):
    # load dataset
    prompt = load_prompt(prompt_file)
    ds = load_dataset(dataset, split="train")
    ds = ds.map(lambda ex: {"prompt": make_prompt(ex["informal"], ex["formal"], prompt)})
    if n_examples is not None:
        ds = ds.shuffle(seed).select(range(n_examples))
    prompts = ds["prompt"]

    # launch async jobs
    raw_outputs = asyncio.run(
        async_run(
            model_url,
            prompts,
            model=model,
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning={"reasoning_effort": "high"},
        )
    )

    # parse and save outputs
    parsed_outputs = [try_json_loads(out.output[1].content[0].text) for out in raw_outputs]
    df = pd.DataFrame.from_records(parsed_outputs)
    df['label'] = ds['label']
    df.to_json(output)

    # evaluate score
    predictions = df['result']
    golds = df['label']

    report = classification_report(golds, predictions)
    print(report)
    with open(Path(output).with_suffix('.metrics'), "w") as f:
        f.write(report)


if __name__ == "__main__":
    app()
