"""Run alignment scoring with LLM evaluation."""

import asyncio
import logging
import time
import tempfile
from pathlib import Path
from typing import Any
import uuid
import json

import pandas as pd
import typer
from datasets import load_dataset
from openai import AsyncOpenAI, OpenAI
from tqdm.asyncio import tqdm

app = typer.Typer(pretty_exceptions_show_locals=False)
logger = logging.getLogger("run_alignment_score")
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


def parse(x):
    try:
        return json.loads(x)
    except Exception as e:
        return {'result': 'misaligned', 'error': e, 'error_response': x}

def write_batch_file(
    prompts: list[list[dict[str, str]]],
    output_file: Path,
    model: str,
    max_tokens: int,
):
    """Write prompts to OpenAI Batch API JSONL format."""
    with open(output_file, "w", encoding="utf-8") as f:
        for prompt in prompts:
            req = {
                "custom_id": str(uuid.uuid4()),
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "input": prompt,
                    "max_output_tokens": max_tokens,
                    "text": {"format": {"type": "json_schema", **SCHEMA}},
                },
            }
            f.write(json.dumps(req) + "\n")

    logger.info("Wrote batch file to `%s`", output_file)

def generate_alignment_scores(
    prompts: list[list[dict[str, str]]],
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    model_url: str,
) -> list[str]:
    """Generate alignment scores using LLM API."""
    client = AsyncOpenAI(base_url=model_url)
    semaphore = asyncio.Semaphore(30)

    async def complete(prompt):
        async with semaphore:
            response = await client.responses.create(
                model=model,
                input=prompt,
                reasoning={"effort": "high"},
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=top_p,
                text={"format": {"type": "json_schema", **SCHEMA}},
            )
        return response

    async def _run(prompts):
        tasks = []
        for prompt in prompts:
            task = complete(prompt)
            tasks.append(task)

        results = await tqdm.gather(*tasks)
        return results

    llm_outputs: list = asyncio.run(_run(prompts))
    raw_outputs = [out.output_text for out in llm_outputs]

    return raw_outputs


@app.command()
def run(
    prompt_path: Path = typer.Option(..., help="Path to the prompt file (system message)."),
    input_path: Path = typer.Option(..., help="Path to the input prediction JSON file."),
    model: str = typer.Option(..., help="Model name."),
    model_url: str = typer.Option(..., help="Model URL for API."),
    output_path: Path | None = typer.Option(None, help="Output JSON path for results."),
    batch_mode: bool = typer.Option(False, help="If set, write a batch JSONL file instead of calling the API."),
    max_tokens: int = typer.Option(4096, help="Max tokens to generate."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    top_p: float = typer.Option(1.0, help="Top-p sampling."),
    n_examples: int | None = typer.Option(None, help="Number of examples to run (for debugging)."),
    entity_types: list[str] | None = typer.Option(None, help="Entity type(s) to autoformalize; repeatable or 'all'."),
):
    """Run LLM-based alignment scoring on predictions."""

    if not batch_mode:
        assert output_path is not None, "If not using --batch-mode, then you must supply an --output-path to save results"

    # Load system prompt
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()
    logger.info("Loaded system prompt from `%s`", prompt_path)

    # Load math-atlas to get informal inputs
    ds = load_dataset('offendo/math-atlas', split='train').to_pandas().set_index('uuid')

    # Load predictions
    df = pd.read_json(input_path)
    if n_examples is not None:
        df = df.sample(n_examples, random_state=1337)
    if entity_types is not None and 'all' not in entity_types:
        df = df[df.type.isin(entity_types)]
    logger.info("Loaded predictions from `%s` (%d rows)", input_path, len(df))

    # Get only passing predictions
    passing = df[df.compiler_output.apply(lambda x: x['verified'])]

    # Prepare prompts
    prompts = []
    for idx, row in passing.iterrows():
        informal = ds.loc[row["uuid"]]['text']
        formal = row["parsed_output"]["text"]
        user_content = f"Informal:\n{informal}\n\nFormal:\n{formal}"
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        prompts.append(prompt)

    # Generate scores with batch mode if enabled
    if batch_mode:
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_output = str(Path(tmpdir, Path(input_path).with_suffix('.jsonl').name))
            write_batch_file(
                prompts,
                output_file=batch_output,
                model=model,
                max_tokens=max_tokens,
            )
            logger.info("Batch mode enabled — skipping API calls.")

            client = OpenAI()
            file = client.files.create(
                file=open(batch_output, "rb"),
                purpose="batch"
            )
        batch = client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/responses",
            completion_window="24h"  # can also be "1h"
        )
        print('Successfully launched: ', batch.id)
        batch = client.batches.retrieve(batch.id)
        print('Initial status: ', batch.status)
        while True:
            batch = client.batches.retrieve(batch.id)
            print("Status:", batch.status)
            if batch.status in ["completed", "failed", "expired"]:
                break
            time.sleep(10)
    else:
        # Generate scores in real time
        raw_outputs = generate_alignment_scores(
            prompts,
            model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            model_url=model_url,
        )
        logger.info("Finished generating alignment scores!")

        # Add to dataframe
        df.loc[passing.index, "alignment_output"] = raw_outputs
        df['alignment_output'].fillna(None)
        df['aligned'] = df.alignment_output.apply(lambda x: parse(x)['result'])

        print("Alignment score distribution:")
        print(df.aligned.value_counts(normalize=True))
        print(df.aligned.value_counts())

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", indent=2)
        logger.info("Saved results to `%s`", output_path)


if __name__ == "__main__":
    app()
