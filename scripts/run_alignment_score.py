"""Run alignment scoring with LLM evaluation."""

import asyncio
import logging
from pathlib import Path
from typing import Any
import json

import pandas as pd
import typer
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

app = typer.Typer(pretty_exceptions_show_locals=False)
logger = logging.getLogger("run_alignment_score")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.WARNING)
logger.setLevel(logging.INFO)


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
                reasoning={"effort": "low"},
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=top_p,
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
    output_path: Path = typer.Option(..., help="Output JSON path for results."),
    max_tokens: int = typer.Option(4096, help="Max tokens to generate."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    top_p: float = typer.Option(1.0, help="Top-p sampling."),
    n_examples: int | None = typer.Option(None, help="Number of examples to run (for debugging)."),
    entity_types: list[str] = typer.Option(..., help="Entity type(s) to autoformalize; repeatable or 'all'."),
):
    """Run LLM-based alignment scoring on predictions."""

    # Load system prompt
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()
    logger.info("Loaded system prompt from `%s`", prompt_path)

    # Load predictions
    df = pd.read_json(input_path)
    if n_examples is not None:
        df = df.sample(n_examples, random_state=1337)
    if entity_types != ["all"]:
        df = df[df.entity_type.isin(entity_types)]
    logger.info("Loaded predictions from `%s` (%d rows)", input_path, len(df))

    # Get only passing predictions
    passing = df[df.compiler_output.apply(lambda x: x['verified'])]

    # Prepare prompts
    prompts = []
    for idx, row in passing.iterrows():
        informal = row["text"]
        formal = row["parsed_output"]["text"]
        user_content = f"Informal:\n{informal}\n\nFormal:\n{formal}"
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        prompts.append(prompt)

    # Generate scores
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
    df.loc[passing.index, "alignment_score"] = raw_outputs

    print("Alignment score distribution:")
    print(df.alignment_score.value_counts(normalize=True))

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", indent=2)
    logger.info("Saved results to `%s`", output_path)


if __name__ == "__main__":
    app()