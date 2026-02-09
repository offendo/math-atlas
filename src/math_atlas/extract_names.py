import asyncio
import json
from pathlib import Path
import pandas as pd
import typer
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

app = typer.Typer(help="Extract names from entities.")

# Limit concurrent requests to avoid hanging the server/client
SEM_LIMIT = 10


async def complete_async(
    client: AsyncOpenAI, model: str, messages: list, **kwargs
) -> list[str]:
    # Corrected schema: 'required' must match the defined property 'names'
    schema = {
        "type": "object",
        "properties": {
            "names": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["names"],
        "additionalProperties": False,
    }

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "name_extraction",
                    "strict": True,
                    "schema": schema,
                },
            },
            **kwargs,
        )
        content = response.choices[0].message.content
        return json.loads(content)["names"] if content else []
    except Exception as e:
        print(f"Error during API call: {e}")
        return []


async def run_async(
    input_path: Path,
    system_prompt_path: Path,
    output_path: Path,
    model: str,
    temperature: float,
    max_gen_tokens: int,
    seed: int,
    server_url: str,
):
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()

    client = AsyncOpenAI(base_url=server_url)
    df = pd.read_json(input_path)

    semaphore = asyncio.Semaphore(SEM_LIMIT)

    async def process_chunk(text: str):
        async with semaphore:  # This controls the flow
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
            return await complete_async(
                client,
                model=model,
                messages=prompt,
                max_tokens=max_gen_tokens,  # OpenAI uses max_tokens
                temperature=temperature,
                seed=seed,
            )

    # Wrap tasks in tqdm.gather to see the bar move
    tasks = [process_chunk(row.text) for idx, row in df.iterrows()]

    all_results = await tqdm.gather(*tasks, desc="Processing Entities")

    df["names"] = all_results
    output_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_json(output_path, orient="records", indent=4)
    print(f"Saved names to {output_path}")


@app.command()
def run(
    input_path: Path = typer.Option(..., help="Path to input file"),
    system_prompt_path: Path = typer.Option(..., help="Path to system prompt file."),
    output_path: Path = typer.Option(..., help="Path to output file"),
    model: str = typer.Option(..., help="Model name."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    max_gen_tokens: int = typer.Option(1000, help="Max tokens."),
    seed: int = typer.Option(1337, help="Random seed."),
    server_url: str = typer.Option("http://localhost:8001/v1", help="vLLM URL."),
):
    asyncio.run(
        run_async(
            input_path,
            system_prompt_path,
            output_path,
            model,
            temperature,
            max_gen_tokens,
            seed,
            server_url,
        )
    )


if __name__ == "__main__":
    app()
