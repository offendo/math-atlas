import asyncio
import json
from pathlib import Path
from typing import Any

import pandas as pd
import tiktoken
import typer
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

app = typer.Typer(help="Formalize math-atlas definitions with vLLM.")


async def complete_async(
    client: AsyncOpenAI, model: str, messages: list, **kwargs
) -> list[dict[str, str]]:
    """
    Sends a request to the vLLM server and returns the parsed JSON response.
    """
    schema = {
        "name": "math_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "A list of mathematical blocks extracted from the text.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "definition",
                                    "theorem",
                                    "proof",
                                    "example",
                                    "exercise",
                                ],
                                "description": "The category of the mathematical content.",
                            },
                            "identifier": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "description": "The label or name of the block (e.g., 'Theorem 1.1'), if available.",
                            },
                            "text": {
                                "type": "string",
                                "description": "The full text content of the block, including LaTeX.",
                            },
                        },
                        "required": ["type", "identifier", "text"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        },
    }
    response = await client.responses.create(
        model=model,
        input=messages,
        reasoning={"effort": "low"},
        text={"format": {"type": "json_schema", **schema}},  # type:ignore
    )
    content = response.output_text
    return json.loads(content)["items"] if content else []


def split_file_into_token_chunks(
    file_path, max_tokens, model_name, separators=["\n\n", "\n", " ", ""]
):
    """Calculates token-based chunks with O(1) length checks."""
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    enc = tiktoken.encoding_for_model(model_name.split("/")[-1])
    full_tokens = enc.encode(full_text)

    def recursive_split(start_idx, end_idx, seps):
        if (end_idx - start_idx) <= max_tokens:
            return [(start_idx, end_idx, enc.decode(full_tokens[start_idx:end_idx]))]
        if not seps:
            return [
                (
                    i,
                    min(i + max_tokens, end_idx),
                    enc.decode(full_tokens[i : min(i + max_tokens, end_idx)]),
                )
                for i in range(start_idx, end_idx, max_tokens)
            ]

        current_sep = seps[0]
        segment_text = enc.decode(full_tokens[start_idx:end_idx])
        parts = segment_text.split(current_sep)

        if len(parts) == 1:
            return recursive_split(start_idx, end_idx, seps[1:])

        final_triples = []
        curr_group_start = start_idx
        running_idx = start_idx
        for i, part in enumerate(parts):
            p_len = len(enc.encode(part))
            s_len = len(enc.encode(current_sep)) if i < len(parts) - 1 else 0

            if (running_idx + p_len + s_len) - curr_group_start > max_tokens:
                if running_idx > curr_group_start:
                    final_triples.append(
                        (
                            curr_group_start,
                            running_idx,
                            enc.decode(full_tokens[curr_group_start:running_idx]),
                        )
                    )
                    curr_group_start = running_idx
                if p_len > max_tokens:
                    final_triples.extend(
                        recursive_split(running_idx, running_idx + p_len, seps[1:])
                    )
                    curr_group_start = running_idx + p_len + s_len
            running_idx += p_len + s_len

        if curr_group_start < end_idx:
            final_triples.append(
                (
                    curr_group_start,
                    end_idx,
                    enc.decode(full_tokens[curr_group_start:end_idx]),
                )
            )
        return final_triples

    return recursive_split(0, len(full_tokens), separators)


def format_prompt(block: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": block},
    ]


async def run_async(
    input_dir: Path,
    system_prompt_path: Path,
    output_dir: Path,
    model: str,
    max_block_size: int,
    temperature: float,
    max_gen_tokens: int,
    seed: int,
    server_url: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()

    if not server_url:
        raise typer.BadParameter("--server-url is required.")

    client = AsyncOpenAI(base_url=server_url)

    # Process files one by one to fulfill 'one output per input'
    # but batch the chunks within each file for speed.
    for file_path in input_dir.glob("*.mmd"):
        print(f"Processing {file_path.name}...")

        file_id = file_path.stem.replace("(mmd)", "").strip()

        # Prepare batch for this specific file
        chunks = split_file_into_token_chunks(file_path, max_block_size, model)

        all_records = []

        async def process_chunk(start: int, end: int, text: str):
            prompt = format_prompt(text, system_prompt)
            output: list[dict[str, Any]] = await complete_async(
                client,
                model=model,
                messages=prompt,
                max_output_tokens=max_gen_tokens,
                temperature=temperature,
            )
            records = []
            for item in output:
                item.update(
                    {"block_start": start, "block_end": end, "file_id": file_id}
                )
                if (offset := text.find(item["text"])) != -1:
                    item.update({"item_start": start + offset})
                else:
                    item.update({"item_start": None})
                records.append(item)
            return records

        tasks = [
            asyncio.create_task(process_chunk(start, end, text))
            for start, end, text in chunks
        ]

        for records in await tqdm.gather(*tasks, desc=file_id):
            all_records.extend(records)

        # Save to JSON via Pandas
        df = pd.DataFrame(all_records)
        output_file = output_dir / file_path.with_suffix(".json").name
        df.to_json(output_file)
        print(f"Saved {len(all_records)} records to {output_file}")


@app.command()
def run(
    input_dir: Path = typer.Option(..., help="Directory of input files"),
    system_prompt_path: Path = typer.Option(..., help="Path to system prompt file."),
    output_dir: Path = typer.Option(..., help="Output directory"),
    model: str = typer.Option(..., help="Model name."),
    max_block_size: int = typer.Option(2500, help="Maximum tokens per input block."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    max_gen_tokens: int = typer.Option(16000, help="Maximum tokens to generate."),
    seed: int = typer.Option(1337, help="Random seed."),
    server_url: str = typer.Option(
        "http://localhost:8001/v1", help="vLLM server base URL."
    ),
):
    asyncio.run(
        run_async(
            input_dir=input_dir,
            system_prompt_path=system_prompt_path,
            output_dir=output_dir,
            model=model,
            max_block_size=max_block_size,
            temperature=temperature,
            max_gen_tokens=max_gen_tokens,
            seed=seed,
            server_url=server_url,
        )
    )


if __name__ == "__main__":
    app()
