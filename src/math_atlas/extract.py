import asyncio
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import tiktoken
import typer
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

app = typer.Typer(help="Extract math-atlas entities, names, and references with vLLM.")

SEM_LIMIT = 10
DEFAULT_SERVER_URL = "http://localhost:8001/v1"
ITEMS_SCHEMA = json.load(open("schemas/items_schema.json", "r"))
NAMES_SCHEMA = json.load(open("schemas/names_schema.json", "r"))
REFERENCES_SCHEMA = json.load(open("schemas/references_schema.json", "r"))


def read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def format_prompt(block: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": block},
    ]


async def complete(
    client: AsyncOpenAI, model: str, messages: list, schema: dict, **kwargs
) -> dict[str, Any]:
    response = await client.responses.create(
        model=model,
        input=messages,
        reasoning={"effort": "low"},
        text={"format": {"type": "json_schema", **schema}},  # type:ignore
        **kwargs,
    )
    content = response.output_text
    return json.loads(content) if content else {}


def split_file_into_token_chunks(
    file_path: Path,
    max_tokens: int,
    model_name: str,
    separators: Iterable[str] = ("\n\n", "\n", " ", ""),
):
    """Calculates token-based chunks with O(1) length checks."""
    full_text = read_text(file_path)

    enc = tiktoken.encoding_for_model(model_name.split("/")[-1])
    full_tokens = enc.encode(full_text)

    def recursive_split(start_idx: int, end_idx: int, seps: list[str]):
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

    return recursive_split(0, len(full_tokens), list(separators))


async def run_dataframe_extraction(
    df: pd.DataFrame,
    *,
    system_prompt: str,
    client: AsyncOpenAI,
    model: str,
    temperature: float,
    max_gen_tokens: int,
    seed: int,
    desc: str,
    schema: dict,
    result_key: str,
):
    semaphore = asyncio.Semaphore(SEM_LIMIT)

    async def process_text(text: str):
        async with semaphore:
            prompt = format_prompt(text, system_prompt)
            payload = await complete(
                client,
                model=model,
                messages=prompt,
                schema=schema,
                max_output_tokens=max_gen_tokens,
                temperature=temperature,
                seed=seed,
            )
            return payload.get(result_key, [])

    tasks = [process_text(row.text) for _, row in df.iterrows()]
    return await tqdm.gather(*tasks, desc=desc)


async def run_items_async(
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

    system_prompt = read_text(system_prompt_path)

    if not server_url:
        raise typer.BadParameter("--server-url is required.")

    client = AsyncOpenAI(base_url=server_url)

    for file_path in input_dir.glob("*.mmd"):
        print(f"Processing {file_path.name}...")

        file_id = file_path.stem.replace("(mmd)", "").strip()
        chunks = split_file_into_token_chunks(file_path, max_block_size, model)

        all_records: list[dict[str, Any]] = []

        async def process_chunk(start: int, end: int, text: str):
            prompt = format_prompt(text, system_prompt)
            payload = await complete(
                client,
                model=model,
                messages=prompt,
                schema=ITEMS_SCHEMA,
                max_output_tokens=max_gen_tokens,
                temperature=temperature,
            )
            output = payload.get("items", [])
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

        df = pd.DataFrame(all_records)
        output_file = output_dir / file_path.with_suffix(".json").name
        df.to_json(output_file)
        print(f"Saved {len(all_records)} records to {output_file}")


async def run_names_async(
    input_path: Path,
    system_prompt_path: Path,
    output_path: Path,
    model: str,
    temperature: float,
    max_gen_tokens: int,
    seed: int,
    server_url: str,
):
    system_prompt = read_text(system_prompt_path)

    client = AsyncOpenAI(base_url=server_url)
    df = pd.read_json(input_path)

    all_results = await run_dataframe_extraction(
        df,
        system_prompt=system_prompt,
        client=client,
        model=model,
        temperature=temperature,
        max_gen_tokens=max_gen_tokens,
        seed=seed,
        desc="Processing Entities",
        schema=NAMES_SCHEMA,
        result_key="names",
    )

    df["names"] = all_results
    output_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_json(output_path, orient="records", indent=4)
    print(f"Saved names to {output_path}")


async def run_references_async(
    input_path: Path,
    system_prompt_path: Path,
    output_path: Path,
    model: str,
    temperature: float,
    max_gen_tokens: int,
    seed: int,
    server_url: str,
    n_examples: int | None,
    filter_references: bool,
):
    system_prompt = read_text(system_prompt_path)

    client = AsyncOpenAI(base_url=server_url)
    df = pd.read_json(input_path)
    if n_examples:
        df = df.sample(n_examples, random_state=seed)

    all_results = await run_dataframe_extraction(
        df,
        system_prompt=system_prompt,
        client=client,
        model=model,
        temperature=temperature,
        max_gen_tokens=max_gen_tokens,
        seed=seed,
        desc="Processing Entities",
        schema=REFERENCES_SCHEMA,
        result_key="references",
    )

    df["references"] = all_results
    output_path.parent.mkdir(exist_ok=True, parents=True)

    def is_reference_valid(term: str, text: str):
        pattern = re.compile(r"\b{}\b".format(re.escape(term)))
        return re.findall(pattern, text)

    if filter_references:
        df["filtered_references"] = df.apply(
            lambda row: [
                x
                for x in row["references"]
                if is_reference_valid(term=x["term"], text=row["text"])
            ],
            axis=1,
        )

    df.to_json(output_path, orient="records", indent=4)
    print(f"Saved references to {output_path}")


@app.command("items")
def items(
    input_dir: Path = typer.Option(..., help="Directory of input files"),
    system_prompt_path: Path = typer.Option(..., help="Path to system prompt file."),
    output_dir: Path = typer.Option(..., help="Output directory"),
    model: str = typer.Option(..., help="Model name."),
    max_block_size: int = typer.Option(2500, help="Maximum tokens per input block."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    max_gen_tokens: int = typer.Option(16000, help="Maximum tokens to generate."),
    seed: int = typer.Option(1337, help="Random seed."),
    server_url: str = typer.Option(DEFAULT_SERVER_URL, help="vLLM server base URL."),
):
    asyncio.run(
        run_items_async(
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


@app.command("names")
def names(
    input_path: Path = typer.Option(..., help="Path to input file"),
    system_prompt_path: Path = typer.Option(..., help="Path to system prompt file."),
    output_path: Path = typer.Option(..., help="Path to output file"),
    model: str = typer.Option(..., help="Model name."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    max_gen_tokens: int = typer.Option(1000, help="Max tokens."),
    seed: int = typer.Option(1337, help="Random seed."),
    server_url: str = typer.Option(DEFAULT_SERVER_URL, help="vLLM URL."),
):
    asyncio.run(
        run_names_async(
            input_path=input_path,
            system_prompt_path=system_prompt_path,
            output_path=output_path,
            model=model,
            temperature=temperature,
            max_gen_tokens=max_gen_tokens,
            seed=seed,
            server_url=server_url,
        )
    )


@app.command("references")
def references(
    input_path: Path = typer.Option(..., help="Path to input file"),
    system_prompt_path: Path = typer.Option(..., help="Path to system prompt file."),
    output_path: Path = typer.Option(..., help="Path to output file"),
    model: str = typer.Option(..., help="Model name."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    max_gen_tokens: int = typer.Option(1000, help="Max tokens."),
    seed: int = typer.Option(1337, help="Random seed."),
    server_url: str = typer.Option(DEFAULT_SERVER_URL, help="vLLM URL."),
    n_examples: int = typer.Option(None, help="Number of examples to process."),
    filter_references: bool = typer.Option(False, help="Filter references"),
):
    asyncio.run(
        run_references_async(
            input_path=input_path,
            system_prompt_path=system_prompt_path,
            output_path=output_path,
            model=model,
            temperature=temperature,
            max_gen_tokens=max_gen_tokens,
            seed=seed,
            server_url=server_url,
            n_examples=n_examples,
            filter_references=filter_references,
        )
    )


if __name__ == "__main__":
    app()
