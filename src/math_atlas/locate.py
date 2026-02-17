from pathlib import Path
import edlib
import pandas as pd
import tiktoken
import typer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import string

enc = tiktoken.encoding_for_model("gpt-oss-120b")

app = typer.Typer(
    help="Identify start location using fuzzy match.",
    pretty_exceptions_show_locals=False,
)

# Per-process cache
_doc_cache = {}

WORD_CHARS = set(string.ascii_letters + string.digits + "_\\")
SENTENCE_END = {".", "!", "?"}


def is_word_char(c: str) -> bool:
    return c in WORD_CHARS


def expand_to_word_boundaries(document: str, start: int, end: int) -> tuple[int, int]:
    # Expand start backward
    while start > 0 and is_word_char(document[start - 1]):
        start -= 1

    # Expand end forward
    doc_len = len(document)
    while end < doc_len and is_word_char(document[end]):
        end += 1

    return start, end


def locate_entity(entity_text: str, document: str) -> tuple[int, int, str]:
    result = edlib.align(entity_text, document, task="locations", mode="HW")

    if not result["locations"]:
        return -1, -1, entity_text

    start, end = result["locations"][-1]
    if start is None or end is None:
        return -1, -1, entity_text

    end += 1  # edlib end is inclusive

    # fix word clipping
    start, end = expand_to_word_boundaries(document, start, end)

    return start, end, document[start:end]


def get_document(file_id: str, start: int, end: int, mmd_dir: Path):
    key = (file_id, start, end)

    if key in _doc_cache:
        return _doc_cache[key]

    with open(mmd_dir / f"(mmd) {file_id}.mmd", "r") as f:
        contents = f.read()
        document = enc.decode(enc.encode(contents)[start : end + 1])
        offset = contents.index(document)

    _doc_cache[key] = (document, offset)
    return document, offset


def process_row(row_dict, mmd_dir: Path):
    doc, offset = get_document(
        row_dict["file_id"],
        row_dict["block_start"],
        row_dict["block_end"],
        mmd_dir,
    )

    start, end, new_text = locate_entity(row_dict["text"], doc)

    return {
        "idx": row_dict["idx"],
        "item_start": start + offset,
        "item_end": end + offset,
        "text": new_text,
    }


@app.command()
def locate(
    input_path: Path = typer.Option(...),
    output_path: Path = typer.Option(...),
    mmd_dir: Path = typer.Option(...),
    workers: int = typer.Option(os.cpu_count(), help="Number of parallel workers"),
):
    df = pd.read_json(input_path)

    # Convert to serializable dicts
    rows = [
        {
            "idx": idx,
            "file_id": row.file_id,
            "block_start": row.block_start,
            "block_end": row.block_end,
            "text": row.text,
        }
        for idx, row in df.iterrows()
    ]

    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_row, row, mmd_dir) for row in rows]

        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())

    # Apply results
    for r in results:
        df.loc[r["idx"], "item_start"] = r["item_start"]
        df.loc[r["idx"], "item_end"] = r["item_end"]
        df.loc[r["idx"], "text"] = r["text"]

    output_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_json(output_path)


if __name__ == "__main__":
    app()
