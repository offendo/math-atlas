from pathlib import Path
import edlib
import pandas as pd
import tiktoken
import typer
from tqdm import tqdm

enc = tiktoken.encoding_for_model("gpt-oss-120b")

app = typer.Typer(
    help="Identify start location using fuzzy match.",
    pretty_exceptions_show_locals=False,
)


# document = enc.decode(enc.encode(file)[x.block_start:x.block_end+1])
def locate_entity(entity_text: str, document: str) -> tuple[int, int, str]:
    result = edlib.align(entity_text, document, task="locations", mode="HW")
    # if we didn't find anything, give up
    if len(result["locations"]):
        return -1, -1, entity_text

    start, end = result["locations"][-1]
    return start, end, document[start:end]


def get_document(file_id: str, start: int, end: int, mmd_dir: Path):
    with open(mmd_dir / f"(mmd) {file_id}.mmd", "r") as f:
        contents = f.read()
        document = enc.decode(enc.encode(contents)[start : end + 1])
    return document


@app.command()
def locate(
    input_path: Path = typer.Option(..., help="Path to input JSON file."),
    output_path: Path = typer.Option(..., help="Path to output JSON file."),
    mmd_dir: Path = typer.Option(..., help="Path to directory containing MMDs"),
):
    df = pd.read_json(input_path)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        doc = get_document(row.file_id, row.block_start, row.block_end, mmd_dir=mmd_dir)
        start, end, new_text = locate_entity(row.text, doc)
        df.loc[idx, "item_start"] = start
        df.loc[idx, "item_end"] = end
        df.loc[idx, "text"] = new_text

    output_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_json(output_path)


if __name__ == "__main__":
    app()
