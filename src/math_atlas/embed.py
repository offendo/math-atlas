from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from uuid import uuid4

import pandas as pd
import typer
from langchain_chroma import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain_core.documents import Document
from langchain_localai import LocalAIEmbeddings
from more_itertools import chunked
from tqdm import tqdm

METADATA_COLUMNS = ("type", "identifier", "block_start", "block_end", "file_id")


@dataclass(frozen=True)
class LoadedDocuments:
    documents: Sequence[Document]
    ids: Sequence[str]


def format_row(row):
    text = row["text"]
    type = row["type"]
    names = row["names"]
    file_id = row["file_id"]
    identifier = row["identifier"] or "[none]"
    return f"""-- Type: {type}
-- Names: {", ".join(names)}
-- File ID: {file_id}
-- Entity identifier: {identifier}
-- Document:
{text}"""


def load_documents_from_json(json_path: str | Path) -> LoadedDocuments:
    """Load documents from a JSON file into LangChain Documents.

    The "text" column is used as the main content. Metadata is pulled from
    type/identifier/block_start/block_end/file_id if present. The "item_start"
    column is explicitly excluded.
    """
    path = Path(json_path)
    df = pd.read_json(path)

    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column in the JSON file.")

    # replace document with formatted text with metadata so it includes it in embedding
    df["text"] = df.apply(format_row, axis=1)

    metadata_cols = [col for col in METADATA_COLUMNS if col in df.columns]
    keep_cols = ["text", *metadata_cols]
    df = df[keep_cols]

    if "item_start" in df.columns:
        df = df.drop(columns=["item_start"])

    loader = DataFrameLoader(df, page_content_column="text")
    documents = loader.load()

    ids: list[str] = []
    for doc in documents:
        doc_id = str(uuid4())
        doc.metadata["uuid"] = doc_id
        if hasattr(doc, "id"):
            try:
                doc.id = doc_id
            except Exception:
                pass
        ids.append(doc_id)

    return LoadedDocuments(documents=documents, ids=ids)


def add_documents_to_chroma(
    documents: list[Document],
    *,
    collection_name: str,
    localai_url: str,
    model: str,
    persist_directory: str | Path,
    ids: list[str] | None = None,
) -> Chroma:
    """Add documents to a Chroma collection using LocalAI embeddings."""
    embeddings = LocalAIEmbeddings(openai_api_base=localai_url, model=model)

    chroma = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    docs_list = list(documents)
    if ids is None:
        ids = [doc.metadata.get("uuid", str(uuid4())) for doc in docs_list]

    for batch in tqdm(
        chunked(zip(docs_list, list(ids)), n=5000), total=int(len(docs_list) / 5000) + 1
    ):
        batch_docs, batch_ids = zip(*batch)
        chroma.add_documents(list(batch_docs), ids=batch_ids)
    return chroma


app = typer.Typer(
    add_completion=False,
    help="Embed entities in mathatlas",
    pretty_exceptions_show_locals=False,
)


@app.command("embed-json")
def embed_json(
    json_path: Path,
    *,
    collection_name: str = typer.Option(..., "--collection-name"),
    localai_url: str = typer.Option(..., "--localai-url"),
    model: str = typer.Option(..., "--model"),
    persist_directory: Path = typer.Option(..., "--persist-directory"),
) -> None:
    """Embed a JSON file into a new Chroma collection."""
    loaded = load_documents_from_json(json_path)
    add_documents_to_chroma(
        list(loaded.documents),
        collection_name=collection_name,
        localai_url=localai_url,
        model=model,
        persist_directory=persist_directory,
        ids=list(loaded.ids),
    )


if __name__ == "__main__":
    app()
