import asyncio
import json
from pathlib import Path

import chromadb
import pandas as pd
import typer
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_localai import LocalAIEmbeddings
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm


class MathAtlasRetriever:
    def __init__(
        self,
        chromadb_path: str | Path,
        embedding_url: str,
        linker_url: str,
        embedding_model: str,
        linker_model: str,
        collection_name: str = "mathatlas",
    ):
        self.embeddings = LocalAIEmbeddings(
            openai_api_base=embedding_url, model=embedding_model
        )
        self.chroma_client = chromadb.PersistentClient(str(chromadb_path))
        self.chroma = self.chroma_client.get_collection(collection_name)
        self.async_client = AsyncOpenAI(base_url=linker_url)
        self.linker_model = linker_model
        with open("prompts/link_validation_prompt.txt", "r") as f:
            self.link_prompt = f.read()
        with open("schemas/link_schema.json", "r") as f:
            self.link_schema = json.load(f)

    def retrieve(self, references: list[str], n_results: int, type: str | None = None):
        queries = [f"""Find the definition of \"{ref}\"""" for ref in references]
        filter = {"type": type} if type else None
        embeddings = self.embeddings.embed_documents(queries)
        results = self.chroma.query(
            query_embeddings=embeddings,
            n_results=n_results,
            where=filter,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def _format_link_prompt(
        self, reference: str, file_id: str, context: str, search_results: list[str]
    ):
        header = f"""**Search term:** {reference}\n**File ID:** {file_id}\n**Context:**\n{context}"""
        parts = [header]
        for i, doc in enumerate(search_results):
            parts.append(f"{i}. {doc}")
        return "\n".join(parts)

    def link_batch(
        self,
        examples: list[dict[str, str]],
        type: str | None = None,
    ):
        return asyncio.run(self.link_batch_async(examples, type))

    async def link_batch_async(
        self,
        examples: list[dict[str, str]],
        type: str | None = None,
    ):
        # Retrieve candidate results for each example
        references = [ex["reference"] for ex in examples]
        results = self.retrieve(references, 10, type=type)

        documents = results["documents"] or []
        metadatas = results["metadatas"] or []

        tasks = []
        for ex, docs in zip(examples, documents):
            messages = [
                {"role": "system", "content": self.link_prompt},
                {
                    "role": "user",
                    "content": self._format_link_prompt(
                        reference=ex["reference"],
                        file_id=ex["file_id"],
                        context=ex["context"],
                        search_results=docs,
                    ),
                },
            ]
            task = self.async_client.responses.create(
                model=self.linker_model,
                input=messages,
                reasoning={"effort": "low"},
                max_output_tokens=1000,
                text={"format": {"type": "json_schema", **self.link_schema}},  # type:ignore
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        batch_records = []
        for response, docs, metas in zip(responses, documents, metadatas):
            content = response.output_text
            matches = json.loads(content)["best_match"] if content else []
            records = []
            for i in matches:
                records.append({"document": docs[i], "metadata": metas[i]})
            batch_records.append(records)

        return batch_records


app = typer.Typer(add_completion=False, help="Link references to math-atlas entries.")


def _normalize_reference(reference: object) -> str:
    if isinstance(reference, dict):
        term = reference.get("term")
        return term if term is not None else json.dumps(reference)
    return str(reference)


@app.command("link")
def link(
    input_path: Path = typer.Option(..., help="Path to input JSON file."),
    output_path: Path = typer.Option(..., help="Path to output JSON file."),
    chromadb_path: Path = typer.Option(..., help="ChromaDB persistence path."),
    embedding_url: str = typer.Option(..., help="Embedding server URL."),
    linker_url: str = typer.Option(..., help="Linker server URL."),
    embedding_model: str = typer.Option(..., help="Embedding model name."),
    linker_model: str = typer.Option(..., help="Linker model name."),
    collection_name: str = typer.Option("mathatlas", help="Chroma collection name."),
    batch_size: int = typer.Option(500, help="Number of rows per batch."),
) -> None:
    df = pd.read_json(input_path)

    retriever = MathAtlasRetriever(
        chromadb_path=chromadb_path,
        embedding_url=embedding_url,
        linker_url=linker_url,
        embedding_model=embedding_model,
        linker_model=linker_model,
        collection_name=collection_name,
    )

    linked_results = [None] * len(df)
    rows = list(df.itertuples(index=True, name="Row"))

    for start in tqdm(range(0, len(rows), batch_size), desc="Processing batches"):
        batch_rows = rows[start : start + batch_size]
        batch_examples: list[dict[str, str]] = []
        example_row_indices: list[int] = []

        for row in batch_rows:
            references = row.references if isinstance(row.references, list) else []
            for reference in references:
                if reference.get("reference_type") != "object_reference":
                    continue
                batch_examples.append(
                    {
                        "reference": _normalize_reference(reference),
                        "context": row.text,
                        "file_id": row.file_id,
                    }
                )
                example_row_indices.append(row.Index)

        batch_links = (
            retriever.link_batch(batch_examples, type="definition")
            if batch_examples
            else []
        )

        per_row_links: dict[int, list] = {row.Index: [] for row in batch_rows}
        for row_index, linked in zip(example_row_indices, batch_links):
            per_row_links[row_index].append(linked)

        for row in batch_rows:
            linked_results[row.Index] = per_row_links[row.Index]

    df["object_links"] = linked_results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path)


if __name__ == "__main__":
    app()
