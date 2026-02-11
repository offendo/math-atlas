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
from tqdm.asyncio import tqdm as async_tqdm

LinkType = Literal["object", "entity"]

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

    def retrieve(self, references: list[str], query_template: str, n_results: int, filter_type: dict | None = None):
        queries = [query_template.format(ref) for ref in references]
        embeddings = self.embeddings.embed_documents(queries)
        results = self.chroma.query(
            query_embeddings=embeddings,  # type:ignore
            n_results=n_results,
            where=filter_type,  # type:ignore
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

    async def link_batch_async(
        self,
        examples: list[dict[str, str]],
        link_type: LinkType = 'object',
        filter_type: str | dict | None = None,
    ):
        # Retrieve candidate results for each example
        if link_type == 'object':
            query_template = "Find the definition for \"{}\""
        else:
            query_template = "Retrieve the document which matches the entity called \"{}\"."

        references = [ex["reference"] for ex in examples]
        results = self.retrieve(references, 10, query_template=query_template, filter_type=filter_type)

        documents = results["documents"] or []
        metadatas = results["metadatas"] or []

        tasks = []
        for ex, docs in zip(examples, documents):
            prompt = self.object_link_prompt if link_type == 'object' else self.entity_link_prompt
            messages = [
                {"role": "system", "content": self.object_link_prompt},
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
                input=messages,  # type:ignore
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
                records.append(metas[i]["uuid"])
            batch_records.append(records)

        return batch_records


app = typer.Typer(add_completion=False, help="Link references to math-atlas entries.")


async def _run_linker(retriever: MathAtlasRetriever, df: pd.DataFrame):
    semaphore = asyncio.Semaphore(20)

    async def process_async(row):
        async with semaphore:
            # Link object refs
            batch = [
                {
                    "reference": reference,
                    "context": row["text"],
                    "file_id": row["file_id"],
                }
                for reference in row.object_references
            ]
            if not batch:
                object_links = []
            else:
                object_links = await retriever.link_batch_async(batch, link_type='object', filter_type={'type': 'definition'})

            # Link entity refs
            batch = [
                {
                    "reference": reference,
                    "context": row["text"],
                    "file_id": row["file_id"],
                }
                for reference in row.entity_references
            ]
            if not batch:
                entity_links = [] 
            else:
                entity_links = await retriever.link_batch_async(batch, link_type='entity', filter_type={'$and': [{'file_id': }]})

            return {'object_links': object_links, 'entity_links': entity_links}

    linked_results = await async_tqdm.gather(
        *[process_async(row) for idx, row in df.iterrows()], desc="Linking"
    )
    return linked_results


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

    links = asyncio.run(_run_linker(retriever, df))
    df["object_links"] = [l['object_links'] for l in links]
    df["entity_links"] = [l['entity_links'] for l in links]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path)


if __name__ == "__main__":
    app()
