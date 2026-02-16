import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

import chromadb
import pandas as pd
import typer
from langchain_localai import LocalAIEmbeddings
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm

LinkType = Literal["object", "entity"]


class MathAtlasLinker:
    def __init__(self, linker_url: str, linker_model: str) -> None:
        self.async_client = AsyncOpenAI(base_url=linker_url)
        self.linker_model = linker_model
        with open("prompts/object_link_validation_prompt.txt", "r") as f:
            self.object_link_prompt = f.read()
        with open("prompts/entity_link_validation_prompt.txt", "r") as f:
            self.entity_link_prompt = f.read()
        with open("schemas/object_link_schema.json", "r") as f:
            self.object_link_schema = json.load(f)
        with open("schemas/entity_link_schema.json", "r") as f:
            self.entity_link_schema = json.load(f)

    def format_link_prompt(
        self, reference: str, file_id: str, context: str, search_results: list[str]
    ):
        header = f"""**Search term:** {reference}\n**File ID:** {file_id}\n**Context:**\n{context}"""
        parts = [header]
        for i, doc in enumerate(search_results):
            parts.append(f"{i}. {doc}")
        return "\n".join(parts)

    async def link_proof(
        self,
        reference: str,
        file_id: str,
        context: str,
        candidates: dict[str, list[Any]],
        link_type: LinkType = "object",
    ):
        pass
    async def link_entity(
        self,
        reference: str,
        file_id: str,
        context: str,
        candidates: dict[str, list[Any]],
        link_type: LinkType = "object",
    ):
        pass

    async def link_object(
        self,
        reference: str,
        file_id: str,
        context: str,
        candidates: dict[str, list[Any]],
        link_type: LinkType = "object",
    ):
        pass

    async def link(
        self,
        reference: str,
        file_id: str,
        context: str,
        candidates: dict[str, list[Any]],
        link_type: LinkType = "object",
    ):
        documents = candidates["documents"][0]
        metadatas = candidates["metadatas"][0]

        match link_type:
            case "object":
                prompt = self.object_link_prompt
                schema = self.object_link_schema

            case "entity":
                prompt = self.entity_link_prompt
                schema = self.entity_link_schema

        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": self.format_link_prompt(
                    reference=reference,
                    file_id=file_id,
                    context=context,
                    search_results=documents,
                ),
            },
        ]

        response = await self.async_client.responses.create(
            model=self.linker_model,
            input=messages,  # type:ignore
            reasoning={"effort": "low"},
            max_output_tokens=5000,
            text={"format": {"type": "json_schema", **schema}},  # type:ignore
        )

        content = response.output_text
        try:
            matches = json.loads(content)["best_match"] if content else []
            matches = matches if isinstance(matches, list) else [matches]
            linked_ids = [metadatas[i]["uuid"] for i in matches]
            return linked_ids
        except Exception as e:
            logger.error(e)
            return []


class MathAtlasRetriever:
    def __init__(
        self,
        chromadb_path: str | Path,
        embedding_url: str,
        embedding_model: str,
        collection_name: str = "mathatlas",
    ):
        self.embeddings = LocalAIEmbeddings(
            openai_api_base=embedding_url, model=embedding_model
        )
        self.chroma_client = chromadb.PersistentClient(str(chromadb_path))
        self.chroma = self.chroma_client.get_collection(collection_name)

    async def retrieve(
        self,
        query: str,
        n_results: int,
        filter_type: dict | None = None,
    ) -> chromadb.QueryResult:
        embeddings = await self.embeddings.aembed_query(query)
        results = self.chroma.query(
            query_embeddings=embeddings,  # type:ignore
            n_results=n_results,
            where=filter_type,  # type:ignore
            include=["documents", "metadatas", "distances"],
        )
        return results


app = typer.Typer(
    add_completion=False,
    help="Link references to math-atlas entries.",
    pretty_exceptions_show_locals=False,
)


async def _run_linker(
    retriever: MathAtlasRetriever,
    linker: MathAtlasLinker,
    df: pd.DataFrame,
):
    semaphore = asyncio.Semaphore(20)

    async def _helper(idx: int, row: pd.Series):
        async with semaphore:
            try:
                # Link object refs
                object_tasks = []
                for reference in row.object_references:
                    object_tasks.append(
                        retriever.retrieve(
                            query=f'Find the definition for "{reference}"',
                            n_results=10,
                            filter_type={"type": "definition"},
                        )
                    )

                entity_tasks = []
                for reference in row.entity_references:
                    entity_tasks.append(
                        retriever.retrieve(
                            query=f'Retrieve the document which defines "{reference}"',
                            n_results=10,
                            filter_type={
                                "$and": [
                                    {"file_id": row.file_id},
                                    {"block_end": {"$lte": int(row.block_end)}},
                                    {
                                        "$or": [
                                            {"type": "definition"},
                                            {"type": "example"},
                                            {"type": "theorem"},
                                        ]
                                    },
                                ]
                            },
                        )
                    )
                object_candidates = await asyncio.gather(*object_tasks)
                entity_candidates = await asyncio.gather(*entity_tasks)

                # Run object linker
                object_link_tasks = []
                for reference, candidates in zip(row.object_references, object_candidates):
                    object_link_tasks.append(
                        linker.link(
                            reference,
                            file_id=row.file_id,
                            context=row.text,
                            candidates=candidates,
                            link_type="object",
                        )
                    )
                entity_link_tasks = []
                for reference, candidates in zip(row.entity_references, entity_candidates):
                    entity_link_tasks.append(
                        linker.link(
                            reference,
                            file_id=row.file_id,
                            context=row.text,
                            candidates=candidates,
                            link_type="entity",
                        )
                    )

                object_links = await asyncio.gather(*object_link_tasks)
                entity_links = await asyncio.gather(*entity_link_tasks)

                return {
                    "object_links": object_links,
                    "entity_links": [link[0] if len(link) > 0 else None for link in entity_links],
                }
            except Exception as e:
                logger.exception(
                    "Linking failed for row idx=%s file_id=%s block_end=%s: %s",
                    idx,
                    getattr(row, "file_id", "?"),
                    getattr(row, "block_end", "?"),
                    e,
                )
                return {
                    "object_links": None,
                    "entity_links": None,
                    "error": str(e),
                }

    linked_results = await async_tqdm.gather(
        *[_helper(idx, row) for idx, row in df.iterrows()], desc="Linking"
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
        embedding_model=embedding_model,
        collection_name=collection_name,
    )
    linker = MathAtlasLinker(
        linker_url=linker_url,
        linker_model=linker_model,
    )

    links = asyncio.run(_run_linker(retriever, linker, df))
    df["object_links"] = [r["object_links"] for r in links]
    df["entity_links"] = [r["entity_links"] for r in links]
    df["linking_errors"] = [r.get("error") for r in links]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path)


if __name__ == "__main__":
    app()
