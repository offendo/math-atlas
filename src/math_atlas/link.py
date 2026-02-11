import asyncio
import json
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_localai import LocalAIEmbeddings
from openai import AsyncOpenAI, OpenAI
from tqdm.asyncio import tqdm


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
        self.async_client = OpenAI(base_url=linker_url)
        self.linker_model = linker_model
        with open("prompts/link_validation_prompt.txt", "r") as f:
            self.link_prompt = f.read()
        with open("schemas/link_schema.json", "r") as f:
            self.link_prompt = json.load(f)

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

    async def link_batch_async(
        self,
        examples: list[dict[str, str]],
        type: str | None = None,
    ):
        # Retrieve candidate results for each example
        references = [ex["reference"] for ex in examples]
        results = self.retrieve(references, 10, type=type)

        docs = results["documents"] or []
        metadatas = results["metadatas"] or []

        tasks = []
        for ex, docs in zip(examples, docs):
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
        for response, docs, metas in zip(responses, docs, metadatas):
            content = response.output_text
            matches = json.loads(content)["best_match"] if content else []
            records = []
            for i in matches:
                records.append({"document": docs[i], "metadata": metas[i]})
            batch_records.append(records)

        return batch_records

    def link(self, reference: str, context: str, file_id: str, type: str | None = None):
        search_results = self.retrieve([reference], 10, type=type)
        messages = [
            {"role": "system", "content": self.link_prompt},
            {
                "role": "user",
                "content": self._format_link_prompt(
                    reference=reference,
                    file_id=file_id,
                    context=context,
                    search_results=search_results,
                ),
            },
        ]
        response = self.client.responses.create(
            model=self.linker_model,
            input=messages,
            reasoning={"effort": "low"},
            max_output_tokens=1000,
            text={"format": {"type": "json_schema", **self.link_schema}},  # type:ignore
        )
        content = response.output_text
        matches = json.loads(content)["best_match"] if content else []
        return [search_results[int(i)] for i in matches]
