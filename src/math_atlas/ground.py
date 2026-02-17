"""Ground entities to Mathlib 4"""

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal

import pandas as pd
import requests
import typer
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

app = typer.Typer(
    help="Ground definitions to Mathlib 4.", pretty_exceptions_show_locals=False
)


@dataclass
class LeanSearchResult:
    distance: float
    module_name: list[str]
    kind: Literal[
        "abbrev",
        "axiom",
        "classInductive",
        "definition",
        "example",
        "inductive",
        "instance",
        "opaque",
        "structure",
        "theorem",
        "proofWanted",
    ]
    name: list[str]
    start: int | None
    stop: int | None
    signature: str
    type: str
    value: str | None
    docstring: str | None
    informal_name: str
    informal_description: str


def format_lean_search_result(res: LeanSearchResult) -> str:
    module_name = ".".join(res.module_name)
    formal_name = ".".join(res.name)
    name = f"{module_name}.{formal_name}"

    return dedent(
        f"""
        Distance: {res.distance}
        {res.kind} {name} {res.signature}
        Elaborated type: {res.type}
        {res.informal_name} : {res.informal_description}
        """.strip()
    )


class MathlibGrounder:
    def __init__(
        self,
        base_url: str,
        model: str = "openai/gpt-oss-120b",
        lean_search_url: str = "http://localhost:2021/search",
    ) -> None:
        self.client = AsyncOpenAI(base_url=base_url)
        self.model = model
        self.lean_search_url = lean_search_url
        with open("./prompts/grounding_prompt.txt") as f:
            self.grounding_instructions = f.read()
        with open("./prompts/augment_prompt.txt") as f:
            self.augment_instructions = f.read()

    async def search_mathlib(self, query: str) -> list[LeanSearchResult]:
        # Delegate search to leansearch Retriever.
        body = {"query": [query], "num_results": 10}
        response = requests.post(self.lean_search_url, json=body)
        assert response.ok, (response.status_code, response.content)
        search_results = response.json()

        # if for some reason we fail the search, return an empty list
        if len(search_results) == 0 or len(search_results[0]) == 0:
            return []

        results = []
        for r in search_results[0]:
            del r["result"]["index"]
            results.append(LeanSearchResult(distance=r["distance"], **r["result"]))
        return results

    async def complete(
        self,
        messages: list,
        schema: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if schema is not None:
            text = {"format": {"type": "json_schema", **schema}}
        else:
            text = None
        response = await self.client.responses.create(
            model=self.model,
            input=messages,
            reasoning={"effort": "low"},
            text=text,
            **kwargs,
        )
        content = response.output_text
        if schema:
            return json.loads(content) if content else {}

        return content

    async def augment_query(self, name: str, text: str):
        prompt = f"""Input: "{name} : {text}" """
        messages = [
            {"role": "system", "content": self.augment_instructions},
            {"role": "user", "content": prompt},
        ]
        return await self.complete(messages)

    async def ground_item_against_mathlib(
        self, name: str, text: str
    ) -> LeanSearchResult | None:

        query = await self.augment_query(name, text)
        candidates = search_mathlib(query)

        formatted_candidates = "\n\n".join(
            [f"{i}. {format_lean_search_result(c)}" for i, c in enumerate(candidates)]
        )

        prompt = dedent(
            f"""
        **Concept to find:**
        {name} : {text}

        **Search Candidates from `mathlib`:**
        {formatted_candidates}
        """.strip()
        )
        schema = {
            "properties": {
                "reasoning": {"title": "Reasoning", "type": "string"},
                "best_match": {
                    "title": "Best Match",
                    "type": "integer",
                    "enum": list(range(len(candidates))),
                },
            },
            "required": ["reasoning", "best_match"],
            "title": "GroundingResponse",
            "type": "object",
        }

        messages = [
            {"role": "system", "content": self.grounding_instructions},
            {"role": "user", "content": prompt},
        ]
        response = await self.complete(messages, schema=schema)
        best_index = response.get("best_match", None)
        if best_index is None:
            return None

        return asdict(candidates[best_index])


@app.command("ground")
def ground(
    model_url: str = typer.Option(..., help="url on which generator model is hosted"),
    model: str = typer.Option(..., help="generator model name"),
    lean_search_url: str = typer.Option(..., help="url on which LeanSearch is running"),
    input_path: Path = typer.Option(..., help="Path to input json"),
    output_path: Path = typer.Option(..., help="Path to output json"),
):
    df = pd.read_json(input_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    grounder = MathlibGrounder(
        base_url=model_url, model=model, lean_search_url=lean_search_url
    )

    async def async_helper() -> list[LeanSearchResult]:
        tasks = []
        definitions = df[df.type == "definition"]
        for idx, row in definitions.iterrows():
            row_tasks = []
            for name in row.names:
                row_tasks.append(grounder.ground_item_against_mathlib(name, row.text))
            tasks.append(asyncio.gather(*row_tasks))

        results = await tqdm.gather(*tasks)
        return results

    results = asyncio.run(async_helper())

    df["mathlib_links"] = results
    df.to_json(output_path)


if __name__ == "__main__":
    app()
