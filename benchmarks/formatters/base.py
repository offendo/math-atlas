#!/usr/bin/env python3

from enum import Enum
from dataclasses import dataclass


class ItemType(Enum):
    DEFINITION = "definition"
    THEOREM = "theorem"
    EXAMPLE = "example"
    EXERCISE = "exercise"
    PROOF = "proof"


@dataclass
class Output:
    thinking: str | None
    text: str


class BaseFormatter:
    def __init__(self, *args, **kwargs):
        pass

    def format(
        self,
        informal: str,
        item_type: ItemType,
        names: list[str] | None = None,
        *args,
        **kwargs,
    ) -> str | list[dict[str, str]]:
        raise NotImplementedError()

    def parse_output(self, output: str) -> Output:
        raise NotImplementedError()

    def format_batch(
        self,
        batch,
        id2tokens: dict[str, list[int]] | None = None,
        id2text: dict[str, str] | None = None,
        n_tokens: int | None = None,
    ) -> dict[str, list[str | list[dict[str, str]]]]:
        prompts = [
            self.format(
                text,
                ItemType[item_type.upper()],
                names,
                start_index=start_index,
                file_id=file_id,
                id2tokens=id2tokens,
                id2text=id2text,
                n_tokens=n_tokens,
            )
            for text, item_type, names, file_id, start_index in zip(
                batch["text"],
                batch["type"],
                batch["names"],
                batch["file_id"],
                batch["item_start"],
            )
        ]
        return {"prompt": prompts}
