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

    def format(self, informal: str, item_type: ItemType, names: list[str] | None = None) -> list[dict[str, str]]:
        raise NotImplementedError()

    def parse_output(self, output: str) -> Output:
        raise NotImplementedError()

    def format_batch(self, batch) -> dict[str, list[list[dict[str, str]]]]:
        prompts = [
            self.format(text, ItemType[item_type.upper()], names)
            for text, item_type, names in zip(batch["text"], batch["type"], batch["names"])
        ]
        return {"prompt": prompts}
