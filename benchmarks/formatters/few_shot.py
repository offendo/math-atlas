import os
import re
from .base import BaseFormatter, ItemType, Output

PROMPT_DIR = "benchmarks/prompts"


class FewShotFormatter(BaseFormatter):
    """Generic formatter for models that use a few‑shot prompt template.

    Templates are expected to live in `benchmarks/prompts` and be named
    `few_shot_{item_type.value}.txt`.  The formatter simply selects the
    appropriate file based on the provided `ItemType`.
    """

    def __init__(self, prompt_dir: str | None = None):
        self.prompt_dir = prompt_dir or PROMPT_DIR

    def format(self, informal: str, item_type: ItemType, names: list[str] | None = None) -> list[dict[str, str]]:
        # choose template file according to the item type
        match item_type:
            case ItemType.THEOREM | ItemType.EXAMPLE | ItemType.EXERCISE:
                filename = f"few_shot_theorem.txt"
            case ItemType.DEFINITION:
                filename = f"few_shot_definition.txt"
            case ItemType.PROOF:
                raise NotImplementedError("No few-shot template for PROOF yet.")

        path = os.path.join(self.prompt_dir, filename)
        with open(path, "r") as f:
            template = f.read()

        messages = [
            {"role": "system", "content": "You are an expert at Lean 4 and Mathematics."},
            {"role": "user", "content": template.format(text=informal)},
        ]
        return messages

    def parse_output(self, output: str) -> Output:
        pattern = re.compile(r"<\|channel\|>(\w+?)<\|message\|>(.*?)<\|end\|>")
        channels = re.findall(pattern, output, flags=re.DOTALL)

        thinking = None
        text = None
        for channel, content in channels:
            if channel == "analysis":
                thinking = thinking + content if thinking else content
            elif channel == "final":
                text = content

        # fall back to entire thing if needed
        if text is None:
            text = output

        return Output(thinking=thinking, text=text)
