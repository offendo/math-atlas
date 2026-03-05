import re
from .base import BaseFormatter, ItemType, Output

INSTRUCTION = """Think step by step to translate the mathematical problem in natural language to Lean 4, and verify the consistency."""


class ReformFormatter(BaseFormatter):
    def format(self, informal: str, item_type: ItemType, names: list[str] | None = None) -> list[dict[str, str]]:
        assert item_type in {ItemType.THEOREM, ItemType.EXAMPLE, ItemType.EXERCISE}, "ReformFormatter only supports THEOREM, EXAMPLE, and EXERCISEs"
        prompt = INSTRUCTION + "\n" + informal
        messages = [
            {"role": "user", "content": prompt},
        ]

        return messages

    def parse_output(self, output: str) -> Output:
        """Extract the last Lean 4 code block from ```lean4...``` tags."""
        pattern = r"```lean4\s*(.*?)\s*```"
        matches = re.findall(pattern, output, re.DOTALL)
        code = matches[-1].strip() if matches else output
        return Output(thinking=None, text=code)
