import os
import re
from .base import BaseFormatter, ItemType, Output

class FewShotFormatter(BaseFormatter):
    """Generic formatter for models that use a few‑shot prompt template.

    Templates are expected to live in `benchmarks/prompts` and be named
    `few_shot_{item_type.value}.txt`.  The formatter simply selects the
    appropriate file based on the provided `ItemType`.
    """

    def __init__(
        self,
        model: str,
        prompt_file: str,
    ):
        self.model = model
        self.prompt_file = prompt_file

        if "qwen" in self.model.lower():
            self.parse_output = self.parse_output_qwen
        elif "gpt" in self.model.lower():
            self.parse_output = self.parse_output_gpt
        else:
            raise NotImplementedError(f"Output parser for model `{model}` not implemented yet.")

    def format(
        self,
        informal: str,
        item_type: ItemType,
        names: list[str] | None = None,
        file_id: str | None = None,
        start_index: int | None = None,
        id2tokens: dict[str, list[int]] | None = None,
        id2text: dict[str, str] | None = None,
        n_tokens: int | None = None,
        *args,
        **kwargs,
    ) -> list[dict[str, str]]:
        # If a forced prompt_file is provided, use it regardless of item_type or context
        with open(self.prompt_file, "r") as f:
            template = f.read()
        
        messages = [
            {"role": "system", "content": "You are an expert at Lean 4 and Mathematics."},
            {"role": "user", "content": template.format(text=informal)},
        ]

        # Otherwise, use the old behavior: select based on item_type and context
        if n_tokens:
            assert id2text is not None, "id2text must be provided if n_tokens is specified"
            assert start_index is not None, "start_index must be provided if n_tokens is specified"
            assert file_id is not None, "file_id must be provided if n_tokens is specified"
            all_text = id2text[file_id]
            context = all_text[max(0, start_index-(n_tokens * 4)):start_index]
            messages = [
                {"role": "system", "content": "You are an expert at Lean 4 and Mathematics."},
                {"role": "user", "content": template.format(context=context, text=informal)},
            ]
        return messages

    def parse_output_gpt(self, output: str) -> Output:
        pattern = re.compile(r"<\|channel\|>(\w+?)<\|message\|>(.*?)(<\|end\|>|$)", flags=re.DOTALL)
        channels = re.finditer(pattern, output)

        thinking = None
        text = None
        for match in channels:
            channel, content = match.group(1), match.group(2)
            if channel == "analysis":
                thinking = thinking + content if thinking else content
            elif channel == "final":
                text = content

        # fall back to entire thing if needed
        if text is None:
            text = output

        return Output(thinking=thinking, text=text)

    def parse_output_qwen(self, output: str) -> Output:
        thinking, sep, text = output.split("</think>")
        if text:
            return Output(thinking=thinking + sep, text=text)
        else:
            return Output(thinking=None, text=output)
