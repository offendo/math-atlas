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

    def __init__(self, model: str, prompt_dir: str | None = None):
        self.prompt_dir = prompt_dir or PROMPT_DIR
        self.model = model

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
        n_tokens: int | None = None,
        *args,
        **kwargs,
    ) -> list[dict[str, str]]:

        if n_tokens and id2tokens and file_id is not None and start_index is not None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            tokens = id2tokens[file_id]
            context_tokens = tokens[max(0, start_index - n_tokens) : start_index]
            context = tokenizer.decode(context_tokens, skip_special_tokens=True)

            # choose template file according to the item type
            match item_type:
                case ItemType.THEOREM | ItemType.EXAMPLE | ItemType.EXERCISE:
                    filename = f"few_shot_theorem_with_context.txt"
                case ItemType.DEFINITION:
                    filename = f"few_shot_definition_with_context.txt"
                case ItemType.PROOF:
                    raise NotImplementedError("No few-shot template for PROOF yet.")

            path = os.path.join(self.prompt_dir, filename)
            with open(path, "r") as f:
                template = f.read()

            messages = [
                {"role": "system", "content": "You are an expert at Lean 4 and Mathematics."},
                {"role": "user", "content": template.format(local_context=context, text=informal)},
            ]
        else:
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
