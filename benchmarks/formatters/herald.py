import re
from transformers import AutoTokenizer
from .base import BaseFormatter, ItemType, Output

MODEL_NAME = "FrenzyMath/Herald_translator"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


class HeraldFormatter(BaseFormatter):
    def format(self, informal: str, item_type: ItemType, names: list[str] | None = None) -> list[dict[str, str]]:
        assert item_type in {ItemType.THEOREM, ItemType.EXAMPLE, ItemType.EXERCISE}, "ATLASFormatter only supports THEOREM, EXAMPLE, and EXERCISEs"
        if names is None or len(names) == 0:
            match = re.search(r"\*\*(.*?)\*\*", informal)
            if match is not None:
                name = match.group(1)
            else:
                name = "unknown"
        else:
            name = names[0]
        template = "Please translate the natural language statement to Lean4 code with the header\n**Name**\n{informal_name}\n**Informal statement**\n{informal_statement}\n"
        messages = [
            {"role": "system", "content": "You are an expert at Lean 4 and Mathematics."},
            {"role": "user", "content": template.format(informal_name=name, informal_statement=informal)},
        ]

        # return TOKENIZER.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        return messages

    def parse_output(self, output: str) -> Output:
        return Output(thinking=None, text=output)
