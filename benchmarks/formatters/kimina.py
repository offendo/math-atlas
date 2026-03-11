from transformers import AutoTokenizer
from .base import BaseFormatter, ItemType, Output

MODEL_NAME = "AI-MO/Kimina-Autoformalizer-7B"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


class KiminaFormatter(BaseFormatter):
    def format(self, informal: str, item_type: ItemType, names: list[str] | None = None) -> list[dict[str, str]]:
        assert item_type in {ItemType.THEOREM, ItemType.EXAMPLE, ItemType.EXERCISE}, "ATLASFormatter only supports THEOREM, EXAMPLE, and EXERCISEs"
        prompt = "Please autoformalize the following problem in Lean 4 with a header. Use the following theorem name: thm_example.\n\n"
        prompt += informal

        messages = [
            {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
            {"role": "user", "content": prompt},
        ]

        # return TOKENIZER.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)
        return messages

    def parse_output(self, output: str) -> Output:
        return Output(thinking=None, text=output)
