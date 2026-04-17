import re
from transformers import AutoTokenizer
from .base import BaseFormatter, ItemType, Output

MODEL_NAME = "Goedel-LM/Goedel-Formalizer-V2-32B"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


class GoedelFormatter(BaseFormatter):
    def format(self, informal: str, item_type: ItemType, names: list[str] | None = None, *args, **kwargs) -> list[dict[str, str]]:
        assert item_type in {ItemType.THEOREM, ItemType.EXAMPLE, ItemType.EXERCISE}, "GoedelFormatter only supports THEOREM, EXAMPLE, and EXERCISEs"
        # Construct the prompt for the model
        user_prompt_content = (
            f"Please autoformalize the following natural language problem statement in Lean 4. "
            f"Use the following theorem name: generated_thm\n"
            f"The natural language statement is: \n"
            f"{informal}"
            f"Think before you provide the lean statement."
        )

        return [{"role": "user", "content": user_prompt_content}]

    def parse_output(self, output: str) -> Output:
        """Extracts the last Lean 4 code block from the model's output."""
        try:
            thinking_match = re.match(r"<think>(.*?)</think>", output, flags=re.DOTALL)
            thinking = thinking_match.group(1) if thinking_match else output

            # code is either the match, or everything after the thinking.
            matches = re.findall(r"```lean4\n(.*?)\n```", output, re.DOTALL)
            code = matches[-1].strip() if matches else output[len(thinking):]
            return Output(thinking=thinking, text=code)
        except Exception:
            return Output(thinking=output, text="error: unable to parse model output")
