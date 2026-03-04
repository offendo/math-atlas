import re
from transformers import AutoTokenizer

MODEL_NAME = "Goedel-LM/Goedel-Formalizer-V2-32B"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


def format_example(informal: str, names: list[str] = None):
    # Construct the prompt for the model
    user_prompt_content = (
        f"Please autoformalize the following natural language problem statement in Lean 4. "
        f"Use the following theorem name: generated_thm\n"
        f"The natural language statement is: \n"
        f"{informal}"
        f"Think before you provide the lean statement."
    )

    chat = [
        {"role": "user", "content": user_prompt_content},
    ]

    # return TOKENIZER.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return chat


def parse_output(text):
    """Extracts the last Lean 4 code block from the model's output."""
    try:
        matches = re.findall(r"```lean4\n(.*?)\n```", text, re.DOTALL)
        code = matches[-1].strip() if matches else text
        thinking = re.match(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        return thinking, code
    except Exception:
        return None, text
