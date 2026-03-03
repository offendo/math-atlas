import re
from transformers import AutoTokenizer

MODEL_NAME = "FrenzyMath/Herald_translator"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


def format_example(informal, names: list[str] = None):
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


def parse_output(text):
    return None, text
