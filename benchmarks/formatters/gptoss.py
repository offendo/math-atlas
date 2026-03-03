import re
from transformers import AutoTokenizer

MODEL_NAME = "openai/gpt-oss-120b"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


def format_example(informal, names: list[str] = None):
    with open("benchmarks/prompts/few_shot.txt", "r") as f:
        template = f.read()
    messages = [
        {"role": "system", "content": "You are an expert at Lean 4 and Mathematics."},
        {"role": "user", "content": template.format(text=informal)},
    ]

    # return TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages


def parse_output(text):
    pattern = re.compile(r"<\|channel\|>(\w+?)<\|message\|>(.*?)<\|end\|>")
    channels = re.findall(pattern, text, flags=re.DOTALL)

    thinking = None
    output = None
    for channel, content in channels:
        if channel == "analysis":
            thinking = thinking + content
        elif channel == "final":
            output = content

    # fall back to entire thing if needed
    if output is None:
        output = text

    return thinking, output
