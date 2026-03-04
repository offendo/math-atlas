import re
from transformers import AutoTokenizer


def format_example(informal, names: list[str] = None):
    with open("benchmarks/prompts/few_shot.txt", "r") as f:
        template = f.read()
    messages = [
        {"role": "system", "content": "You are an expert at Lean 4 and Mathematics."},
        {"role": "user", "content": template.format(text=informal)},
    ]

    return messages


def parse_output(text):
    return None, text
