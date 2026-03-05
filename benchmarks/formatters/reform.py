import re

INSTRUCTION = """Think step by step to translate the mathematical problem in natural language to Lean 4, and verify the consistency."""

def format_example(informal, names: list[str] = None):
    prompt = INSTRUCTION + '\n' + informal
    messages = [
        {"role": "user", "content": prompt},
    ]

    return messages


def parse_output(text: str) -> tuple[str, str]:
    """Extract the last Lean 4 code block from ```lean4...``` tags."""
    pattern = r"```lean4\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    code = matches[-1].strip() if matches else None
    return None, code