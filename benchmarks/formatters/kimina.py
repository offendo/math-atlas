from transformers import AutoTokenizer

MODEL_NAME = "AI-MO/Kimina-Autoformalizer-7B"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


def format_example(informal, names: list[str] = None):
    prompt = "Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n"
    prompt += informal

    messages = [
        {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
        {"role": "user", "content": prompt},
    ]

    # return TOKENIZER.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)
    return messages


def parse_output(text):
    return None, text
