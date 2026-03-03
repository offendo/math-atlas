from transformers import AutoTokenizer

MODEL_NAME = "XiaoyangLiu-sjtu/ATLAS_Translator_D"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

INSTRUCTION = """You are an expert in the Lean4 theorem prover. Your task is to translate theorems from natural language into formal Lean4 statements. Please follow these guidelines:
1. Carefully analyze the given theorem in natural language.
2. Translate it into a correct and precise Lean4 formal statement.
3. Use the following format for your response:
theorem tm_name :
{{The theorem's Lean4 formal statement}}
:= by sorry
4. Focus solely on the translation. Do not attempt to prove the theorem or provide additional explanations.
5. Ensure that your translation accurately captures all the mathematical concepts and relationships expressed in the natural language version.
6. Use appropriate Lean4 syntax, including correct use of quantifiers, implications, and mathematical symbols.
7. If the theorem involves specific mathematical structures (e.g., groups, rings, topological spaces), use the corresponding Lean4 definitions and notations.
Remember, the goal is to create a syntactically correct and semantically accurate formalization in Lean4. Your translation should be faithful to the meaning of the original theorem while adhering to Lean4 conventions and best practices.
Now please begin by carefully reading the natural language statement provided, and then proceed with your translation into Lean4."""


def format_example(informal, names: list[str] = None):
    messages = [
        {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
        {"role": "user", "content": INSTRUCTION + "\n" + informal},
    ]

    # return TOKENIZER.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return messages


def parse_output(text):
    return None, text
