# scripts/demo_count_model.py

# Demo script for the count-based language model.
# It trains on a tiny corpus, generates token IDs, then decodes them back to text.

from tinylm.count_model import CountLanguageModel
from tinylm.dataset import build_examples
from tinylm.generator import generate_ids
from tinylm.tokenizer import Tokenizer


def main() -> None:
    text = "hello"

    context_size = 4
    max_new_tokens = len(text)

    tokenizer = Tokenizer.from_text(text)
    ids = tokenizer.encode(text)

    examples = build_examples(
        ids=ids,
        start_id=tokenizer.token_to_id["<START>"],
        context_size=context_size,
    )

    model = CountLanguageModel()
    model.fit(examples)

    generated_ids = generate_ids(
        model=model,
        start_id=tokenizer.token_to_id["<START>"],
        context_size=context_size,
        max_new_tokens=max_new_tokens,
    )

    generated_text = tokenizer.decode(generated_ids)

    print("training text:", text)
    print("generated ids:", generated_ids)
    print("generated text:", generated_text)


if __name__ == "__main__":
    main()