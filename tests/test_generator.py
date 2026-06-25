# tests/test_generator.py

from tinylm.count_model import CountLanguageModel
from tinylm.dataset import build_examples
from tinylm.generator import generate_ids
from tinylm.tokenizer import Tokenizer

def test_generate_ids_from_hello_model():
    tokenizer = Tokenizer.from_text("hello")

    ids = tokenizer.encode("hello")
    start_id = tokenizer.token_to_id["<START>"]
    context_size = 4
    max_new_tokens = 5

    examples = build_examples(
        ids=ids,
        start_id=start_id,
        context_size=context_size
    )

    model = CountLanguageModel()
    model.fit(examples)

    generated = generate_ids(
        model=model,
        start_id=start_id,
        context_size=context_size,
        max_new_tokens=max_new_tokens
    )

    assert generated == ids