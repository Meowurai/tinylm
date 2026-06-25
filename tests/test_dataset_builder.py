from tinylm.dataset import build_examples
from tinylm.tokenizer import Tokenizer

def test_build_examples_from_hello():
    examples = build_examples(
        ids=[2, 1, 3, 3, 4],
        start_id=0,
        context_size=4
    )

    assert examples == [
        ([0, 0, 0, 0], 2),
        ([0, 0, 0, 2], 1),
        ([0, 0, 2, 1], 3),
        ([0, 2, 1, 3], 3),
        ([2, 1, 3, 3], 4),
    ]

def test_tokenizer_and_datasetbuilder():
    tokenizer = Tokenizer.from_text("hello")
    ids = tokenizer.encode("hello")

    examples = build_examples(
        ids=ids,
        start_id=tokenizer.token_to_id["<START>"],
        context_size=4,
    )

    assert examples[0] == ([0, 0, 0, 0], 2)
    assert examples[-1] == ([2, 1, 3, 3], 4)

