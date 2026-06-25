import pytest 
from tinylm.dataset_builder import build_examples

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