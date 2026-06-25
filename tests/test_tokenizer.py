# tests/test_tokenizer.py
import pytest
from tinylm.tokenizer import Tokenizer


def test_encode_decode_hello():
    tokenizer = Tokenizer().from_text("hello")

    ids = tokenizer.encode("hello")

    assert ids == [1, 2, 3, 3, 4]
    assert tokenizer.decode(ids) == "hello"
    assert tokenizer.vocabulary_size == 5

def test_encode_unknown_character_fails():
    tokenizer = Tokenizer().from_text("hello")

    with pytest.raises(ValueError):
        tokenizer.encode("help")

    