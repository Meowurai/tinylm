# tests/test_tokenizer.py
import pytest
from tinylm.tokenizer import Tokenizer


def test_encode_decode_hello():
    tokenizer = Tokenizer()

    ids = tokenizer.encode("hello")

    assert ids == [1, 2, 3, 3, 4]
    assert tokenizer.decode(ids) == "hello"

def test_encode_unknown_character_fails():
    tokenizer = Tokenizer()

    with pytest.raises(ValueError):
        tokenizer.encode("help")

    