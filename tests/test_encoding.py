# tests/test_encoding.py

from tinylm.encoding import one_hot

def test_onehot_represents_token_id():
    assert one_hot(token_id=2, vocab_size=5) == [0, 0, 1, 0, 0]

def test_one_hot_start_token():
    assert one_hot(token_id=0, vocab_size=5) == [1, 0, 0, 0, 0]

def test_one_hot_last_token():
    assert one_hot(token_id=4, vocab_size=5) == [0, 0, 0, 0, 1]