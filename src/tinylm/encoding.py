# src/tinlylm/encoding.py

# The job of encoding.py is to transform token IDs into
# numerical representations that can be used as model inputs

def one_hot(token_id: int, vocab_size: int) -> list[int]:
    if token_id >= vocab_size:
        raise ValueError(f"token_id {token_id} cannot be larger than vocab size {vocab_size}")
    data = [0 for _ in range(vocab_size)]
    data[token_id] = 1
    return data