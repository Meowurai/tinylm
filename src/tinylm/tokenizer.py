# src/tinylm/tokenizer.py

# The tokenizer's job is to define the vocabulary of allowed tokens
# and provide a stable mapping between tokens and integer IDs.
# It can encode text into IDs and decode IDs back into text.

class Tokenizer:
    def __init__(self) -> None:
        self.vocabulary = {
            "<start>": 0,
            "h": 1,
            "e": 2,
            "l": 3,
            "o": 4,
        }

        self.vocab_size = len(self.vocabulary)

    def encode(self, text: str) -> list[int]:
        tokens = [char for char in text]
        token_ids = []
        for token in tokens:
            if token not in self.vocabulary:
                raise ValueError(f"Unrecognized token {token}")
            
            token_ids.append(self.vocabulary[token])

        return token_ids
    
    def decode(self, ids: list[int]) -> str:
        tokens = []
        for id in ids:
            id_found = False
            for key, value in self.vocabulary.items():
               if id == value:
                   tokens.append(key)
                   id_found = True 

            if not id_found:
               raise ValueError(f"Unrecognized id {id}")
            
        return "".join(tokens)
           
            
if __name__ == "__main__":
    tokenizer = Tokenizer()

    ids = tokenizer.encode("hello")

    assert tokenizer.decode(ids) == "hello"
    assert tokenizer.vocab_size == 5
    