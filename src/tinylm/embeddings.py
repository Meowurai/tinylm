# src/tinylm/embeddings.py

# The job of EmbeddingTable is to store a learned vector for each token ID
# and return the vector for a given token.

import random

from tinylm.autograd import Value

class EmbeddingTable:
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int
    ) -> None:
        self.vectors = [
            [
                Value(random.uniform(-0.1, 0.1)) 
                for _ in range(embedding_size)
            ]
            for _ in range(vocab_size)
        ]

    def parameters(self) -> list[Value]:
        return [vector for row in self.vectors for vector in row]
    
    def lookup(self, token_id: int) -> list[Value]:
        return self.vectors[token_id]