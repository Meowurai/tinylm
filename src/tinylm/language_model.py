# src/tinylm/language_model.py

from itertools import chain

from tinylm.activations import softmax
from tinylm.autograd import Value
from tinylm.embeddings import EmbeddingTable
from tinylm.linear import Linear


class LanguageModel:
    """
    Learns a function that maps a fixed-size context
    to logits over the vocabulary.
    """
    def __init__(self, context_size: int, vocab_size: int, embedding_size: int, hidden_size: int) -> None:
        self.context_size = context_size
        self.vocab_size = vocab_size

        self.embedding_table = EmbeddingTable(vocab_size, embedding_size)
        
        self.hidden_layer = Linear(
            input_size=context_size * embedding_size,
            output_size=hidden_size
        )

        self.output_layer = Linear(
            input_size=hidden_size,
            output_size=vocab_size
        )

        
    def parameters(self) -> list[Value]:
        return (
            self.embedding_table.parameters()
            + self.hidden_layer.parameters()
            + self.output_layer.parameters()
        )
    
    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0.0
    
    def forward(self, context: list[int]) -> list[Value]:
        # Extract and flatten embedding vectors
        embedding_vectors = list(
            chain.from_iterable(
                self.embedding_table.lookup(token_id)
                for token_id in context
            )
        )

        hidden = self.hidden_layer(embedding_vectors)
        activated = [x.tanh() for x in hidden]
        logits = self.output_layer(activated)

        return logits
    
    def predict_probabilities(self, context: list[int]) -> list[Value]:
        logits = self.forward(context)
        probabilities = softmax(logits)
        return probabilities