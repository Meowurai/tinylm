import random

from itertools import chain

from tinylm.activations import softmax
from tinylm.autograd import Value
from tinylm.embeddings import EmbeddingTable


class LanguageModel:
    """
    Learns a function that maps a fixed-size context
    to logits over the vocabulary.
    """
    def __init__(self, context_size: int, vocab_size: int, embedding_size: int) -> None:
        self.context_size = context_size
        self.vocab_size = vocab_size

        self.embedding_table = EmbeddingTable(vocab_size, embedding_size)
        self.input_size = context_size * embedding_size

        self.weights = [
            [Value(random.uniform(-0.1, 0.1)) for _ in range(self.input_size)] 
            for _ in range(vocab_size)]
        
        self.bias = [
            Value(random.uniform(-0.1, 0.1))
            for _ in range(vocab_size)
        ]

    def parameters(self) -> list[Value]:
        return (
            self.embedding_table.parameters()
            + [
                weight 
                for row in self.weights
                for weight in row
            ] + self.bias
        )
    
    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0.0
    
    def predict_logits(self, context: list[int]) -> list[Value]:
        # Extract and flatten embedding vectors
        embedding_vectors = list(
            chain.from_iterable(
                self.embedding_table.lookup(token_id)
                for token_id in context
            )
        )

        # predict logits
        logits = []
        for output_index, weights in enumerate(self.weights):
            weighted_sum = Value(0.0)
            for input_value, weight in zip(embedding_vectors, weights):
                weighted_sum += weight * input_value

            logit = weighted_sum + self.bias[output_index]
            logits.append(logit)

        return logits
    
    def predict_probabilities(self, context: list[int]) -> list[Value]:
        logits = self.predict_logits(context)
        probabilities = softmax(logits)
        return probabilities