import random

from tinylm.encoding import one_hot
from tinylm.activations import softmax



class LanguageModel:
    """
    Learns a function that maps a fixed-size context
    to logits over the vocabulary.
    """
    def __init__(self, context_size: int, vocab_size: int) -> None:
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.input_size = context_size * vocab_size
        # Small random initialization.
        self.weights = [
            [random.uniform(-0.1, 0.1) for _ in range(self.input_size)] 
            for _ in range(vocab_size)]
        
        self.bias = [
            random.uniform(-0.1, 0.1)
            for _ in range(vocab_size)
        ]

    def predict_logits(self, context: list[int]) -> list[float]:
        encoded_tokens = [one_hot(token_id, self.vocab_size) for token_id in context]

        flat_encoded = []
        for encoded in encoded_tokens:
            for value in encoded:
                flat_encoded.append(value)

        logits = []
        for output_index, neuron_weights in enumerate(self.weights):
            weighted_sum = 0.0
            for input_value, weight in zip(flat_encoded, neuron_weights):
                weighted_sum += input_value * weight

            logit = weighted_sum + self.bias[output_index]
            logits.append(logit)

        return logits
    
    def predict_probabilities(self, context: list[int]) -> list[float]:
        logits = self.predict_logits(context)
        probabilities = softmax(logits)
        return probabilities