import random

from tinylm.encoding import one_hot
from tinylm.activations import softmax
from tinylm.autograd import Value


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
            [Value(random.uniform(-0.1, 0.1)) for _ in range(self.input_size)] 
            for _ in range(vocab_size)]
        
        self.bias = [
            Value(random.uniform(-0.1, 0.1))
            for _ in range(vocab_size)
        ]

    def parameters(self) -> list[Value]:
        return [
            weight 
            for row in self.weights
            for weight in row
        ] + self.bias
    
    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0.0
    
    def predict_logits(self, context: list[int]) -> list[Value]:
        encoded_tokens = [one_hot(token_id, self.vocab_size) for token_id in context]

        flat_encoded = []
        for encoded in encoded_tokens:
            for value in encoded:
                flat_encoded.append(value)

        logits = []
        for output_index, neuron_weights in enumerate(self.weights):
            weighted_sum = Value(0.0)
            for input_value, weight in zip(flat_encoded, neuron_weights):
                weighted_sum += weight * Value(float(input_value))

            logit = weighted_sum + self.bias[output_index]
            logits.append(logit)

        return logits
    
    def predict_probabilities(self, context: list[int]) -> list[Value]:
        logits = self.predict_logits(context)
        probabilities = softmax(logits)
        return probabilities