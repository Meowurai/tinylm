# src/tinlym/activations.py

# The job of activations.py is to provide functions that transform
# raw model outputs into useful numerical representations.
# For now, softmax turns logits into probabilities.

import math

def softmax(logits: list[int|float]) -> list[float]:
    max_logit = max(logits) # too handle large logits
    weights = [math.exp(logit - max_logit) for logit in logits]
    total_weight = sum(weights)

    return [weight / total_weight for weight in weights]
