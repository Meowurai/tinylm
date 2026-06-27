# src/tinlym/activations.py

# The job of activations.py is to provide functions that transform
# raw model outputs into useful numerical representations.
# For now, softmax turns logits into probabilities.

from tinylm.autograd import Value

def softmax(logits: list[Value]) -> list[Value]:
    max_logit = max(logit.data for logit in logits)
    weights = [logit.exp() for logit in logits]
    total_weight = weights[0]
    for weight in weights[1:]:
        total_weight += weight

    return [weight / total_weight for weight in weights]
