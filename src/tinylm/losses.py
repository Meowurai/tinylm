# src/tinylm/losses.py

from tinylm.autograd import Value

def cross_entropy(probabilities: list[Value], target: int) -> Value:
    return Value(-1.0) * probabilities[target].log()

