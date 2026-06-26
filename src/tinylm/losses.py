# src/tinylm/losses.py

import math

def cross_entropy(probabilities: list[float], target: int) -> float:
    EPSILON = 1e-12
    return -math.log(max(probabilities[target], EPSILON))

