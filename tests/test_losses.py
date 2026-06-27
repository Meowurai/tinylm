# tests/test_losses.py

import math 

from tinylm.losses import cross_entropy
from tinylm.autograd import Value

def test_cross_entropy_perfect_prediction():
    loss = cross_entropy([Value(1.0), Value(0.0), Value(0.0)], target=0)

    assert loss.data == 0.0

def test_cross_entropy_increases_when_correct_probability_decreases():
    confident = cross_entropy([Value(0.9), Value(0.1)], target=0)
    uncertain = cross_entropy([Value(0.5), Value(0.5)], target=0)
    wrong = cross_entropy([Value(0.1), Value(0.9)], target=0)

    assert confident.data < uncertain.data < wrong.data

def test_cross_entropy_matches_negative_log():
    probabilities = [Value(0.2), Value(0.7), Value(0.1)]

    assert cross_entropy(probabilities, target=1).data == -math.log(0.7)