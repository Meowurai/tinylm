# tests/test_losses.py

import math 

from tinylm.losses import cross_entropy

def test_cross_entropy_perfect_prediction():
    loss = cross_entropy([1.0, 0.0, 0.0], target=0)

    assert loss == 0.0

def test_cross_entropy_increases_when_correct_probability_decreases():
    confident = cross_entropy([0.9, 0.1], target=0)
    uncertain = cross_entropy([0.5, 0.5], target=0)
    wrong = cross_entropy([0.1, 0.9], target=0)

    assert confident < uncertain < wrong

def test_cross_entropy_matches_negative_log():
    probabilities = [0.2, 0.7, 0.1]

    assert cross_entropy(probabilities, target=1) == -math.log(0.7)