from tinylm.activations import softmax

def test_softmax_handles_large_logits():
    probabilities = softmax([1000.0, 999.0, 998.0])

    assert abs(sum(probabilities) - 1.0) < 1e-9
    assert probabilities[0] > probabilities[1] > probabilities[2] 