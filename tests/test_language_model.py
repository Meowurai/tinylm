# tests/language_model.py

from tinylm.language_model import LanguageModel

def test_langugage_model_predicts_one_probability_per_token():
    model = LanguageModel(context_size=4, vocab_size=5)
    probabilities = model.predict_probabilities([0, 0, 2, 1])

    assert len(probabilities) == 5

def test_langugage_model_probabilities_sum_to_one():
    model = LanguageModel(context_size=4, vocab_size=5)

    probabilities = model.predict_probabilities([0, 0, 2, 1])

    assert abs(sum([probability.data for probability in probabilities]) - 1.0) < 1e-9