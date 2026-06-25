# tests/test_count_model.py

import pytest
from tinylm.count_model import CountLanguageModel, sample_from_probabilities


def test_fit_counts_repeated_context_target():
    model = CountLanguageModel()

    examples = [
        ([0, 1], 2),
        ([0, 1], 2),
        ([0, 1], 3),
    ]

    model.fit(examples)

    assert model.predict_counts([0, 1]) == {
        2: 2,
        3: 1,
    }


def test_fit_tracks_different_contexts_separately():
    model = CountLanguageModel()

    examples = [
        ([0, 1], 2),
        ([1, 2], 3),
    ]

    model.fit(examples)

    assert model.predict_counts([0, 1]) == {2: 1}
    assert model.predict_counts([1, 2]) == {3: 1}


def test_predict_counts_for_unseen_context_returns_empty_dict():
    model = CountLanguageModel()

    examples = [
        ([0, 1], 2),
    ]

    model.fit(examples)

    assert model.predict_counts([9, 9]) == {}


def test_predict_probabilities_normalizes_counts():
    model = CountLanguageModel()

    model.fit([
        ([0, 1], 2),
        ([0, 1], 2),
        ([0, 1], 3),
    ])

    assert model.predict_probabilities([0, 1]) == {
        2: 2 / 3,
        3: 1 / 3,
    }

def test_predict_probabilities_unknown_context():
    model = CountLanguageModel()

    model.fit([
        ([0, 1], 2),
        ([0, 1], 2),
        ([0, 1], 3),
    ])

    assert model.predict_probabilities([4, 1]) == {}

def test_sampling_single_option_returns_that_option():
    assert sample_from_probabilities({2: 1.0}) == 2

def test_sampling_empty_probabilities_fails():
    with pytest.raises(ValueError):
        sample_from_probabilities({})