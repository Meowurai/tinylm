# tests/test_count_model.py

from tinylm.count_model import CountLanguageModel


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