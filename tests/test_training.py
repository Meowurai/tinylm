# tests/test_training.py

from tinylm.language_model import LanguageModel
from tinylm.training import train_one_example


def test_train_one_example_returns_loss_value():
    model = LanguageModel(context_size=4, vocab_size=5, embedding_size=3)

    loss = train_one_example(
        model=model,
        context=[0, 0, 2, 1],
        target=3,
        learning_rate=0.1,
    )

    assert isinstance(loss, float)