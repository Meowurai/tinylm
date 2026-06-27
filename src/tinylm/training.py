# src/tinylm/training.py

# The job of training.py is to coordinate the learning loop:
# forward pass, loss, backwardpass, and parameter update.

from tinylm.language_model import LanguageModel
from tinylm.losses import cross_entropy

def train_one_example(
    model: LanguageModel,
    context: list[int],
    target: int,
    learning_rate: float = 0.1,
) -> float:
    
    model.zero_grad()
    probabilities = model.predict_probabilities(context)

    loss = cross_entropy(probabilities, target)
    loss.backward()

    for parameter in model.parameters():
        # Update parameters to learn
        parameter.data -= learning_rate * parameter.grad

    return loss.data