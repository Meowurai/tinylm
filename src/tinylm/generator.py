# src/tinylm/generator.py

# The generator's job is to repeatedly use a model to produce token IDs.
# It does not know how to decode IDs into text.

from tinylm.count_model import CountLanguageModel, sample_from_probabilities

def generate_ids(
    model,
    start_id: int,
    context_size: int,
    max_new_tokens: int,
) -> list[int]:
    context = [start_id] * context_size
    generated = []

    for _ in range(max_new_tokens):
        probabilities = model.predict_probabilities(context)
        next_id = sample_from_probabilities(probabilities)
        generated.append(next_id)
        context = context[1:] + [next_id]

    return generated