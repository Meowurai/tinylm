# src/tinylm/count_model.py

# The count language model's job is to learn how often each target
# follows each context, then expose those counts for prediction.

class CountLanguageModel:
    def __init__(self) -> None:
        self.next_token_count: dict[tuple[int, ...], dict[int, int]] = {}

    def fit(self, examples: list[tuple[list[int], int]]) -> None:
        for context, target in examples:
            context_key = tuple(context)

            if context_key not in self.next_token_count:
                self.next_token_count[context_key] = {}

            target_counts = self.next_token_count[context_key]
            target_counts[target] = target_counts.get(target, 0) + 1

    def predict_counts(self, context: list[int]) -> dict[int, int]:
        context_key = tuple(context)
        return dict(self.next_token_count.get(context_key, {}))


if __name__ == "__main__":
    model = CountLanguageModel()

    data = [
        ([0, 1], 2),
        ([1, 0], 1),
        ([0, 0], 1),
        ([1, 1], 2),
        ([0, 1], 2),
        ([1, 0], 2),
        ([1, 0], 1),
    ]

    model.fit(data)

    prediction = model.predict_counts([1, 0])
    print(prediction)