from tinylm.dataset import build_examples
from tinylm.language_model import LanguageModel
from tinylm.tokenizer import Tokenizer
from tinylm.training import train_one_example

def print_prediction(model: LanguageModel, tokenizer: Tokenizer, context: list[int]):
    probabilities = model.predict_probabilities(context)

    for token_id, probability in enumerate(probabilities):
        token = tokenizer.id_to_token[token_id]
        print(f"{token!r}: {probability.data:.4f}")

def generate_ids(
    model: LanguageModel,
    start_id: int,
    context_size: int,
    max_new_tokens: int,
) -> list[int]:
    context = [start_id] * context_size
    generated = []

    for _ in range(max_new_tokens):
        probabilities = model.predict_probabilities(context)

        # greedy decoding, pick highest probability
        next_id = max(
            range(len(probabilities)),
            key=lambda token_id: probabilities[token_id].data,
        )

        generated.append(next_id)
        context = context[1:] + [next_id]
    
    return generated

def main() -> None:
    text = "hello"

    context_size = 4
    learning_rate = 0.1
    epochs = 500

    tokenizer = Tokenizer.from_text(text)
    ids = tokenizer.encode(text)

    examples = build_examples(
        ids=ids,
        start_id=tokenizer.token_to_id["<START>"],
        context_size=context_size
    )

    model = LanguageModel(
        context_size=context_size,
        vocab_size=tokenizer.vocabulary_size,
        embedding_size=3
    )

    print("\nBefore training:")
    print_prediction(model, tokenizer, [0, 0, 0, 0])
    print()

    for epoch in range(epochs):
        total_loss = 0.0

        for context, target in examples:
            loss = train_one_example(
                model,
                context,
                target,
                learning_rate,
            )
            total_loss += loss
    
        average_loss = total_loss / len(examples)

        if epoch % 50 == 0:
            print(f"epoch={epoch} loss={average_loss:.4f}")

    print("\nAfter training:")
    print_prediction(model, tokenizer, [0, 0, 0, 0])
    print()

    print("Predictions for all training contexts:")
    for context, target in examples:
        target_token = tokenizer.id_to_token[target]
        print(f"context={context} target={target_token!r}")
        print_prediction(model, tokenizer, context)
        print()

    generated_ids = generate_ids(
        model=model,
        start_id=tokenizer.token_to_id["<START>"],
        context_size=context_size,
        max_new_tokens=len(text),
    )

    generated_text = tokenizer.decode(generated_ids)

    print("Generated ids:", generated_ids)
    print("Generated text:", generated_text)

if __name__ == "__main__":
    main()