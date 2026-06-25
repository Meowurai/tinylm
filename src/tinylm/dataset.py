# src/tinlym/dataset_builder.py

# The dataset builder's job is to turn a sequence of token IDs
# into many fixed-size context -> target training examples.

def build_examples(
    ids: list[int],
    start_id: int,
    context_size: int,
) -> list[tuple[list[int], int]]:
    context = [start_id] * context_size
    examples = []
    print(f"next context: {context}")
    for target in ids:
        example = (context, target)
        examples.append(example)
        
        context = context[1:]
        context.append(target)

    return examples

if __name__ == "__main__":
    ids = [2, 1, 3, 3, 4]
    start_id = 0
    context_size = 4 

    build_examples(ids, start_id, context_size)