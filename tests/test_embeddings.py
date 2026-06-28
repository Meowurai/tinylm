# tests/test_embeddings.py

from tinylm.autograd import Value
from tinylm.embeddings import EmbeddingTable


def test_embedding_table_has_one_vector_per_token():
    table = EmbeddingTable(vocab_size=5, embedding_size=3)

    assert len(table.vectors) == 5

def test_each_embedding_has_embedding_size_values():
    table = EmbeddingTable(vocab_size=5, embedding_size=3)

    assert len(table.vectors[0]) == 3

def test_embeddings_are_values():
    table = EmbeddingTable(vocab_size=5, embedding_size=3)

    assert isinstance(table.vectors[0][0], Value)

def test_lookup_returns_embedding_for_token():
    table = EmbeddingTable(vocab_size=5, embedding_size=3)

    embedding = table.lookup(2)

    assert len(embedding) == 3
    assert embedding[0] is table.vectors[2][0]