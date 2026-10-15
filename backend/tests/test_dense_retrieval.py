from types import SimpleNamespace

from app.services.retrieval.dense import dense_search, embed_and_index_chunks, remove_chunks_from_dense_index
from tests.fakes import FakeVectorIndex, deterministic_embed_query, deterministic_embed_texts


def make_chunk(id_, text):
    return SimpleNamespace(id=id_, text=text)


def test_embed_and_index_then_search_ranks_by_relevance():
    index = FakeVectorIndex()
    chunks = [
        make_chunk(1, "Product X supports MySQL and PostgreSQL databases."),
        make_chunk(2, "The office is located in Mumbai and Pune."),
        make_chunk(3, "PostgreSQL is a powerful open source database."),
    ]
    embed_and_index_chunks(chunks, index=index, embed_fn=deterministic_embed_texts, persist=False)
    assert index.size == 3

    results = dense_search(
        "Does the product support PostgreSQL?", top_k=3, index=index, embed_fn=deterministic_embed_query
    )
    ids_by_score = [r.chunk_id for r in results]
    assert ids_by_score[0] in (1, 3)
    assert 2 in ids_by_score


def test_search_respects_top_k():
    index = FakeVectorIndex()
    chunks = [make_chunk(i, f"chunk number {i} about topic {i % 3}") for i in range(10)]
    embed_and_index_chunks(chunks, index=index, embed_fn=deterministic_embed_texts, persist=False)
    results = dense_search("topic 1", top_k=4, index=index, embed_fn=deterministic_embed_query)
    assert len(results) == 4


def test_search_on_empty_index_returns_empty_list():
    index = FakeVectorIndex()
    results = dense_search("anything", top_k=5, index=index, embed_fn=deterministic_embed_query)
    assert results == []


def test_reindexing_same_chunk_ids_is_idempotent_not_duplicated():
    index = FakeVectorIndex()
    chunks = [make_chunk(1, "alpha beta"), make_chunk(2, "gamma delta")]
    embed_and_index_chunks(chunks, index=index, embed_fn=deterministic_embed_texts, persist=False)
    embed_and_index_chunks(chunks, index=index, embed_fn=deterministic_embed_texts, persist=False)
    assert index.size == 2


def test_remove_chunks_from_dense_index():
    index = FakeVectorIndex()
    chunks = [make_chunk(1, "alpha"), make_chunk(2, "beta")]
    embed_and_index_chunks(chunks, index=index, embed_fn=deterministic_embed_texts, persist=False)
    remove_chunks_from_dense_index([1], index=index, persist=False)
    assert index.size == 1
    remaining_ids = [cid for cid, _ in index.search(deterministic_embed_query("beta"), top_k=5)]
    assert remaining_ids == [2]


def test_empty_chunk_list_indexes_nothing():
    index = FakeVectorIndex()
    embed_and_index_chunks([], index=index, embed_fn=deterministic_embed_texts, persist=False)
    assert index.size == 0
