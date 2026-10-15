from app.services.retrieval.bm25_index import BM25Index, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Product-X supports PostgreSQL!") == ["product", "x", "supports", "postgresql"]


def test_tokenize_keeps_numbers_and_ids():
    assert tokenize("Invoice #A102 due 2024-01-15") == ["invoice", "a102", "due", "2024", "01", "15"]


def test_exact_term_match_outranks_semantically_similar_but_term_absent_doc():
    idx = BM25Index(k1=1.5, b=0.75)
    idx.add(
        ids=[1, 2],
        texts=[
            "Product X supports PostgreSQL and MySQL databases.",
            "Our platform integrates with a variety of relational data stores.",
        ],
    )
    results = idx.search("PostgreSQL support", top_k=2)
    assert results[0][0] == 1
    assert results[0][1] > 0
    assert dict(results).get(2, 0.0) == 0.0


def test_rare_term_scores_higher_than_common_term():
    idx = BM25Index()
    idx.add(
        ids=[1, 2, 3, 4],
        texts=[
            "the quick brown fox",
            "the lazy dog sleeps",
            "the cat and the mouse",
            "a rare xenoblade reference here",
        ],
    )
    assert idx._idf("xenoblade") > idx._idf("the")


def test_term_frequency_saturates_but_still_increases_score():
    idx = BM25Index(k1=1.5, b=0.75)
    idx.add(ids=[1, 2], texts=["postgresql postgresql postgresql database", "postgresql database engine works well"])
    s1 = idx.score(["postgresql"], 1)
    s2 = idx.score(["postgresql"], 2)
    assert s1 > s2
    idx2 = BM25Index(k1=1.5, b=0.75)
    idx2.add(ids=[1], texts=["postgresql " * 1 + "database"])
    idx3 = BM25Index(k1=1.5, b=0.75)
    idx3.add(ids=[1], texts=["postgresql " * 3 + "database"])
    assert idx3.score(["postgresql"], 1) < 3 * idx2.score(["postgresql"], 1)


def test_longer_document_is_penalised_via_length_normalization():
    idx = BM25Index(k1=1.5, b=0.75)
    idx.add(
        ids=[1, 2],
        texts=[
            "postgresql database",
            "postgresql database " + " ".join(f"filler{i}" for i in range(50)),
        ],
    )
    assert idx.score(["postgresql"], 1) > idx.score(["postgresql"], 2)


def test_remove_then_search_excludes_removed_doc():
    idx = BM25Index()
    idx.add(ids=[1, 2], texts=["alpha beta", "alpha gamma"])
    idx.remove([1])
    results = dict(idx.search("alpha", top_k=5))
    assert 1 not in results and 2 in results


def test_reindexing_same_id_does_not_duplicate_or_accumulate_df():
    idx = BM25Index()
    idx.add(ids=[1], texts=["alpha beta"])
    idx.add(ids=[1], texts=["alpha beta"])
    assert idx.size == 1


def test_search_on_empty_index_returns_empty():
    assert BM25Index().search("anything", top_k=5) == []


def test_query_with_no_recognizable_tokens_returns_empty():
    idx = BM25Index()
    idx.add(ids=[1], texts=["alpha beta"])
    assert idx.search("!!!   ", top_k=5) == []


def test_top_k_is_respected():
    idx = BM25Index()
    idx.add(ids=list(range(10)), texts=[f"shared term doc{i}" for i in range(10)])
    assert len(idx.search("shared", top_k=3)) == 3


def test_save_and_load_roundtrip(tmp_path):
    idx = BM25Index(k1=1.2, b=0.8)
    idx.add(ids=[1, 2], texts=["alpha beta", "gamma delta"])
    path = tmp_path / "bm25.pkl"
    idx.save(path)
    loaded = BM25Index.load_or_create(path)
    assert loaded.size == 2
    assert loaded.k1 == 1.2 and loaded.b == 0.8
    assert dict(loaded.search("alpha", top_k=5)).get(1, 0.0) > 0


def test_load_or_create_with_missing_path_returns_empty_index(tmp_path):
    idx = BM25Index.load_or_create(tmp_path / "does_not_exist.pkl")
    assert idx.size == 0
