"""
Covers the happy path of POST /documents/{id}/index, which test_documents_api.py
only exercises for the 409 (no-chunks) failure case. The real embedding/FAISS/
BM25 calls are monkeypatched here -- document_indexer.py's own orchestration
logic (status transitions, rollback on failure) is what's under test, not the
ML models themselves.
"""

BASE = "/api/v1/documents"


def upload(client, name, data):
    return client.post(f"{BASE}/upload", files={"file": (name, data, "text/plain")})


def test_index_endpoint_transitions_status_to_indexed(client, monkeypatch):
    import app.services.retrieval.document_indexer as indexer_module

    monkeypatch.setattr(indexer_module, "embed_and_index_chunks", lambda chunks: None)
    monkeypatch.setattr(indexer_module, "index_chunks_bm25", lambda chunks: None)

    doc = upload(client, "a.txt", b"Product X supports PostgreSQL and MySQL databases.").json()
    assert doc["status"] == "chunking"

    r = client.post(f"{BASE}/{doc['id']}/index")
    assert r.status_code == 200
    assert r.json()["status"] == "indexed"


def test_index_endpoint_marks_failed_on_embedding_error(client, monkeypatch):
    import app.services.retrieval.document_indexer as indexer_module

    def boom(chunks):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(indexer_module, "embed_and_index_chunks", boom)

    doc = upload(client, "a.txt", b"Product X supports PostgreSQL and MySQL databases.").json()
    r = client.post(f"{BASE}/{doc['id']}/index")
    assert r.status_code == 500

    refetched = client.get(f"{BASE}/{doc['id']}").json()
    assert refetched["status"] == "failed"
    assert "model download failed" in refetched["error_message"]


def test_index_then_delete_removes_from_indexes(client, monkeypatch):
    import app.api.v1.documents as documents_api_module
    import app.services.retrieval.document_indexer as indexer_module

    monkeypatch.setattr(indexer_module, "embed_and_index_chunks", lambda chunks: None)
    monkeypatch.setattr(indexer_module, "index_chunks_bm25", lambda chunks: None)

    removed = {"called": False}

    def fake_remove(db, document_id):
        removed["called"] = True

    # Patched on the api module, not index_manager itself: documents.py imported the function
    # by name at module-load time, so that's the reference the endpoint actually calls.
    monkeypatch.setattr(documents_api_module, "remove_document_from_indexes", fake_remove)

    doc = upload(client, "a.txt", b"Some content to index and then delete.").json()
    client.post(f"{BASE}/{doc['id']}/index")
    r = client.delete(f"{BASE}/{doc['id']}")
    assert r.status_code == 204
    assert removed["called"] is True
