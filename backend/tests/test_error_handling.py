"""
Covers the global exception handler in main.py: an unhandled exception from
inside a route must never reach the client as a raw traceback or internal
exception string -- just a generic message (plus a debug_detail field, only
in development mode, deliberately not asserted here since it's dev-only).
"""

BASE = "/api/v1/chat"


def test_unhandled_exception_returns_generic_message_not_a_traceback(client, monkeypatch):
    import app.api.v1.chat as chat_module

    def boom(question, chunk_meta_fn, **kwargs):
        raise RuntimeError("psycopg2.OperationalError: could not connect to server at /secret/path")

    monkeypatch.setattr(chat_module, "run_query_pipeline", boom)

    r = client.post(f"{BASE}/query", json={"question": "anything"})
    assert r.status_code == 500
    body = r.json()
    assert "psycopg2" not in body["detail"]
    assert "/secret/path" not in body["detail"]
    assert body["detail"] == "An internal error occurred. Please try again."



def test_question_max_length_matches_settings(client, monkeypatch):
    from app.core.config import get_settings
    import app.api.v1.chat as chat_module

    def fake_pipeline(question, chunk_meta_fn, **kwargs):
        return {
            "answer": "Test answer",
            "citations": [],
            "claims": [],
        }

    monkeypatch.setattr(chat_module, "run_query_pipeline", fake_pipeline)

    limit = get_settings().MAX_QUESTION_CHARS

    r_ok = client.post(f"{BASE}/query", json={"question": "x" * limit})
    r_over = client.post(f"{BASE}/query", json={"question": "x" * (limit + 1)})

    assert r_ok.status_code != 422
    assert r_over.status_code == 422

