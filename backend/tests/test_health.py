from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check_returns_ok():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app_name"] == "GroundedRAG"
    assert "environment" in body


def test_root_returns_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    assert "GroundedRAG" in response.json()["message"]
