"""
Test suite for the Ticket Triager API.
Run from backend/: pytest tests/ -v
"""
import os
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))
from main import app  # noqa: E402


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_classify_missing_text_returns_400(client):
    resp = client.post("/api/classify", json={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_classify_empty_text_returns_400(client):
    resp = client.post("/api/classify", json={"text": "   "})
    assert resp.status_code == 400


def test_classify_text_too_long_returns_400(client):
    resp = client.post("/api/classify", json={"text": "a" * 5001})
    assert resp.status_code == 400


def test_classify_baseline_billing_ticket(client):
    resp = client.post(
        "/api/classify",
        json={"text": "I was charged twice for my subscription, please refund me.", "method": "baseline"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["category"] == "billing"
    assert body["priority"] in {"low", "medium", "high", "urgent"}
    assert body["method"] == "baseline"


def test_classify_baseline_technical_ticket(client):
    resp = client.post(
        "/api/classify",
        json={"text": "The app crashes every time I try to upload a large file.", "method": "baseline"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["category"] == "technical"


def test_classify_baseline_account_ticket(client):
    resp = client.post(
        "/api/classify",
        json={"text": "I'm locked out of my account and need help verifying my identity.", "method": "baseline"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["category"] == "account"


def test_classify_llm_without_api_key_returns_502(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resp = client.post(
        "/api/classify",
        json={"text": "My account was suspended, please help.", "method": "llm"},
    )
    assert resp.status_code == 502


def test_classify_auto_falls_back_to_baseline_without_api_key(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resp = client.post(
        "/api/classify",
        json={"text": "I was charged twice for my subscription.", "method": "auto"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["method"] == "baseline"
