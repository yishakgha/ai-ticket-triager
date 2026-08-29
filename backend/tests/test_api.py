"""
Test suite for the Ticket Triager API.
Run from backend/: pytest tests/ -v
"""
import io
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


def test_classify_baseline_includes_confidence_scores(client):
    resp = client.post(
        "/api/classify",
        json={"text": "I was charged twice for my subscription, please refund me.", "method": "baseline"},
    )
    body = resp.get_json()
    assert "category_confidence" in body
    assert "priority_confidence" in body
    assert 0.0 <= body["category_confidence"] <= 1.0
    assert 0.0 <= body["priority_confidence"] <= 1.0


def test_analytics_starts_reflecting_classifications(client):
    client.post("/api/classify", json={"text": "The app crashes on upload.", "method": "baseline"})
    resp = client.get("/api/analytics")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["total_classified"] >= 1
    assert "by_category" in body
    assert "by_priority" in body
    assert "by_method" in body


def test_batch_classify_valid_csv(client):
    csv_content = "text\nI was charged twice for my subscription\nThe app crashes on upload\n"
    data = {"file": (io.BytesIO(csv_content.encode()), "tickets.csv")}
    resp = client.post("/api/classify/batch", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["count"] == 2
    assert len(body["results"]) == 2
    assert all("category" in r for r in body["results"])


def test_batch_classify_handles_notepad_bom_encoding(client):
    # Windows Notepad saves CSVs as "UTF-8 with BOM" by default, which
    # prepends an invisible byte-order-mark before the first header. Without
    # handling this, the 'text' column check fails even on a valid file.
    csv_content = "text\r\nI was charged twice for my subscription\r\nThe app crashes on upload\r\n"
    bom_bytes = b"\xef\xbb\xbf" + csv_content.encode("utf-8")
    data = {"file": (io.BytesIO(bom_bytes), "tickets.csv")}
    resp = client.post("/api/classify/batch", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert resp.get_json()["count"] == 2


def test_batch_classify_handles_utf16_encoding(client):
    # Some Windows locales/versions default Notepad to UTF-16 instead of
    # UTF-8. The batch endpoint tries multiple encodings before giving up.
    csv_content = "text\r\nI was charged twice for my subscription\r\nThe app crashes on upload\r\n"
    utf16_bytes = csv_content.encode("utf-16")
    data = {"file": (io.BytesIO(utf16_bytes), "tickets.csv")}
    resp = client.post("/api/classify/batch", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert resp.get_json()["count"] == 2


def test_batch_classify_missing_text_column_returns_400(client):
    data = {"file": (io.BytesIO(b"foo\nbar\n"), "bad.csv")}
    resp = client.post("/api/classify/batch", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_batch_classify_rejects_non_csv_file(client):
    data = {"file": (io.BytesIO(b"hello"), "notes.txt")}
    resp = client.post("/api/classify/batch", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_batch_classify_no_file_returns_400(client):
    resp = client.post("/api/classify/batch", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_batch_classify_handles_utf8_bom(client):
    # Windows Notepad and Excel commonly save CSVs with a UTF-8 byte-order-mark
    # (BOM) prefix, which silently corrupts the first header name (e.g. "text"
    # becomes "\ufefftext") unless explicitly stripped on read.
    csv_content = "text\nI was charged twice for my subscription\n"
    data = {"file": (io.BytesIO(csv_content.encode("utf-8-sig")), "tickets.csv")}
    resp = client.post("/api/classify/batch", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["count"] == 1


def test_batch_classify_skips_blank_lines(client):
    # Python's csv.DictReader silently skips fully blank lines (no row is
    # yielded for them at all), so a file with a blank line in the middle
    # should still classify only the non-blank rows, with no error or gap.
    csv_content = "text\nI was charged twice for my subscription\n\nThe app crashes on upload\n"
    data = {"file": (io.BytesIO(csv_content.encode()), "tickets.csv")}
    resp = client.post("/api/classify/batch", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["count"] == 2
    assert all("category" in r for r in body["results"])


def test_batch_classify_flags_row_with_empty_text_value(client):
    # A row where the 'text' column itself is empty (as opposed to a fully
    # blank line) IS yielded by csv.DictReader, and should be flagged with
    # an error rather than crashing or silently skipping.
    csv_content = 'text\n""\nThe app crashes on upload\n'
    data = {"file": (io.BytesIO(csv_content.encode()), "tickets.csv")}
    resp = client.post("/api/classify/batch", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    flagged = [r for r in body["results"] if r.get("error")]
    assert len(flagged) == 1


def test_batch_classify_updates_analytics(client):
    csv_content = "text\nI was charged twice for my subscription\nThe app crashes on upload\n"
    data = {"file": (io.BytesIO(csv_content.encode()), "tickets.csv")}
    client.post("/api/classify/batch", data=data, content_type="multipart/form-data")

    resp = client.get("/api/analytics")
    body = resp.get_json()
    assert body["total_classified"] >= 2
    assert body["by_method"].get("baseline", 0) >= 2
