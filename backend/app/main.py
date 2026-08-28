"""
Flask backend for the AI Support Ticket Triager.

Endpoints:
  POST /api/classify        -> classify a single ticket (category, priority, reasoning, confidence, method)
  POST /api/classify/batch  -> classify a CSV file of tickets (baseline model only, for speed/cost)
  GET  /api/analytics       -> summary counts of everything classified this session (by category/priority)
  GET  /api/health          -> health check
"""
import os
import sys
import csv
import io
import logging
from collections import Counter
from datetime import datetime, timezone

from flask import Flask, request, jsonify, send_from_directory
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(__file__))
from models.llm_classifier import classify_with_llm  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# Load baseline models once at startup
category_clf = joblib.load(os.path.join(MODEL_DIR, "category_clf.joblib"))
category_vectorizer = joblib.load(os.path.join(MODEL_DIR, "category_vectorizer.joblib"))
priority_clf = joblib.load(os.path.join(MODEL_DIR, "priority_clf.joblib"))
priority_vectorizer = joblib.load(os.path.join(MODEL_DIR, "priority_vectorizer.joblib"))

# In-memory history of every classification made this session, used to power
# the /api/analytics endpoint. This resets on server restart (e.g. Render
# free-tier spin-down) -- acceptable for a demo/portfolio deployment; a
# production version would persist this to a database instead.
classification_history = []
MAX_HISTORY = 1000  # cap memory usage


def record_history(category: str, priority: str, method: str):
    classification_history.append(
        {
            "category": category,
            "priority": priority,
            "method": method,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    if len(classification_history) > MAX_HISTORY:
        classification_history.pop(0)


def classify_with_baseline(ticket_text: str):
    cat_vec = category_vectorizer.transform([ticket_text])
    category = category_clf.predict(cat_vec)[0]
    category_confidence = float(category_clf.predict_proba(cat_vec).max())

    pri_vec = priority_vectorizer.transform([ticket_text])
    priority = priority_clf.predict(pri_vec)[0]
    priority_confidence = float(priority_clf.predict_proba(pri_vec).max())

    return {
        "category": category,
        "priority": priority,
        "reasoning": "Predicted via TF-IDF + Logistic Regression baseline model (no reasoning text available).",
        "category_confidence": round(category_confidence, 3),
        "priority_confidence": round(priority_confidence, 3),
    }


@app.route("/", methods=["GET"])
def serve_frontend():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/classify", methods=["POST"])
def classify():
    data = request.get_json(silent=True) or {}
    ticket_text = (data.get("text") or "").strip()

    if not ticket_text:
        return jsonify({"error": "Field 'text' is required and cannot be empty."}), 400
    if len(ticket_text) > 5000:
        return jsonify({"error": "Ticket text too long (max 5000 characters)."}), 400

    method = data.get("method", "auto")  # "baseline", "llm", or "auto"

    if method == "baseline":
        result = classify_with_baseline(ticket_text)
        result["method"] = "baseline"
        record_history(result["category"], result["priority"], "baseline")
        return jsonify(result)

    if method == "llm":
        try:
            result = classify_with_llm(ticket_text)
            result["method"] = "llm"
            record_history(result["category"], result["priority"], "llm")
            return jsonify(result)
        except Exception as e:
            logger.exception("LLM classification failed (method=llm)")
            return jsonify({"error": f"LLM classification failed: {str(e)}"}), 502

    # auto: try LLM, fall back to baseline
    try:
        result = classify_with_llm(ticket_text)
        result["method"] = "llm"
    except Exception as e:
        logger.exception("LLM classification failed in auto mode, falling back to baseline: %s", e)
        result = classify_with_baseline(ticket_text)
        result["method"] = "baseline"

    record_history(result["category"], result["priority"], result["method"])
    return jsonify(result)


@app.route("/api/classify/batch", methods=["POST"])
def classify_batch():
    """
    Accepts a CSV file upload (multipart/form-data, field name 'file') with a
    'text' column, one ticket per row. Classifies every row using the
    baseline model (LLM is intentionally not used here -- a batch of many
    tickets run through a paid API would be slow and costly; the baseline
    model is instant and free, which is the right tradeoff for bulk triage).
    Returns a JSON array of results, one per row, in the same order.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Expected a multipart field named 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "File must be a .csv file."}), 400

    try:
        stream = io.StringIO(file.stream.read().decode("utf-8-sig"), newline=None)
        reader = csv.DictReader(stream)
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {str(e)}"}), 400

    if reader.fieldnames is None or "text" not in [f.strip() for f in reader.fieldnames]:
        return jsonify({"error": "CSV must have a 'text' column."}), 400

    results = []
    MAX_ROWS = 500  # guard against extremely large uploads on a free-tier instance

    for i, row in enumerate(reader):
        if i >= MAX_ROWS:
            break
        ticket_text = (row.get("text") or "").strip()
        if not ticket_text:
            results.append({"text": "", "error": "empty row, skipped"})
            continue
        result = classify_with_baseline(ticket_text[:5000])
        result["method"] = "baseline"
        result["text"] = ticket_text
        record_history(result["category"], result["priority"], "baseline")
        results.append(result)

    return jsonify({"results": results, "count": len(results)})


@app.route("/api/analytics", methods=["GET"])
def analytics():
    """
    Returns aggregate counts of every classification made this server
    session: totals by category, by priority, and by method used. Backs the
    simple analytics view in the frontend. Session-scoped, not persisted --
    see the note on classification_history above.
    """
    category_counts = Counter(h["category"] for h in classification_history)
    priority_counts = Counter(h["priority"] for h in classification_history)
    method_counts = Counter(h["method"] for h in classification_history)

    return jsonify(
        {
            "total_classified": len(classification_history),
            "by_category": dict(category_counts),
            "by_priority": dict(priority_counts),
            "by_method": dict(method_counts),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
