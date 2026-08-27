"""
Flask backend for the AI Support Ticket Triager.

Endpoints:
  POST /api/classify   -> classify a ticket (category, priority, reasoning, method used)
  GET  /api/health      -> health check
"""
import os
import sys
from flask import Flask, request, jsonify, send_from_directory
import joblib

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


def classify_with_baseline(ticket_text: str):
    cat_vec = category_vectorizer.transform([ticket_text])
    category = category_clf.predict(cat_vec)[0]

    pri_vec = priority_vectorizer.transform([ticket_text])
    priority = priority_clf.predict(pri_vec)[0]

    return {
        "category": category,
        "priority": priority,
        "reasoning": "Predicted via TF-IDF + Logistic Regression baseline model (no reasoning text available).",
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
        return jsonify(result)

    if method == "llm":
        try:
            result = classify_with_llm(ticket_text)
            result["method"] = "llm"
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": f"LLM classification failed: {str(e)}"}), 502

    # auto: try LLM, fall back to baseline
    try:
        result = classify_with_llm(ticket_text)
        result["method"] = "llm"
    except Exception:
        result = classify_with_baseline(ticket_text)
        result["method"] = "baseline"

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
