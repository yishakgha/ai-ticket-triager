"""
LLM-based ticket classifier — the second AI approach, used for comparison
against the TF-IDF + Logistic Regression baseline, and as the source of the
human-readable reasoning/summary shown to the user (a use case the baseline
model can't provide alone).

Requires an ANTHROPIC_API_KEY environment variable at runtime. If no key is
set (e.g. running fully offline), classify() falls back to the baseline
model so the app degrades gracefully rather than crashing.
"""
import os
import json

CATEGORIES = ["billing", "technical", "account", "feature_request", "general_inquiry"]
PRIORITIES = ["low", "medium", "high", "urgent"]

SYSTEM_PROMPT = f"""You are a support ticket triage assistant. Given a ticket's text,
classify it into exactly one category from {CATEGORIES} and one priority from {PRIORITIES}.
Also provide a one-sentence reasoning for your decision.

Respond ONLY with valid JSON in this exact shape, no other text:
{{"category": "...", "priority": "...", "reasoning": "..."}}
"""


def classify_with_llm(ticket_text: str):
    """Calls the Anthropic API to classify a ticket. Raises on failure so the
    caller can decide whether to fall back to the baseline model."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    import anthropic  # imported lazily so the app can run without the package installed

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": ticket_text}],
    )
    raw = response.content[0].text.strip()
    # Defensive parsing in case the model wraps JSON in markdown fences
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    parsed = json.loads(raw)

    if parsed.get("category") not in CATEGORIES:
        parsed["category"] = "general_inquiry"
    if parsed.get("priority") not in PRIORITIES:
        parsed["priority"] = "low"
    return parsed
