# Design & Testing Document — AI Support Ticket Triager

## 1. Overview

This document covers the design/architecture decisions and the testing carried
out for the AI Support Ticket Triager, submitted as the MSSE Capstone Project.

## 2. Problem & Users

Support teams receive tickets that need to be categorized and prioritized
before being routed to the right person. Doing this manually is slow and
inconsistent. This system takes raw ticket text and returns a predicted
**category**, **priority**, and a **reasoning** string, so a human agent can
quickly verify and act on the suggestion rather than triaging from scratch.

## 3. Architecture & Technology Choices

| Decision | Choice | Reasoning |
|---|---|---|
| Backend framework | Flask | Lightweight, minimal boilerplate for a small REST API, well suited to a solo 5-day build. (FastAPI was considered but was unavailable in the offline dev sandbox used during initial development; Flask has no such constraint and is equally production-appropriate for this scope.) |
| Baseline ML approach | TF-IDF + Logistic Regression (scikit-learn) | Fast to train, cheap to run at inference time, fully interpretable, and requires no external API or network access — a reliable fallback. |
| Secondary AI approach | Claude API (LLM), prompted for structured JSON output | Handles novel phrasing the baseline model wasn't trained on, and — unlike the baseline — produces a human-readable reasoning string, which is valuable for agent trust/auditability. |
| Classification strategy | `auto` mode: try LLM first, fall back to baseline on any failure (missing API key, network error, malformed response) | Ensures the system is never fully unavailable — a key reliability requirement for a triage tool that may run in environments without external network access. |
| Frontend | Single-page vanilla HTML/CSS/JS, served directly by Flask (`static_folder`) | Avoids a separate frontend build/deploy step and a second CORS-configured service — one deployable unit, simpler for a free-tier host and a 5-day timeline. |
| Data storage | No database; models loaded from disk (`.joblib`) at startup | The system is stateless per-request; no ticket history needs to persist for this scope, so a database would add complexity without benefit. |
| Deployment | Single Flask service via `gunicorn`, deployed to [Render/Railway — free tier] | Free-tier hosting is sufficient for demo-scale traffic; a single service (API + static frontend) minimizes moving parts to deploy and keep alive within a 5-day window. |
| Deployment cost | $0 (free tier) | Free tier is adequate for a portfolio/demo deployment; a production version handling real traffic volumes would need a paid tier for reliability (no cold-start sleep) and to support LLM API call costs at scale, which scale per-request rather than being a fixed hosting cost. |

### Software/architectural patterns used
- **Strategy pattern** (informally): the two classifiers (`classify_with_baseline`,
  `classify_with_llm`) are interchangeable strategies selected by the `method`
  request parameter, with `auto` implementing a fallback chain between them.
- **Separation of concerns**: model training (`train_baseline.py`) is fully
  decoupled from inference (`main.py`), which only loads pre-trained artifacts.
- **Graceful degradation**: the LLM path never crashes the request pipeline —
  failures are caught and routed to the baseline model in `auto` mode.

## 4. Data

A labeled synthetic dataset of 200 support tickets (40 per category) was
generated (`data/generate_dataset.py`) from hand-written realistic ticket
templates across the 5 categories, with priority labels assigned per example
and light textual variation (prefixes/suffixes) applied for diversity.
Synthetic data was used (per the Capstone handbook's guidance that simulated
data is acceptable) since no real support-ticket dataset with the needed
labels was available.

**Known limitation:** because the dataset is template-based with limited
underlying examples per category, the baseline classifier achieves very high
accuracy (100% on category) that would likely be lower on truly novel,
messier real-world ticket text. This is an honest limitation, not a claim of
production-grade accuracy — it's the reason the LLM-based classifier is
included as a second, more generalizable approach.

## 5. Model Evaluation (Baseline Classifier)

Evaluated on a held-out 20% test split (40 tickets), stratified by label.

**Category classifier:**
- Accuracy: **100%**
- Precision/Recall/F1: 1.00 across all 5 classes
- The categories are lexically distinct enough (e.g. "charged"/"refund" vs.
  "crashes"/"error") that this task is fully separable on this dataset.

**Priority classifier:**
- Accuracy: **92.5%**
- Weighted F1: 0.91
- Weaker on the `urgent` class (33% recall) — the smallest class in the
  dataset and the one most dependent on subtle tone/urgency language rather
  than clear keywords, which is a reasonable real-world failure mode to flag
  rather than hide.

Full classification reports and confusion matrices are produced by
`backend/app/models/train_baseline.py` and were captured during training.

## 6. Testing

Automated tests (`backend/tests/test_api.py`, run via `pytest`) — 21 tests
covering:
- Health check endpoint returns 200
- Missing/empty ticket text returns 400 with a clear error
- Oversized ticket text (>5000 chars) returns 400
- Baseline classification returns the expected category for representative
  billing, technical, and account tickets
- Baseline classification includes valid confidence scores (0.0–1.0) for
  both category and priority
- LLM classification correctly returns 502 when no API key is configured
- `auto` mode correctly falls back to the baseline model when the LLM path
  is unavailable
- Batch CSV upload: valid file classifies all rows correctly
- Batch CSV upload: handles UTF-8-with-BOM and UTF-16 encodings (both
  produced by Windows Notepad depending on locale/version — this was a real
  issue hit during manual testing and fixed as a result, see Section 9)
- Batch CSV upload: rejects a file missing the required `text` column,
  rejects non-CSV files, rejects a request with no file attached
- Batch CSV upload: correctly skips fully blank lines and flags rows with
  an empty `text` value rather than crashing
- Batch CSV upload: classifications are correctly recorded into the session
  analytics history
- Analytics endpoint reflects classifications made via both the single and
  batch endpoints

These tests run automatically via GitHub Actions CI (`.github/workflows/ci.yml`)
on every push and pull request to `main`, so regressions are caught before
merge.

**Manual testing:** the full application (backend + frontend) was smoke-tested
end-to-end locally — submitting tickets through the UI and confirming correct
category/priority/reasoning display, error states, and the loading state on
the submit button.

## 7. Known Limitations & Future Work

- The baseline model's high accuracy reflects the synthetic dataset's
  regularity, not necessarily real-world generalization — a production
  version would need a labeled real-ticket dataset.
- No authentication/rate-limiting is implemented, which would be required
  before handling untrusted public traffic.
- No persistence layer — ticket history/audit trail isn't stored, which
  would likely be a next feature for a real support team.

## 8. Extended Features (beyond the minimum requirements)

Three additional capabilities were built beyond the core single-ticket
classification flow, to better demonstrate real-world triage tooling:

**Confidence scores.** Every baseline classification now returns a
`category_confidence` and `priority_confidence` (the model's max predicted
probability), shown in the UI next to each badge. This gives an agent a
signal for when to trust a prediction versus review it manually — a
prediction at 35% confidence should be treated very differently from one at
95%, even though both return the same label. (The LLM path does not return a
numeric confidence score, since the underlying API does not expose one;
this is a real, documented tradeoff between the two approaches rather than
an oversight.)

**Batch CSV upload.** A `/api/classify/batch` endpoint accepts a CSV file
(with a `text` column) and classifies every row using the baseline model,
returning results as JSON that the frontend renders as a table with a
"download results" button. The LLM was deliberately excluded from the batch
path: running potentially hundreds of rows through a paid, rate-limited API
would be slow and costly, whereas the baseline model classifies a full batch
in milliseconds at zero marginal cost. This is a concrete example of
choosing the right AI approach for the job rather than defaulting to the
most powerful one everywhere.

**Session analytics.** A `/api/analytics` endpoint aggregates every
classification made during the current server session (single or batch) and
returns counts by category, priority, and method used. The frontend renders
this as a simple bar-chart view. This is intentionally session-scoped and
resets on server restart — a documented limitation rather than a persistence
layer, since adding a database was out of scope for this project's timeline
but would be the natural next step for a production version.

## 9. A Real Debugging Story (production incident during development)

During deployment testing, the live LLM classifier silently fell back to the
baseline model on every request, with no visible error. Investigation (via
added logging, see `main.py`) traced this to an `anthropic.BadRequestError`
caused by an empty API credit balance on the Anthropic account used for
testing — not a code defect. This incident is included here because it is a
genuine validation of the system's graceful-degradation design: even under
a real, unplanned failure of the LLM dependency, the API continued
returning valid 200 responses to the frontend by falling back to the
baseline model, exactly as designed in Section 3's "auto mode" strategy.
Once credits were added, the LLM path was confirmed working end-to-end in
production.
