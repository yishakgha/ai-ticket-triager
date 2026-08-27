# AI Support Ticket Triager

An AI system that classifies incoming support tickets by **category** (billing,
technical, account, feature_request, general_inquiry) and **priority** (low,
medium, high, urgent), and provides a short reasoning for each decision — built
for the MSSE Capstone Project (Quantic School of Business and Technology).

## Why this project

Support teams triage incoming tickets manually, which is slow and inconsistent
at scale. This system automates the first pass of triage so tickets can be
routed and prioritized faster, using two complementary AI approaches so the
tradeoffs between them can be evaluated directly (see `docs/design_and_testing.md`).

## Architecture

- **Backend:** Flask REST API (`backend/app/main.py`)
- **Two classification approaches:**
  1. **Baseline ML model** — TF-IDF + Logistic Regression, trained on a labeled
     dataset (`backend/app/models/train_baseline.py`). Fast, cheap, no external
     API dependency, but no natural-language reasoning.
  2. **LLM-based classifier** — Claude API with a structured prompt
     (`backend/app/models/llm_classifier.py`). Produces a human-readable
     reasoning string, more flexible to novel ticket phrasing, but requires an
     API key and has per-request cost/latency.
  - Default mode (`auto`) tries the LLM first and falls back to the baseline
    model if no API key is configured or the call fails — so the app always
    works, even fully offline.
- **Frontend:** single-page HTML/CSS/JS app (`frontend/index.html`), served
  directly by Flask.

## Running locally

```bash
cd backend
pip install -r requirements.txt

# Train the baseline models (only needed once, or after regenerating the dataset)
cd app/models
python train_baseline.py
cd ../..

# (Optional) enable LLM classification
cp .env.example .env   # then add your ANTHROPIC_API_KEY

# Run the app
cd app
python main.py
```

Visit `http://localhost:5000`.

## Running tests

```bash
cd backend
pytest tests/ -v
```

## Regenerating the dataset

```bash
cd data
python generate_dataset.py
```

## Deployment

Deployed as a single Flask service (serves both the API and the frontend) to
a free-tier host. See `Procfile` for the production start command
(`gunicorn`). Live URL: **[add your deployed link here]**

## Project links

- Deployed app: [add link]
- Agile task board (Trello): [add link]
- Design & testing document: [`docs/design_and_testing.md`](docs/design_and_testing.md)
- Demo video: [add link]

## Repository access

This repository is shared with the GitHub account `quantic-grader` per
Capstone submission requirements.
