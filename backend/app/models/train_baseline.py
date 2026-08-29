"""
Trains two baseline classifiers (TF-IDF + Logistic Regression):
  1. category classifier
  2. priority classifier

Saves both models + vectorizers to disk as .joblib files and prints
evaluation metrics (accuracy, precision/recall/F1, confusion matrix)
for inclusion in the design & testing document.

Run from backend/app/models/: python train_baseline.py
"""
import csv
import os
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "tickets.csv")
MODEL_DIR = os.path.dirname(__file__)


def load_data() -> tuple[list[str], list[str], list[str]]:
    """Load the labeled ticket dataset from data/tickets.csv, returning
    three parallel lists: ticket texts, category labels, priority labels."""
    texts, categories, priorities = [], [], []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            categories.append(row["category"])
            priorities.append(row["priority"])
    return texts, categories, priorities


def train_and_eval(texts: list[str], labels: list[str], label_name: str) -> float:
    """Train a TF-IDF + Logistic Regression classifier for one label type
    (either 'category' or 'priority'), print an evaluation report on a
    held-out 20% test split, save the fitted model and vectorizer to disk as
    .joblib files, and return the test-set accuracy."""
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)

    print(f"\n=== {label_name.upper()} CLASSIFIER ===")
    print(f"Accuracy: {acc:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, preds, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    labels_sorted = sorted(set(labels))
    print("Labels order:", labels_sorted)
    print(confusion_matrix(y_test, preds, labels=labels_sorted))

    joblib.dump(clf, os.path.join(MODEL_DIR, f"{label_name}_clf.joblib"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, f"{label_name}_vectorizer.joblib"))
    return acc


if __name__ == "__main__":
    texts, categories, priorities = load_data()
    cat_acc = train_and_eval(texts, categories, "category")
    pri_acc = train_and_eval(texts, priorities, "priority")
    print(f"\nSummary: category_acc={cat_acc:.3f}, priority_acc={pri_acc:.3f}")
