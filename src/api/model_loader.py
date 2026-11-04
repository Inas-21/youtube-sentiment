"""Chargement unique du modèle et logique de prédiction."""

import json
import os
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from src.utils.text import clean_text

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
SENTIMENTS = {-1: "negative", 0: "neutral", 1: "positive"}


class SentimentModel:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.vectorizer = joblib.load(model_dir / "vectorizer.joblib")
        self.model = joblib.load(model_dir / "model.joblib")
        metadata_path = model_dir / "model_metadata.json"
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        self.classes = [int(c) for c in self.model.classes_]

    @property
    def version(self) -> str:
        return self.metadata.get("model_version", "unknown")

    def predict(self, comments: list[str]) -> list[dict]:
        X = self.vectorizer.transform([clean_text(c) for c in comments])
        probas = self.model.predict_proba(X)
        results = []
        for text, proba in zip(comments, probas):
            best = int(np.argmax(proba))
            label = self.classes[best]
            results.append({
                "text": text,
                "label": label,
                "sentiment": SENTIMENTS[label],
                "confidence": round(float(proba[best]), 4),
                "probabilities": {SENTIMENTS[c]: round(float(p), 4) for c, p in zip(self.classes, proba)},
            })
        return results


def compute_statistics(predictions: list[dict]) -> dict:
    total = len(predictions)
    counts = Counter(p["sentiment"] for p in predictions)
    counts = {s: counts.get(s, 0) for s in SENTIMENTS.values()}
    return {
        "total": total,
        "counts": counts,
        "percentages": {s: round(100 * n / total, 2) for s, n in counts.items()},
        "average_confidence": round(sum(p["confidence"] for p in predictions) / total, 4),
        "sentiment_score": round(sum(p["label"] for p in predictions) / total, 4),
        "dominant_sentiment": max(counts, key=counts.get),
    }
