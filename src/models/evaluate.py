"""Évaluation finale du modèle sauvegardé sur le test set + benchmark d'inférence.

Entrée  : models/vectorizer.joblib, models/model.joblib, data/processed/test.csv
Sorties : reports/metrics.json, reports/classification_report.txt
          reports/figures/confusion_matrix.png

Usage :
    uv run python -m src.models.evaluate
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from src.utils.logger import get_logger
from src.utils.text import clean_text

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
LABELS = [-1, 0, 1]
NAMES = ["négatif", "neutre", "positif"]

# Critères de performance du sujet
MIN_ACCURACY = 0.80
MIN_F1_PER_CLASS = 0.75
MAX_LATENCY_MS_50 = 100

logger = get_logger("evaluate", "training.log")


def load_artifacts():
    return joblib.load(MODELS_DIR / "vectorizer.joblib"), joblib.load(MODELS_DIR / "model.joblib")


def plot_confusion(y_true, y_pred) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, normalize, title in ((axes[0], None, "Effectifs"), (axes[1], "true", "Normalisée (rappel par classe)")):
        cm = confusion_matrix(y_true, y_pred, labels=LABELS, normalize=normalize)
        ConfusionMatrixDisplay(cm, display_labels=NAMES).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format=".2f" if normalize else "d")
        ax.set_title(title)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
    fig.suptitle("Matrice de confusion — test set")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def benchmark(vectorizer, model, texts: list[str], batch_size: int = 50, runs: int = 100) -> dict:
    """Latence d'un batch de 50 commentaires bruts : nettoyage + vectorisation + prédiction."""
    rng = np.random.default_rng(42)
    timings = []
    for _ in range(runs):
        batch = list(rng.choice(texts, size=batch_size, replace=False))
        start = time.perf_counter()
        X = vectorizer.transform([clean_text(t) for t in batch])
        model.predict_proba(X)
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "batch_size": batch_size,
        "runs": runs,
        "mean_ms": round(float(np.mean(timings)), 2),
        "p95_ms": round(float(np.percentile(timings, 95)), 2),
        "max_ms": round(float(np.max(timings)), 2),
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer, model = load_artifacts()
    test = pd.read_csv(TEST_PATH)
    X_test, y_test = test["text"], test["label"]

    y_pred = model.predict(vectorizer.transform(X_test))
    acc = accuracy_score(y_test, y_pred)
    f1_per_class = f1_score(y_test, y_pred, labels=LABELS, average=None)
    report_txt = classification_report(y_test, y_pred, labels=LABELS, target_names=NAMES, digits=4)
    report = classification_report(y_test, y_pred, labels=LABELS, target_names=NAMES, output_dict=True)

    plot_confusion(y_test, y_pred)
    latency = benchmark(vectorizer, model, X_test.tolist())

    criteria = {
        f"accuracy >= {MIN_ACCURACY}": bool(acc >= MIN_ACCURACY),
        f"f1 par classe > {MIN_F1_PER_CLASS}": bool((f1_per_class > MIN_F1_PER_CLASS).all()),
        f"latence p95 (50 comm.) < {MAX_LATENCY_MS_50} ms": bool(latency["p95_ms"] < MAX_LATENCY_MS_50),
    }
    metrics = {
        "n_test": len(test),
        "accuracy": round(acc, 4),
        "f1_macro": round(report["macro avg"]["f1-score"], 4),
        "f1_weighted": round(report["weighted avg"]["f1-score"], 4),
        "per_class": {n: {k: round(v, 4) for k, v in report[n].items()} for n in NAMES},
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=LABELS).tolist(),
        "latency_batch_50": latency,
        "criteria": criteria,
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORTS_DIR / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    print("\n=== Évaluation sur le test set ===")
    print(report_txt)
    print(f"Latence batch de 50 : moyenne {latency['mean_ms']} ms, p95 {latency['p95_ms']} ms")
    print("\n=== Critères du sujet ===")
    for name, ok in criteria.items():
        print(f"  {'OK ' if ok else 'KO '} {name}")
    logger.info("Évaluation : acc=%.4f f1_macro=%.4f", acc, metrics["f1_macro"])


if __name__ == "__main__":
    main()
