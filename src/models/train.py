"""Entraînement du modèle de sentiment : comparaison d'algorithmes + GridSearchCV.

Entrée  : data/processed/train.csv
Sorties : models/vectorizer.joblib, models/model.joblib, models/model_metadata.json
          reports/model_comparison.csv, reports/gridsearch_results.csv
          reports/figures/model_comparison.png

Toute la sélection se fait par validation croisée sur le TRAIN uniquement ;
le test set est réservé à src/models/evaluate.py.

Usage :
    uv run python -m src.models.train [--skip-comparison]
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.utils.logger import get_logger

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
SEED = 42
CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
SCORING = {"accuracy": "accuracy", "f1_macro": "f1_macro"}

logger = get_logger("train", "training.log")


def make_pipeline(clf, **tfidf_params) -> Pipeline:
    params = {"sublinear_tf": True, "min_df": 2, **tfidf_params}
    return Pipeline([("tfidf", TfidfVectorizer(**params)), ("clf", clf)])


def compare_models(X: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Compare plusieurs algorithmes avec le même TF-IDF par défaut (CV 3 folds)."""
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "LinearSVC": LinearSVC(class_weight="balanced"),
        "ComplementNB": ComplementNB(),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=SEED),
    }
    rows = []
    for name, clf in candidates.items():
        logger.info("CV : %s", name)
        scores = cross_validate(make_pipeline(clf), X, y, cv=CV, scoring=SCORING, n_jobs=1)
        rows.append({
            "model": name,
            "cv_accuracy": scores["test_accuracy"].mean(),
            "cv_f1_macro": scores["test_f1_macro"].mean(),
            "cv_f1_macro_std": scores["test_f1_macro"].std(),
            "fit_time_s": scores["fit_time"].mean(),
        })
        logger.info("  acc=%.4f  f1_macro=%.4f  (%.1fs/fold)",
                    rows[-1]["cv_accuracy"], rows[-1]["cv_f1_macro"], rows[-1]["fit_time_s"])
    return pd.DataFrame(rows).sort_values("cv_f1_macro", ascending=False)


def plot_comparison(df: pd.DataFrame) -> None:
    long = df.melt(id_vars="model", value_vars=["cv_accuracy", "cv_f1_macro"],
                   var_name="métrique", value_name="score")
    long["métrique"] = long["métrique"].map({"cv_accuracy": "Accuracy", "cv_f1_macro": "F1 macro"})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=long, x="model", y="score", hue="métrique", palette=["#4c72b0", "#dd8452"], ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    ax.axhline(0.80, ls="--", c="grey", lw=1)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("")
    ax.set_title("Comparaison des algorithmes (validation croisée 3 folds, train)")
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


def grid_search(X: pd.Series, y: pd.Series) -> GridSearchCV:
    """Optimise TF-IDF + Logistic Regression (le modèle retenu par le sujet)."""
    pipeline = make_pipeline(LogisticRegression(max_iter=5000, class_weight="balanced"))
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "tfidf__max_features": [None, 50_000],
        "tfidf__sublinear_tf": [True, False],
        "clf__C": [1, 5, 10, 20],
    }
    search = GridSearchCV(pipeline, param_grid, cv=CV, scoring=SCORING, refit="f1_macro",
                          n_jobs=-1, verbose=1)
    start = time.perf_counter()
    search.fit(X, y)
    logger.info("GridSearch terminé en %.0fs — meilleur f1_macro CV = %.4f",
                time.perf_counter() - start, search.best_score_)
    logger.info("Meilleurs paramètres : %s", search.best_params_)
    return search


def save_model(search: GridSearchCV, n_train: int) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    best = search.best_estimator_
    joblib.dump(best.named_steps["tfidf"], MODELS_DIR / "vectorizer.joblib", compress=3)
    joblib.dump(best.named_steps["clf"], MODELS_DIR / "model.joblib", compress=3)

    idx = search.best_index_
    metadata = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": "TF-IDF + LogisticRegression",
        "sklearn_version": sklearn.__version__,
        "classes": [int(c) for c in best.named_steps["clf"].classes_],
        "label_names": {"-1": "negative", "0": "neutral", "1": "positive"},
        "best_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in search.best_params_.items()},
        "cv_f1_macro": round(float(search.cv_results_["mean_test_f1_macro"][idx]), 4),
        "cv_accuracy": round(float(search.cv_results_["mean_test_accuracy"][idx]), 4),
        "n_train": n_train,
        "vocabulary_size": len(best.named_steps["tfidf"].vocabulary_),
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Modèle sauvegardé dans %s (vocabulaire : %d termes)", MODELS_DIR, metadata["vocabulary_size"])


def main(skip_comparison: bool) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH)
    X, y = train["text"], train["label"]
    logger.info("Train : %d commentaires", len(train))

    if not skip_comparison:
        comparison = compare_models(X, y)
        comparison.round(4).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
        plot_comparison(comparison)
        print("\n=== Comparaison des algorithmes (CV) ===")
        print(comparison.round(4).to_string(index=False))

    search = grid_search(X, y)
    cols = ["params", "mean_test_accuracy", "mean_test_f1_macro", "std_test_f1_macro",
            "mean_fit_time", "rank_test_f1_macro"]
    (pd.DataFrame(search.cv_results_)[cols].sort_values("rank_test_f1_macro")
     .round(4).to_csv(REPORTS_DIR / "gridsearch_results.csv", index=False))

    print("\n=== GridSearchCV — 5 meilleures configurations ===")
    top = pd.DataFrame(search.cv_results_).sort_values("rank_test_f1_macro").head(5)
    for _, r in top.iterrows():
        print(f"  f1={r.mean_test_f1_macro:.4f} acc={r.mean_test_accuracy:.4f}  {r.params}")

    save_model(search, len(train))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-comparison", action="store_true", help="ne lancer que le GridSearch")
    main(parser.parse_args().skip_comparison)
