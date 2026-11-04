"""Analyse exploratoire du dataset brut (data/raw/reddit.csv).

Produit :
    reports/figures/class_distribution.png
    reports/figures/text_length_distribution.png
    reports/figures/text_length_boxplot.png
    reports/figures/top_words_per_class.png
    reports/eda_summary.json

Usage :
    uv run python -m src.data.eda
"""

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : on sauvegarde uniquement
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.data.download_data import LABEL_NAMES, RAW_PATH
from src.utils.logger import get_logger

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "reports" / "figures"
SUMMARY_PATH = ROOT / "reports" / "eda_summary.json"
PALETTE = {"négatif": "#d9534f", "neutre": "#9e9e9e", "positif": "#5cb85c"}
ORDER = ["négatif", "neutre", "positif"]

logger = get_logger("eda", "data_pipeline.log")


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns={"clean_comment": "text", "category": "label"})
    df["sentiment"] = df["label"].map(LABEL_NAMES)
    return df


def data_quality(df: pd.DataFrame) -> dict:
    text = df["text"].fillna("")
    return {
        "n_rows": len(df),
        "missing_text": int(df["text"].isna().sum()),
        "empty_text": int((text.str.strip() == "").sum()) - int(df["text"].isna().sum()),
        "duplicate_texts": int(df.duplicated(subset="text").sum()),
        "conflicting_duplicates": int(  # même texte, labels différents
            df.dropna(subset=["text"]).groupby("text")["label"].nunique().gt(1).sum()
        ),
        "contains_url": int(text.str.contains(r"http\S+|www\.\S+", regex=True).sum()),
        "contains_mention": int(text.str.contains(r"(?:^|\s)[@u]/\w+|@\w+", regex=True).sum()),
        "contains_non_ascii": int(text.str.contains(r"[^\x00-\x7F]", regex=True).sum()),
    }


def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df["sentiment"].value_counts().reindex(ORDER)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[s] for s in counts.index])
    for bar, n in zip(bars, counts.values):
        ax.annotate(f"{n:,}\n({n / len(df):.1%})", (bar.get_x() + bar.get_width() / 2, n),
                    ha="center", va="bottom", fontsize=10)
    ax.set_title("Distribution des classes de sentiment")
    ax.set_ylabel("Nombre de commentaires")
    ax.set_ylim(0, counts.max() * 1.18)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_text_lengths(df: pd.DataFrame) -> None:
    clip = df["n_words"].quantile(0.99)  # on coupe la longue traîne pour la lisibilité
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data=df[df["n_words"] <= clip], x="n_words", hue="sentiment", hue_order=ORDER,
                 palette=PALETTE, bins=60, element="step", stat="density", common_norm=False, ax=ax)
    ax.set_title(f"Longueur des commentaires (mots, ≤ percentile 99 = {clip:.0f})")
    ax.set_xlabel("Nombre de mots")
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "text_length_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=df, x="sentiment", y="n_words", hue="sentiment", order=ORDER,
                palette=PALETTE, showfliers=False, legend=False, ax=ax)
    ax.set_title("Nombre de mots par classe (sans valeurs extrêmes)")
    ax.set_xlabel("")
    ax.set_ylabel("Nombre de mots")
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "text_length_boxplot.png", dpi=150)
    plt.close(fig)


def top_words(df: pd.DataFrame, n: int = 15) -> dict:
    result = {}
    for sentiment in ORDER:
        tokens = Counter()
        for text in df.loc[df["sentiment"] == sentiment, "text"]:
            tokens.update(w for w in re.findall(r"[a-z']+", text)
                          if w not in ENGLISH_STOP_WORDS and len(w) > 2)
        result[sentiment] = tokens.most_common(n)
    return result


def plot_top_words(words: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, sentiment in zip(axes, ORDER):
        labels, counts = zip(*words[sentiment])
        ax.barh(labels[::-1], counts[::-1], color=PALETTE[sentiment])
        ax.set_title(sentiment.capitalize())
        ax.tick_params(axis="y", labelsize=9)
        sns.despine(ax=ax)
    fig.suptitle("Mots les plus fréquents par classe (hors stopwords)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_words_per_class.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    quality = data_quality(df)

    df = df.dropna(subset=["text"]).copy()
    df["n_words"] = df["text"].str.split().str.len()
    df["n_chars"] = df["text"].str.len()

    length_stats = (df.groupby("sentiment")["n_words"]
                    .describe(percentiles=[0.5, 0.95])
                    .reindex(ORDER).round(1))

    plot_class_distribution(df)
    plot_text_lengths(df)
    words = top_words(df)
    plot_top_words(words)

    counts = df["sentiment"].value_counts().reindex(ORDER)
    summary = {
        "quality": quality,
        "class_counts": counts.to_dict(),
        "class_ratio": (counts / counts.sum()).round(3).to_dict(),
        "imbalance_ratio_max_min": round(counts.max() / counts.min(), 2),
        "words_per_comment": length_stats.to_dict(orient="index"),
        "top_words": {k: [w for w, _ in v] for k, v in words.items()},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Qualité des données ===")
    for k, v in quality.items():
        print(f"  {k:<24}: {v:,}")
    print("\n=== Classes ===")
    for s in ORDER:
        print(f"  {s:<8}: {counts[s]:>6,} ({counts[s] / counts.sum():.1%})")
    print(f"  ratio max/min : {summary['imbalance_ratio_max_min']}")
    print("\n=== Nombre de mots par commentaire ===")
    print(length_stats.to_string())
    print("\n=== Top mots ===")
    for s in ORDER:
        print(f"  {s:<8}: {', '.join(summary['top_words'][s][:10])}")
    logger.info("Figures sauvegardées dans %s, résumé dans %s", FIG_DIR, SUMMARY_PATH)


if __name__ == "__main__":
    main()
