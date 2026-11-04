"""Nettoyage du dataset brut et split train/test reproductible.

Entrée  : data/raw/reddit.csv
Sorties : data/processed/reddit_clean.csv, train.csv, test.csv (colonnes 'text', 'label')
          reports/preprocessing_summary.json

Usage :
    uv run python -m src.data.preprocess [--test-size 0.2] [--seed 42]
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.download_data import LABEL_NAMES, RAW_PATH
from src.utils.logger import get_logger
from src.utils.text import clean_text

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
SUMMARY_PATH = ROOT / "reports" / "preprocessing_summary.json"
MIN_PER_CLASS = 300  # exigence du sujet

logger = get_logger("preprocess", "data_pipeline.log")


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    steps = {"raw": len(df)}

    df = df.rename(columns={"clean_comment": "text", "category": "label"})[["text", "label"]]
    df = df.dropna(subset=["text", "label"])
    steps["after_dropna"] = len(df)

    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin(LABEL_NAMES)]

    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    steps["after_drop_empty"] = len(df)

    # Déduplication APRÈS nettoyage (deux textes peuvent devenir identiques)
    # et AVANT le split, pour éviter qu'un même texte soit à la fois en train et en test.
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    steps["after_dedup"] = len(df)

    return df, steps


def check_requirements(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts()
    missing = [LABEL_NAMES[l] for l in LABEL_NAMES if counts.get(l, 0) < MIN_PER_CLASS]
    if len(df) < 1000 or missing:
        raise ValueError(f"Dataset insuffisant : {len(df)} lignes, classes < {MIN_PER_CLASS} : {missing}")


def main(test_size: float, seed: int) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df, steps = clean_dataset(pd.read_csv(RAW_PATH))
    check_requirements(df)

    train, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)

    df.to_csv(PROCESSED_DIR / "reddit_clean.csv", index=False)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    def dist(d: pd.DataFrame) -> dict:
        return {LABEL_NAMES[k]: int(v) for k, v in d["label"].value_counts().sort_index().items()}

    summary = {
        "steps": steps,
        "test_size": test_size,
        "random_state": seed,
        "stratified": True,
        "n_train": len(train),
        "n_test": len(test),
        "distribution_train": dist(train),
        "distribution_test": dist(test),
        "imbalance_strategy": "class_weight='balanced' à l'entraînement (pas de ré-échantillonnage)",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Nettoyage ===")
    for name, n in steps.items():
        print(f"  {name:<18}: {n:>6,}")
    print(f"\n=== Split stratifié {1 - test_size:.0%}/{test_size:.0%} (seed={seed}) ===")
    for name, part in (("train", train), ("test", test)):
        ratios = part["label"].value_counts(normalize=True).sort_index()
        print(f"  {name:<5}: {len(part):>6,}  " +
              "  ".join(f"{LABEL_NAMES[k]}={v:.1%}" for k, v in ratios.items()))
    logger.info("Données sauvegardées dans %s", PROCESSED_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.test_size, args.seed)
