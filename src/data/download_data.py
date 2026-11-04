"""Téléchargement du dataset Reddit Sentiment Analysis vers data/raw/reddit.csv.

Usage :
    uv run python -m src.data.download_data [--force]
"""

import argparse
from pathlib import Path

import pandas as pd
import requests

from src.utils.logger import get_logger

DATA_URL = (
    "https://raw.githubusercontent.com/Himanshu-1703/"
    "reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
)
ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "reddit.csv"
LABEL_NAMES = {-1: "négatif", 0: "neutre", 1: "positif"}

logger = get_logger("download_data", "data_pipeline.log")


def download(url: str = DATA_URL, dest: Path = RAW_PATH, force: bool = False) -> Path:
    if dest.exists() and not force:
        logger.info("Fichier déjà présent : %s (utiliser --force pour re-télécharger)", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Téléchargement depuis %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    dest.write_bytes(response.content)
    logger.info("Sauvegardé dans %s (%.1f Mo)", dest, len(response.content) / 1e6)
    return dest


def print_stats(path: Path) -> None:
    df = pd.read_csv(path)
    text_col, label_col = df.columns[0], df.columns[1]

    print("\n=== Statistiques du dataset ===")
    print(f"Fichier          : {path}")
    print(f"Colonnes         : {list(df.columns)}")
    print(f"Commentaires     : {len(df):,}")
    print(f"Textes manquants : {df[text_col].isna().sum()}")
    print(f"Doublons         : {df.duplicated(subset=text_col).sum()}")

    print("\nDistribution des labels :")
    counts = df[label_col].value_counts().sort_index()
    for label, n in counts.items():
        name = LABEL_NAMES.get(label, "?")
        print(f"  {label:>2} ({name:<8}) : {n:>6,}  ({n / len(df):.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-télécharger même si le fichier existe")
    args = parser.parse_args()

    print_stats(download(force=args.force))
