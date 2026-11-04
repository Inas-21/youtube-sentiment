"""Tests du modèle : performance sur le test set, cas limites, temps d'inférence."""

import json
import time
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.api.model_loader import SentimentModel

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "processed" / "test.csv"


@pytest.fixture(scope="module")
def model():
    return SentimentModel()


# ---------- Performance sur le test set ----------

@pytest.mark.skipif(not TEST_PATH.exists(), reason="lancer src.data.preprocess d'abord")
def test_performance_on_test_set(model):
    test = pd.read_csv(TEST_PATH)
    y_pred = model.model.predict(model.vectorizer.transform(test["text"]))
    assert accuracy_score(test["label"], y_pred) >= 0.80
    assert (f1_score(test["label"], y_pred, average=None) > 0.75).all()


def test_metadata_consistent(model):
    meta = json.loads((ROOT / "models" / "model_metadata.json").read_text(encoding="utf-8"))
    assert meta["classes"] == model.classes == [-1, 0, 1]
    assert len(model.vectorizer.vocabulary_) == meta["vocabulary_size"]


# ---------- Cas limites : le modèle ne doit jamais planter ----------

EDGE_CASES = {
    "un caractère": "a",
    "très court": "ok",
    "ponctuation seule": "!!!",
    "chiffres": "12345",
    "url + mention seules": "https://youtu.be/x @john",
    "très long": "great video " * 2000,
    "emojis seuls": "😍😍😍",
    "texte + emoji": "I love it ❤️",
    "français + anglais": "Ce film est génial, I love it",
    "hinglish": "yeh video bahut accha hai, great job",
    "devanagari": "बहुत अच्छा वीडियो",
    "arabe": "مرحبا رائع",
    "html": "Tom &amp; Jerry &#39;rocks&#39;",
    "sauts de ligne": "first line\n\nsecond line\t tab",
}


@pytest.mark.parametrize("text", EDGE_CASES.values(), ids=EDGE_CASES.keys())
def test_edge_cases_valid_output(model, text):
    [pred] = model.predict([text])
    assert pred["sentiment"] in {"negative", "neutral", "positive"}
    assert 0 <= pred["confidence"] <= 1
    assert abs(sum(pred["probabilities"].values()) - 1) < 1e-3


@pytest.mark.parametrize("text,expected", [
    ("I love it ❤️", "positive"),
    ("this is bad 👎", "negative"),
    ("best tutorial ever thank you", "positive"),
    ("I hate this so much, worst video", "negative"),
    ("Ce film est génial, I love it", "positive"),
    ("yeh video bahut accha hai, great job", "positive"),
    ("ok", "neutral"),
    ("great video " * 2000, "positive"),
])
def test_expected_sentiments(model, text, expected):
    assert model.predict([text])[0]["sentiment"] == expected


# Limites connues, documentées dans le rapport (xfail = échec attendu)
@pytest.mark.xfail(reason="unigrammes : la négation 'not good' n'est pas capturée", strict=True)
def test_known_limit_negation(model):
    assert model.predict(["not good at all"])[0]["sentiment"] == "negative"


@pytest.mark.xfail(reason="emojis seuls : mots issus de demojize trop rares dans le corpus Reddit", strict=True)
def test_known_limit_emoji_only(model):
    assert model.predict(["😍😍😍"])[0]["sentiment"] == "positive"


# ---------- Temps d'inférence ----------

def test_inference_time_batch_50(model):
    batch = [f"this video is really nice number {i} 😂" for i in range(50)]
    model.predict(batch)  # échauffement
    timings = []
    for _ in range(20):
        start = time.perf_counter()
        model.predict(batch)
        timings.append((time.perf_counter() - start) * 1000)
    assert sorted(timings)[int(0.95 * len(timings)) - 1] < 100  # p95 < 100 ms
