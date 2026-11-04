import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import MAX_BATCH_SIZE


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # le "with" déclenche le lifespan (chargement du modèle)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_batch_structure(client):
    comments = ["I love this video, amazing work!", "This is terrible and boring", "posted at 5pm"]
    r = client.post("/predict_batch", json={"comments": comments})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 3
    for pred, text in zip(body["predictions"], comments):
        assert pred["text"] == text
        assert pred["sentiment"] in {"negative", "neutral", "positive"}
        assert 0 <= pred["confidence"] <= 1
        assert abs(sum(pred["probabilities"].values()) - 1) < 1e-3
    stats = body["statistics"]
    assert stats["total"] == 3
    assert sum(stats["counts"].values()) == 3


def test_obvious_sentiments(client):
    r = client.post("/predict_batch", json={"comments": ["awesome great love it", "horrible worst hate"]})
    preds = r.json()["predictions"]
    assert preds[0]["sentiment"] == "positive"
    assert preds[1]["sentiment"] == "negative"


@pytest.mark.parametrize("size", [1, 50, MAX_BATCH_SIZE])
def test_batch_sizes(client, size):
    r = client.post("/predict_batch", json={"comments": ["nice video"] * size})
    assert r.status_code == 200
    assert r.json()["statistics"]["total"] == size


@pytest.mark.parametrize("payload", [
    {"comments": []},                              # batch vide
    {"comments": ["ok", "   "]},                   # commentaire vide
    {"comments": ["x"] * (MAX_BATCH_SIZE + 1)},    # batch trop grand
    {"comments": "pas une liste"},                 # mauvais type
    {"texts": ["mauvaise clé"]},                   # champ manquant
])
def test_invalid_payloads(client, payload):
    assert client.post("/predict_batch", json=payload).status_code == 422


def test_long_comment_truncated_not_rejected(client):
    r = client.post("/predict_batch", json={"comments": ["great " * 3000]})
    assert r.status_code == 200
    assert len(r.json()["predictions"][0]["text"]) == 5000


def test_response_time_batch_100(client):
    comments = [f"really nice video {i}" for i in range(100)]
    client.post("/predict_batch", json={"comments": comments})  # échauffement
    r = client.post("/predict_batch", json={"comments": comments})
    assert r.status_code == 200
    assert r.json()["processing_time_ms"] < 200
    assert float(r.headers["X-Process-Time-Ms"]) < 500


def test_unknown_route_404(client):
    assert client.get("/does-not-exist").status_code == 404


def test_wrong_method_405(client):
    assert client.get("/predict_batch").status_code == 405


def test_cors_header(client):
    r = client.options("/predict_batch", headers={
        "Origin": "chrome-extension://abcdef", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") in {"*", "chrome-extension://abcdef"}
