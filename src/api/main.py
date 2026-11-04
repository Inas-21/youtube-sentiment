"""API REST d'analyse de sentiment des commentaires YouTube.

Lancement local :
    uv run uvicorn src.api.main:app --reload --port 8000
Documentation interactive : http://localhost:8000/docs
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.model_loader import SentimentModel, compute_statistics
from src.api.schemas import HealthResponse, PredictBatchRequest, PredictBatchResponse
from src.utils.logger import get_logger

logger = get_logger("api", "api.log")
START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement UNE seule fois au démarrage (et non à chaque requête)
    try:
        app.state.model = SentimentModel()
        logger.info("Modèle chargé (version %s)", app.state.model.version)
    except Exception:
        app.state.model = None
        logger.exception("Échec du chargement du modèle : l'API démarre en mode dégradé")
    yield


app = FastAPI(
    title="YouTube Sentiment API",
    description="Analyse de sentiment (négatif / neutre / positif) de commentaires YouTube — "
                "TF-IDF + Logistic Regression.",
    version="1.0.0",
    lifespan=lifespan,
)

# L'extension Chrome appelle l'API depuis une origine chrome-extension://<id>.
# ALLOWED_ORIGINS permet de restreindre en production (liste séparée par des virgules).
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.info("%s %s -> %d (%.1f ms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Erreur non gérée sur %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Erreur interne du serveur"})


def get_model(request: Request) -> SentimentModel:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    return model


@app.get("/", include_in_schema=False)
def root():
    return {"name": app.title, "version": app.version, "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health(request: Request):
    """Vérifie l'état de l'API et du modèle."""
    model = request.app.state.model
    return HealthResponse(
        status="ok" if model else "degraded",
        model_loaded=model is not None,
        model_version=model.version if model else None,
        algorithm=model.metadata.get("algorithm") if model else None,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.post("/predict_batch", response_model=PredictBatchResponse, tags=["prediction"])
def predict_batch(payload: PredictBatchRequest, request: Request):
    """Analyse un batch de commentaires et retourne le sentiment de chacun + des statistiques globales."""
    model = get_model(request)
    start = time.perf_counter()
    predictions = model.predict(payload.comments)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return PredictBatchResponse(
        predictions=predictions,
        statistics=compute_statistics(predictions),
        model_version=model.version,
        processing_time_ms=round(elapsed_ms, 2),
    )
