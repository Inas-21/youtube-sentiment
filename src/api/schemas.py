"""Schémas Pydantic : validation des requêtes et documentation automatique (/docs)."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_BATCH_SIZE = 500
MAX_COMMENT_CHARS = 5000

Sentiment = Literal["negative", "neutral", "positive"]


class PredictBatchRequest(BaseModel):
    comments: list[str] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE,
        description=f"Liste de commentaires bruts (1 à {MAX_BATCH_SIZE})",
        examples=[["This video is amazing, thanks! 😍", "Worst tutorial ever.", "Uploaded at 5pm"]],
    )

    @field_validator("comments")
    @classmethod
    def no_blank_comments(cls, comments: list[str]) -> list[str]:
        blank = [i for i, c in enumerate(comments) if not c.strip()]
        if blank:
            raise ValueError(f"commentaires vides aux positions {blank}")
        # Les commentaires très longs sont tronqués plutôt que rejetés
        return [c[:MAX_COMMENT_CHARS] for c in comments]


class Prediction(BaseModel):
    text: str
    label: Literal[-1, 0, 1]
    sentiment: Sentiment
    confidence: float = Field(..., ge=0, le=1, description="probabilité de la classe prédite")
    probabilities: dict[Sentiment, float]


class BatchStatistics(BaseModel):
    total: int
    counts: dict[Sentiment, int]
    percentages: dict[Sentiment, float]
    average_confidence: float
    sentiment_score: float = Field(..., description="moyenne des labels, de -1 (négatif) à 1 (positif)")
    dominant_sentiment: Sentiment


class PredictBatchResponse(BaseModel):
    predictions: list[Prediction]
    statistics: BatchStatistics
    model_version: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None
    algorithm: str | None = None
    uptime_seconds: float
