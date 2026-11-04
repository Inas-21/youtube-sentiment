# Image de production de l'API YouTube Sentiment
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860

WORKDIR /app

# 1) Dépendances d'abord : cette couche reste en cache tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) Utilisateur non-root (sécurité) + dossier de logs accessible en écriture
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/logs \
    && chown appuser:appuser /app/logs

# 3) Code et modèles en dernier : ce sont eux qui changent le plus souvent
COPY --chown=appuser:appuser models/vectorizer.joblib models/model.joblib models/model_metadata.json ./models/
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser app_api.py .

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health', timeout=4)"

CMD ["python", "app_api.py"]
