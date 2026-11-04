# YouTube Sentiment Analysis — Projet MLOps

Analyse automatique du sentiment (négatif / neutre / positif) des commentaires YouTube :
modèle scikit-learn (TF-IDF + Logistic Regression), API FastAPI déployée sur Hugging Face Spaces (Docker)
et extension Chrome.

## Structure

```
data/raw, data/processed   # données brutes / nettoyées (non versionnées)
models/                    # modèle + vectoriseur (.joblib)
src/data                   # téléchargement, nettoyage, split
src/models                 # entraînement, évaluation
src/api                    # API FastAPI
src/utils                  # fonctions partagées (nettoyage texte, logs)
chrome-extension/          # frontend
tests/                     # tests pytest
notebooks/                 # EDA
reports/figures/           # graphiques et métriques
logs/                      # fichiers de log
```

## Installation (avec [uv](https://docs.astral.sh/uv/))

```bash
uv sync                # crée .venv (Python 3.10) et installe toutes les dépendances
uv run pytest          # lance les tests
```

Fichiers de dépendances :
- `pyproject.toml` + `uv.lock` : source de vérité
- `requirements.txt` : dépendances **prod** (Docker) — `uv export --no-dev --no-hashes --no-emit-project -o requirements.txt`
- `requirements-dev.txt` : dev complet — `uv export --no-hashes --no-emit-project -o requirements-dev.txt`
