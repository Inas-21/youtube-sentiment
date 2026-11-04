"""Point d'entrée de production de l'API (utilisé par le Dockerfile).

Le port est configurable via la variable d'environnement PORT (7860 par défaut).
    python app_api.py
"""

import os

import uvicorn

from src.api.main import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",  # écouter sur toutes les interfaces : obligatoire dans un conteneur
        port=int(os.getenv("PORT", "7860")),
        workers=1,       # un seul processus : le modèle n'est chargé qu'une fois en mémoire
        proxy_headers=True,  # derrière le proxy HTTPS de Codespaces
        forwarded_allow_ips="*",
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
