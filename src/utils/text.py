"""Nettoyage de texte partagé entre l'entraînement et l'API.

IMPORTANT : toute modification ici impose de ré-entraîner le modèle, sinon le texte
vu en production ne ressemblera plus à celui vu à l'entraînement.
"""

import html
import re

import emoji

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[\w.\-]+|\b[ur]/\w+")  # @user (YouTube), u/user et r/sub (Reddit)
NON_WORD_RE = re.compile(r"[^\w\s]|_")  # ponctuation et symboles ; \w garde les lettres de tous les alphabets
WHITESPACE_RE = re.compile(r"\s+")


def emojis_to_words(text: str) -> str:
    """😂 -> ' face with tears of joy ' : les mots produits existent dans le vocabulaire
    d'entraînement (joy, heart, angry...), ce qui conserve le signal de sentiment."""
    return emoji.demojize(text, delimiters=(" ", " ")).replace("_", " ")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)  # &amp; &#39; ... (fréquent dans le HTML YouTube)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = emojis_to_words(text)
    text = text.lower()
    text = NON_WORD_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()
