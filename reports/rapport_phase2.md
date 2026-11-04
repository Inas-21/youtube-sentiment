# Phase 2 — Pipeline de données et préparation

## 2.1 Collecte des données

Le jeu de données utilisé est le *Reddit Sentiment Analysis Dataset*, composé de commentaires Reddit
annotés selon trois classes de sentiment : **-1 (négatif)**, **0 (neutre)** et **1 (positif)**.
Le téléchargement est automatisé par le script `src/data/download_data.py`, qui récupère le fichier CSV
depuis le dépôt GitHub source, le sauvegarde dans `data/raw/reddit.csv` et affiche des statistiques
descriptives (nombre de commentaires, distribution des labels). Le script ne retélécharge pas le
fichier s'il est déjà présent (option `--force` pour forcer la mise à jour), et chaque exécution est
journalisée dans `logs/data_pipeline.log`.

Le fichier brut contient **37 249 commentaires** répartis sur deux colonnes (`clean_comment`,
`category`), renommées `text` et `label` conformément aux spécifications du projet.

## 2.2 Analyse exploratoire des données (EDA)

L'analyse exploratoire, implémentée dans `src/data/eda.py`, a porté sur trois axes : la qualité des
données, la distribution des classes et les caractéristiques des textes. Les figures sont générées
automatiquement dans `reports/figures/` et un résumé chiffré est exporté dans `reports/eda_summary.json`.

### Qualité des données

| Indicateur | Valeur |
|---|---|
| Commentaires | 37 249 |
| Textes manquants | 100 |
| Textes vides | 121 (tous étiquetés *neutre*) |
| Doublons | 449 (aucun avec des labels contradictoires) |
| Textes contenant une URL | 28 |
| Textes contenant des caractères non-ASCII | 1 950 (dont 78 avec emojis, 115 en hindi) |

Nous avons constaté que les textes sont **déjà partiellement pré-traités** dans le dataset d'origine :
ils sont en minuscules, sans ponctuation, et certains mots très courts ont été retirés. Ce constat est
important, car les commentaires YouTube reçus en production seront bruts. Le nettoyage doit donc
ramener ces commentaires au même format que les données d'entraînement.

### Distribution des classes

> **[Figure 1 — `class_distribution.png`]** *Distribution des classes de sentiment dans le dataset.*

Le dataset présente un **déséquilibre modéré** : 42,6 % de commentaires positifs, 35,1 % de neutres et
22,3 % de négatifs, soit un ratio de 1,9 entre la classe majoritaire et la classe minoritaire.
Ce déséquilibre n'est pas extrême, mais il risque de biaiser le modèle en faveur de la classe positive
et de dégrader le F1-score de la classe négative. Toutes les classes dépassent largement le minimum
exigé (300 exemples par classe).

### Longueur des textes

> **[Figure 2 — `text_length_distribution.png`]** *Distribution du nombre de mots par commentaire, par classe.*
>
> **[Figure 3 — `text_length_boxplot.png`]** *Nombre de mots par classe (hors valeurs extrêmes).*

| Classe | Médiane (mots) | Moyenne | 95ᵉ percentile | Maximum |
|---|---|---|---|---|
| Négatif | 19 | 34,8 | 110 | 865 |
| Neutre | 6 | 9,8 | 27 | 518 |
| Positif | 21 | 42,8 | 147 | 1 307 |

La longueur des textes est très discriminante : les commentaires **neutres sont nettement plus courts**
(médiane de 6 mots) que les commentaires positifs et négatifs (19 à 21 mots). Les distributions
présentent une longue traîne, avec quelques commentaires très longs (plus de 1 000 mots) mais rares.

### Vocabulaire

> **[Figure 4 — `top_words_per_class.png`]** *Mots les plus fréquents par classe (hors mots vides).*

Les mots les plus fréquents (*people, india, modi, bjp, congress*) révèlent un corpus **fortement
centré sur la politique indienne**. Les classes se distinguent par des termes spécifiques : *good,
right, think* pour la classe positive, des termes injurieux pour la classe négative, et des mots en
hindi translittéré (*hai*) pour la classe neutre. Cette spécificité thématique constitue une **limite
de généralisation** : le modèle pourrait être moins performant sur des commentaires YouTube portant sur
d'autres sujets (jeux vidéo, musique, technologie…).

## 2.3 Nettoyage des données

Le nettoyage repose sur une fonction unique, `clean_text()` (`src/utils/text.py`), **partagée entre le
pipeline d'entraînement et l'API de production**. Ce choix garantit que le modèle reçoit en production
un texte transformé exactement comme lors de l'entraînement, évitant ainsi le *training/serving skew*.
Les transformations appliquées sont, dans l'ordre :

1. **Décodage des entités HTML** (`&amp;`, `&#39;`), fréquentes dans le texte extrait des pages YouTube ;
2. **Suppression des URLs** (`http…`, `www…`) ;
3. **Suppression des mentions** YouTube (`@utilisateur`) et Reddit (`u/utilisateur`, `r/communauté`) ;
4. **Conversion des emojis en mots** avec la bibliothèque `emoji` (😂 → *face with tears of joy*,
   ❤️ → *red heart*) ;
5. **Mise en minuscules** ;
6. **Suppression de la ponctuation et des caractères spéciaux**, en conservant les lettres de tous les
   alphabets (latin, devanagari, arabe…) ;
7. **Normalisation des espaces**.

Le choix de **convertir les emojis plutôt que de les supprimer** se justifie par leur forte charge
émotionnelle sur YouTube. Les mots produits (*joy, heart, angry*) existent déjà dans le vocabulaire
d'entraînement, ce qui permet au modèle d'exploiter leur sentiment bien que le dataset Reddit contienne
très peu d'emojis.

La fonction est couverte par six tests unitaires (`tests/test_text.py`) : URLs et mentions, emojis,
ponctuation, entités HTML, alphabets non latins et entrées vides ou invalides.

Le script `src/data/preprocess.py` applique ensuite les opérations suivantes au dataset :

| Étape | Commentaires restants | Supprimés |
|---|---|---|
| Données brutes | 37 249 | — |
| Suppression des textes manquants | 37 149 | 100 |
| Suppression des textes vides après nettoyage | 36 986 | 163 |
| Suppression des doublons | **36 568** | 418 |

La suppression des textes vides est essentielle : les 121 textes vides d'origine étaient **tous
étiquetés neutres**, ce qui aurait appris au modèle un raccourci erroné (« texte vide = neutre »).
La déduplication est effectuée **après le nettoyage**, car deux textes différents peuvent devenir
identiques une fois nettoyés, et **avant le split**, pour éviter qu'un même commentaire se retrouve à
la fois dans les ensembles d'entraînement et de test, ce qui gonflerait artificiellement les scores.

## 2.4 Gestion du déséquilibre des classes

Compte tenu du déséquilibre modéré (ratio 1,9), nous avons choisi de **ne pas ré-échantillonner** les
données (ni sur-échantillonnage, ni sous-échantillonnage), afin de ne pas dupliquer artificiellement
des exemples ni perdre d'information. Le déséquilibre sera compensé lors de l'entraînement par une
**pondération des classes** (`class_weight='balanced'`), qui donne plus de poids aux erreurs sur la
classe minoritaire (négative).

## 2.5 Séparation entraînement / test

Les données nettoyées sont séparées en un ensemble d'entraînement et un ensemble de test selon un
ratio **80 / 20**, avec **stratification** sur le label et une **graine aléatoire fixe**
(`random_state = 42`) pour garantir la reproductibilité.

| Ensemble | Commentaires | Négatif | Neutre | Positif |
|---|---|---|---|---|
| Entraînement | 29 254 | 22,5 % | 34,5 % | 43,0 % |
| Test | 7 314 | 22,5 % | 34,5 % | 43,0 % |

Avec 7 314 exemples, l'ensemble de test est suffisamment grand pour estimer de façon fiable les
métriques de chaque classe (environ 1 650 exemples négatifs), tout en conservant l'essentiel des
données pour l'apprentissage. L'optimisation des hyperparamètres sera réalisée par validation croisée
sur l'ensemble d'entraînement uniquement ; l'ensemble de test ne sera utilisé qu'une seule fois, pour
l'évaluation finale du modèle.

Les fichiers produits sont stockés au format CSV avec les colonnes `text` et `label` :
`data/processed/reddit_clean.csv`, `train.csv` et `test.csv`. Les statistiques du pipeline sont
exportées dans `reports/preprocessing_summary.json`.

## 2.6 Bilan

À l'issue de cette phase, nous disposons d'un pipeline de données **automatisé, reproductible et
testé**, qui transforme le fichier brut en ensembles d'entraînement et de test propres, équilibrés en
proportions et sans fuite de données. Les commandes suivantes suffisent à régénérer l'ensemble des
données et des analyses :

```bash
uv run python -m src.data.download_data
uv run python -m src.data.eda
uv run python -m src.data.preprocess
```
