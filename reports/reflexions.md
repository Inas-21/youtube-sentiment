# Questions de réflexion — réponses (pour le rapport technique)

## Phase 1 — Configuration de l'environnement MLOps

**Pourquoi séparer les données raw et processed ?**
Les données brutes (`data/raw`) sont la source de vérité et ne sont jamais modifiées. Les données
traitées (`data/processed`) sont un produit dérivé, régénérable à tout moment par `src/data/preprocess.py`.
Cette séparation garantit la reproductibilité (on peut changer le nettoyage et relancer le pipeline),
la traçabilité (on sait d'où vient chaque donnée d'entraînement) et évite d'écraser l'original par erreur.

**Quels fichiers ne doivent jamais être commités sur Git ?**
Les secrets (tokens Hugging Face, `.env`, clés), l'environnement virtuel (`.venv/`), les caches
(`__pycache__`, `.pytest_cache`), les logs et les données volumineuses re-téléchargeables. Ils sont
exclus via `.gitignore`. Les modèles `.joblib` finaux sont en revanche versionnés car ils sont
petits, font partie des livrables et sont nécessaires au déploiement.

**Quelle est l'importance d'un environnement virtuel ?**
Il isole les dépendances du projet de celles du système et des autres projets, évitant les conflits
de versions. Avec `uv`, le fichier `uv.lock` fige les versions exactes : n'importe qui (ou le
conteneur Docker) reproduit exactement le même environnement, en Python 3.10 comme l'image de
production `python:3.10-slim`.

## Phase 2 — Pipeline de données et préparation

**Pourquoi nettoyer les URLs et mentions ?**
- *Elles ne portent pas de sentiment* : `https://youtu.be/xyz` ou `@john_doe` ne disent rien de
  l'opinion exprimée. Les garder ajoute du bruit au vocabulaire TF-IDF.
- *Elles sont quasi uniques* : chaque URL ou pseudo n'apparaît qu'une ou deux fois, ce qui gonfle le
  vocabulaire de tokens rares inutiles (plus de mémoire, risque de sur-apprentissage sur des
  identifiants plutôt que sur le langage).
- *Elles ne généralisent pas* : les pseudos Reddit du dataset d'entraînement (`u/...`) n'existent pas
  sur YouTube (`@...`). Un modèle qui s'y appuierait échouerait en production.
- *Vie privée* : les mentions sont des données personnelles qu'il est préférable de ne pas exploiter.

Dans notre pipeline, `clean_text()` (`src/utils/text.py`) supprime les URLs (`http…`, `www…`), les
mentions YouTube (`@user`) et Reddit (`u/user`, `r/sub`), décode les entités HTML (`&amp;`, `&#39;`)
et retire la ponctuation. **La même fonction est appliquée à l'entraînement et dans l'API**, sinon le
modèle verrait en production un texte différent de celui appris (*training/serving skew*).

**Comment gérer les emojis dans le texte ?**
Trois options existent : les supprimer, les garder comme caractères, ou les convertir en texte.
Nous avons choisi la **conversion en mots** avec `emoji.demojize()` : 😂 → `face with tears of joy`,
❤️ → `red heart`, 😡 → `enraged face`.
- Les emojis portent un sentiment fort, surtout sur YouTube où ils sont omniprésents ; les supprimer
  ferait perdre ce signal (« 😂😂😂 » deviendrait un commentaire vide).
- Les garder bruts ne fonctionne pas bien avec TF-IDF : le tokenizer par défaut les ignore, et le
  dataset Reddit en contient très peu (78 commentaires sur 37 249), donc le modèle ne pourrait pas
  apprendre leur sens.
- Convertis en mots, ils produisent des termes (`joy`, `love`, `heart`, `angry`, `sad`) **déjà
  présents dans le vocabulaire d'entraînement** : le modèle exploite donc leur sentiment même sans en
  avoir vu beaucoup. Nous séparons les mots (`face_with_tears_of_joy` → `face with tears of joy`)
  précisément pour profiter de ce vocabulaire existant.

Limite : certains mots neutres apparaissent (`face`, `with`) et l'ironie (😂 sarcastique) reste mal
gérée — une piste d'amélioration serait un lexique emoji→sentiment dédié.

**Quel ratio train/test est approprié ?**
Nous utilisons **80 % / 20 %**, **stratifié** et avec une **graine fixe** (`random_state=42`) :
- Avec 36 568 commentaires après nettoyage, 20 % donne 7 314 exemples de test (≈ 1 650 négatifs) :
  largement assez pour une estimation fiable des métriques par classe, tout en gardant 29 254
  exemples pour l'apprentissage.
- Pour un petit dataset (quelques milliers de lignes), on garderait plutôt 80/20 avec validation
  croisée ; pour des millions de lignes, 90/10 ou 98/2 suffirait. 70/30 priverait inutilement le
  modèle de données.
- La **stratification** conserve les proportions des classes (22,5 % / 34,5 % / 43,0 %) dans les deux
  ensembles, indispensable avec un déséquilibre.
- La **graine fixe** rend le split reproductible.
- L'optimisation des hyperparamètres se fera par **validation croisée sur le train uniquement** : le
  test set n'est utilisé qu'une fois, pour l'évaluation finale.

**Précautions supplémentaires prises :**
- Déduplication **avant** le split (449 doublons bruts, 418 supprimés après nettoyage) pour éviter qu'un
  même texte soit à la fois en train et en test, ce qui gonflerait artificiellement les scores.
- Suppression des 121 textes vides, **tous étiquetés neutre** : ils auraient appris au modèle un
  raccourci « vide = neutre ».
- Déséquilibre modéré (ratio 1,9) : traité par `class_weight='balanced'` à l'entraînement plutôt que
  par sur/sous-échantillonnage, pour ne pas dupliquer ni jeter de données.
