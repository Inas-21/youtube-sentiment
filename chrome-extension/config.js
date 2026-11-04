// Configuration par défaut (modifiable depuis le panneau ⚙ du popup, sauvegardée dans chrome.storage).
// Après le déploiement, remplacer apiUrl par l'URL publique du Codespace
// (https://<nom>-7860.app.github.dev) ou la saisir dans le panneau ⚙.
const DEFAULT_SETTINGS = {
  apiUrl: "http://localhost:8000",
  maxComments: 100,
  includeReplies: false,
  autoScroll: true,
};

const API_BATCH_SIZE = 100;      // l'API accepte 500 max ; on découpe en batchs de 100
const API_TIMEOUT_MS = 60000;    // marge pour un conteneur cloud qui vient de démarrer
