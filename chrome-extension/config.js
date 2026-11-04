// Configuration par défaut (modifiable depuis le panneau ⚙ du popup, sauvegardée dans chrome.storage).
// Après le déploiement (Étape 7), remplacer API_URL par l'URL du Space Hugging Face.
const DEFAULT_SETTINGS = {
  apiUrl: "http://localhost:8000",
  maxComments: 100,
  includeReplies: false,
  autoScroll: true,
};

const API_BATCH_SIZE = 100;      // l'API accepte 500 max ; on découpe en batchs de 100
const API_TIMEOUT_MS = 60000;    // un Space HF en veille peut mettre ~30 s à se réveiller
