// Logique du popup : extraction (content script) -> API -> affichage.

const SENTIMENTS = {
  positive: { label: "Positif", cssVar: "--positive" },
  neutral: { label: "Neutre", cssVar: "--neutral" },
  negative: { label: "Négatif", cssVar: "--negative" },
};
const ORDER = ["positive", "neutral", "negative"];

const $ = (id) => document.getElementById(id);
const state = { settings: { ...DEFAULT_SETTINGS }, result: null, filter: "all", search: "", sort: "default" };

// ---------- Paramètres & thème ----------

async function loadSettings() {
  const stored = await chrome.storage.local.get(["settings", "theme", "lastResult"]);
  state.settings = { ...DEFAULT_SETTINGS, ...stored.settings };
  $("apiUrl").value = state.settings.apiUrl;
  $("maxComments").value = String(state.settings.maxComments);
  $("includeReplies").checked = state.settings.includeReplies;
  $("autoScroll").checked = state.settings.autoScroll;

  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(stored.theme || (prefersDark ? "dark" : "light"));
  return stored.lastResult;
}

function saveSettings() {
  state.settings = {
    apiUrl: $("apiUrl").value.trim().replace(/\/+$/, "") || DEFAULT_SETTINGS.apiUrl,
    maxComments: Number($("maxComments").value),
    includeReplies: $("includeReplies").checked,
    autoScroll: $("autoScroll").checked,
  };
  chrome.storage.local.set({ settings: state.settings });
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("themeBtn").textContent = theme === "dark" ? "☀️" : "🌙";
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(next);
  chrome.storage.local.set({ theme: next });
  if (state.result) renderDonut(state.result.stats); // les couleurs dépendent du thème
}

// ---------- UI : états ----------

function setLoading(text) {
  $("loading").hidden = !text;
  $("loadingText").textContent = text || "";
  $("analyzeBtn").disabled = !!text;
}

function showError(message) {
  $("error").hidden = !message;
  $("error").textContent = message || "";
}

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (el.hidden = true), 1800);
}

// ---------- Extraction & API ----------

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

async function extractComments(tab) {
  await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] });
  const response = await chrome.tabs.sendMessage(tab.id, {
    type: "EXTRACT_COMMENTS",
    maxComments: state.settings.maxComments,
    includeReplies: state.settings.includeReplies,
    autoScroll: state.settings.autoScroll,
  });
  if (response?.error === "NOT_VIDEO") throw new Error("Ouvrez une vidéo YouTube (URL youtube.com/watch?v=…).");
  if (response?.error) throw new Error(`Extraction impossible : ${response.error}`);
  return response;
}

async function fetchJson(url, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), API_TIMEOUT_MS);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body);
      throw new Error(`Erreur API ${res.status} : ${detail}`);
    }
    return body;
  } catch (e) {
    if (e.name === "AbortError") throw new Error(`L'API ne répond pas (délai de ${API_TIMEOUT_MS / 1000} s dépassé).`);
    if (e instanceof TypeError) {
      throw new Error(`API injoignable à ${state.settings.apiUrl}.\nVérifiez qu'elle est lancée et que l'URL (⚙) est correcte.`);
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

async function predict(comments) {
  const batches = [];
  for (let i = 0; i < comments.length; i += API_BATCH_SIZE) batches.push(comments.slice(i, i + API_BATCH_SIZE));

  const start = performance.now();
  const responses = await Promise.all(batches.map((batch) =>
    fetchJson(`${state.settings.apiUrl}/predict_batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments: batch.map((c) => c.text) }),
    })
  ));
  return {
    predictions: responses.flatMap((r) => r.predictions),
    modelVersion: responses[0]?.model_version,
    serverMs: responses.reduce((sum, r) => sum + r.processing_time_ms, 0),
    totalMs: performance.now() - start,
    batches: batches.length,
  };
}

function computeStats(items) {
  const counts = { positive: 0, neutral: 0, negative: 0 };
  let confidence = 0;
  let score = 0;
  for (const it of items) {
    counts[it.sentiment]++;
    confidence += it.confidence;
    score += it.label;
  }
  const total = items.length;
  const sorted = ORDER.slice().sort((a, b) => counts[b] - counts[a]);
  return {
    total,
    counts,
    percentages: Object.fromEntries(ORDER.map((s) => [s, total ? (100 * counts[s]) / total : 0])),
    averageConfidence: total ? confidence / total : 0,
    score: total ? score / total : 0,
    dominant: counts[sorted[0]] === counts[sorted[1]] ? "mixed" : sorted[0],
  };
}

async function analyze() {
  showError("");
  saveSettings();
  try {
    const tab = await getActiveTab();
    if (!tab?.url?.startsWith("https://www.youtube.com/")) {
      throw new Error("Cette extension fonctionne uniquement sur youtube.com.\nOuvrez une vidéo puis relancez l'analyse.");
    }

    setLoading(state.settings.autoScroll ? "Chargement des commentaires (défilement)…" : "Extraction des commentaires…");
    const page = await extractComments(tab);
    if (!page.comments.length) {
      throw new Error("Aucun commentaire trouvé.\nFaites défiler jusqu'aux commentaires (ou activez le défilement auto dans ⚙) puis réessayez.");
    }

    setLoading(`Analyse de ${page.comments.length} commentaires…`);
    const api = await predict(page.comments);

    const items = page.comments.map((c, i) => ({ ...c, ...api.predictions[i], index: i }));
    state.result = {
      videoId: page.videoId,
      title: page.title,
      url: page.url,
      items,
      stats: computeStats(items),
      timing: { totalMs: api.totalMs, serverMs: api.serverMs, batches: api.batches },
      modelVersion: api.modelVersion,
      analyzedAt: new Date().toISOString(),
    };
    chrome.storage.local.set({ lastResult: state.result });
    state.filter = "all";
    renderResults();
  } catch (e) {
    showError(e.message);
  } finally {
    setLoading("");
  }
}

async function testApi() {
  saveSettings();
  $("apiStatus").textContent = "Test en cours…";
  try {
    const start = performance.now();
    const health = await fetchJson(`${state.settings.apiUrl}/health`);
    const ms = Math.round(performance.now() - start);
    $("apiStatus").textContent = `✅ ${health.status} — modèle ${health.model_version} (${ms} ms)`;
  } catch (e) {
    $("apiStatus").textContent = `❌ ${e.message}`;
  }
}

// ---------- Rendu ----------

const color = (sentiment) => getComputedStyle(document.documentElement).getPropertyValue(SENTIMENTS[sentiment].cssVar).trim();
const pct = (x) => `${x.toFixed(1)} %`;

function renderResults() {
  const { title, stats, timing, analyzedAt } = state.result;
  $("results").hidden = false;
  $("videoTitle").textContent = title;
  const when = new Date(analyzedAt).toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "short" });
  $("metaLine").textContent =
    `Analysé le ${when} · API ${Math.round(timing.totalMs)} ms (modèle ${timing.serverMs.toFixed(1)} ms, ${timing.batches} batch${timing.batches > 1 ? "s" : ""})`;

  $("dominant").textContent = stats.dominant === "mixed" ? "Mitigée" : SENTIMENTS[stats.dominant].label;
  $("kpiTotal").textContent = stats.total;
  $("kpiScore").textContent = (stats.score > 0 ? "+" : "") + stats.score.toFixed(2);
  $("kpiConfidence").textContent = `${Math.round(stats.averageConfidence * 100)} %`;

  const legend = $("legend");
  legend.replaceChildren(...ORDER.map((s) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="dot"></span><span class="name"></span><span class="pct"></span>`;
    li.querySelector(".dot").style.background = `var(${SENTIMENTS[s].cssVar})`;
    li.querySelector(".name").textContent = `${SENTIMENTS[s].label} (${stats.counts[s]})`;
    li.querySelector(".pct").textContent = pct(stats.percentages[s]);
    return li;
  }));

  renderDonut(stats);
  renderFilters();
  renderList();
}

function renderDonut(stats) {
  const svg = $("donut");
  const NS = "http://www.w3.org/2000/svg";
  const r = 38;
  const circumference = 2 * Math.PI * r;
  svg.replaceChildren();

  let offset = 0;
  for (const s of ORDER) {
    const fraction = stats.total ? stats.counts[s] / stats.total : 0;
    if (!fraction) continue;
    const circle = document.createElementNS(NS, "circle");
    Object.entries({
      cx: 50, cy: 50, r, fill: "none", stroke: color(s), "stroke-width": 14,
      "stroke-dasharray": `${fraction * circumference} ${circumference}`,
      "stroke-dashoffset": -offset,
      transform: "rotate(-90 50 50)",
    }).forEach(([k, v]) => circle.setAttribute(k, v));
    svg.appendChild(circle);
    offset += fraction * circumference;
  }

  const label = document.createElementNS(NS, "text");
  label.setAttribute("x", 50); label.setAttribute("y", 51);
  label.setAttribute("text-anchor", "middle"); label.setAttribute("class", "donut-label");
  label.textContent = `${Math.round(stats.percentages.positive)}%`;
  const sub = document.createElementNS(NS, "text");
  sub.setAttribute("x", 50); sub.setAttribute("y", 62);
  sub.setAttribute("text-anchor", "middle"); sub.setAttribute("class", "donut-sub");
  sub.textContent = "positifs";
  svg.append(label, sub);
}

function renderFilters() {
  const { counts, total } = state.result.stats;
  const options = [["all", `Tous (${total})`], ...ORDER.map((s) => [s, `${SENTIMENTS[s].label} (${counts[s]})`])];
  $("filters").replaceChildren(...options.map(([value, text]) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.role = "tab";
    b.textContent = text;
    b.setAttribute("aria-selected", String(state.filter === value));
    b.addEventListener("click", () => { state.filter = value; renderFilters(); renderList(); });
    return b;
  }));
}

function visibleItems() {
  const q = state.search.toLowerCase();
  const items = state.result.items.filter((it) =>
    (state.filter === "all" || it.sentiment === state.filter) &&
    (!q || it.text.toLowerCase().includes(q) || (it.author || "").toLowerCase().includes(q)));

  const sorters = {
    default: (a, b) => a.index - b.index,
    confidence: (a, b) => b.confidence - a.confidence,
    negative: (a, b) => b.probabilities.negative - a.probabilities.negative,
    positive: (a, b) => b.probabilities.positive - a.probabilities.positive,
  };
  return items.sort(sorters[state.sort]);
}

function renderList() {
  const items = visibleItems();
  $("listInfo").textContent = `${items.length} commentaire${items.length > 1 ? "s" : ""} affiché${items.length > 1 ? "s" : ""}`;

  const fragment = document.createDocumentFragment();
  for (const it of items) {
    const li = document.createElement("li");
    li.className = "comment";
    li.style.setProperty("--c", `var(${SENTIMENTS[it.sentiment].cssVar})`);
    li.innerHTML = `
      <div class="comment-head">
        <span class="badge"></span>
        <span class="author"></span>
        <span class="reply-tag"></span>
        <span class="conf"><span class="conf-bar"><span></span></span><span class="conf-val"></span></span>
      </div>
      <p class="comment-text"></p>`;
    li.querySelector(".badge").textContent = SENTIMENTS[it.sentiment].label;
    li.querySelector(".author").textContent = it.author || "Anonyme";
    li.querySelector(".reply-tag").textContent = it.isReply ? "réponse" : "";
    li.querySelector(".conf-bar > span").style.width = `${Math.round(it.confidence * 100)}%`;
    li.querySelector(".conf-val").textContent = `${Math.round(it.confidence * 100)} %`;
    const text = li.querySelector(".comment-text");
    text.textContent = it.text;
    text.title = "Cliquer pour déplier";
    text.addEventListener("click", () => text.classList.toggle("expanded"));
    fragment.appendChild(li);
  }
  $("commentList").replaceChildren(fragment);
}

// ---------- Copier / exporter ----------

function copyResults() {
  const { title, url, stats } = state.result;
  const items = visibleItems();
  const lines = [
    `Analyse de sentiment — ${title}`,
    url,
    `Total : ${stats.total} | ` + ORDER.map((s) => `${SENTIMENTS[s].label} : ${pct(stats.percentages[s])}`).join(" | "),
    `Score : ${stats.score.toFixed(2)} | Confiance moyenne : ${Math.round(stats.averageConfidence * 100)} %`,
    `--- ${items.length} commentaires (filtre : ${state.filter === "all" ? "tous" : SENTIMENTS[state.filter].label}) ---`,
    ...items.map((it) => `[${SENTIMENTS[it.sentiment].label} ${Math.round(it.confidence * 100)}%] ${it.author || "Anonyme"} : ${it.text}`),
  ];
  navigator.clipboard.writeText(lines.join("\n"))
    .then(() => toast("Résultats copiés ✓"))
    .catch(() => toast("Copie impossible"));
}

function exportCsv() {
  const escape = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const header = ["author", "text", "sentiment", "label", "confidence", "p_negative", "p_neutral", "p_positive", "is_reply", "likes"];
  const rows = visibleItems().map((it) => [
    it.author, it.text, it.sentiment, it.label, it.confidence,
    it.probabilities.negative, it.probabilities.neutral, it.probabilities.positive, it.isReply, it.likes,
  ].map(escape).join(","));
  const blob = new Blob(["﻿" + [header.join(","), ...rows].join("\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `sentiment_${state.result.videoId || "video"}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
  toast("CSV exporté ✓");
}

// ---------- Initialisation ----------

document.addEventListener("DOMContentLoaded", async () => {
  const lastResult = await loadSettings();

  $("analyzeBtn").addEventListener("click", analyze);
  $("themeBtn").addEventListener("click", toggleTheme);
  $("settingsBtn").addEventListener("click", () => ($("settings").hidden = !$("settings").hidden));
  $("testApiBtn").addEventListener("click", testApi);
  ["apiUrl", "maxComments", "includeReplies", "autoScroll"].forEach((id) => $(id).addEventListener("change", saveSettings));
  $("search").addEventListener("input", (e) => { state.search = e.target.value; renderList(); });
  $("sort").addEventListener("change", (e) => { state.sort = e.target.value; renderList(); });
  $("copyBtn").addEventListener("click", copyResults);
  $("csvBtn").addEventListener("click", exportCsv);

  // Le popup se ferme dès qu'on clique ailleurs : on réaffiche la dernière analyse de la même vidéo.
  const tab = await getActiveTab();
  const videoId = tab?.url ? new URL(tab.url).searchParams.get("v") : null;
  if (lastResult && videoId && lastResult.videoId === videoId) {
    state.result = lastResult;
    renderResults();
  }
});
