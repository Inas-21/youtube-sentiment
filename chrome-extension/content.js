// Injecté à la demande par le popup (chrome.scripting) dans l'onglet YouTube.
// Les commentaires YouTube sont chargés dynamiquement au scroll : on fait défiler la page
// jusqu'à en avoir assez (ou plus aucun nouveau), puis on lit le DOM.
(() => {
  if (window.__ytSentimentInjected) return;
  window.__ytSentimentInjected = true;

  const SEL = {
    section: "ytd-comments#comments",
    thread: "ytd-comment-thread-renderer",
    comment: "ytd-comment-view-model, ytd-comment-renderer",
    text: "#content-text",
    author: "#author-text",
    likes: "#vote-count-middle",
    replies: "#replies",
    title: "h1.ytd-watch-metadata, #title h1",
  };

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const clean = (s) => (s || "").replace(/\s+/g, " ").trim();

  function collect(includeReplies) {
    const results = [];
    const seen = new Set();
    for (const thread of document.querySelectorAll(SEL.thread)) {
      const repliesEl = thread.querySelector(SEL.replies);
      for (const textEl of thread.querySelectorAll(SEL.text)) {
        const isReply = !!repliesEl && repliesEl.contains(textEl);
        if (isReply && !includeReplies) continue;

        const text = clean(textEl.innerText);
        if (!text) continue;
        const box = textEl.closest(SEL.comment) || thread;
        const author = clean(box.querySelector(SEL.author)?.innerText);
        const key = `${author}|${text}`;
        if (seen.has(key)) continue;
        seen.add(key);

        results.push({
          text,
          author,
          likes: clean(box.querySelector(SEL.likes)?.innerText),
          isReply,
        });
      }
    }
    return results;
  }

  async function loadMore(max) {
    const section = document.querySelector(SEL.section);
    if (!section) return;
    const startY = window.scrollY;
    section.scrollIntoView();

    let last = -1;
    let stable = 0;
    for (let i = 0; i < 40 && stable < 3; i++) {
      await sleep(i === 0 ? 1500 : 1000);
      const n = document.querySelectorAll(SEL.thread).length;
      if (n >= max) break;
      stable = n === last ? stable + 1 : 0; // 3 tours sans nouveau commentaire -> fin
      last = n;
      window.scrollTo(0, document.documentElement.scrollHeight);
    }
    window.scrollTo(0, startY);
  }

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg.type !== "EXTRACT_COMMENTS") return;

    (async () => {
      if (location.pathname !== "/watch") {
        sendResponse({ error: "NOT_VIDEO" });
        return;
      }
      if (msg.autoScroll) await loadMore(msg.maxComments);
      const comments = collect(msg.includeReplies).slice(0, msg.maxComments);
      sendResponse({
        videoId: new URLSearchParams(location.search).get("v"),
        title: clean(document.querySelector(SEL.title)?.innerText) || document.title,
        url: location.href,
        comments,
      });
    })().catch((e) => sendResponse({ error: e.message }));

    return true; // réponse asynchrone
  });
})();
