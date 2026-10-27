/* MAFDET app: the shell. API, translations, identity shapes, navigation, toasts,
   polling of the app's state. chat.js and views.js add the views. */
(() => {
  "use strict";

  const M = (window.M = { views: {}, state: {}, dict: {}, fallback: {}, lang: "en" });
  const $ = (M.$ = (id) => document.getElementById(id));

  // ---- server ----------------------------------------------------------------------

  M.api = (path, body) =>
    fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers: { "X-Mafdet-Token": window.MAFDET_TOKEN, "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    }).then((r) => {
      if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
      return r;
    });
  M.get = (path) => M.api(path).then((r) => r.json());
  M.post = (path, body = {}) => M.api(path, body).then((r) => r.json());

  // ---- events: views listen to state changes ---------------------------------------

  const listeners = {};
  M.on = (name, fn) => (listeners[name] = listeners[name] || []).push(fn);
  M.emit = (name, data) => (listeners[name] || []).forEach((fn) => { try { fn(data); } catch (e) { console.error(e); } });

  // ---- translations ------------------------------------------------------------------

  const loadDict = (lang) => fetch(`/static/i18n/${lang}.json`).then((r) => r.json());

  M.t = (key, vars = {}) => {
    const text = M.dict[key] ?? M.fallback[key] ?? key;
    return text.replace(/\{(\w+)\}/g, (_, k) => (vars[k] ?? `{${k}}`));
  };
  const t = M.t;

  M.applyI18n = (root = document) => {
    root.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
    root.querySelectorAll("[data-i18n-placeholder]").forEach((el) => { el.placeholder = t(el.dataset.i18nPlaceholder); });
    root.querySelectorAll("[data-i18n-aria-label]").forEach((el) => el.setAttribute("aria-label", t(el.dataset.i18nAriaLabel)));
  };

  M.setLang = async (lang, save = true) => {
    M.dict = lang === "en" ? M.fallback : await loadDict(lang).catch(() => M.fallback);
    M.lang = lang;
    document.documentElement.lang = lang;
    M.applyI18n();
    renderLang();
    if (save) M.state.settings = await M.post("/api/settings", { language: lang }).catch(() => M.state.settings);
    M.emit("lang", lang);
  };

  // ---- numbers, sizes, durations in the reader's language ----------------------------

  M.num = (n, digits = 0) =>
    new Intl.NumberFormat(M.lang, { maximumFractionDigits: digits, minimumFractionDigits: 0 }).format(n);
  M.bytes = (b) => {
    const gb = b / 2 ** 30;
    return gb >= 1 ? `${M.num(gb, gb < 10 ? 1 : 0)} GB` : `${M.num(b / 2 ** 20)} MB`;
  };
  M.mb = (mb) => M.bytes(mb * 2 ** 20);
  M.dur = (s) => {
    if (s == null) return "–";
    if (s < 60) return t("unit.s", { n: M.num(s, s < 10 ? 1 : 0) });
    if (s < 3600) return t("unit.min", { n: M.num(s / 60, s < 600 ? 1 : 0) });
    return t("unit.h", { n: M.num(s / 3600, 1) });
  };
  M.esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  // Count up to a number: stats arrive with a short roll instead of popping in.
  M.countUp = (el, to, digits = 0, suffix = "") => {
    const from = parseFloat(el.dataset.v || "0") || 0;
    el.dataset.v = to;
    if (matchMedia("(prefers-reduced-motion: reduce)").matches || from === to) {
      el.innerHTML = M.num(to, digits) + suffix;
      return;
    }
    const t0 = performance.now(), d = 700;
    const step = (now) => {
      const k = Math.min(1, (now - t0) / d), e = 1 - (1 - k) ** 3;
      el.innerHTML = M.num(from + (to - from) * e, digits) + suffix;
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };

  // ---- identity: the shapes of mafdet/identity.py, drawn as inline SVG ---------------

  const ID = window.MAFDET_SHAPES;
  M.mark = (name, size, mono = false) => {
    const paths = ID.shapes[name]
      .filter(([, accent, open]) => !(accent && mono && open))
      .map(([d, accent]) => `<path fill-rule="evenodd" d="${d}" fill="${accent && !mono ? "var(--amber)" : "currentColor"}"${accent ? ' class="eye"' : ""}/>`)
      .join("");
    return `<svg class="mark" width="${size}" height="${size}" viewBox="0 0 100 100" aria-hidden="true">${paths}</svg>`;
  };
  M.wordmark = (height) => {
    const [w, h] = ID.wordmark.size;
    const paths = ID.wordmark.letters
      .map(([tx, d], i) => `<g transform="translate(${tx} 0)"><path style="--i:${i}" d="${d}" fill="currentColor"/></g>`).join("");
    return `<svg class="wordmark" width="${(height * w) / h}" height="${height}" viewBox="0 0 ${w} ${h}" role="img" aria-label="MAFDET">${paths}</svg>`;
  };
  M.version = ID.version;

  const GLYPHS = {
    send: "M50 4 L94 48 L68 48 L68 96 L32 96 L32 48 L6 48 Z",
    down: "M50 96 L94 52 L68 52 L68 4 L32 4 L32 52 L6 52 Z",
    stop: "M18 18 L82 18 L82 82 L18 82 Z",
    plus: "M42 8 L58 8 L58 42 L92 42 L92 58 L58 58 L58 92 L42 92 L42 58 L8 58 L8 42 L42 42 Z",
    copy: "M30 4 L96 4 L96 70 L30 70 Z M42 16 L42 58 L84 58 L84 16 Z M4 30 L20 30 L20 82 L72 82 L72 96 L4 96 Z",
    check: "M4 52 L18 38 L38 58 L82 14 L96 28 L38 86 Z",
    x: "M18 6 L50 38 L82 6 L94 18 L62 50 L94 82 L82 94 L50 62 L18 94 L6 82 L38 50 L6 18 Z",
    bolt: "M58 2 L14 58 L44 58 L36 98 L86 38 L54 38 Z",
  };
  M.glyph = (name, size) =>
    `<svg width="${size}" height="${size}" viewBox="0 0 100 100" aria-hidden="true"><path fill-rule="evenodd" fill="currentColor" d="${GLYPHS[name]}"/></svg>`;

  // ---- toasts --------------------------------------------------------------------------

  M.toast = (text, kind = "") => {
    const el = document.createElement("div");
    el.className = `toast ${kind}`;
    el.textContent = text;
    $("toasts").appendChild(el);
    setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 400); }, 4200);
  };

  // ---- navigation ----------------------------------------------------------------------

  M.view = "chat";
  M.go = (view) => {
    M.view = view;
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.dataset.view === view));
    const items = [...document.querySelectorAll(".nav-item")];
    items.forEach((b) => { b.classList.toggle("active", b.dataset.view === view); b.setAttribute("aria-selected", b.dataset.view === view); });
    const on = items.find((b) => b.dataset.view === view);
    if (on) $("nav-glider").style.transform = `translateY(${on.offsetTop}px)`;
    M.emit("view", view);
  };

  function renderRail() {
    $("rail-logo").innerHTML = M.mark("mask", 30) + M.wordmark(12);
    document.querySelectorAll(".nav-icon").forEach((el) => { el.innerHTML = M.mark(el.dataset.icon, 18, true); });
    document.querySelectorAll(".nav-item").forEach((b) => b.addEventListener("click", () => M.go(b.dataset.view)));
    $("model-pill").addEventListener("click", () => M.go("models"));
  }

  function renderLang() {
    $("lang").innerHTML = ["en", "it", "fr", "de"]
      .map((l) => `<button type="button" role="radio" aria-checked="${l === M.lang}" class="${l === M.lang ? "on" : ""}" data-lang="${l}">${l}</button>`).join("");
    $("lang").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => M.setLang(b.dataset.lang)));
  }

  // The model in the rail: a diamond that says if it is ready, and its name.
  M.modelName = (id) => {
    const m = (M.state.catalog?.models || []).find((x) => x.id === id);
    return m ? `${m.name} · ${m.quant}` : id || "";
  };
  function renderPill(s) {
    const dot = $("model-dot"), text = $("model-pill-text");
    dot.className = "dot " + (s.state === "ready" ? "ready" : s.state === "loading" ? "loading" : s.state === "error" ? "error" : "");
    text.textContent = s.state === "ready" ? (s.model || M.modelName(s.model_id))
      : s.state === "loading" ? t("top.loading", { model: M.modelName(s.model_id) || "…" })
      : s.state === "error" ? t("top.error") : t("top.no_model");
  }

  // ---- state: polled from the server ----------------------------------------------------

  M.refresh = {
    catalog: () => M.get("/api/catalog").then((c) => { M.state.catalog = c; M.emit("catalog", c); }).catch(() => {}),
    hardware: (fresh) => (fresh ? M.post("/api/hardware/refresh") : M.get("/api/hardware"))
      .then((h) => { M.state.hardware = h; M.emit("hardware", h); }).catch(() => {}),
    runtime: () => M.get("/api/runtime").then((r) => { M.state.runtime = r; M.emit("runtime", r); }).catch(() => {}),
  };

  let lastState = "";
  M.setStatus = (s) => {
    const prev = M.state.status;
    M.state.status = s;
    renderPill(s);
    if (s.state !== lastState) {
      if (s.state === "ready" && lastState === "loading") M.toast(t("models.toast.loaded", { model: s.model || M.modelName(s.model_id) }), "ok");
      lastState = s.state;
      M.refresh.catalog();
    }
    M.emit("status", { s, prev });
  };

  async function pollStatus() {
    if (!document.hidden) {
      try { M.setStatus(await M.get("/api/status")); } catch (e) { /* server busy or gone */ }
    }
    const s = M.state.status || {};
    const fast = s.state === "loading" || (s.state === "ready" && M.view === "chat");
    setTimeout(pollStatus, fast ? 200 : 900);
  }

  // Background jobs: progress for the cards, a toast when one ends.
  const seen = {};
  M.pendingUse = new Set();
  async function pollTasks() {
    let tasks = [];
    try { tasks = await M.get("/api/tasks"); } catch (e) { /* next round */ }
    M.state.tasks = tasks;
    for (const task of tasks) {
      const before = seen[task.id];
      seen[task.id] = task.state;
      if (before === "running" && task.state !== "running") taskEnded(task);
    }
    M.emit("tasks", tasks);
    setTimeout(pollTasks, tasks.some((x) => x.state === "running") ? 350 : 1500);
  }

  function taskEnded(task) {
    // A download from the Hugging Face search has "repo:file" as target, and the model's id in its result.
    const id = task.result?.id || task.target;
    const name = task.result?.id ? M.modelName(id) : M.modelName(task.target).split("/").pop();
    if (task.state === "error") M.toast(t("models.toast.failed", { model: name || task.kind, error: task.error }), "error");
    else if (task.state === "done") {
      if (task.kind === "download") M.toast(t("models.toast.downloaded", { model: name }), "ok");
      if (task.kind === "bench") M.toast(t("models.toast.measured", { model: name, n: M.num(task.result.decode_tps, 1) }), "ok");
      if (task.kind === "llama") M.toast(t("hardware.toast.installed", { build: task.result.build }), "ok");
    }
    // "Download and use": the model loads as soon as its file is complete.
    if (task.kind === "download" && task.state === "done" && M.pendingUse.has(task.target)) {
      M.pendingUse.delete(task.target);
      M.post("/api/model/load", { model: id }).then(M.setStatus).catch(() => {});
    }
    M.refresh.catalog();
    if (task.kind === "llama") M.refresh.runtime();
    M.emit("task-ended", task);
  }
  M.taskFor = (kind, target) =>
    (M.state.tasks || []).filter((x) => x.kind === kind && (target === undefined || x.target === target)).pop();
  M.startTask = async (path, body) => {
    const task = await M.post(path, body);
    seen[task.id] = "running";
    M.state.tasks = [...(M.state.tasks || []).filter((x) => x.id !== task.id), task];
    M.emit("tasks", M.state.tasks);
    return task;
  };

  // ---- splash --------------------------------------------------------------------------

  function splash() {
    $("splash-mark").innerHTML = M.mark("mask", 120);
    $("splash-word").innerHTML = M.wordmark(30);
  }
  function hideSplash() {
    $("splash").classList.add("gone");
    setTimeout(() => $("splash").remove(), 700);
  }

  // ---- start ---------------------------------------------------------------------------

  async function boot() {
    splash();
    const t0 = performance.now();
    M.fallback = await loadDict("en");
    M.dict = M.fallback;
    M.state.settings = await M.get("/api/settings").catch(() => ({ language: "en" }));
    renderRail();
    await M.setLang(M.state.settings.language || "en", false);
    await Promise.all([M.refresh.catalog(), M.refresh.hardware(), M.refresh.runtime(),
      M.get("/api/status").then(M.setStatus).catch(() => {})]);
    M.emit("ready");
    M.go("chat");
    // The splash stays long enough for its animation to be seen, never longer.
    setTimeout(() => {
      hideSplash();
      if (!M.state.settings.first_run_done && M.views.onboarding) M.views.onboarding.open();
    }, Math.max(0, 1300 - (performance.now() - t0)));
    pollStatus();
    pollTasks();
    window.addEventListener("resize", () => M.go(M.view));
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
