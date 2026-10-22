/* MAFDET app: the chat view. Every change to the draft goes to the server, which keeps
   the model's cache following it; the status line shows what the model is doing with it. */
(() => {
  "use strict";

  const { $, t } = M;
  const scroller = $("scroll"), thread = $("thread"), input = $("input"), sendBtn = $("send");

  // ---- safe Markdown -------------------------------------------------------------
  // Everything is escaped first: the model's text can never become markup or script.

  const esc = M.esc;

  // Simple TeX, as the models write it ($3 \times 4 = 12$, \frac{a}{b}, \boxed{11}), shown as plain
  // symbols. A pair of $ is math only if it holds a command or an operator: "$5 and $10" stays text.
  const TEX = { times: "×", cdot: "·", div: "÷", pm: "±", leq: "≤", le: "≤", geq: "≥", ge: "≥", neq: "≠", ne: "≠",
    approx: "≈", infty: "∞", to: "→", rightarrow: "→", Rightarrow: "⇒", implies: "⇒", pi: "π", alpha: "α",
    beta: "β", theta: "θ", lambda: "λ", mu: "μ", sigma: "σ", Delta: "Δ", sum: "Σ", quad: "  ", qquad: "    ",
    ldots: "…", dots: "…", circ: "°" };
  const SUP = { 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹", n: "ⁿ", "-": "⁻" };
  const BOX_OPEN = "\u0001", BOX_CLOSE = "\u0002";
  function tex(src) {
    let s = src;
    for (let k = 0; k < 4; k++) {                          // nested \frac{\sqrt{2}}{2}: a few passes
      s = s.replace(/\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}/g, (_, a, b) => `${a.length > 1 ? `(${a})` : a}⁄${b.length > 1 ? `(${b})` : b}`)
        .replace(/\\sqrt\{([^{}]*)\}/g, "√($1)")
        .replace(/\\(?:text|mathrm|mathbf|operatorname)\{([^{}]*)\}/g, "$1")
        .replace(/\\boxed\{([^{}]*)\}/g, `${BOX_OPEN}$1${BOX_CLOSE}`);
    }
    return s.replace(/\^\{?([0-9n-]+)\}?/g, (_, e) => [...e].map((c) => SUP[c] ?? c).join(""))
      .replace(/\\left|\\right|\\,|\\;|\\!/g, "")
      .replace(/\\([a-zA-Z]+)/g, (m, w) => TEX[w] ?? m)
      .replace(/\\%/g, "%").replace(/[{}]/g, "");
  }
  const MATH = /\$\$([^$]+)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$([^$\n]+?)\$(?!\d)/g;
  const looksMath = (s) => /\\[a-zA-Z]|[=+*/^<>]|\d\s*-\s*\d/.test(s);
  function math(text) {
    return text
      .replace(MATH, (all, a, b, c, d) => {
        const body = a ?? b ?? c ?? d;
        return d !== undefined && !looksMath(d) ? all : `<span class="math">${tex(body).trim()}</span>`;
      })
      .replace(/\\boxed\{([^{}]*)\}/g, `${BOX_OPEN}$1${BOX_CLOSE}`)
      .replace(new RegExp(`${BOX_OPEN}([^${BOX_CLOSE}]*)${BOX_CLOSE}`, "g"), "<strong>$1</strong>");
  }

  function inline(src) {
    return src.split(/(`[^`\n]+`)/g).map((part, i) => {
      if (i % 2) return `<code>${esc(part.slice(1, -1))}</code>`;
      return math(esc(part))
        .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2">$1</a>')
        .replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, "<strong>$1</strong>")
        .replace(/__(?=\S)([\s\S]*?\S)__/g, "<strong>$1</strong>")
        .replace(/(^|[^*\w])\*(?=\S)([^*\n]*?\S)\*(?![*\w])/g, "$1<em>$2</em>")
        .replace(/(^|[^_\w])_(?=\S)([^_\n]*?\S)_(?![_\w])/g, "$1<em>$2</em>")
        .replace(/~~(?=\S)([\s\S]*?\S)~~/g, "<del>$1</del>");
    }).join("");
  }

  const LIST_ITEM = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/;
  const indent = (l) => l.match(/^\s*/)[0].length;
  const startsBlock = (l) =>
    /^\s*(#{1,6}\s|```|~~~|>|\$\$|\\\[|([-*+]|\d{1,9}[.)])\s|(---|\*\*\*|___)\s*$)/.test(l) || /^\s*\|.*\|\s*$/.test(l);

  function codeBlock(code, lang) {
    return `<div class="code"><div class="code-head"><span class="label">${esc(lang || "code")}</span>` +
      `<button class="copy label" type="button">${M.glyph("copy", 11)}<span>${t("action.copy")}</span></button></div>` +
      `<pre><code>${esc(code)}</code></pre></div>`;
  }

  function list(lines, i) {
    const first = lines[i].match(LIST_ITEM);
    const base = first[1].length, ordered = /\d/.test(first[2]);
    const items = [];
    let content = 0;
    while (i < lines.length) {
      const l = lines[i], m = l.match(LIST_ITEM);
      if (m && m[1].length === base && /\d/.test(m[2]) === ordered) {
        items.push([m[3]]);
        content = base + m[2].length + 1;
      } else if (!l.trim()) {
        const next = lines[i + 1];
        if (next === undefined || !next.trim() || (indent(next) <= base && !LIST_ITEM.test(next))) break;
        items[items.length - 1].push("");
      } else if (indent(l) > base) {
        items[items.length - 1].push(l.slice(Math.min(indent(l), content)));
      } else if (!startsBlock(l)) {
        items[items.length - 1].push(l.trim());           // lazy continuation
      } else break;
      i++;
    }
    const tag = ordered ? "ol" : "ul";
    const start = ordered && parseInt(first[2], 10) !== 1 ? ` start="${parseInt(first[2], 10)}"` : "";
    const body = items.map((it) => `<li>${markdown(it.join("\n")).replace(/^<p>([\s\S]*?)<\/p>/, "$1")}</li>`).join("");
    return [`<${tag}${start}>${body}</${tag}>`, i];
  }

  function table(lines, i) {
    const cells = (l) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
    const head = cells(lines[i]);
    const align = cells(lines[i + 1]).map((c) => (/^:-+:$/.test(c) ? "center" : /-:$/.test(c) ? "right" : ""));
    const td = (tag, c, k) => `<${tag}${align[k] ? ` style="text-align:${align[k]}"` : ""}>${inline(c)}</${tag}>`;
    let html = `<div class="table"><table><thead><tr>${head.map((c, k) => td("th", c, k)).join("")}</tr></thead><tbody>`;
    i += 2;
    while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
      html += `<tr>${cells(lines[i++]).map((c, k) => td("td", c, k)).join("")}</tr>`;
    }
    return [html + "</tbody></table></div>", i];
  }

  function markdown(src) {
    const lines = src.replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let i = 0, m;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i++; continue; }
      if ((m = line.match(/^\s*(```+|~~~+)\s*([^\s`]*)/))) {
        const fence = m[1], body = [];
        i++;
        while (i < lines.length && !lines[i].trim().startsWith(fence)) body.push(lines[i++]);
        i++;                                               // closing fence (absent while streaming)
        out.push(codeBlock(body.join("\n"), m[2]));
      } else if (/^\s*(\$\$|\\\[)/.test(line)) {
        // Display math over several lines: $$ on its own line, the formula, $$ (or \[ ... \]).
        const open = line.trim().startsWith("$$") ? "$$" : "\\[", close = open === "$$" ? "$$" : "\\]";
        const rest = line.trim().slice(open.length), body = [];
        i++;
        if (rest.includes(close)) body.push(rest.slice(0, rest.indexOf(close)));
        else {
          if (rest.trim()) body.push(rest);
          while (i < lines.length && !lines[i].includes(close)) body.push(lines[i++]);
          if (i < lines.length) body.push(lines[i].slice(0, lines[i].indexOf(close)));
          i++;                                             // the closing line (absent while streaming)
        }
        out.push(`<div class="math-block">${math(`\\[${esc(body.join(" "))}\\]`)}</div>`);
      } else if ((m = line.match(/^\s*(#{1,6})\s+(.*?)\s*#*\s*$/))) {
        out.push(`<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`);
        i++;
      } else if (/^\s*(---|\*\*\*|___)\s*$/.test(line)) {
        out.push("<hr>");
        i++;
      } else if (/^\s*>/.test(line)) {
        const body = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) body.push(lines[i++].replace(/^\s*>\s?/, ""));
        out.push(`<blockquote>${markdown(body.join("\n"))}</blockquote>`);
      } else if (/^\s*\|.*\|\s*$/.test(line) && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1] || "")) {
        [m, i] = table(lines, i);
        out.push(m);
      } else if (LIST_ITEM.test(line)) {
        [m, i] = list(lines, i);
        out.push(m);
      } else {
        const para = [line.trim()];
        i++;
        while (i < lines.length && lines[i].trim() && !startsBlock(lines[i])) para.push(lines[i++].trim());
        out.push(`<p>${para.map(inline).join("<br>")}</p>`);
      }
    }
    return out.join("");
  }

  // ---- what the thread shows when there is no conversation -----------------------------

  let shown = "";                 // which empty state is on screen, so polling does not redraw it

  function empty(kind, html) {
    if (shown === kind) return;
    shown = kind;
    thread.innerHTML = html;
    M.applyI18n(thread);
    thread.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", () => M.go(b.dataset.go)));
    thread.querySelector("[data-install]")?.addEventListener("click", installRuntime);
    thread.querySelector("[data-retry]")?.addEventListener("click", () => M.post("/api/model/load", { model: M.state.status.model_id || "" }));
    thread.querySelector("[data-get]")?.addEventListener("click", (e) => getAndUse(e.currentTarget.dataset.get));
  }

  async function installRuntime() {
    await M.startTask("/api/runtime/install");
    shown = "";
    render();
  }

  async function getAndUse(id) {
    M.pendingUse.add(id);
    await M.startTask("/api/download", { model: id });
    shown = "";
    render();
  }

  const hasConversation = () => !!thread.querySelector(".msg");

  function hero() {
    const k = (key, label) => `<b>${key}</b> ${t(label)}`;
    empty("hero",
      `<section class="hero">${M.mark("mask", 112)}${M.wordmark(26)}` +
      `<p class="hero-title">${t("app.tagline")}</p>` +
      `<p class="hero-body">${t("chat.hero.body")}</p>` +
      `<p class="hero-keys label">${[k("Enter", "chat.key.send"), k("Shift+Enter", "chat.key.newline"), k("Esc", "chat.key.stop"), k("Ctrl+N", "chat.key.new")].join(" · ")}</p></section>`);
  }

  function progressBlock(task) {
    if (!task) return "";
    const p = Math.round(100 * (task.progress || 0));
    const text = task.kind === "download" && task.total_bytes
      ? t("models.downloading", { done: M.bytes(task.done_bytes), total: M.bytes(task.total_bytes), speed: M.bytes(task.speed_bps), eta: M.dur(task.eta_s) })
      : t("hardware.installing", { stage: t(`hardware.stage.${task.stage || "release"}`), p });
    return `<div class="progress" style="max-width:420px"><div class="progress-bar${p ? "" : " indeterminate"}"><i style="width:${p}%"></i></div>` +
      `<div class="progress-text"><span>${esc(text)}</span><span>${p}%</span></div></div>`;
  }

  // The bar is updated in place: redrawing the whole state would replay its entrance.
  function progress(task) {
    const slot = thread.querySelector(".prog-slot");
    if (slot && task) slot.innerHTML = progressBlock(task);
  }

  function render() {
    const s = M.state.status || {};
    if (hasConversation() && s.state === "ready") return;
    if (s.state === "ready") return hero();
    if (s.state === "no-runtime") {
      const task = M.taskFor("llama");
      const running = task && task.state === "running";
      empty(running ? "rt-running" : "no-runtime",
        `<div class="empty">${M.mark("awaken", 72)}<h2>${t("state.no_runtime.title")}</h2><p>${t("state.no_runtime.body")}</p>` +
        (running ? `<div class="prog-slot"></div>` : `<button class="btn accent" data-install>${t("action.install_runtime")}</button>`) + `</div>`);
      return progress(task);
    }
    if (s.state === "no-model") {
      const id = s.model_id;
      const task = id && M.taskFor("download", id);
      if (task && task.state === "running") {
        empty(`dl-${id}`, `<div class="empty">${M.mark("race", 72)}<h2>${esc(M.modelName(id))}</h2><div class="prog-slot"></div></div>`);
        return progress(task);
      }
      const missing = id && !(M.state.catalog?.models || []).find((m) => m.id === id)?.installed;
      return empty(`no-model-${id}-${missing}`,
        `<div class="empty">${M.mark("race", 72)}<h2>${t("state.no_model.title")}</h2>` +
        `<p>${missing ? t("state.missing.body", { model: esc(M.modelName(id)) }) : t("state.no_model.body")}</p>` +
        `<div class="hero-cta">${missing ? `<button class="btn accent" data-get="${esc(id)}">${t("action.download_and_use")}</button>` : ""}` +
        `<button class="btn${missing ? "" : " accent"}" data-go="models">${t("action.open_models")}</button></div></div>`);
    }
    if (s.state === "loading") {
      return empty(`loading-${s.model_id}`,
        `<div class="empty">${M.mark("mask", 88).replace('class="mark"', 'class="mark breathe"')}<h2>${t("state.loading.title")}</h2>` +
        `<p>${t("state.loading.body", { model: esc(M.modelName(s.model_id) || "…") })}</p><div class="loader"><i></i></div></div>`);
    }
    if (s.state === "error") {
      return empty(`error-${s.error}`,
        `<div class="empty">${M.mark("judge", 72)}<h2>${t("state.error.title")}</h2><p>${t(`error.${s.error_code || "load_failed"}`)}</p>` +
        `<p class="error-detail">${esc(s.error || "")}</p>` +
        `<div class="hero-cta"><button class="btn accent" data-retry>${t("action.retry")}</button><button class="btn" data-go="models">${t("action.open_models")}</button></div></div>`);
    }
  }

  // ---- conversation ----------------------------------------------------------------

  function addMessage(role) {
    if (!hasConversation()) { thread.innerHTML = ""; shown = ""; }
    const el = document.createElement("article");
    el.className = `msg ${role}`;
    el.innerHTML = role === "user"
      ? `<div class="msg-label label">${t("chat.you")}</div><div class="msg-body"></div>`
      : `<div class="msg-label label">${M.mark("mask", 16)}MAFDET</div>` +
        `<details class="think" hidden><summary class="label"></summary><div class="think-body"></div></details>` +
        `<div class="msg-body md"></div><div class="msg-meta label" hidden></div>`;
    thread.appendChild(el);
    return el;
  }

  const nearBottom = () => scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 64;
  const toBottom = () => { scroller.scrollTop = scroller.scrollHeight; };

  function renderAnswer(el, text, writing) {
    // Follow the answer only if the reader is already at the bottom: scrolling up is never undone.
    const follow = nearBottom();
    const body = el.querySelector(".msg-body");
    body.innerHTML = markdown(text);
    if (writing) {
      const spots = body.querySelectorAll("p, li, h1, h2, h3, h4, h5, h6, pre code, td, blockquote");
      const caret = document.createElement("span");
      caret.className = "caret";
      (spots[spots.length - 1] || body).appendChild(caret);
    }
    if (follow) toBottom();
    updateJump();
  }

  const fmt = (n) => M.num(Math.round(n));

  function renderThink(el, text, thinking, d) {
    const box = el.querySelector(".think");
    if (!text) return;
    const follow = nearBottom();
    box.hidden = false;
    box.querySelector(".think-body").textContent = text;
    const summary = box.querySelector("summary");
    summary.classList.toggle("pulse", thinking);
    summary.textContent = thinking ? t("chat.thinking")
      : d && d.think_tokens ? t("chat.reasoning_stats", { total: fmt(d.think_tokens), ahead: fmt(d.think_ahead) }) : t("chat.reasoning");
    if (follow) toBottom();
  }

  function metaLine(el, d) {
    const meta = el.querySelector(".msg-meta");
    const parts = [`<span class="diamond">◆</span>` + (d.think_tokens
      ? t("chat.meta.first_word", { ms: fmt(d.answer_ms) }) : t("chat.meta.first_token", { ms: fmt(d.ttft_ms) }))];
    if (d.message_tokens) parts.push(t("chat.meta.read", { read: fmt(d.read), total: fmt(d.message_tokens) }));
    if (d.think_tokens) parts.push(t("chat.meta.think_ahead", { ahead: fmt(d.think_ahead), total: fmt(d.think_tokens) }));
    if (d.ahead) parts.push(t("chat.meta.answer_ahead", { n: fmt(d.ahead) }));
    parts.push(t("chat.meta.speed", { n: fmt(d.tokens), tps: M.num(d.tps, 1) }));
    if (d.stopped) parts.push(t("chat.meta.stopped"));
    if (d.cut) parts.push(t("chat.meta.cut"));
    meta.innerHTML = parts.join('<span class="sep">/</span>');
    meta.hidden = false;
  }

  function errorLine(el, message) {
    const meta = el.querySelector(".msg-meta");
    meta.innerHTML = `<span class="msg-error">${esc(message)}</span>`;
    meta.hidden = false;
  }

  // ---- draft: every change goes to the server, which decides when to read ----------

  let draftBusy = false, draftAgain = false;
  async function pushDraft() {
    if (draftBusy) { draftAgain = true; return; }
    draftBusy = true;
    do {
      draftAgain = false;
      try { M.setStatus(await M.post("/api/draft", { text: input.value })); } catch (e) { /* next poll */ }
    } while (draftAgain);
    draftBusy = false;
  }

  function autosize() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, window.innerHeight * 0.4)}px`;
  }

  // ---- send -------------------------------------------------------------------------

  let generating = false;
  const ready = () => (M.state.status || {}).state === "ready";

  async function send() {
    const text = input.value;
    if (!text.trim() || generating || !ready()) return;
    generating = true;
    setSendMode();
    addMessage("user").querySelector(".msg-body").textContent = text;
    input.value = "";
    autosize();
    pushDraft();
    const bot = addMessage("bot");
    toBottom();
    let answer = "", reasoning = "", frame = 0;
    const paint = () => {
      frame = 0;
      renderThink(bot, reasoning, !answer);
      renderAnswer(bot, answer, true);
    };
    renderAnswer(bot, "", true);
    try {
      const reader = (await M.api("/api/send", { text })).body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let cut;
        while ((cut = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, cut);
          buf = buf.slice(cut + 2);
          if (!chunk.startsWith("data: ")) continue;
          const ev = JSON.parse(chunk.slice(6));
          if (ev.type === "token" || ev.type === "think") {
            if (ev.type === "token") answer += ev.text; else reasoning += ev.text;
            if (!frame) frame = requestAnimationFrame(paint);
          } else if (ev.type === "done") {
            renderThink(bot, reasoning, false, ev);
            metaLine(bot, ev);
          } else if (ev.type === "error") {
            errorLine(bot, ev.code ? t(`error.${ev.code}`) : ev.message);
          }
        }
      }
    } catch (e) {
      errorLine(bot, t("chat.connection_lost", { error: e.message }));
    } finally {
      if (frame) cancelAnimationFrame(frame);
      bot.querySelector(".think summary")?.classList.remove("pulse");
      renderAnswer(bot, answer, false);
      generating = false;
      setSendMode();
      input.focus();
    }
  }

  function setSendMode() {
    sendBtn.classList.toggle("stop", generating);
    sendBtn.innerHTML = M.glyph(generating ? "stop" : "send", generating ? 14 : 16);
    sendBtn.disabled = !generating && (!ready() || !input.value.trim());
    input.disabled = !ready() && !generating;
  }

  const stop = () => { if (generating) M.post("/api/stop").catch(() => {}); };

  async function newChat() {
    stop();
    while (generating) await new Promise((r) => setTimeout(r, 30));
    await M.post("/api/new").catch(() => {});
    thread.innerHTML = "";
    shown = "";
    render();
    input.focus();
    pushDraft();
  }

  // ---- status line ---------------------------------------------------------------------

  const statusIcon = $("status-icon"), statusText = $("status-text"), meter = $("meter"), meterRead = $("meter-read");
  let lastIcon = "";

  function setIcon(name, on) {
    if (name !== lastIcon) { statusIcon.innerHTML = M.mark(name, 14, true); lastIcon = name; }
    statusIcon.classList.toggle("on", on);
  }

  const diamond = (text) => `<span class="diamond">◆</span> ${esc(text)}`;

  function showStatus(s) {
    statusText.classList.toggle("error", s.state === "error");
    statusText.classList.remove("pulse");
    meter.hidden = true;
    $("field").classList.remove("ahead");
    $("topbar-model").textContent = s.state === "ready" ? s.model : "";
    $("status-side").textContent = s.state === "ready" ? t("status.context", { n: fmt(s.context), max: fmt(s.ctx) }) : "";
    if (s.state === "loading") {
      setIcon("awaken", false);
      statusText.textContent = t("status.loading");
      statusText.classList.add("pulse");
    } else if (s.state === "error") {
      setIcon("judge", false);
      statusText.textContent = t(`error.${s.error_code || "load_failed"}`);
    } else if (s.state === "no-runtime") {
      setIcon("awaken", false);
      statusText.textContent = t("status.no_runtime");
    } else if (s.state !== "ready") {
      setIcon("race", false);
      statusText.textContent = t("status.no_model");
    } else if (generating || s.generating) {
      setIcon("strike", true);
      statusText.textContent = t("status.answering");
    } else if (!s.has_text) {
      setIcon("watch", false);
      statusText.textContent = t("status.watching");
    } else {
      const done = !s.busy && s.pending === 0;
      setIcon("watch", true);
      meter.hidden = false;
      meter.classList.toggle("ready", done);
      meterRead.style.width = `${s.message_tokens ? (100 * s.read) / s.message_tokens : 0}%`;
      const read = t("status.read", { read: fmt(s.read), total: fmt(s.message_tokens) });
      const thought = s.thinking_ahead ? t("status.thought", { n: fmt(s.thinking_ahead) }) : "";
      // "Ready" only for what is really ready: the answer, or (no reasoning) the prompt read.
      if (s.ahead_writing) {
        setIcon("strike", true);
        $("field").classList.add("ahead");
        statusText.innerHTML = diamond(s.gatherer ? t("status.gathering", { n: fmt(s.thinking_ahead) })
          : s.think && !s.ahead ? t("status.thinking_ahead", { n: fmt(s.thinking_ahead) })
          : t("status.answering_ahead", { n: fmt(s.ahead) }));
        statusText.classList.add("pulse");
      } else if (s.busy) {
        statusText.textContent = t("status.reading", { read, thought });
        statusText.classList.add("pulse");
      } else if (s.answer_ready) {
        setIcon("strike", true);
        statusText.innerHTML = diamond(t("status.answer_ready"));
      } else if (s.ahead) {
        setIcon("strike", true);
        statusText.innerHTML = diamond(t("status.answer_started", { n: fmt(s.ahead) }));
      } else if (s.think && s.thinking_ahead) {
        statusText.textContent = t("status.paused", { n: fmt(s.thinking_ahead) });
      } else if (done) {
        statusText.innerHTML = diamond(t("status.ready", { read, left: fmt(s.left), thought }));
      } else {
        statusText.textContent = t("status.waiting", { read, thought });
      }
    }
    setSendMode();
  }

  // ---- events --------------------------------------------------------------------------

  let modelSeen = null;
  M.on("status", ({ s }) => {
    // A different model is a different session: its conversation starts over.
    if (s.state === "ready" && modelSeen !== null && s.model_id !== modelSeen) { thread.innerHTML = ""; shown = ""; }
    if (s.state === "ready") modelSeen = s.model_id;
    showStatus(s);
    render();
  });
  M.on("tasks", () => { if (!hasConversation()) render(); });
  M.on("lang", () => {
    shown = "";
    $("new-chat").innerHTML = M.glyph("plus", 10) + t("action.new_chat");
    $("jump").innerHTML = M.glyph("down", 10) + `<span class="label">${t("action.latest")}</span>`;
    if (M.state.status) showStatus(M.state.status);
    render();
  });
  M.on("view", (v) => { if (v === "chat" && ready()) setTimeout(() => input.focus(), 50); });

  input.addEventListener("input", () => { autosize(); pushDraft(); setSendMode(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
  });
  sendBtn.addEventListener("click", () => (generating ? stop() : send()));
  $("new-chat").addEventListener("click", newChat);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") stop();
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "n") { e.preventDefault(); M.go("chat"); newChat(); }
  });

  function updateJump() { $("jump").hidden = nearBottom(); }
  scroller.addEventListener("scroll", updateJump, { passive: true });
  $("jump").addEventListener("click", () => scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" }));

  thread.addEventListener("click", (e) => {
    const link = e.target.closest("a[href]");
    if (link) {                                            // open links in the browser, not in this window
      e.preventDefault();
      M.post("/api/open", { url: link.getAttribute("href") }).catch(() => {});
      return;
    }
    const copy = e.target.closest(".copy");
    if (copy) {
      const code = copy.closest(".code").querySelector("pre code").textContent;
      navigator.clipboard.writeText(code).then(() => {
        copy.innerHTML = `${M.glyph("check", 11)}<span>${t("action.copied")}</span>`;
        setTimeout(() => { copy.innerHTML = `${M.glyph("copy", 11)}<span>${t("action.copy")}</span>`; }, 1500);
      }).catch(() => {});
    }
  });

  M.views.chat = { render };
  autosize();
})();
