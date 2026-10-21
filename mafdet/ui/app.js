/* MAFDET chat window: talks to the local server started by `mafdet watch`. */
(() => {
  "use strict";

  const TOKEN = window.MAFDET_TOKEN;
  const ID = window.MAFDET_SHAPES;
  const $ = (id) => document.getElementById(id);
  const scroller = $("scroll"), thread = $("thread"), input = $("input"), sendBtn = $("send");

  // ---- identity: the shapes of mafdet/identity.py, drawn as inline SVG --------

  function mark(name, size, mono = false) {
    const paths = ID.shapes[name]
      .filter(([, accent, open]) => !(accent && mono && open))
      .map(([d, accent]) => `<path fill-rule="evenodd" d="${d}" fill="${accent && !mono ? "var(--amber)" : "currentColor"}"/>`)
      .join("");
    return `<svg class="mark" width="${size}" height="${size}" viewBox="0 0 100 100" aria-hidden="true">${paths}</svg>`;
  }

  function wordmark(height) {
    const [w, h] = ID.wordmark.size;
    const paths = ID.wordmark.letters
      .map(([tx, d]) => `<path transform="translate(${tx} 0)" d="${d}" fill="currentColor"/>`).join("");
    return `<svg class="wordmark" width="${(height * w) / h}" height="${height}" viewBox="0 0 ${w} ${h}" role="img" aria-label="MAFDET">${paths}</svg>`;
  }

  // Interface glyphs, same grammar: one solid mass from straight segments.
  const GLYPHS = {
    send: "M50 4 L94 48 L68 48 L68 96 L32 96 L32 48 L6 48 Z",
    down: "M50 96 L94 52 L68 52 L68 4 L32 4 L32 52 L6 52 Z",
    stop: "M18 18 L82 18 L82 82 L18 82 Z",
    plus: "M42 8 L58 8 L58 42 L92 42 L92 58 L58 58 L58 92 L42 92 L42 58 L8 58 L8 42 L42 42 Z",
    copy: "M30 4 L96 4 L96 70 L30 70 Z M42 16 L42 58 L84 58 L84 16 Z M4 30 L20 30 L20 82 L72 82 L72 96 L4 96 Z",
    check: "M4 52 L18 38 L38 58 L82 14 L96 28 L38 86 Z",
  };
  const glyph = (name, size) =>
    `<svg width="${size}" height="${size}" viewBox="0 0 100 100" aria-hidden="true"><path fill-rule="evenodd" fill="currentColor" d="${GLYPHS[name]}"/></svg>`;

  $("lockup").innerHTML = mark("mask", 30) + wordmark(12);
  $("new-chat").innerHTML = glyph("plus", 10) + "New chat";
  $("jump").innerHTML = glyph("down", 10) + '<span class="label">Latest</span>';
  sendBtn.innerHTML = glyph("send", 16);

  // ---- server ----------------------------------------------------------------

  function api(path, body) {
    return fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers: { "X-Mafdet-Token": TOKEN, "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    }).then((r) => {
      if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
      return r;
    });
  }

  // ---- safe Markdown -------------------------------------------------------------
  // Everything is escaped first: the model's text can never become markup or script.

  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  function inline(src) {
    return src.split(/(`[^`\n]+`)/g).map((part, i) => {
      if (i % 2) return `<code>${esc(part.slice(1, -1))}</code>`;
      return esc(part)
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
    /^\s*(#{1,6}\s|```|~~~|>|([-*+]|\d{1,9}[.)])\s|(---|\*\*\*|___)\s*$)/.test(l) || /^\s*\|.*\|\s*$/.test(l);

  function codeBlock(code, lang) {
    return `<div class="code"><div class="code-head"><span class="label">${esc(lang || "code")}</span>` +
      `<button class="copy label" type="button">${glyph("copy", 11)}<span>Copy</span></button></div>` +
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

  // ---- conversation ----------------------------------------------------------------

  function hero() {
    thread.innerHTML =
      `<section class="hero">${mark("mask", 112)}${wordmark(26)}` +
      `<p class="hero-title">You type. It already knows.</p>` +
      `<p class="hero-body">MAFDET reads your message while you write it, so your local model is ready the moment you press Enter.</p>` +
      `<p class="hero-keys label"><b>Enter</b> send · <b>Shift+Enter</b> new line · <b>Esc</b> stop</p></section>`;
  }

  function addMessage(role) {
    thread.querySelector(".hero")?.remove();
    const el = document.createElement("article");
    el.className = `msg ${role}`;
    el.innerHTML = role === "user"
      ? `<div class="msg-label label">You</div><div class="msg-body"></div>`
      : `<div class="msg-label label">${mark("mask", 16)}MAFDET</div>` +
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

  const fmt = (n) => Math.round(n).toLocaleString("en-US");

  // The reasoning (thinking mode): folded above the answer, open while the model is still thinking.
  function renderThink(el, text, thinking, d) {
    const box = el.querySelector(".think");
    if (!text) return;
    const follow = nearBottom();
    box.hidden = false;
    box.querySelector(".think-body").textContent = text;
    const summary = box.querySelector("summary");
    summary.classList.toggle("pulse", thinking);
    summary.textContent = thinking ? "Thinking"
      : d && d.think_tokens ? `Reasoning · ${fmt(d.think_tokens)} tokens, ${fmt(d.think_ahead)} before Enter` : "Reasoning";
    if (follow) toBottom();
  }

  function metaLine(el, d) {
    const meta = el.querySelector(".msg-meta");
    const parts = [d.think_tokens
      ? `<span class="diamond">◆</span>first word ${fmt(d.answer_ms)} ms`
      : `<span class="diamond">◆</span>first token ${fmt(d.ttft_ms)} ms`];
    if (d.message_tokens) parts.push(`${fmt(d.read)} of ${fmt(d.message_tokens)} tokens read while you typed`);
    if (d.think_tokens) parts.push(`${fmt(d.think_ahead)} of ${fmt(d.think_tokens)} reasoning tokens before Enter`);
    if (d.ahead) parts.push(`${fmt(d.ahead)} answer tokens before Enter`);
    parts.push(`${fmt(d.tokens)} tokens · ${d.tps} tok/s`);
    if (d.stopped) parts.push("stopped");
    if (d.cut) parts.push("cut at the token limit");
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
      try { showStatus(await (await api("/api/draft", { text: input.value })).json()); } catch (e) { /* next poll */ }
    } while (draftAgain);
    draftBusy = false;
  }

  function autosize() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, window.innerHeight * 0.4)}px`;
  }

  // ---- send -------------------------------------------------------------------------

  let state = "loading", generating = false;

  async function send() {
    const text = input.value;
    if (!text.trim() || generating || state !== "ready") return;
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
      const reader = (await api("/api/send", { text })).body.getReader();
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
            errorLine(bot, ev.message);
          }
        }
      }
    } catch (e) {
      errorLine(bot, `Connection lost: ${e.message}`);
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
    sendBtn.innerHTML = glyph(generating ? "stop" : "send", generating ? 14 : 16);
    sendBtn.setAttribute("aria-label", generating ? "Stop" : "Send");
    sendBtn.disabled = !generating && (state !== "ready" || !input.value.trim());
  }

  const stop = () => { if (generating) api("/api/stop", {}).catch(() => {}); };

  async function newChat() {
    stop();
    while (generating) await new Promise((r) => setTimeout(r, 30));
    await api("/api/new", {}).catch(() => {});
    hero();
    input.focus();
    pushDraft();
  }

  // ---- status line ---------------------------------------------------------------------

  const statusIcon = $("status-icon"), statusText = $("status-text"), meter = $("meter"), meterRead = $("meter-read");
  let lastIcon = "";

  function setIcon(name, on) {
    if (name !== lastIcon) { statusIcon.innerHTML = mark(name, 14); lastIcon = name; }
    statusIcon.classList.toggle("on", on);
  }

  function showStatus(s) {
    state = s.state;
    $("model").textContent = s.state === "ready" ? s.model : s.state === "loading" ? "Waking up" : "Error";
    statusText.classList.toggle("error", s.state === "error");
    statusText.classList.remove("pulse");
    meter.hidden = true;
    $("status-side").textContent = s.state === "ready" ? `context ${fmt(s.context)} / ${fmt(s.ctx)}` : "";
    if (s.state === "loading") {
      setIcon("awaken", false);
      statusText.textContent = "Waking up · starting llama.cpp and loading the model";
      statusText.classList.add("pulse");
    } else if (s.state === "error") {
      setIcon("judge", false);
      statusText.textContent = s.error;
      statusText.title = s.error;
    } else if (generating || s.generating) {
      setIcon("strike", true);
      statusText.textContent = "Answering · Esc to stop";
    } else if (!s.has_text) {
      setIcon("watch", false);
      statusText.textContent = "Watching · start typing, MAFDET reads as you write";
    } else {
      const ready = !s.busy && s.pending === 0;
      setIcon("watch", true);
      meter.hidden = false;
      meter.classList.toggle("ready", ready);
      meterRead.style.width = `${s.message_tokens ? (100 * s.read) / s.message_tokens : 0}%`;
      const read = `${fmt(s.read)} of ${fmt(s.message_tokens)} tokens read`;
      const thought = s.thinking_ahead ? ` · ${fmt(s.thinking_ahead)} reasoning tokens ahead` : "";
      // "Ready" only for what is really ready: the answer, or (no reasoning) the prompt read.
      if (s.ahead_writing) {
        setIcon("strike", true);
        statusText.innerHTML = s.think && !s.ahead
          ? `<span class="diamond">◆</span> Thinking ahead · ${fmt(s.thinking_ahead)} reasoning tokens`
          : `<span class="diamond">◆</span> Answering ahead · ${fmt(s.ahead)} answer tokens written`;
        statusText.classList.add("pulse");
      } else if (s.busy) {
        statusText.textContent = `Reading · ${read}${thought}`;
        statusText.classList.add("pulse");
      } else if (s.answer_ready) {
        setIcon("strike", true);
        statusText.innerHTML = `<span class="diamond">◆</span> Answer ready on Enter · written ahead in full`;
      } else if (s.ahead) {
        setIcon("strike", true);
        statusText.innerHTML = `<span class="diamond">◆</span> Answer started · ${fmt(s.ahead)} tokens written ahead, the rest after Enter`;
      } else if (s.think && s.thinking_ahead) {
        statusText.textContent = `Paused · ${fmt(s.thinking_ahead)} reasoning tokens ahead, more thinking after Enter`;
      } else if (ready) {
        statusText.innerHTML = `<span class="diamond">◆</span> Ready on Enter · ${read}, ${fmt(s.left)} left${thought}`;
      } else {
        statusText.textContent = `Waiting for a pause · ${read}${thought}`;
      }
    }
    setSendMode();
  }

  async function poll() {
    if (!document.hidden) {
      try { showStatus(await (await api("/api/status")).json()); } catch (e) { /* server busy or gone */ }
    }
    setTimeout(poll, state === "ready" ? 200 : 500);
  }

  // ---- events --------------------------------------------------------------------------

  input.addEventListener("input", () => { autosize(); pushDraft(); setSendMode(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
  });
  sendBtn.addEventListener("click", () => (generating ? stop() : send()));
  $("new-chat").addEventListener("click", newChat);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") stop();
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "n") { e.preventDefault(); newChat(); }
  });

  function updateJump() { $("jump").hidden = nearBottom(); }
  scroller.addEventListener("scroll", updateJump, { passive: true });
  $("jump").addEventListener("click", () => scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" }));

  thread.addEventListener("click", (e) => {
    const link = e.target.closest("a[href]");
    if (link) {                                            // open links in the browser, not in this window
      e.preventDefault();
      api("/api/open", { url: link.getAttribute("href") }).catch(() => {});
      return;
    }
    const copy = e.target.closest(".copy");
    if (copy) {
      const code = copy.closest(".code").querySelector("pre code").textContent;
      navigator.clipboard.writeText(code).then(() => {
        copy.innerHTML = `${glyph("check", 11)}<span>Copied</span>`;
        setTimeout(() => { copy.innerHTML = `${glyph("copy", 11)}<span>Copy</span>`; }, 1500);
      }).catch(() => {});
    }
  });

  hero();
  autosize();
  setSendMode();
  input.focus();
  poll();
})();
