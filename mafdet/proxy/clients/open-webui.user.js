// ==UserScript==
// @name         MAFDET drafts for Open WebUI
// @namespace    https://github.com/Mafdet-AI/mafdet
// @version      0.1.0
// @description  Sends what you are typing in Open WebUI to the MAFDET proxy, so it reads ahead before Enter.
// @match        http://localhost:8080/*
// @match        http://127.0.0.1:8080/*
// @grant        none
// @run-at       document-idle
// ==/UserScript==
//
// Without CORS on the proxy: replace "@grant none" above with the two lines
//   // @grant        GM_xmlhttpRequest
//   // @connect      127.0.0.1
// (and @connect localhost if PROXY uses that name), then set USE_GM = true below.

(() => {
  'use strict';

  const PROXY = 'http://127.0.0.1:8077';  // where `python -m mafdet.proxy` listens
  const USE_GM = false;                    // true: GM_xmlhttpRequest instead of fetch (see above)
  const DEBOUNCE_MS = 250;                 // send after this long without a change...
  const MAX_WAIT_MS = 600;                 // ...and at least this often while typing goes on
  const RETRY_MS = 5000;                   // after a failed send, stay quiet this long
  const INPUT = '#chat-input';             // Open WebUI's message box (TipTap editor, or an old textarea)

  // "new" on a new chat (/ or /?...), the chat id on /c/<id>, null elsewhere (notes, channels...).
  function conversationId() {
    const p = location.pathname;
    if (p === '/' || p === '') return 'new';
    const m = p.match(/^\/c\/([^/]+)/);
    return m ? decodeURIComponent(m[1]) : null;
  }

  // The text as Open WebUI will send it. The editor's HTML goes through turndown
  // (RichTextInput.svelte, serializeContent): one "\n" between paragraphs, "  \n" for a
  // Shift+Enter line break, "\n\n" around other blocks, outer whitespace trimmed.
  function readText(el) {
    if (el.tagName === 'TEXTAREA') return el.value;
    let out = '', sep = 0;
    for (const block of el.children) {
      const isP = block.tagName === 'P';
      let text = isP ? inline(block) : block.tagName.match(/^H([1-6])$/)
        ? '#'.repeat(+block.tagName[1]) + ' ' + inline(block) : block.innerText;
      if (isP && text === '') text = '  ';  // an empty paragraph becomes a <br>: "  \n"
      const s = isP ? 1 : 2;
      out += (out ? '\n'.repeat(Math.max(sep, s)) : '') + text;
      sep = s;
    }
    return out.replace(/ /g, ' ').replace(/^[\t\r\n]+/, '').replace(/\s+$/, '');
  }

  function inline(node) {
    let s = '';
    for (const n of node.childNodes) {
      if (n.nodeType === Node.TEXT_NODE) { s += n.data; continue; }
      if (n.nodeType !== Node.ELEMENT_NODE) continue;
      if (n.dataset.type === 'mention') {  // turndown rule "mentions": <@id>, "/" becomes "$"
        const ch = n.dataset.mentionSuggestionChar || '@';
        s += `<${ch === '/' ? '$' : ch}${n.dataset.id || ''}>`;
      } else if (n.tagName === 'BR') {
        if (!n.classList.contains('ProseMirror-trailingBreak')) s += '  \n';
      } else if (n.className && String(n.className).startsWith('ProseMirror-')) {
        // editor scaffolding (separators, widgets): not text
      } else if (n.tagName === 'STRONG' || n.tagName === 'B') s += `**${inline(n)}**`;
      else if (n.tagName === 'EM' || n.tagName === 'I') s += `_${inline(n)}_`;
      else if (n.tagName === 'CODE') s += '`' + inline(n) + '`';
      else s += inline(n);
    }
    return s;
  }

  // Sending: fire and forget, one warning per outage, quiet retries.
  let lastKey = null, downUntil = 0, warned = false;

  function post(body) {
    const ok = () => { downUntil = 0; warned = false; };
    const fail = () => {
      downUntil = Date.now() + RETRY_MS;
      lastKey = null;  // resend the same text once the proxy is back
      if (!warned) { warned = true; console.warn(`[mafdet] proxy unreachable at ${PROXY}; drafts paused, retrying quietly`); }
    };
    if (USE_GM && typeof GM_xmlhttpRequest === 'function') {
      GM_xmlhttpRequest({ method: 'POST', url: PROXY + '/v1/drafts', data: body,
        headers: { 'Content-Type': 'application/json' }, onload: ok, onerror: fail, ontimeout: fail, timeout: 5000 });
    } else {
      fetch(PROXY + '/v1/drafts', { method: 'POST', mode: 'cors', body,
        headers: { 'Content-Type': 'application/json' } }).then(ok, fail);
    }
  }

  function send(el) {
    const conv = conversationId();
    if (!conv || !el.isConnected || Date.now() < downUntil) return;
    const text = readText(el);
    if (!text.trim()) return;                 // nothing typed, or just cleared by Enter
    const key = conv + '\u0000' + text;
    if (key === lastKey) return;              // nothing changed since the last draft
    lastKey = key;
    post(JSON.stringify({ conversation_id: conv, text }));
  }

  // Debounce: DEBOUNCE_MS after the last change, but never more than MAX_WAIT_MS of silence while typing.
  let timer = 0, pendingSince = 0;

  function flush(el) { clearTimeout(timer); pendingSince = 0; send(el); }

  function changed(el) {
    const now = Date.now();
    if (!pendingSince) pendingSince = now;
    clearTimeout(timer);
    const wait = Math.min(DEBOUNCE_MS, MAX_WAIT_MS - (now - pendingSince));
    timer = setTimeout(() => flush(el), Math.max(0, wait));
  }

  // The editor applies most edits itself (Enter, Backspace, undo, paste) without an "input"
  // event, so watch the box's DOM. Open WebUI re-creates the box (new chat, settings change):
  // re-attach whenever an input box gets focus or a key, at the document level.
  const watched = new WeakSet();

  function attach(el) {
    if (!el || watched.has(el)) return;
    watched.add(el);
    if (el.tagName === 'TEXTAREA') el.addEventListener('input', () => changed(el));
    else new MutationObserver(() => changed(el))
      .observe(el, { childList: true, subtree: true, characterData: true });
  }

  const target = (e) => (e.target instanceof Element ? e.target.closest(INPUT) : null);
  for (const type of ['focusin', 'keydown', 'input']) {
    document.addEventListener(type, (e) => attach(target(e)), true);
  }
  document.addEventListener('paste', (e) => {
    const el = target(e);
    if (!el) return;
    attach(el);
    setTimeout(() => flush(el), 0);  // a paste goes out at once, after the editor has inserted it
  }, true);
  attach(document.querySelector(INPUT));
})();
