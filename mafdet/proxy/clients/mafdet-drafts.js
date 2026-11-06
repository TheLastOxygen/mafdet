// mafdet-drafts.js: send what the user is typing to the MAFDET proxy, for any chat front end.
//
//   const drafts = mafdetDrafts({ proxy: 'http://127.0.0.1:8077' });
//   input.addEventListener('input', () => drafts.typed(chatId, input.value));
//   input.addEventListener('paste', () => setTimeout(() => drafts.pasted(chatId, input.value)));
//   // on Enter: send the usual POST /v1/chat/completions, through the proxy, then
//   drafts.sent();
//
// chatId: your conversation's id, or "new" before it has one. The text must be the user message
// exactly as the request on Enter will carry it (the proxy matches the two). The proxy puts the
// text after the conversation it saw pass through, so only the text is sent. Fire and forget:
// nothing here ever throws, and a proxy that is down costs one console warning.
// Works in browsers and in Node 18+ (global fetch). No dependencies.

(function (root) {
  'use strict';

  function mafdetDrafts(options) {
    const o = Object.assign({ proxy: 'http://127.0.0.1:8077', debounceMs: 250, maxWaitMs: 600,
                              retryMs: 5000, fetch: root.fetch && root.fetch.bind(root) }, options || {});
    let timer = null, pendingSince = 0, pending = null, lastKey = null, downUntil = 0, warned = false;

    function post(conversationId, text) {
      if (!text || !text.trim() || Date.now() < downUntil) return;
      const key = conversationId + '\u0000' + text;
      if (key === lastKey) return;                     // nothing changed since the last draft
      lastKey = key;
      Promise.resolve()
        .then(() => o.fetch(o.proxy + '/v1/drafts', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ conversation_id: String(conversationId || 'new'), text }) }))
        .then(() => { downUntil = 0; warned = false; },
              () => {
                downUntil = Date.now() + o.retryMs;
                lastKey = null;                        // send it again once the proxy is back
                if (!warned) { warned = true; console.warn('[mafdet] proxy unreachable at ' + o.proxy); }
              });
    }

    function flush() {
      clearTimeout(timer); timer = null; pendingSince = 0;
      if (pending) { const p = pending; pending = null; post(p.id, p.text); }
    }

    return {
      // The text changed: send it after a pause, and at least every maxWaitMs while typing goes on.
      typed(conversationId, text) {
        const now = Date.now();
        pending = { id: conversationId, text };
        if (!pendingSince) pendingSince = now;
        clearTimeout(timer);
        timer = setTimeout(flush, Math.max(0, Math.min(o.debounceMs, o.maxWaitMs - (now - pendingSince))));
      },
      // A paste (or anything big at once): send it now.
      pasted(conversationId, text) { pending = { id: conversationId, text }; flush(); },
      // Enter: the request is on its way; nothing left to send for this message.
      sent() { clearTimeout(timer); timer = null; pendingSince = 0; pending = null; lastKey = null; },
    };
  }

  if (typeof module === 'object' && module.exports) module.exports = mafdetDrafts;
  else root.mafdetDrafts = mafdetDrafts;
})(typeof globalThis !== 'undefined' ? globalThis : this);
