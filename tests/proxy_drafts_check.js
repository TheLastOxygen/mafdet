const mafdetDrafts = require(process.argv[2]);
const failures = [];
const check = (cond, what) => { if (!cond) failures.push(what); };
const sent = [];
let fail = false;
const fakeFetch = (url, init) => fail ? Promise.reject(new Error('down')) : (sent.push(JSON.parse(init.body)), Promise.resolve({ ok: true }));
const d = mafdetDrafts({ fetch: fakeFetch, debounceMs: 50, maxWaitMs: 120, retryMs: 100 });
const sleep = ms => new Promise(r => setTimeout(r, ms));
(async () => {
  // steady typing every 30 ms: at least one draft per 120 ms, the last text after the pause
  let text = '';
  for (const ch of 'how many apples are left') { text += ch; d.typed('new', text); await sleep(30); }
  await sleep(80);
  const typedDrafts = sent.length;
  check(typedDrafts >= 4 && typedDrafts < 25, 'drafts while typing: ' + typedDrafts);
  check(sent.at(-1).text === 'how many apples are left', 'last draft is the whole text');
  // unchanged text is not sent again; empty text never
  d.typed('new', 'how many apples are left'); d.typed('new', '   '); await sleep(80);
  check(sent.length === typedDrafts, 'no resend: ' + sent.length);
  // a paste goes at once
  d.pasted('c1', 'a long document'); await sleep(5);
  check(sent.at(-1).conversation_id === 'c1' && sent.at(-1).text === 'a long document', 'paste at once');
  // sent() drops what is pending
  d.typed('c1', 'a long document and more'); d.sent(); await sleep(80);
  check(sent.at(-1).text === 'a long document', 'pending dropped on Enter');
  // proxy down: one warning, quiet, then back
  fail = true; d.pasted('c1', 'x1'); await sleep(5); d.pasted('c1', 'x2'); await sleep(5);
  fail = false; await sleep(120); d.pasted('c1', 'x3'); await sleep(5);
  check(sent.at(-1).text === 'x3', 'resumes after the retry window');
  if (failures.length) { console.error('FAILED: ' + failures.join('; ')); process.exit(1); }
  console.log('ok', sent.length, 'drafts');
})();
