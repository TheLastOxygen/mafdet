/* MAFDET app: the Models, Hardware and Settings views, and the first-start setup. */
(() => {
  "use strict";

  const { $, t, esc } = M;
  const after = (fn) => requestAnimationFrame(() => requestAnimationFrame(fn));   // next paint: transitions run

  // =====================================================================================
  // Models
  // =====================================================================================

  let filter = "all";                     // all, fits, installed
  let family = "all";                     // all, or a family id
  const picked = {};                      // model id → quantization chosen on its card
  let animateNext = true;

  function groups() {
    const out = new Map();
    for (const e of M.state.catalog?.models || []) {
      if (!out.has(e.model_id)) out.set(e.model_id, []);
      out.get(e.model_id).push(e);
    }
    return [...out.values()];
  }

  // The variant a card shows: the one picked, else the one in use, else a downloaded one, else the default.
  function shownVariant(variants) {
    const id = variants[0].model_id, loaded = (M.state.status || {}).model_id;
    return variants.find((v) => v.quant === picked[id]) || variants.find((v) => v.id === loaded) ||
      variants.find((v) => v.installed && v.default) || variants.find((v) => v.installed) ||
      variants.find((v) => v.default) || variants[0];
  }

  const keep = {
    all: () => true,
    fits: (vs) => vs.some((v) => v.placement === "gpu"),
    installed: (vs) => vs.some((v) => v.installed),
  };

  const inFamily = (vs) => family === "all" || vs[0].family === family;

  // Two rows of chips: the family, then what to show of it.
  function renderFilter() {
    const gs = groups();
    const fams = M.state.catalog?.families || [];
    const chip = (attr, val, on, label, n) =>
      `<button type="button" class="chip${on ? " on" : ""}" ${attr}="${val}" role="radio" aria-checked="${on}">` +
      `${esc(label)}<span class="count">${n}</span></button>`;
    $("models-filter").innerHTML =
      `<div class="chips">${chip("data-fam", "all", family === "all", t("models.filter.all"), gs.length)}` +
      fams.map((f) => chip("data-fam", f.id, family === f.id, f.name, gs.filter((vs) => vs[0].family === f.id).length)).join("") + `</div>` +
      `<div class="chips">${["fits", "installed"].map((f) =>
        chip("data-f", f, filter === f, t(`models.filter.${f}`), gs.filter(inFamily).filter(keep[f]).length)).join("")}</div>`;
    $("models-filter").querySelectorAll("[data-fam]").forEach((b) => b.addEventListener("click", () => {
      family = b.dataset.fam;
      animateNext = true;
      renderModels();
    }));
    $("models-filter").querySelectorAll("[data-f]").forEach((b) => b.addEventListener("click", () => {
      filter = filter === b.dataset.f ? "all" : b.dataset.f;       // a second click shows everything again
      animateNext = true;
      renderModels();
    }));
  }

  function renderSubtitle() {
    const hw = M.state.hardware;
    if (!hw) return;
    const others = M.state.catalog?.vram_others_mb || 0;
    $("models-sub").textContent = hw.gpu
      ? t("models.subtitle_gpu", { gpu: hw.gpu.name.replace(/^(NVIDIA (GeForce )?|AMD (Radeon )?)/, ""), ram: M.mb(hw.ram_mb),
          others: others > 1024 ? t("models.vram_others", { n: M.mb(others) }) : "" })
      : t("models.subtitle_cpu", { cpu: hw.cpu, ram: M.mb(hw.ram_mb) });
  }

  function placementBadge(e) {
    const text = e.placement === "partial" ? t("models.placement.partial", { p: Math.round(e.gpu_share * 100) })
      : t(`models.placement.${e.placement}`);
    return `<span class="badge ${e.placement}">${esc(text)}</span>`;
  }

  // What can be done with a variant right now: download, use, measure, or the progress of one of them.
  function actions(e) {
    const s = M.state.status || {};
    const dl = M.taskFor("download", e.id), bench = M.taskFor("bench", e.id);
    if (dl && dl.state === "running") {
      const p = Math.round(100 * dl.progress);
      const text = dl.total_bytes
        ? t("models.downloading", { done: M.bytes(dl.done_bytes), total: M.bytes(dl.total_bytes), speed: M.bytes(dl.speed_bps), eta: M.dur(dl.eta_s) })
        : t("models.download_starting");
      return `<div class="progress"><div class="progress-bar${p ? "" : " indeterminate"}"><i style="width:${p}%"></i></div>` +
        `<div class="progress-text"><span>${esc(text)}</span><button type="button" class="btn ghost" data-cancel="${dl.id}">${t("action.cancel")}</button></div></div>`;
    }
    if (bench && bench.state === "running") {
      return `<div class="progress"><div class="progress-bar indeterminate"><i></i></div>` +
        `<div class="progress-text"><span>${esc(t("models.benchmarking", { stage: t(`models.bench_stage.${bench.stage || "starting"}`) }))}</span></div></div>`;
    }
    if (!e.installed) {
      return `<button type="button" class="btn accent" data-get="${e.id}">${t("action.download_and_use")}</button>` +
        `<button type="button" class="btn ghost" data-dl="${e.id}">${t("action.download")}</button>`;
    }
    const inUse = s.model_id === e.id;
    const main = inUse && s.state === "ready" ? `<button type="button" class="btn" disabled>${M.glyph("check", 10)}${t("action.in_use")}</button>`
      : inUse && s.state === "loading" ? `<button type="button" class="btn" disabled>${t("action.loading")}</button>`
      : `<button type="button" class="btn primary" data-use="${e.id}">${t("action.use")}</button>`;
    return main + `<button type="button" class="btn ghost" data-bench="${e.id}">${t(e.measured ? "action.remeasure" : "action.benchmark")}</button>`;
  }

  function card(vs, i, animate) {
    const e = shownVariant(vs);
    const s = M.state.status || {};
    // The overall pick in the full catalog; the best of the family when one family is shown.
    const pick = family === "all" ? M.state.catalog?.recommended : M.state.catalog?.family_best?.[family];
    const rec = (pick || "").startsWith(e.model_id + ":");
    const ribbon = family === "all" ? t("models.recommended") : t("models.recommended_family", { family: e.family_name });
    const loaded = vs.some((v) => v.id === s.model_id && s.state === "ready");
    const params = e.active_b < e.params_b ? t("models.params_moe", { n: M.num(e.params_b, 1), a: M.num(e.active_b, 1) })
      : t("models.params", { n: M.num(e.params_b, 1) });
    const without = e.without_s || 0, withM = e.with_mafdet_s || 0;
    const stamp = e.measured ? `<span class="tag-measured">${t("models.measured")}</span>` : t("models.estimated");
    return `<article class="card${rec ? " recommended" : ""}${loaded ? " loaded" : ""}${e.placement === "too_big" ? " dim" : ""}${animate ? " enter" : ""}" style="--i:${i}" data-model="${e.model_id}">
      <div class="card-head">
        <div><h3 class="card-name">${esc(e.name)}</h3><div class="card-params label">${esc(params)}</div>
          <div class="card-vendor">${esc(e.vendor)} · ${esc(e.license)}</div></div>
        ${rec ? `<span class="ribbon">${M.glyph("bolt", 10)}${esc(ribbon)}</span>` : ""}
      </div>
      <p class="card-blurb">${t(`models.tier.${e.tier}`)}</p>
      <div class="card-tags">${placementBadge(e)}<span class="badge">${M.bytes(e.bytes)}</span><span class="badge">◆ ${t(e.reason_ahead ? "models.reason_ahead" : "models.answers_ahead")}</span></div>
      <div class="gauge">
        <div class="gauge-row"><span class="label">${t("models.vram_fit")}</span><span class="label">${Math.round(e.gpu_share * 100)}%</span></div>
        <div class="gauge-bar ${e.placement}"><i data-w="${Math.round(e.gpu_share * 100)}"></i></div>
      </div>
      <div class="stats">
        <div class="stat"><span class="label">${t("models.speed")}</span><span class="stat-value" data-count="${e.decode_tps}" data-digits="${e.decode_tps < 20 ? 1 : 0}" data-suffix="<small>tok/s</small>"></span><span class="stat-note label">${stamp}</span></div>
        <div class="stat"><span class="label">${t("models.read_speed")}</span><span class="stat-value" data-count="${e.prefill_tps}" data-suffix="<small>tok/s</small>"></span><span class="stat-note label">${stamp}</span></div>
        ${e.reason_ahead ? `<div class="wait-compare">
          <div class="gauge-row"><span class="label">${t("models.wait")}</span><span class="stat-value">${t("models.wait_value", { s: M.dur(withM) })}</span></div>
          <div class="wait-bars">
            <div class="wait-bar with"><i data-w="${without ? Math.max(2, (100 * withM) / without) : 0}"></i></div>
            <div class="wait-bar without"><i data-w="${without ? 100 : 0}"></i></div>
          </div>
          <span class="stat-note">${t("models.wait_without", { s: M.dur(without) })}</span>
        </div>` : `<div class="wait-compare">
          <div class="gauge-row"><span class="label">${t("models.answer_time")}</span><span class="stat-value">${t("models.wait_value", { s: M.dur(e.answer_s) })}</span></div>
          <span class="stat-note">${t("models.answer_note")}</span>
        </div>`}
      </div>
      ${vs.length > 1 ? `<div class="quants">${vs.map((v) => `<button type="button" data-q="${v.quant}" class="${v.id === e.id ? "on" : ""}">${v.quant}${v.installed ? '<span class="tick">●</span>' : ""}</button>`).join("")}</div>` : ""}
      <p class="card-hint">${t(`models.hint.${e.placement}`)}</p>
      <div class="card-actions" data-actions="${e.id}">${actions(e)}</div>
    </article>`;
  }

  // Bars grow and numbers roll in when they appear; later redraws set them straight away.
  function settle(root, animate) {
    root.querySelectorAll("[data-w]").forEach((el) => {
      if (animate) after(() => { el.style.width = `${el.dataset.w}%`; });
      else { el.style.transition = "none"; el.style.width = `${el.dataset.w}%`; }
    });
    root.querySelectorAll("[data-count]").forEach((el) => {
      const v = parseFloat(el.dataset.count), digits = +(el.dataset.digits || 0), suffix = el.dataset.suffix || "";
      if (animate) M.countUp(el, v, digits, suffix);
      else el.innerHTML = M.num(v, digits) + suffix;
    });
  }

  function bindCard(root) {
    root.querySelectorAll(".quants button").forEach((b) => b.addEventListener("click", () => {
      picked[b.closest(".card").dataset.model] = b.dataset.q;
      renderModels();
    }));
  }

  function renderModels() {
    const grid = $("models-grid");
    renderFilter();
    renderSubtitle();
    // The recommended model first (overall, or of the family shown), then the catalog's order.
    const best = ((family === "all" ? M.state.catalog?.recommended : M.state.catalog?.family_best?.[family]) || "").split(":")[0];
    const gs = groups().filter(inFamily).filter(keep[filter]).sort((a, b) => (b[0].model_id === best) - (a[0].model_id === best));
    const animate = animateNext && M.view === "models";
    animateNext = false;
    grid.innerHTML = gs.length ? gs.map((vs, i) => card(vs, i, animate)).join("")
      : `<p class="page-sub" style="padding:8px 2px">${t("models.empty")}</p>`;
    bindCard(grid);
    settle(grid, animate);
  }

  // Progress changes only the action row of a card, not the whole grid.
  function updateActions() {
    document.querySelectorAll("[data-actions]").forEach((el) => {
      const e = (M.state.catalog?.models || []).find((x) => x.id === el.dataset.actions);
      if (!e) return;
      const html = actions(e);
      if (el.dataset.html !== html) { el.innerHTML = html; el.dataset.html = html; }
    });
  }

  // One listener for every card button, wherever the card is (Models view or setup).
  document.addEventListener("click", async (ev) => {
    const b = ev.target.closest("[data-get],[data-dl],[data-use],[data-bench],[data-cancel]");
    if (!b || !b.closest(".card, .ob-model")) return;
    if (b.dataset.get) { M.pendingUse.add(b.dataset.get); await M.startTask("/api/download", { model: b.dataset.get }); }
    else if (b.dataset.dl) await M.startTask("/api/download", { model: b.dataset.dl });
    else if (b.dataset.bench) await M.startTask("/api/bench", { model: b.dataset.bench });
    else if (b.dataset.cancel) await M.post("/api/tasks/cancel", { id: b.dataset.cancel });
    else if (b.dataset.use) { M.setStatus(await M.post("/api/model/load", { model: b.dataset.use })); M.go("chat"); }
    updateActions();
  });

  M.on("catalog", () => renderModels());
  M.on("hardware", renderSubtitle);
  M.on("tasks", updateActions);
  M.on("status", ({ s, prev }) => { if (!prev || prev.state !== s.state || prev.model_id !== s.model_id) updateActions(); });
  M.on("view", (v) => { if (v === "models") { animateNext = true; renderModels(); M.refresh.catalog(); } });
  M.on("lang", () => renderModels());
  M.views.models = { card, settle, actions };

  // =====================================================================================
  // Hardware
  // =====================================================================================

  function meterBlock(label, used, total, right) {
    const p = total ? Math.round((100 * used) / total) : 0;
    return `<div class="gauge"><div class="gauge-row"><span class="label">${label}</span><span class="label">${right}</span></div>` +
      `<div class="gauge-bar ${p > 85 ? "partial" : ""}"><i data-w="${p}"></i></div></div>`;
  }

  function runtimeBlock() {
    const r = M.state.runtime || {};
    const task = M.taskFor("llama");
    let side;
    if (task && task.state === "running") {
      const p = Math.round(100 * task.progress);
      side = `<div class="progress"><div class="progress-bar${p ? "" : " indeterminate"}"><i style="width:${p}%"></i></div>` +
        `<div class="progress-text"><span>${esc(t("hardware.installing", { stage: t(`hardware.stage.${task.stage || "release"}`), p }))}</span>` +
        `<button type="button" class="btn ghost" data-cancel-rt="${task.id}">${t("action.cancel")}</button></div></div>`;
    } else {
      side = `<button type="button" class="btn ${r.installed ? "" : "accent"}" data-install-rt>${t(r.installed ? "action.reinstall" : "action.install_runtime")}</button>`;
    }
    const where = r.installed ? `<p class="hw-line mono">${esc(t(r.managed ? "hardware.runtime_managed" : "hardware.runtime_own", { path: r.path }))}</p>` : "";
    return `<div class="hw-runtime"><div>
        <h3 class="hw-name">${r.installed ? esc(t("hardware.runtime_installed", { build: r.build || "?" })) : t("hardware.runtime_missing")}</h3>
        <p class="hw-line">${esc(t("hardware.runtime_tested", { build: r.tested_build || "" }))} ${esc(t("hardware.runtime_build", { label: r.recommended || "" }))}</p>${where}
      </div><div class="rt-side" style="min-width:260px">${side}</div></div>`;
  }

  let hwAnimated = false;
  function renderHardware() {
    const hw = M.state.hardware;
    if (!hw) return;
    const g = hw.gpu;
    const cards = [
      `<div class="hw-card enter" style="--i:0"><div class="hw-head">${M.mark("strike", 22)}<span class="label">${t("hardware.gpu")}</span></div>` +
      (g ? `<h3 class="hw-name">${esc(g.name)}</h3><p class="hw-line">${esc(t("hardware.driver", { v: g.driver }))}</p>` +
        meterBlock(t("hardware.vram"), g.vram_used_mb, g.vram_mb, `${M.mb(g.vram_mb)} · ${esc(t("hardware.in_use", { used: M.mb(g.vram_used_mb) }))}`) +
        `<div class="hw-figures"><div class="stat"><span class="label">${t("hardware.bandwidth")}</span><span class="stat-value">${esc(t("hardware.bandwidth_value", { n: M.num(g.bandwidth_gbs) }))}</span>` +
        `${g.bandwidth_known ? "" : `<span class="stat-note label">${t("hardware.estimated")}</span>`}</div></div>`
        : `<p class="hw-line">${t("hardware.no_gpu")}</p>`) + `</div>`,
      `<div class="hw-card enter" style="--i:1"><div class="hw-head">${M.mark("hunt", 22)}<span class="label">${t("hardware.cpu")}</span></div>` +
      `<h3 class="hw-name">${esc(hw.cpu)}</h3><p class="hw-line">${esc(t("hardware.cores", { cores: hw.cores, threads: hw.threads }))}</p></div>`,
      `<div class="hw-card enter" style="--i:2"><div class="hw-head">${M.mark("race", 22)}<span class="label">${t("hardware.ram")}</span></div>` +
      `<h3 class="hw-name">${M.mb(hw.ram_mb)}</h3>` +
      meterBlock(esc(t("hardware.in_use", { used: M.mb(hw.ram_mb - hw.ram_free_mb) })), hw.ram_mb - hw.ram_free_mb, hw.ram_mb,
        esc(t("hardware.ram_free", { free: M.mb(hw.ram_free_mb) }))) +
      `<div class="hw-figures"><div class="stat"><span class="label">${t("hardware.bandwidth")}</span><span class="stat-value">${esc(t("hardware.bandwidth_value", { n: M.num(hw.ram_bandwidth_gbs) }))}</span><span class="stat-note label">${t("hardware.estimated")}</span></div></div></div>`,
      `<div class="hw-card enter" style="--i:3"><div class="hw-head">${M.mark("mask", 22)}<span class="label">${t("hardware.system")}</span></div>` +
      `<h3 class="hw-name">${esc(hw.os)}</h3><p class="hw-line mono">${esc(hw.arch)}</p></div>`,
      `<div class="hw-card wide enter" style="--i:4" id="hw-runtime"><div class="hw-head">${M.mark("awaken", 22)}<span class="label">${t("hardware.runtime")}</span></div>${runtimeBlock()}</div>`,
    ];
    const animate = !hwAnimated && M.view === "hardware";
    const grid = $("hw-grid");
    grid.innerHTML = cards.join("");
    if (!animate) grid.querySelectorAll(".enter").forEach((el) => el.classList.remove("enter"));
    settle(grid, animate);
    if (animate) hwAnimated = true;
  }

  function updateRuntime() {
    const box = $("hw-runtime");
    if (!box) return;
    const html = runtimeBlock();
    const holder = box.querySelector(".hw-runtime");
    if (holder && holder.outerHTML !== html) holder.outerHTML = html;
  }

  $("hw-grid").addEventListener("click", async (e) => {
    if (e.target.closest("[data-install-rt]")) await M.startTask("/api/runtime/install");
    const c = e.target.closest("[data-cancel-rt]");
    if (c) await M.post("/api/tasks/cancel", { id: c.dataset.cancelRt });
    updateRuntime();
  });
  $("hw-refresh").addEventListener("click", () => { hwAnimated = false; M.refresh.hardware(true); M.refresh.runtime(); });
  M.on("hardware", renderHardware);
  M.on("runtime", renderHardware);
  M.on("tasks", updateRuntime);
  M.on("lang", renderHardware);
  M.on("view", (v) => { if (v === "hardware") { hwAnimated = false; renderHardware(); } });

  // =====================================================================================
  // Settings
  // =====================================================================================

  const LANG_NAMES = { en: "English", it: "Italiano", fr: "Français", de: "Deutsch" };
  const RELOAD = ["think", "ahead", "think_ahead", "max_tokens", "ctx", "device", "recycle", "gatherer", "branches"];

  async function save(changes) {
    const before = M.state.settings || {};
    M.state.settings = await M.post("/api/settings", changes);
    const reload = Object.keys(changes).some((k) => RELOAD.includes(k) && before[k] !== changes[k]) &&
      ["ready", "loading"].includes((M.state.status || {}).state);
    M.toast(t(reload ? "settings.toast.reload" : "settings.toast.saved"), "ok");
  }

  const row = (name, hint, control, exp = false) =>
    `<div class="set-row"><div class="set-name">${t(name)}${exp ? `<span class="label exp">β</span>` : ""}</div>` +
    `<p class="set-hint">${t(hint)}</p><div class="set-control">${control}</div></div>`;
  const section = (title, rows) => `<section class="set-section enter"><div class="label">${t(title)}</div>${rows.join("")}</section>`;
  const toggle = (key, on) => `<button type="button" class="switch${on ? " on" : ""}" role="switch" aria-checked="${on}" data-key="${key}"></button>`;
  const range = (key, v, min, max, step, show) =>
    `<input type="range" data-key="${key}" min="${min}" max="${max}" step="${step}" value="${v}"><span class="set-value" data-show="${key}">${show(v)}</span>`;
  const seg = (key, options, v) => `<div class="seg" data-key="${key}">${options.map(([val, label]) =>
    `<button type="button" data-val="${val}" class="${val === v ? "on" : ""}">${label}</button>`).join("")}</div>`;

  const shows = {
    ahead: (v) => t("settings.tokens", { n: M.num(+v) }),
    think_ahead: (v) => t("settings.tokens", { n: M.num(+v) }),
    max_tokens: (v) => (+v ? t("settings.tokens", { n: M.num(+v) }) : t("settings.unlimited")),
  };

  function renderSettings() {
    const st = M.state.settings;
    if (!st) return;
    const small = (M.state.catalog?.models || []).filter((m) => m.installed && m.family === "qwen3" && m.params_b <= 1.7);
    const gatherer = `<select data-key="gatherer"><option value="">${t("settings.gatherer.none")}</option>` +
      small.map((m) => `<option value="${m.id}"${st.gatherer === m.id ? " selected" : ""}>${esc(m.name)} · ${m.quant}</option>`).join("") + `</select>`;
    const ctx = `<select data-key="ctx">${[4096, 8192, 16384, 32768].map((n) => `<option value="${n}"${st.ctx === n ? " selected" : ""}>${M.num(n)}</option>`).join("")}</select>`;
    $("settings").innerHTML = [
      section("settings.section.general", [
        row("settings.language", "settings.language.hint", seg("language", Object.entries(LANG_NAMES), st.language)),
      ]),
      section("settings.section.thinking", [
        row("settings.think", "settings.think.hint", toggle("think", st.think)),
        row("settings.think_ahead", "settings.think_ahead.hint", range("think_ahead", st.think_ahead, 256, 4096, 256, shows.think_ahead)),
        row("settings.ahead", "settings.ahead.hint", range("ahead", st.ahead, 0, 1024, 64, shows.ahead)),
        row("settings.max_tokens", "settings.max_tokens.hint", range("max_tokens", st.max_tokens, 0, 8192, 256, shows.max_tokens)),
      ]),
      section("settings.section.model", [
        row("settings.ctx", "settings.ctx.hint", ctx),
        row("settings.device", "settings.device.hint", seg("device", [["auto", t("settings.device.auto")], ["cuda", t("settings.device.cuda")], ["cpu", t("settings.device.cpu")]], st.device)),
      ]),
      section("settings.section.experimental", [
        row("settings.recycle", "settings.recycle.hint", toggle("recycle", st.recycle), true),
        row("settings.gatherer", "settings.gatherer.hint", gatherer, true),
        row("settings.branches", "settings.branches.hint", toggle("branches", st.branches), true),
      ]),
      `<section class="set-section enter"><div class="label">${t("settings.section.about")}</div><div class="about">` +
      `<div style="display:flex;align-items:center;gap:12px">${M.mark("mask", 28)}${M.wordmark(12)}<span class="label">${esc(t("settings.about.version", { v: M.version }))}</span></div>` +
      `<p style="margin:0">${t("settings.about.body")}</p>` +
      `<p style="margin:0" class="label">${t("settings.about.license")} · <a href="https://github.com/TheLastOxygen/mafdet" data-link>${t("settings.about.source")}</a></p>` +
      `<div><button type="button" class="btn" data-setup>${t("settings.setup_again")}</button></div></div></section>`,
    ].join("");
    $("settings").querySelectorAll(".enter").forEach((el, i) => el.style.setProperty("--i", i));
  }

  $("settings").addEventListener("click", (e) => {
    const sw = e.target.closest(".switch");
    if (sw) {
      const on = !sw.classList.contains("on");
      sw.classList.toggle("on", on);
      sw.setAttribute("aria-checked", on);
      save({ [sw.dataset.key]: on });
    }
    const opt = e.target.closest(".seg button");
    if (opt) {
      const key = opt.closest(".seg").dataset.key, val = opt.dataset.val;
      opt.parentElement.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === opt));
      if (key === "language") M.setLang(val); else save({ [key]: val });
    }
    const link = e.target.closest("[data-link]");
    if (link) { e.preventDefault(); M.post("/api/open", { url: link.href }); }
    if (e.target.closest("[data-setup]")) M.views.onboarding.open();
  });
  $("settings").addEventListener("input", (e) => {
    const k = e.target.dataset.key;
    if (e.target.type === "range" && shows[k]) $("settings").querySelector(`[data-show="${k}"]`).textContent = shows[k](e.target.value);
  });
  $("settings").addEventListener("change", (e) => {
    const k = e.target.dataset.key;
    if (!k) return;
    save({ [k]: e.target.type === "range" || k === "ctx" ? +e.target.value : e.target.value });
  });
  M.on("view", (v) => { if (v === "settings") renderSettings(); });
  M.on("lang", renderSettings);

  // =====================================================================================
  // First start: welcome, machine, runtime, model, ready
  // =====================================================================================

  const STEPS = ["welcome", "hardware", "runtime", "model", "ready"];
  let step = 0, scanned = false;

  function stepper() {
    $("ob-steps").innerHTML = STEPS.map((s, i) =>
      `<li class="${i < step ? "done" : i === step ? "on" : ""}">${t(`onboarding.step.${s}`)}</li>`).join("");
  }

  function buttons(next, { back = step > 0, skip = step < STEPS.length - 1, nextLabel = "action.next", disabled = false } = {}) {
    $("ob-actions").innerHTML =
      `<div>${skip ? `<button type="button" class="btn ghost" data-ob="skip">${t("action.skip")}</button>` : ""}</div>` +
      `<div class="right">${back ? `<button type="button" class="btn" data-ob="back">${t("action.back")}</button>` : ""}` +
      `<button type="button" class="btn accent" data-ob="next"${disabled ? " disabled" : ""}>${t(nextLabel)}</button></div>`;
    $("ob-actions").querySelector('[data-ob="next"]').onclick = next;
  }

  function show() {
    stepper();
    const body = $("ob-body");
    const name = STEPS[step];
    const go = () => { step++; show(); };
    if (name === "welcome") {
      body.innerHTML = `<div class="splash-mark">${M.mark("mask", 96)}</div><h2>${t("onboarding.welcome.title")}</h2><p>${t("onboarding.welcome.body")}</p>`;
      buttons(go);
    } else if (name === "hardware") {
      const hw = M.state.hardware || {};
      const items = [
        [t("hardware.gpu"), hw.gpu ? hw.gpu.name : "–"],
        [t("hardware.vram"), hw.gpu ? M.mb(hw.gpu.vram_mb) : "–"],
        [t("hardware.cpu"), hw.cpu || "–"],
        [t("hardware.ram"), M.mb(hw.ram_mb || 0)],
      ];
      body.innerHTML = `<h2>${t("onboarding.hardware.title")}</h2><div class="scan-list">` +
        items.map(([k], i) => `<div class="scan-item" data-i="${i}"><span class="label">${esc(k)}</span><div class="loader"><i></i></div></div>`).join("") +
        `</div><p id="ob-found" style="min-height:3em"></p>`;
      buttons(go, { disabled: !scanned });
      // The findings appear one by one, as if read off the machine.
      items.forEach(([, v], i) => setTimeout(() => {
        const el = body.querySelector(`.scan-item[data-i="${i}"]`);
        if (!el) return;
        el.querySelector(".loader").outerHTML = `<span class="val">${esc(v)}</span>`;
        if (i === items.length - 1) {
          scanned = true;
          $("ob-found").textContent = hw.gpu ? t("onboarding.hardware.found_gpu", { gpu: hw.gpu.name, vram: M.mb(hw.gpu.vram_mb) })
            : t("onboarding.hardware.found_cpu");
          buttons(go);
        }
      }, scanned ? 0 : 450 + i * 380));
    } else if (name === "runtime") {
      const r = M.state.runtime || {};
      const task = M.taskFor("llama");
      const running = task && task.state === "running";
      body.innerHTML = `<div>${M.mark("awaken", 64)}</div><h2>${t("onboarding.runtime.title")}</h2>` +
        (r.installed ? `<p>${esc(t("onboarding.runtime.done", { build: r.build }))}</p>`
          : `<p>${esc(t("onboarding.runtime.body", { label: r.recommended || "" }))}</p><div id="ob-rt">` +
            (running ? "" : `<button type="button" class="btn accent" id="ob-install">${t("action.install_runtime")}</button>`) + `</div>`);
      body.querySelector("#ob-install")?.addEventListener("click", async () => { await M.startTask("/api/runtime/install"); show(); });
      buttons(go, { disabled: !r.installed });
      if (running) runtimeProgress();
    } else if (name === "model") {
      const best = M.state.catalog?.recommended;
      const vs = (M.state.catalog?.models || []).filter((m) => best && m.model_id === best.split(":")[0]);
      // The recommended quantization only until one is picked: show() runs again after every click.
      if (best && !picked[best.split(":")[0]]) picked[best.split(":")[0]] = best.split(":")[1];
      body.innerHTML = `<h2>${t("onboarding.model.title")}</h2><p>${t("onboarding.model.body")}</p>` +
        `<div class="ob-model">${vs.length ? M.views.models.card(vs, 0, false) : ""}</div>`;
      M.views.models.settle(body, true);
      body.querySelectorAll(".quants button").forEach((b) => b.addEventListener("click", () => { picked[b.closest(".card").dataset.model] = b.dataset.q; show(); }));
      buttons(go);
    } else {
      body.innerHTML = `<div>${M.mark("mask", 96)}</div><h2>${t("onboarding.ready.title")}</h2><p>${t("onboarding.ready.body")}</p>`;
      buttons(finish, { back: false, skip: false, nextLabel: "action.start_chat" });
    }
    $("ob-actions").querySelector('[data-ob="back"]')?.addEventListener("click", () => { step--; show(); });
    $("ob-actions").querySelector('[data-ob="skip"]')?.addEventListener("click", finish);
  }

  function runtimeProgress() {
    const slot = $("ob-rt"), task = M.taskFor("llama");
    if (!slot || !task) return;
    const p = Math.round(100 * task.progress);
    slot.innerHTML = `<div class="progress"><div class="progress-bar${p ? "" : " indeterminate"}"><i style="width:${p}%"></i></div>` +
      `<div class="progress-text"><span>${esc(t("hardware.installing", { stage: t(`hardware.stage.${task.stage || "release"}`), p }))}</span><span>${p}%</span></div></div>`;
  }

  async function finish() {
    $("onboarding").hidden = true;
    M.state.settings = await M.post("/api/settings", { first_run_done: true }).catch(() => M.state.settings);
    M.go("chat");
    const s = M.state.status || {};
    const want = M.state.settings.model;
    const ready = (M.state.catalog?.models || []).find((m) => m.id === want && m.installed);
    if (ready && s.state !== "ready" && s.state !== "loading") M.setStatus(await M.post("/api/model/load", { model: want }));
  }

  M.views.onboarding = {
    open() {
      step = 0;
      scanned = false;
      $("onboarding").hidden = false;
      show();
    },
  };
  M.on("tasks", () => {
    if ($("onboarding").hidden) return;
    if (STEPS[step] === "runtime") {
      const task = M.taskFor("llama");
      if (task && task.state === "running") runtimeProgress();
    }
    if (STEPS[step] === "model") {
      const el = document.querySelector(".ob-model [data-actions]");
      if (el) el.innerHTML = M.views.models.actions((M.state.catalog?.models || []).find((x) => x.id === el.dataset.actions));
    }
  });
  M.on("runtime", () => { if (!$("onboarding").hidden && STEPS[step] === "runtime") show(); });
  M.on("lang", () => { if (!$("onboarding").hidden) show(); });
})();
