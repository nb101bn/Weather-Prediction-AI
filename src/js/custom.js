/* Skew-T Point Forecast HUD - talks to Flask (/hud/*), which forwards to the FastAPI service. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const LEADS = [1, 2, 3, 6, 12, 24, 36, 48];
  const LEVELS = ["SFC", "1000 hPa", "925 hPa", "850 hPa", "700 hPa", "500 hPa", "400 hPa", "300 hPa", "250 hPa", "200 hPa"];
  const state = { component: "dewpoint", lead: 2, source: "wyoming", level: "SFC", components: [], last: null, busy: false };
  const SVGNS = "http://www.w3.org/2000/svg";

  // ------------------------------------------------------------------ utilities
  const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
  const signed = (v, d = 1) => (v === null || v === undefined ? "—" : (v > 0 ? "+" : "") + Number(v).toFixed(d));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const decimals = (c) => (c === "height" || c === "wind_direction" || c === "relative_humidity" ? 0 : 1);
  const compMeta = (id) => state.components.find((c) => c.id === id) || { id, label: id, units: "" };

  function log(message, isError = false) {
    const li = document.createElement("li");
    li.textContent = `${new Date().toISOString().slice(11, 19)}Z  ${message}`;
    if (isError) li.className = "err";
    $("log").prepend(li);
    while ($("log").children.length > 60) $("log").lastChild.remove();
  }

  function detailText(detail) {
    if (Array.isArray(detail)) return detail.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
    return detail || "Unknown error";
  }

  async function api(method, path, body) {
    const t0 = performance.now();
    const res = await fetch(path, { method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
    let data;
    try { data = await res.json(); } catch { data = { detail: `HTTP ${res.status}` }; }
    if (data && data.backend) setPill("pill-backend", `BACKEND: ${data.backend.toUpperCase()}`, data.backend === "fastapi" ? "ok" : "");
    if (!res.ok) { const err = new Error(detailText(data.detail)); err.status = res.status; throw err; }
    return { data, ms: Math.round(performance.now() - t0) };
  }

  function setPill(id, text, cls) { const p = $(id); p.textContent = text; p.className = `pill ${cls || ""}`; }

  // ------------------------------------------------------------------ controls
  function renderComponents() {
    const grid = $("component-grid");
    grid.innerHTML = "";
    state.components.forEach((c) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = `component${c.id === state.component ? " active" : ""}`;
      b.dataset.component = c.id;
      b.title = `Keywords: ${c.synonyms.join(", ")}\nModel: ${c.model}`;
      b.innerHTML = `${esc(c.label)}<span class="units">${esc(c.units)}</span><span class="dot ${c.model === "trained" ? "trained" : ""}"></span>`;
      b.addEventListener("click", () => setComponent(c.id));
      grid.appendChild(b);
    });
  }

  function setComponent(id) {
    state.component = id;
    document.querySelectorAll(".component").forEach((b) => b.classList.toggle("active", b.dataset.component === id));
    const c = compMeta(id);
    setPill("pill-model", `MODEL: ${c.model === "trained" ? "TRAINED" : "PERSISTENCE"}`, c.model === "trained" ? "ok" : "warn");
  }

  function setLead(hours) {
    const h = Math.max(1, Math.min(48, Math.round(Number(hours) || 1)));
    state.lead = h;
    $("lead-range").value = h;
    $("lead-number").value = h;
    $("lead-display").textContent = `+${h} HR${h === 1 ? "" : "s"}`;
    document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", Number(c.dataset.lead) === h));
  }

  function setLevel(level) { state.level = level; $("level-select").value = level; }

  async function setSource(source) {
    state.source = source;
    document.querySelectorAll("#source-toggle button").forEach((b) => b.classList.toggle("active", b.dataset.source === source));
    setPill("pill-source", `SOURCE: ${source.toUpperCase()}`, source === "synthetic" ? "bad" : "ok");
    await loadComponents();
  }

  async function loadComponents() {
    try {
      const { data } = await api("GET", `/hud/components?source=${state.source}`);
      state.components = data.components;
      renderComponents();
      setComponent(state.component);
    } catch (e) { log(`components: ${e.message}`, true); }
  }

  async function loadStations() {
    try {
      const { data } = await api("GET", "/hud/stations");
      $("station-select").innerHTML = data.stations.map((s) => `<option value="${esc(s.wmo)}">${esc(s.wmo)} ${esc(s.id)} – ${esc(s.name)}</option>`).join("");
    } catch (e) { log(`stations: ${e.message}`, true); }
  }

  const station = () => ($("station-custom").value.trim() || $("station-select").value || "72451").toUpperCase();
  const initTime = () => ($("init-latest").checked ? null : `${$("init-date").value}T${$("init-hour").value}:00`);

  // ------------------------------------------------------------------ command line
  let resolveTimer = null;
  async function resolveCommand(text, apply) {
    if (!text.trim()) { $("command-feedback").textContent = "Type a component and lead time, then press Enter."; $("command-feedback").className = "feedback"; return null; }
    try {
      const { data } = await api("POST", "/hud/resolve", { query: text });
      const parts = [data.label ? `${data.label} (matched "${data.matched_keyword}")` : "component ?", data.lead_hours ? `+${data.lead_hours} h` : "lead ?", data.level || ""];
      $("command-feedback").textContent = `→ ${parts.filter(Boolean).join(" · ")}`;
      $("command-feedback").className = `feedback ${data.component ? "ok" : "bad"}`;
      if (apply) {
        if (data.component) setComponent(data.component);
        if (data.lead_hours) setLead(data.lead_hours);
        if (data.level) setLevel(data.level);
      }
      return data;
    } catch (e) { $("command-feedback").textContent = e.message; $("command-feedback").className = "feedback bad"; return null; }
  }

  // ------------------------------------------------------------------ forecast
  async function runForecast() {
    if (state.busy) return;
    state.busy = true;
    $("run").disabled = true;
    $("run").textContent = "RUNNING…";
    const payload = { component: state.component, lead_hours: state.lead, station: station(), source: state.source, init_time: initTime(), level: state.level };
    log(`forecast ${payload.component} +${payload.lead_hours}h @ ${payload.station} (${payload.source})`);
    try {
      const { data, ms } = await api("POST", "/hud/forecast", payload);
      state.last = data;
      render(data);
      log(`ok in ${ms} ms via ${data.backend}; init ${data.init_time}`);
    } catch (e) {
      $("warnings").innerHTML = `<div class="warning error">Forecast failed: ${esc(e.message)}</div>`;
      log(`error: ${e.message}`, true);
    } finally {
      state.busy = false;
      $("run").disabled = false;
      $("run").textContent = "RUN FORECAST";
    }
  }

  function render(d) {
    $("warnings").innerHTML = d.warnings.map((w) => `<div class="warning ${w.startsWith("SYNTHETIC") ? "critical" : ""}">${esc(w)}</div>`).join("");
    const dp = decimals(d.component), h = d.headline, up = (h.change || 0) >= 0;
    const valid = d.valid_time.replace("T", " ").slice(0, 16), init = d.init_time.replace("T", " ").slice(0, 16);
    $("headline").innerHTML = `
      <div class="hl-value">${fmt(h.forecast, dp)}<small>${esc(d.units)}</small></div>
      <div><div class="hl-title">${esc(d.label)} · ${esc(h.level)} — +${d.lead_hours} HRs</div>
        <div class="hl-meta">Station ${esc(d.station)} · valid ${esc(valid)}Z · init ${esc(init)}Z</div>
        <div class="hl-meta">${d.model.status === "trained" ? "NN forecast" : "Persistence (untrained node)"} · ${esc(d.source)} data</div></div>
      <div class="hl-side"><div>Observed (init): ${fmt(h.observed, dp)} ${esc(d.units)}</div>
        <div class="${up ? "up" : "down"}">Change: ${signed(h.change, dp)} ${esc(d.units)}</div>
        <div class="muted">${esc(h.pressure_hpa)} hPa</div></div>`;
    $("level-table").innerHTML = d.profile.map((r) => `<tr class="${r.level === h.level ? "selected" : ""} ${r.below_ground ? "below" : ""}">
      <td>${esc(r.level)}${r.below_ground ? " (bg)" : ""}</td><td>${fmt(r.pressure_hpa, 0)}</td><td>${fmt(r.observed, dp)}</td><td>${fmt(r.forecast, dp)}</td><td>${signed(r.change, dp)}</td></tr>`).join("");
    renderModel(d.model);
    setPill("pill-model", `MODEL: ${d.model.status === "trained" ? "TRAINED" : "PERSISTENCE"}`, d.model.status === "trained" ? "ok" : "warn");
    $("skewt-caption").textContent = `${d.station} · ${init}Z`;
    $("profile-caption").textContent = `${d.label} · +${d.lead_hours} h`;
    drawSkewT($("skewt"), d);
    drawProfile($("profile-chart"), d);
  }

  function renderModel(m) {
    if (m.status !== "trained") { $("model-info").innerHTML = `<p>${esc(m.note)}</p>`; return; }
    const x = m.metrics || {}, u = compMeta(state.last.component).units;
    const mae = (a, b) => (a === undefined ? "—" : `${fmt(a, 2)} vs ${fmt(b, 2)} ${u}`);
    $("model-info").innerHTML = `<dl>
      <dt>Status</dt><dd>trained (${esc(m.source)})</dd><dt>Stations</dt><dd>${esc((m.stations || []).join(", "))}</dd>
      <dt>Data</dt><dd>${esc((m.date_range || []).map((s) => s.slice(0, 10)).join(" → "))}</dd>
      <dt>Pairs</dt><dd>${m.n_native_pairs} real · ${m.n_interpolated_pairs} time-interp.</dd>
      <dt>MAE real</dt><dd>${mae(x.mae_nn_all_native, x.mae_persistence_all_native)}</dd>
      <dt>MAE all</dt><dd>${mae(x.mae_nn_all, x.mae_persistence_all)}</dd>
      <dt>Trained</dt><dd>${esc(m.trained_at)} · best ep ${x.best_epoch ?? "—"}/${m.epochs}</dd></dl>
      <p class="muted">MAE = validation error, NN vs persistence (held-out, later dates).</p>`;
  }

  // ------------------------------------------------------------------ SVG helpers
  function el(tag, attrs, parent, text) {
    const e = document.createElementNS(SVGNS, tag);
    Object.entries(attrs || {}).forEach(([k, v]) => e.setAttribute(k, v));
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  const P_TOP = 100, P_BOT = 1050, TOP = 15, BOT = 530;
  const yP = (p) => TOP + ((Math.log(p) - Math.log(P_TOP)) / (Math.log(P_BOT) - Math.log(P_TOP))) * (BOT - TOP);
  const ISOBARS = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100];

  function drawSkewT(svg, d) {
    svg.innerHTML = "";
    const L = 40, R = 490, TMIN = -40, TMAX = 50, sx = (R - L) / (TMAX - TMIN);
    const xT = (t, p) => L + (t - TMIN) * sx + (BOT - yP(p));
    const clip = el("clipPath", { id: "skewt-clip" }, el("defs", {}, svg));
    el("rect", { x: L, y: TOP, width: R - L, height: BOT - TOP }, clip);
    const g = el("g", { "clip-path": "url(#skewt-clip)" }, svg);
    for (let t = -130; t <= 50; t += 10) el("line", { x1: xT(t, P_BOT), y1: yP(P_BOT), x2: xT(t, P_TOP), y2: yP(P_TOP), class: `isotherm${t === 0 ? " zero" : ""}` }, g);
    for (let th = 250; th <= 460; th += 10) {
      const pts = [];
      for (let p = P_BOT; p >= P_TOP; p -= 25) pts.push(`${xT(th * Math.pow(p / 1000, 0.2857) - 273.15, p)},${yP(p)}`);
      el("polyline", { points: pts.join(" "), class: "adiabat" }, g);
    }
    ISOBARS.forEach((p) => { el("line", { x1: L, y1: yP(p), x2: R, y2: yP(p), class: "grid-line" }, svg); el("text", { x: L - 4, y: yP(p) + 3, "text-anchor": "end", class: "isobar-label" }, svg, p); });
    for (let t = -40; t <= 50; t += 10) el("text", { x: xT(t, P_BOT), y: BOT + 14, "text-anchor": "middle", class: "axis-label" }, svg, t);
    el("rect", { x: L, y: TOP, width: R - L, height: BOT - TOP, fill: "none", stroke: "#2a3a4c" }, svg);

    const o = d.observed_sounding, trace = (key) => o.PRES.map((p, i) => [p, o[key][i]]).filter(([p, v]) => p !== null && v !== null && p >= P_TOP);
    const line = (pairs, cls) => pairs.length > 1 && el("polyline", { points: pairs.map(([p, t]) => `${xT(t, p)},${yP(p)}`).join(" "), class: cls }, g);
    line(trace("TEMP"), "trace temp");
    line(trace("DWPT"), "trace dew");
    if (d.component === "temperature" || d.component === "dewpoint") {
      const fc = d.profile.filter((r) => !r.below_ground && r.forecast !== null).map((r) => [r.pressure_hpa, r.forecast]);
      line(fc, "trace forecast");
      fc.forEach(([p, t]) => el("circle", { cx: xT(t, p), cy: yP(p), r: 3, class: "pt forecast" }, g));
    }
    // wind barbs (observed) on the right margin
    let lastY = -99;
    o.PRES.forEach((p, i) => {
      if (p === null || p < P_TOP || o.DRCT[i] === null || o.SKNT[i] === null || Math.abs(yP(p) - lastY) < 16) return;
      lastY = yP(p);
      windBarb(svg, 525, yP(p), o.DRCT[i], o.SKNT[i]);
    });
    const lg = [["Temp (obs)", "var(--temp)", ""], ["Dewpt (obs)", "var(--dew)", ""]];
    if (d.component === "temperature" || d.component === "dewpoint") lg.push([`${d.label} +${d.lead_hours}h`, "var(--fc)", "6 4"]);
    lg.forEach(([name, color, dash], i) => {
      el("line", { x1: L + 10, y1: TOP + 14 + i * 15, x2: L + 32, y2: TOP + 14 + i * 15, stroke: color, "stroke-width": 2.5, "stroke-dasharray": dash }, svg);
      el("text", { x: L + 38, y: TOP + 18 + i * 15, class: "legend" }, svg, name);
    });
  }

  function windBarb(svg, x, y, dir, spd) {
    const g = el("g", { class: "barb" }, svg);
    if (spd < 2.5) { el("circle", { cx: x, cy: y, r: 3, fill: "none" }, g); return; }
    const r = (dir * Math.PI) / 180, ux = Math.sin(r), uy = -Math.cos(r), px = Math.cos(r), py = Math.sin(r), len = 26;
    const at = (s) => [x + ux * s, y + uy * s];
    el("line", { x1: x, y1: y, x2: at(len)[0], y2: at(len)[1] }, g);
    let s = Math.round(spd / 5) * 5, pos = len;
    while (s >= 50) { const [a, b] = at(pos), [c, e] = at(pos - 6); el("polygon", { points: `${a},${b} ${a + px * 10},${b + py * 10} ${c},${e}` }, g); pos -= 8; s -= 50; }
    while (s >= 10) { const [a, b] = at(pos); el("line", { x1: a, y1: b, x2: a + px * 10 + ux * 3, y2: b + py * 10 + uy * 3 }, g); pos -= 4; s -= 10; }
    if (s >= 5) { if (pos === len) pos -= 4; const [a, b] = at(pos); el("line", { x1: a, y1: b, x2: a + px * 5 + ux * 1.5, y2: b + py * 5 + uy * 1.5 }, g); }
  }

  function drawProfile(svg, d) {
    svg.innerHTML = "";
    const L = 52, R = 400, rows = d.profile.filter((r) => !r.below_ground && r.observed !== null);
    const vals = rows.flatMap((r) => [r.observed, r.forecast]).filter((v) => v !== null);
    if (!vals.length) { el("text", { x: 210, y: 280, "text-anchor": "middle", class: "empty-chart" }, svg, "No data"); return; }
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (d.component === "wind_direction") { lo = 0; hi = 360; } else { const pad = Math.max((hi - lo) * 0.1, 1); lo -= pad; hi += pad; }
    const xV = (v) => L + ((v - lo) / (hi - lo)) * (R - L);
    ISOBARS.forEach((p) => { el("line", { x1: L, y1: yP(p), x2: R, y2: yP(p), class: "grid-line" }, svg); el("text", { x: L - 4, y: yP(p) + 3, "text-anchor": "end", class: "isobar-label" }, svg, p); });
    const raw = (hi - lo) / 5, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = d.component === "wind_direction" ? 90 : [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {  // "nice" tick values
      el("line", { x1: xV(v), y1: TOP, x2: xV(v), y2: BOT, class: "grid-line" }, svg);
      el("text", { x: xV(v), y: BOT + 14, "text-anchor": "middle", class: "axis-label" }, svg, +v.toFixed(2));
    }
    el("rect", { x: L, y: TOP, width: R - L, height: BOT - TOP, fill: "none", stroke: "#2a3a4c" }, svg);
    const sel = rows.find((r) => r.level === d.headline.level);
    if (sel) el("rect", { x: L, y: yP(sel.pressure_hpa) - 7, width: R - L, height: 14, fill: "rgba(56,189,248,.12)" }, svg);
    [["observed", "observed"], ["forecast", "forecast"]].forEach(([key, cls]) => {
      const pts = rows.filter((r) => r[key] !== null);
      el("polyline", { points: pts.map((r) => `${xV(r[key])},${yP(r.pressure_hpa)}`).join(" "), class: `trace ${cls}` }, svg);
      pts.forEach((r) => el("circle", { cx: xV(r[key]), cy: yP(r.pressure_hpa), r: 3, class: `pt ${cls}` }, svg));
    });
    [["Observed (init)", "#c8d6e3", ""], [`Forecast +${d.lead_hours}h`, "var(--fc)", "6 4"]].forEach(([name, color, dash], i) => {
      el("line", { x1: R - 130, y1: TOP + 14 + i * 15, x2: R - 108, y2: TOP + 14 + i * 15, stroke: color, "stroke-width": 2.5, "stroke-dasharray": dash }, svg);
      el("text", { x: R - 102, y: TOP + 18 + i * 15, class: "legend" }, svg, name);
    });
    el("text", { x: (L + R) / 2, y: BOT + 28, "text-anchor": "middle", class: "axis-label" }, svg, `${d.label} (${d.units})`);
  }

  // ------------------------------------------------------------------ wiring
  function init() {
    LEADS.forEach((h) => {
      const c = document.createElement("button");
      c.type = "button"; c.className = "chip"; c.dataset.lead = h; c.textContent = `+${h}`;
      c.addEventListener("click", () => setLead(h));
      $("lead-chips").appendChild(c);
    });
    $("level-select").innerHTML = LEVELS.map((l) => `<option>${l}</option>`).join("");
    $("level-select").addEventListener("change", (e) => setLevel(e.target.value));
    $("lead-range").addEventListener("input", (e) => setLead(e.target.value));
    $("lead-number").addEventListener("change", (e) => setLead(e.target.value));
    document.querySelectorAll("#source-toggle button").forEach((b) => b.addEventListener("click", () => setSource(b.dataset.source)));
    $("init-latest").addEventListener("change", (e) => { $("init-picker").hidden = e.target.checked; });
    $("init-date").value = new Date().toISOString().slice(0, 10);
    $("run").addEventListener("click", runForecast);
    $("command").addEventListener("input", (e) => { clearTimeout(resolveTimer); resolveTimer = setTimeout(() => resolveCommand(e.target.value, false), 250); });
    $("command-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const parsed = await resolveCommand($("command").value, true);
      if (parsed && parsed.component) runForecast();
    });
    setInterval(() => { $("utc-clock").textContent = `${new Date().toISOString().slice(11, 19)} Z`; }, 1000);
    setLead(2);
    setLevel("SFC");
    loadStations();
    setSource("wyoming");
    api("GET", "/hud/health").then(({ data }) => log(`connected: ${data.service} via ${data.backend}`)).catch((e) => log(`health: ${e.message}`, true));
  }
  document.addEventListener("DOMContentLoaded", init);
})();
