"use strict";

// Shape definitions mirror app/deck.py (make_layout / SHAPE_DIMS); the server re-validates.
// Plan coordinates are feet: x to the right, z away from the house.
const SHAPES = {
  rect: {
    name: "Rectangle",
    defaults: { A: 12, B: 16 },
    controls: [
      { key: "A", badges: ["A", "C"], name: "Depth" },
      { key: "B", badges: ["B"], name: "Width" },
    ],
    limits: () => ({ A: [6, 30], B: [8, 40] }),
    front: "B",
  },
  angled: {
    name: "45° Angled",
    defaults: { A: 10, B: 6, C: 10 },
    controls: [
      { key: "A", badges: ["A", "E"], name: "Sides" },
      { key: "B", badges: ["B", "D"], name: "Corners" },
      { key: "C", badges: ["C"], name: "Front" },
    ],
    limits: (d) => ({
      A: [4, 24],
      B: [2, Math.min(10, Math.floor((48 - d.C) / Math.SQRT2))],
      C: [6, Math.min(30, Math.floor(48 - d.B * Math.SQRT2))],
    }),
    front: "C",
  },
  l_left: {
    name: "L-shape · Left",
    defaults: { A: 16, B: 20, C: 8, D: 10 },
    controls: [
      { key: "A", badges: ["A"], name: "Left side" },
      { key: "B", badges: ["B"], name: "Front" },
      { key: "C", badges: ["C"], name: "Right side" },
      { key: "D", badges: ["D"], name: "Back" },
    ],
    limits: (d) => ({
      A: [Math.max(10, d.C + 4), 30],
      B: [Math.max(12, d.D + 4), 40],
      C: [4, Math.min(26, d.A - 4)],
      D: [4, Math.min(36, d.B - 4)],
    }),
    front: "B",
  },
  l: {
    name: "L-shape · Right",
    defaults: { A: 8, B: 20, C: 16, D: 10 },
    controls: [
      { key: "A", badges: ["A"], name: "Left side" },
      { key: "B", badges: ["B"], name: "Front" },
      { key: "C", badges: ["C"], name: "Right side" },
      { key: "D", badges: ["D"], name: "Back" },
    ],
    limits: (d) => ({
      A: [4, Math.min(26, d.C - 4)],
      B: [Math.max(12, d.D + 4), 40],
      C: [Math.max(10, d.A + 4), 30],
      D: [4, Math.min(36, d.B - 4)],
    }),
    front: "B",
  },
  u: {
    name: "U-shape",
    defaults: { A: 16, B: 28, D: 8, E: 8, wall: 8 },
    controls: [
      { key: "A", badges: ["A", "C"], name: "Sides" },
      { key: "B", badges: ["B"], name: "Front" },
      { key: "E", badges: ["E"], name: "Back left" },
      { key: "D", badges: ["D"], name: "Back right" },
      { key: "wall", badges: [], name: "House depth", house: true },
    ],
    limits: (d) => ({
      A: [Math.max(10, d.wall + 4), 30],
      B: [Math.max(16, d.D + d.E + 6), 48],
      D: [4, Math.min(20, d.B - d.E - 6)],
      E: [4, Math.min(20, d.B - d.D - 6)],
      wall: [4, Math.min(26, d.A - 4)],
    }),
    front: "B",
  },
};

// Mirrors app/deck.py
const STAIR_WIDTH = 4;
const STAIR_MIN_EDGE = 5.5;
const STAIR_RUN = 10 / 12;
const HEIGHTS = [2, 3, 4, 6];

const state = { step: 1, shape: null, dims: {}, height: 3, stairEdge: null, model: null };

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const arSupported = (() => {
  const a = document.createElement("a");
  return !!(a.relList && a.relList.supports && a.relList.supports("ar"));
})();
// Android has no Quick Look; its counterpart is Google's Scene Viewer, opened with an intent link
const isAndroid = /Android/i.test(navigator.userAgent);

// ---------- geometry ----------

function layout(shape = state.shape, d = state.dims) {
  if (shape === "rect") {
    return { pts: [[0, 0], [0, d.A], [d.B, d.A], [d.B, 0]], labels: ["A", "B", "C", null], house: null };
  }
  if (shape === "l") {
    const w = d.B, dep = d.C, hw = w - d.D, hd = dep - d.A;
    return {
      pts: [[0, hd], [0, dep], [w, dep], [w, 0], [hw, 0], [hw, hd]],
      labels: ["A", "B", "C", "D", null, null],
      house: [0, 0, hw, hd],
    };
  }
  if (shape === "l_left") {
    // Mirror image of "l": the house sits in the top-right notch
    const w = d.B, dep = d.A, hx = d.D, hd = dep - d.C;
    return {
      pts: [[0, 0], [0, dep], [w, dep], [w, hd], [hx, hd], [hx, 0]],
      labels: ["A", "B", "C", null, null, "D"],
      house: [hx, 0, w, hd],
    };
  }
  if (shape === "u") {
    const w = d.B, dep = d.A, left = d.E, right = w - d.D;
    return {
      pts: [[0, 0], [0, dep], [w, dep], [w, 0], [right, 0], [right, d.wall], [left, d.wall], [left, 0]],
      labels: ["A", "B", "C", "D", null, null, null, "E"],
      house: [left, 0, right, d.wall],
    };
  }
  const c = d.B / Math.SQRT2, w = d.C + 2 * c;
  return {
    pts: [[0, 0], [0, d.A], [c, d.A + c], [w - c, d.A + c], [w, d.A], [w, 0]],
    labels: ["A", "B", "C", "D", "E", null],
    house: null,
  };
}

function area(pts) {
  let a = 0;
  pts.forEach((p, i) => {
    const q = pts[(i + 1) % pts.length];
    a += p[0] * q[1] - q[0] * p[1];
  });
  return a / 2;
}

function edgeInfo(lay) {
  const sign = area(lay.pts) > 0 ? 1 : -1;
  return lay.pts.map((a, i) => {
    const b = lay.pts[(i + 1) % lay.pts.length];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const u = [(b[0] - a[0]) / len, (b[1] - a[1]) / len];
    const out = [u[1] * sign, -u[0] * sign]; // outward normal
    return { i, a, b, len, u, out, mid: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], label: lay.labels[i] };
  });
}

function stairRisers(height = state.height) {
  return Math.max(2, Math.ceil((height * 12) / 7.5 - 1e-9));
}

function stairPolygon(edge) {
  const depth = (stairRisers() - 1) * STAIR_RUN + 0.3;
  const h = STAIR_WIDTH / 2;
  const p = (s, t) => [edge.mid[0] + edge.u[0] * s + edge.out[0] * t, edge.mid[1] + edge.u[1] * s + edge.out[1] * t];
  const treads = [];
  for (let k = 1; k < stairRisers(); k++) treads.push([p(-h, k * STAIR_RUN), p(h, k * STAIR_RUN)]);
  return { poly: [p(-h, 0), p(h, 0), p(h, depth), p(-h, depth)], treads };
}

function stairEdges() {
  return edgeInfo(layout()).filter((e) => e.label);
}

// Why an edge can't take stairs (mirrors Layout.stair_problem), or null if it can
function stairProblem(edge) {
  if (Math.abs(edge.u[0]) > 1e-9 && Math.abs(edge.u[1]) > 1e-9) return "Angled side";
  if (edge.len < STAIR_MIN_EDGE - 1e-6) return "Too short";
  return null;
}

function canHoldStairs(edge) {
  return !stairProblem(edge);
}

function fmtFt(ft) {
  const inches = Math.round(ft * 12);
  const f = Math.floor(inches / 12), i = inches % 12;
  return i ? `${f}′ ${i}″` : `${f}′`;
}

// ---------- plan drawing ----------

// Self-contained (no page CSS) so the same SVG also works as the <img> thumbnail, light and dark
const PLAN_STYLE = `<style>
  svg{--bg:#ffffff;--deck:#eaf2ff;--ink:#1d1d1f;--board:rgba(29,29,31,.12);--house:#e5e5ea;--wall:#8e8e93;--label:#6e6e73;--accent:#007aff}
  @media (prefers-color-scheme: dark){
    svg{--bg:#1c1c1e;--deck:#10243d;--ink:#f5f5f7;--board:rgba(245,245,247,.12);--house:#2c2c2e;--wall:#8e8e93;--label:#98989f;--accent:#0a84ff}
  }
  .plan-bg{fill:var(--bg)}
  .deck{fill:var(--deck);stroke:var(--ink);stroke-width:1.75;stroke-linejoin:round}
  .boards{stroke:var(--board);stroke-width:.6;fill:none}
  .house-fill{fill:var(--house)}
  .house{stroke:var(--wall);stroke-width:4;stroke-linecap:round}
  .house-label{fill:var(--label);font:600 10px -apple-system,sans-serif;letter-spacing:.08em}
  .stairs{fill:var(--bg);stroke:var(--ink);stroke-width:1.5}
  .tread{stroke:var(--ink);stroke-width:1;opacity:.5}
  .edge-on{stroke:var(--accent);stroke-width:4;stroke-linecap:round}
  .badge circle{fill:var(--bg);stroke:var(--ink);stroke-width:1.5}
  .badge text{fill:var(--ink);font:700 11px -apple-system,sans-serif}
  .badge.on circle{fill:var(--accent);stroke:var(--accent)}
  .badge.on text{fill:#ffffff}
  .badge.off{opacity:.3}
  .badge.pick{cursor:pointer}
</style>`;

function planSvg({ stairs = false, pick = false, standalone = false } = {}) {
  const lay = layout();
  const edges = edgeInfo(lay);
  const stairEdge = stairs && state.stairEdge && edges.find((e) => e.label === state.stairEdge);
  const stairShape = stairEdge ? stairPolygon(stairEdge) : null;

  // Fit everything (deck, house band, stairs) into the view
  const W = 320, H = 250, pad = 30, houseBand = 22;
  const all = [...lay.pts, ...(stairShape ? stairShape.poly : [])];
  const minX = Math.min(...all.map((p) => p[0])), maxX = Math.max(...all.map((p) => p[0]));
  const minZ = Math.min(...all.map((p) => p[1])), maxZ = Math.max(...all.map((p) => p[1]));
  const s = Math.min((W - 2 * pad) / (maxX - minX), (H - 2 * pad - houseBand) / (maxZ - minZ));
  const ox = (W - (maxX - minX) * s) / 2 - minX * s;
  const oy = pad + houseBand + (H - 2 * pad - houseBand - (maxZ - minZ) * s) / 2 - minZ * s;
  const X = (x) => +(ox + x * s).toFixed(1), Y = (z) => +(oy + z * s).toFixed(1);
  const pts = (poly) => poly.map(([x, z]) => `${X(x)},${Y(z)}`).join(" ");

  // House: a gray band behind the deck, plus the notch it sits in for L / U shapes
  const deckW = Math.max(...lay.pts.map((p) => p[0]));
  let house = `<rect class="house-fill" x="${X(0) - 12}" y="${Y(0) - houseBand}" width="${X(deckW) - X(0) + 24}" height="${houseBand}"/>`;
  let houseLabel = [(X(0) + X(deckW)) / 2, Y(0) - houseBand / 2 + 4];
  if (lay.house) {
    const [x0, z0, x1, z1] = lay.house;
    const [extL, extR] = [x0 === 0 ? 12 : 0, x1 === deckW ? 12 : 0]; // the house continues past the deck's edge
    house = `<rect class="house-fill" x="${X(x0) - extL}" y="${Y(z0) - houseBand}" width="${X(x1) - X(x0) + extL + extR}" height="${Y(z1) - Y(z0) + houseBand}"/>`;
    houseLabel = [(X(x0) + X(x1) - extL + extR) / 2, (Y(z0) - houseBand + Y(z1)) / 2 + 4];
  }
  const walls = edges
    .filter((e) => !e.label)
    .map((e) => `<line class="house" x1="${X(e.a[0])}" y1="${Y(e.a[1])}" x2="${X(e.b[0])}" y2="${Y(e.b[1])}"/>`)
    .join("");

  // Decking lines clipped to the deck
  const boards = [];
  const step = (5.625 / 12) * s;
  if (step > 2.5) for (let y = Y(minZ) + step; y < Y(maxZ); y += step) boards.push(`M${X(minX)} ${y.toFixed(1)}H${X(maxX)}`);
  const clipId = `clip-${Math.random().toString(36).slice(2, 8)}`;

  let stairsSvg = "";
  if (stairShape) {
    stairsSvg = `<polygon class="stairs" points="${pts(stairShape.poly)}"/>` +
      stairShape.treads.map(([a, b]) => `<line class="tread" x1="${X(a[0])}" y1="${Y(a[1])}" x2="${X(b[0])}" y2="${Y(b[1])}"/>`).join("");
  }
  const highlight = stairEdge
    ? `<line class="edge-on" x1="${X(stairEdge.a[0])}" y1="${Y(stairEdge.a[1])}" x2="${X(stairEdge.b[0])}" y2="${Y(stairEdge.b[1])}"/>`
    : "";

  // Letter badges just outside each open edge; on the stair edge, slid along it to clear the stairs
  const badges = edges
    .filter((e) => e.label)
    .map((e) => {
      const along = stairShape && e === stairEdge ? STAIR_WIDTH / 2 + 12 / s : 0;
      const cx = X(e.mid[0] + e.out[0] * (13 / s) + e.u[0] * along);
      const cy = Y(e.mid[1] + e.out[1] * (13 / s) + e.u[1] * along);
      const cls = ["badge"];
      if (e.label === state.stairEdge && stairs) cls.push("on");
      if (pick) cls.push(canHoldStairs(e) ? "pick" : "off");
      return `<g class="${cls.join(" ")}" data-edge="${e.label}"><circle cx="${cx}" cy="${cy}" r="9"/><text x="${cx}" y="${cy + 4}" text-anchor="middle">${e.label}</text></g>`;
    })
    .join("");

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Deck plan"${standalone ? ` width="${W * 2}" height="${H * 2}"` : ""}>
    ${PLAN_STYLE}
    ${standalone ? `<rect class="plan-bg" width="${W}" height="${H}"/>` : ""}
    <defs><clipPath id="${clipId}"><polygon points="${pts(lay.pts)}"/></clipPath></defs>
    ${house}
    <text class="house-label" x="${houseLabel[0]}" y="${houseLabel[1]}" text-anchor="middle">HOUSE</text>
    ${stairsSvg}
    <polygon class="deck" points="${pts(lay.pts)}"/>
    <path class="boards" clip-path="url(#${clipId})" d="${boards.join("")}"/>
    ${walls}${highlight}${badges}
  </svg>`;
}

// ---------- navigation ----------

function go(step) {
  state.step = step;
  $$(".step").forEach((s) => (s.hidden = Number(s.dataset.step) !== step));
  $$("#progress li").forEach((li, i) => li.classList.toggle("on", i < step));
  $("#back").hidden = step === 1;
  if (step !== 5) cancelPhotoRequests(); // leaving the photo screen cancels pending edits
  // Back on the photo screen with another deck shape: it may snap to something else, so look again;
  // switching between the two L-shapes keeps the analysis but picks a corner that turns the new way
  if (step === 5 && photo.source && photo.kind && photo.kind !== snapKind()) analyzeWalls();
  else if (step === 5 && photo.walls && photo.shape !== state.shape) {
    photo.shape = state.shape;
    choosePlacement();
  }
  ({ 2: renderDesign, 3: renderStairs, 4: renderReview, 5: renderPhoto })[step]?.();
  window.scrollTo({ top: 0 });
}

$("#back").addEventListener("click", () => go(Math.max(1, state.step - 1)));

// ---------- step 1: shape ----------

function renderShapes() {
  const box = $("#shapes");
  box.innerHTML = "";
  for (const [key, def] of Object.entries(SHAPES)) {
    const btn = document.createElement("button");
    btn.className = "card choice";
    btn.dataset.shape = key;
    btn.innerHTML = `<span class="shape-thumb">${thumbSvg(key)}</span>
      <span class="card-title">${def.name}</span>`;
    btn.addEventListener("click", () => {
      if (state.shape !== key) {
        state.shape = key;
        state.dims = { ...def.defaults };
        state.stairEdge = def.front;
      }
      $$("[data-shape]").forEach((b) => b.classList.toggle("selected", b === btn));
      go(2);
    });
    box.appendChild(btn);
  }
}

function thumbSvg(shape) {
  const lay = layout(shape, SHAPES[shape].defaults);
  const xs = lay.pts.map((p) => p[0]), zs = lay.pts.map((p) => p[1]);
  const w = Math.max(...xs), d = Math.max(...zs);
  const s = Math.min(96 / w, 58 / d);
  const ox = (120 - w * s) / 2, oy = 22;
  const P = (poly) => poly.map(([x, z]) => `${(ox + x * s).toFixed(1)},${(oy + z * s).toFixed(1)}`).join(" ");
  let house = `<rect class="t-house" x="${ox - 8}" y="${oy - 12}" width="${w * s + 16}" height="12"/>`;
  if (lay.house) {
    const [x0, z0, x1, z1] = lay.house;
    const [extL, extR] = [x0 === 0 ? 8 : 0, x1 === w ? 8 : 0];
    house = `<rect class="t-house" x="${ox + x0 * s - extL}" y="${oy - 12}" width="${(x1 - x0) * s + extL + extR}" height="${(z1 - z0) * s + 12}"/>`;
  }
  return `<svg viewBox="0 0 120 90" class="shape-icon">${house}<polygon class="t-deck" points="${P(lay.pts)}"/></svg>`;
}

// ---------- step 2: design ----------

function clampDims() {
  // Re-clamp until stable: limits depend on the other dimensions
  const def = SHAPES[state.shape];
  for (let pass = 0; pass < 4; pass++) {
    const lim = def.limits(state.dims);
    for (const [k, [lo, hi]] of Object.entries(lim)) state.dims[k] = Math.min(Math.max(state.dims[k], lo), hi);
  }
}

function renderDesign() {
  const def = SHAPES[state.shape];
  $("#design-plan").innerHTML = planSvg();

  const lim = def.limits(state.dims);
  const fields = $("#fields");
  fields.innerHTML = "";
  for (const ctl of def.controls) {
    const el = document.createElement("div");
    el.className = "field";
    const badges = ctl.house
      ? `<span class="badge-chip house">⌂</span>`
      : ctl.badges.map((b) => `<span class="badge-chip">${b}</span>`).join("");
    el.innerHTML = `
      <div class="field-label">${badges}<span>${ctl.name}</span></div>
      <div class="stepper">
        <button type="button" aria-label="Decrease ${ctl.name}">−</button>
        <output>${state.dims[ctl.key]}′</output>
        <button type="button" aria-label="Increase ${ctl.name}">+</button>
      </div>`;
    const [dec, inc] = $$("button", el);
    const [lo, hi] = lim[ctl.key];
    dec.disabled = state.dims[ctl.key] <= lo;
    inc.disabled = state.dims[ctl.key] >= hi;
    const bump = (delta) => {
      state.dims[ctl.key] = Math.min(Math.max(state.dims[ctl.key] + delta, lo), hi);
      clampDims();
      renderDesign();
    };
    dec.addEventListener("click", () => bump(-1));
    inc.addEventListener("click", () => bump(1));
    fields.appendChild(el);
  }

  const heights = $("#heights");
  heights.innerHTML = "";
  HEIGHTS.forEach((h) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = `${h} ft`;
    b.classList.toggle("selected", state.height === h);
    b.addEventListener("click", () => {
      state.height = h;
      renderDesign();
    });
    heights.appendChild(b);
  });
}

$("#to-stairs").addEventListener("click", () => go(3));

// ---------- step 3: stairs ----------

function ensureStairEdge() {
  // null means "no stairs" and is kept; a side that can no longer take stairs falls back to the front
  if (!state.stairEdge) return;
  const edges = stairEdges();
  const current = edges.find((e) => e.label === state.stairEdge);
  if (current && canHoldStairs(current)) return;
  const front = edges.find((e) => e.label === SHAPES[state.shape].front && canHoldStairs(e));
  state.stairEdge = (front || edges.find(canHoldStairs) || {}).label || null;
}

function renderStairs() {
  ensureStairEdge();
  $("#gen-error").hidden = true;
  const plan = $("#stairs-plan");
  plan.innerHTML = planSvg({ stairs: true, pick: true });
  $$(".badge.pick", plan).forEach((g) => g.addEventListener("click", () => pickStairs(g.dataset.edge)));

  const box = $("#edge-choices");
  box.innerHTML = "";
  const none = document.createElement("button");
  none.type = "button";
  none.className = "edge-choice";
  none.classList.toggle("selected", !state.stairEdge);
  none.textContent = "No stairs";
  none.addEventListener("click", () => pickStairs(null));
  box.appendChild(none);
  for (const e of stairEdges()) {
    const problem = stairProblem(e);
    const b = document.createElement("button");
    b.type = "button";
    b.className = "edge-choice";
    b.disabled = !!problem;
    b.classList.toggle("selected", e.label === state.stairEdge);
    b.innerHTML = `<span class="badge-chip">${e.label}</span><span>${fmtFt(e.len)}</span>${problem ? `<small>${problem}</small>` : ""}`;
    b.addEventListener("click", () => pickStairs(e.label));
    box.appendChild(b);
  }
}

function pickStairs(label) {
  // Tapping the selected side again removes the stairs
  state.stairEdge = label === state.stairEdge ? null : label;
  renderStairs();
}

// Generating the model happens on the way to the review step
$("#to-review").addEventListener("click", async () => {
  const btn = $("#to-review");
  btn.disabled = true;
  btn.textContent = "Generating model…";
  $("#gen-error").hidden = true;
  try {
    const res = await fetch("/api/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ shape: state.shape, dims: state.dims, height_ft: state.height, stair_edge: state.stairEdge }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail?.[0]?.msg || `Server returned ${res.status}`);
    }
    state.model = await res.json();
    go(4);
  } catch (err) {
    $("#gen-error").textContent = `Could not generate the model. ${err.message}`;
    $("#gen-error").hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = "Continue";
  }
});

// ---------- step 4: review ----------

function renderReview() {
  loadPrompts();
  $("#ar-link").href = isAndroid ? sceneViewerUrl(state.model.glb_url) : `${state.model.url}#allowsContentScaling=0`;
  // Scene Viewer only loads models over HTTPS
  $("#ar-https").hidden = !(isAndroid && location.protocol !== "https:");
  $("#ar-thumb").src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(planSvg({ stairs: true, standalone: true }));
}

// Quick Look reads options from the URL fragment; allowsContentScaling=0 locks the model at true scale.
// Scene Viewer takes them as query parameters: resizable=false does the same.
// The intent goes to the Google app, which Android phones sold in the US come with (and which
// <model-viewer> uses too). ar_preferred opens straight into AR; where Google Play Services for AR is
// missing, Scene Viewer offers to install it or shows the model in 3D instead. (Sending the intent to
// com.google.ar.core in ar_only mode skips the Google app, but fails outright without ARCore.)
// Without the Google app, the browser follows the fallback back to this page with #ar-unavailable,
// which the diagnostics below pick up.
const AR_UNAVAILABLE_HASH = "#ar-unavailable";
function sceneViewerUrl(glbUrl) {
  const params = new URLSearchParams({ file: new URL(glbUrl, location.href).href, mode: "ar_preferred", resizable: "false", title: "Deck" });
  const fallback = encodeURIComponent(location.href.split("#")[0] + AR_UNAVAILABLE_HASH);
  return `intent://arvr.google.com/scene-viewer/1.0?${params}#Intent;scheme=https;package=com.google.android.googlequicksearchbox;action=android.intent.action.VIEW;S.browser_fallback_url=${fallback};end;`;
}

// ---------- AR diagnostics (Android) ----------
// On Android, AR Preview hands the page to Scene Viewer through an intent link. When that works the
// page goes to the background within a moment; when it doesn't (no Google app, a browser that
// ignores intent links, ...) nothing visible happens. Either way the page reports what it saw to the
// server, which appends it to generated/diagnostics.jsonl next to every model download. The outcome
// goes out as soon as it is known and the slower environment checks follow in a second report with
// the same ID, so a quick outcome never cuts them short.
const AR_OPEN_TIMEOUT_MS = 3000;
const IN_APP_BROWSER = /MicroMessenger|WeChat|QQ\/|MQQBrowser|UCBrowser|baiduboxapp|Quark|DingTalk|Lark|Feishu|HuaweiBrowser|HeyTapBrowser|MiuiBrowser|VivoBrowser|SamsungBrowser|Weibo/i;

$("#ar-link").addEventListener("click", () => {
  if (isAndroid) diagnoseArLaunch();
});

function diagnoseArLaunch() {
  const id = Math.random().toString(36).slice(2, 8);
  const t0 = performance.now();
  const events = [];
  let sent = false;
  collectArEnv().then((env) => postDiag({ id, kind: "env", ...env }));

  const send = (outcome) => {
    if (sent) return;
    sent = true;
    for (const [type, fn] of listeners) window.removeEventListener(type, fn), document.removeEventListener(type, fn);
    postDiag({ id, kind: "outcome", outcome, events });
  };
  const mark = (name) => {
    events.push({ name, ms: Math.round(performance.now() - t0) });
    if (name === "visibility:hidden" || name === "pagehide") send("left_page");
  };
  // The intent's fallback brings the browser back here with #ar-unavailable: no Google app
  const onFallback = () => {
    if (location.hash !== AR_UNAVAILABLE_HASH) return;
    mark("fallback");
    history.replaceState(null, "", location.pathname + location.search);
    send("fallback");
    $("#ar-diag").textContent = `AR isn't available on this phone: the Google app is missing (ID ${id}).`;
    $("#ar-diag").hidden = false;
  };
  const listeners = [
    ["visibilitychange", () => mark(`visibility:${document.visibilityState}`)],
    ["pagehide", () => mark("pagehide")],
    ["blur", () => mark("blur")],
    ["hashchange", onFallback],
  ];
  for (const [type, fn] of listeners) (type === "visibilitychange" ? document : window).addEventListener(type, fn);

  $("#ar-diag").hidden = true;
  setTimeout(() => {
    if (sent) return;
    send("stayed");
    $("#ar-diag").textContent = `AR didn't open. A diagnostic report was sent (ID ${id}).`;
    $("#ar-diag").hidden = false;
  }, AR_OPEN_TIMEOUT_MS);
}

async function postDiag(report) {
  const body = JSON.stringify(report);
  // keepalive / sendBeacon so the report survives the page being backgrounded or closed
  try {
    await fetch("/api/diag", { method: "POST", headers: { "Content-Type": "application/json" }, body, keepalive: true });
  } catch {
    navigator.sendBeacon?.("/api/diag", new Blob([body], { type: "application/json" }));
  }
}

async function collectArEnv() {
  const glb = new URL(state.model.glb_url, location.href).href;
  const env = {
    page: location.href,
    secureContext: window.isSecureContext,
    ua: navigator.userAgent,
    inAppBrowser: IN_APP_BROWSER.test(navigator.userAgent) ? navigator.userAgent.match(IN_APP_BROWSER)[0] : null,
    language: navigator.language,
    screen: `${screen.width}x${screen.height}@${devicePixelRatio}`,
    intent: $("#ar-link").href,
    glb,
  };
  const probes = [
    // Can the model be downloaded at all, and with what type? (marked so the server log tells it apart)
    (async () => {
      const t = performance.now();
      try {
        const r = await fetch(`${glb}?probe=1`, { method: "HEAD", cache: "no-store" });
        env.glbProbe = { status: r.status, type: r.headers.get("content-type"), bytes: r.headers.get("content-length"), ms: Math.round(performance.now() - t) };
      } catch (err) {
        env.glbProbe = { error: String(err) };
      }
    })(),
    // Browser and phone model, where the browser offers them
    (async () => {
      try {
        const d = navigator.userAgentData;
        if (d) env.uaData = { mobile: d.mobile, platform: d.platform, ...(await d.getHighEntropyValues(["model", "platformVersion", "fullVersionList"])) };
      } catch (err) {
        env.uaData = { error: String(err) };
      }
    })(),
    // WebXR AR support in Chrome implies ARCore (Google Play Services for AR) is installed
    (async () => {
      try {
        env.webxrAr = navigator.xr ? await navigator.xr.isSessionSupported("immersive-ar") : "no navigator.xr";
      } catch (err) {
        env.webxrAr = String(err);
      }
    })(),
  ];
  // A check that hangs shouldn't hold back the rest: send what is known after 5 s
  const done = await Promise.race([Promise.all(probes).then(() => true), new Promise((r) => setTimeout(() => r(false), 5000))]);
  if (!done) env.incomplete = true;
  return env;
}

// ---------- step 5: show in photo ----------
// Mirrors ARDeckAIViewModel on iOS: the first result for a photo is a "design" (the master);
// other styles restyle that master so the deck keeps its shape. Every result is kept.

const MAX_PHOTO = 1536; // matches the server; smaller uploads are faster on mobile data
const photo = {
  styles: [], // [{id, title}] from the server
  style: "cedar",
  source: null, // {blob, url}
  results: [], // [{id, style, blob, url, master}]
  selectedId: null,
  showingOriginal: false,
  pending: new Map(), // style id -> AbortController
  error: null,
  analysis: null, // the server's read of the viewpoint used (from the latest design response)
  mode: "analyzing", // "analyzing" (finding walls) | "placing" (homeowner picks where) | "results"
  kind: null, // snap kind the analysis was for: "wall" | "corner" | "bump"
  shape: null, // deck shape the placement is for (the two L-shapes share corner analyses)
  walls: null, // /api/photo/analyze result: {kind, analyzed, elevation, candidates: [...]} (see photo.analyze_photo)
  place: null, // {wall: index into candidates(), t: deck center along a wall's base line, 0-1 (walls only)}
  analyzing: null, // AbortController of the wall analysis
  guideUrl: null, // the placement drawn on the photo, as sent to Gemini (for the prompt panel)
};
let resultSeq = 0;

fetch("/api/photo/styles")
  .then((r) => r.json())
  .then((styles) => {
    photo.styles = styles;
    if (state.step === 5) renderPhoto();
  })
  .catch(() => {});

// ---------- prompt tuning ----------
// Edited templates live in localStorage (a per-device convenience); unedited ones fall back to the server defaults.

const PROMPT_KEYS = { design: "deckPrompt.design", restyle: "deckPrompt.restyle" };
const prompts = { defaults: null, modelId: null };

function storedPrompt(kind) {
  try {
    return localStorage.getItem(PROMPT_KEYS[kind]);
  } catch {
    return null;
  }
}

function storePrompt(kind, value) {
  try {
    if (value === null || value === prompts.defaults?.[kind]) localStorage.removeItem(PROMPT_KEYS[kind]);
    else localStorage.setItem(PROMPT_KEYS[kind], value);
  } catch {
    /* storage unavailable: edits last for this page only */
  }
}

async function loadPrompts() {
  if (prompts.modelId === state.model.id) return;
  prompts.modelId = state.model.id;
  try {
    prompts.defaults = await (await fetch(`/api/models/${state.model.id}/prompts`)).json();
  } catch {
    return;
  }
  $("#prompt-design").value = storedPrompt("design") ?? prompts.defaults.design;
  $("#prompt-restyle").value = storedPrompt("restyle") ?? prompts.defaults.restyle;
  $("#prompt-deck").textContent = prompts.defaults.deck;
}

function promptFor(kind) {
  // Before the defaults load, only a stored edit is sent; otherwise the server uses its default
  return prompts.defaults ? $(`#prompt-${kind}`).value : storedPrompt(kind);
}

$("#prompt-design").addEventListener("input", (e) => storePrompt("design", e.target.value));
$("#prompt-restyle").addEventListener("input", (e) => storePrompt("restyle", e.target.value));
$("#prompt-reset").addEventListener("click", () => {
  if (!prompts.defaults) return;
  $("#prompt-design").value = prompts.defaults.design;
  $("#prompt-restyle").value = prompts.defaults.restyle;
  storePrompt("design", null);
  storePrompt("restyle", null);
});

const styleTitle = (id) => photo.styles.find((s) => s.id === id)?.title || id;
const selectedResult = () => photo.results.find((r) => r.id === photo.selectedId) || null;
const latestMaster = () => [...photo.results].reverse().find((r) => r.master) || null;

$("#show-in-photo").addEventListener("click", () => $("#photo-input").click());
$("#photo-new").addEventListener("click", () => $("#photo-input").click());
$("#placing-new").addEventListener("click", () => $("#photo-input").click());

$("#photo-input").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  event.target.value = ""; // allow picking the same photo again
  if (!file) return;
  let source;
  try {
    source = await downscale(file);
  } catch {
    photo.error = "That file isn't a photo we can read.";
    renderPhoto();
    return;
  }
  resetPhoto();
  photo.source = source;
  if (state.step !== 5) go(5);
  analyzeWalls();
});

async function downscale(file) {
  // An <img> applies the EXIF orientation, so the canvas copy comes out upright
  const url = URL.createObjectURL(file);
  try {
    // onload rather than img.decode(): decode() can stall while the page is in the background
    const img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = url;
    });
    const scale = Math.min(1, MAX_PHOTO / Math.max(img.naturalWidth, img.naturalHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(img.naturalWidth * scale);
    canvas.height = Math.round(img.naturalHeight * scale);
    canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.85));
    if (!blob) throw new Error("encode failed");
    return { blob, url: URL.createObjectURL(blob), width: canvas.width, height: canvas.height, image: canvas };
  } finally {
    URL.revokeObjectURL(url);
  }
}

function cancelPhotoRequests() {
  photo.pending.forEach((controller) => controller.abort());
  photo.pending.clear();
  photo.analyzing?.abort();
  photo.analyzing = null;
}

function resetPhoto() {
  cancelPhotoRequests();
  if (photo.source) URL.revokeObjectURL(photo.source.url);
  photo.results.forEach((r) => URL.revokeObjectURL(r.url));
  if (photo.guideUrl) URL.revokeObjectURL(photo.guideUrl);
  Object.assign(photo, {
    source: null, results: [], selectedId: null, showingOriginal: false, error: null, analysis: null,
    mode: "analyzing", walls: null, place: null, guideUrl: null,
  });
}

async function requestImage(style, url, body, master, template, extraHeaders = {}) {
  const controller = new AbortController();
  photo.pending.set(style, controller);
  photo.error = null;
  renderPhoto();
  const source = photo.source;
  try {
    const headers = { "Content-Type": "image/jpeg", ...extraHeaders };
    if (template) headers["X-Prompt-Template"] = encodeURIComponent(template);
    const res = await fetch(url, { method: "POST", headers, body, signal: controller.signal });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(typeof data.detail === "string" ? data.detail : `Server returned ${res.status}`);
    }
    const blob = await res.blob();
    if (photo.source !== source) return; // a new photo was picked meanwhile
    const analysis = res.headers.get("X-View-Analysis");
    if (analysis) photo.analysis = JSON.parse(decodeURIComponent(analysis));
    // The server's check of a design against the deck (see photo.design); restyles aren't checked
    const check = res.headers.get("X-Check");
    const result = { id: ++resultSeq, style, blob, url: URL.createObjectURL(blob), master, check: check && JSON.parse(decodeURIComponent(check)) };
    photo.results.push(result);
    // Show it if the homeowner is still on this style (or has nothing to look at yet)
    if (photo.style === style || !selectedResult()) {
      photo.selectedId = result.id;
      photo.showingOriginal = false;
    }
  } catch (err) {
    if (err.name === "AbortError") return;
    photo.error = `${styleTitle(style)}: ${err.message}`;
  } finally {
    if (photo.pending.get(style) === controller) photo.pending.delete(style);
    renderPhoto();
  }
}

function design(style) {
  if (photo.pending.has(style)) return;
  const placement = placementPayload();
  const extra = placement ? { "X-Placement": encodeURIComponent(JSON.stringify(placement)) } : {};
  requestImage(style, `/api/models/${state.model.id}/photo?style=${style}`, photo.source.blob, true, promptFor("design"), extra);
}

// ---------- placing the deck on the photo ----------
// The server finds where this deck shape could go (mirrors photo.py): one wall (rectangle, angled), an
// outside corner (L) or a part of the house that sticks out (U), as ground lines in 0-1 image coordinates.
// Tapping snaps to the nearest candidate; on a wall, dragging also slides the deck along it.

const SNAP_KIND = { rect: "wall", angled: "wall", l: "corner", l_left: "corner", u: "bump" };
// Which way an L deck's side part runs back from the corner; it only fits corners that turn the same way
const CORNER_SIDE = { l: "right", l_left: "left" };
const snapKind = () => SNAP_KIND[state.shape];
const PLACING_HINT = {
  wall: "Tap a highlighted wall to place the deck, then drag to adjust.",
  corner: "Tap a highlighted corner to wrap the deck around it.",
  noCorner: "No corner here turns this way. Try the other L-shape, or Add Deck to place it anyway.",
  bump: "Tap a highlighted part of the house to wrap the deck around it.",
};

async function analyzeWalls() {
  const controller = (photo.analyzing = new AbortController());
  const source = photo.source;
  const kind = snapKind();
  Object.assign(photo, { mode: "analyzing", walls: null, place: null, kind, shape: state.shape });
  renderPhoto();
  try {
    const res = await fetch(`/api/photo/analyze?kind=${kind}`, {
      method: "POST", headers: { "Content-Type": "image/jpeg" }, body: source.blob, signal: controller.signal,
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const found = await res.json();
    if (photo.source !== source) return;
    photo.walls = found;
    if (found.candidates.length) {
      choosePlacement();
    } else {
      startDesign(); // nothing to pick from: the server places it straight on
    }
  } catch (err) {
    if (err.name === "AbortError") return;
    startDesign();
  } finally {
    if (photo.analyzing === controller) photo.analyzing = null;
    renderPhoto();
  }
}

// The candidates this deck shape fits: for an L, only corners turning its way
const candidates = () => (photo.walls?.candidates || []).filter((c) => !CORNER_SIDE[photo.shape] || c.side === CORNER_SIDE[photo.shape]);

function choosePlacement() {
  // Start on the recommended candidate, or the first that fits; none fitting leaves the deck unplaced
  const fits = candidates();
  photo.place = fits.length ? { wall: Math.max(fits.findIndex((c) => c.recommended), 0), t: 0.5 } : null;
  photo.mode = "placing";
}
const along = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
const share = (feet, length, fallback) => (length ? Math.min(Math.max(feet / length, 0.12), 1) : fallback);

function outline(c) {
  // The candidate's ground line(s) as one polyline
  if (photo.kind === "corner") return [c.main_end, c.corner, c.side_end];
  if (photo.kind === "bump") return [c.left_back, c.front_line[0], c.front_line[1], c.right_back];
  return c.line;
}

function deckPath() {
  // Where the deck meets the building, as drawn on the photo and sent as the guide line
  const c = candidates()[photo.place.wall];
  const d = state.dims;
  if (photo.kind === "corner") {
    // Along the main wall for the part next to the house (B - D), back along the side wall by the house's depth
    const houseDepth = Math.abs(d.C - d.A);
    return [along(c.corner, c.main_end, share(d.B - d.D, c.length_ft, 0.5)), c.corner, along(c.corner, c.side_end, share(houseDepth, c.side_length_ft, 0.6))];
  }
  if (photo.kind === "bump") return outline(c);
  const xs = layout().pts.map((p) => p[0]);
  const half = share(Math.max(...xs) - Math.min(...xs), c.length_ft, 0.6) / 2;
  const t = Math.min(Math.max(photo.place.t, half), 1 - half);
  return [along(...c.line, t - half), along(...c.line, t + half)];
}

function placementPayload() {
  if (!photo.place || !candidates().length) return null;
  const c = candidates()[photo.place.wall];
  return {
    kind: photo.kind, label: c.label, azimuth: c.azimuth, length_ft: c.length_ft,
    path: deckPath(), elevation: photo.walls.elevation,
  };
}

function renderPlacement() {
  const svg = $("#placement");
  const { width: w, height: h } = photo.source;
  const H = (1000 * h) / w;
  svg.setAttribute("viewBox", `0 0 1000 ${H}`);
  const P = ([x, y]) => `${x * 1000},${y * H}`;
  const poly = (pts, cls) => `<polyline class="${cls}" points="${pts.map(P).join(" ")}"/>`;
  let html = "";
  candidates().forEach((c, i) => {
    html += poly(outline(c), `wall-band${i === photo.place.wall ? " selected" : ""}`);
    if (c.recommended) {
      const [x, y] = photo.kind === "corner" ? c.corner : along(...(photo.kind === "bump" ? c.front_line : c.line), 0.5);
      html += `<text class="wall-tag" x="${x * 1000}" y="${y * H - 26}" text-anchor="middle">Recommended</text>`;
    }
  });
  if (photo.place) {
    const path = deckPath();
    html += poly(path, "deck-bar-halo") + poly(path, "deck-bar");
  }
  svg.innerHTML = html;
}

function pointerToImage(event) {
  // Client px -> viewBox units -> 0-1 image coordinates (the viewBox letterboxes like object-fit: contain)
  const svg = $("#placement");
  const pt = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
  const vb = svg.viewBox.baseVal;
  return [pt.x / vb.width, pt.y / vb.height];
}

function projectOnto(a, b, [px, py]) {
  // Position along segment a-b (0-1) and distance to it, measured in image pixels
  const { width: w, height: h } = photo.source;
  const [x1, y1, x2, y2] = [a[0] * w, a[1] * h, b[0] * w, b[1] * h];
  const [qx, qy] = [px * w, py * h];
  const len2 = (x2 - x1) ** 2 + (y2 - y1) ** 2 || 1;
  const t = Math.min(Math.max(((qx - x1) * (x2 - x1) + (qy - y1) * (y2 - y1)) / len2, 0), 1);
  return { t, dist: Math.hypot(x1 + (x2 - x1) * t - qx, y1 + (y2 - y1) * t - qy) };
}

function distanceTo(c, p) {
  const pts = outline(c);
  return Math.min(...pts.slice(1).map((b, i) => projectOnto(pts[i], b, p).dist));
}

let dragging = false;
$("#placement").addEventListener("pointerdown", (event) => {
  if (photo.mode !== "placing" || !candidates().length) return;
  const p = pointerToImage(event);
  // Snap to the nearest candidate; on a wall the drag then slides along that wall only
  let best = 0;
  candidates().forEach((c, i) => {
    if (distanceTo(c, p) < distanceTo(candidates()[best], p)) best = i;
  });
  photo.place = { wall: best, t: photo.kind === "wall" ? projectOnto(...candidates()[best].line, p).t : 0.5 };
  dragging = photo.kind === "wall";
  if (dragging) $("#placement").setPointerCapture(event.pointerId);
  renderPlacement();
});
$("#placement").addEventListener("pointermove", (event) => {
  if (!dragging) return;
  photo.place.t = projectOnto(...candidates()[photo.place.wall].line, pointerToImage(event)).t;
  renderPlacement();
});
for (const type of ["pointerup", "pointercancel"]) $("#placement").addEventListener(type, () => (dragging = false));

function startDesign() {
  photo.mode = "results";
  drawGuide();
  design(photo.style);
}

function drawGuide() {
  // The same guide the server draws, for the prompt panel's [PLACEMENT GUIDE] thumbnail
  if (photo.guideUrl) URL.revokeObjectURL(photo.guideUrl);
  photo.guideUrl = null;
  const path = placementPayload()?.path;
  if (!path) return;
  const { image, width: w, height: h } = photo.source;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const g = canvas.getContext("2d");
  g.drawImage(image, 0, 0);
  g.strokeStyle = "rgb(255, 0, 200)";
  g.lineCap = g.lineJoin = "round";
  g.lineWidth = Math.max(8, Math.round(w * 0.012));
  g.beginPath();
  path.forEach(([x, y], i) => (i ? g.lineTo(x * w, y * h) : g.moveTo(x * w, y * h)));
  g.stroke();
  canvas.toBlob((blob) => {
    photo.guideUrl = URL.createObjectURL(blob);
    renderPromptInputs();
  }, "image/jpeg", 0.85);
}

$("#add-deck").addEventListener("click", startDesign);
$("#photo-move").addEventListener("click", () => {
  photo.mode = "placing";
  renderPhoto();
});

function restyle(master, style) {
  if (photo.pending.has(style)) return;
  requestImage(style, `/api/photo/restyle?style=${style}`, master.blob, false, promptFor("restyle"));
}

function chooseStyle(style) {
  photo.style = style;
  const existing = [...photo.results].reverse().find((r) => r.style === style);
  if (existing) {
    photo.selectedId = existing.id;
    photo.showingOriginal = false;
  } else if (photo.source && !photo.pending.has(style)) {
    const master = latestMaster();
    master ? restyle(master, style) : design(style);
  }
  renderPhoto();
}

// The actual input images for each prompt, so the panel shows exactly what goes to Gemini
function renderPromptInputs() {
  const ref = (view) => `/models/${state.model.id}/ref/${view}.png`;
  const tile = (label, url) =>
    url
      ? `<a href="${url}" target="_blank" rel="noopener"><img src="${url}" alt="${label}"><span>${label}</span></a>`
      : `<span class="empty"><span>${label}</span></span>`;
  const a = photo.analysis;
  const matched = a && `/models/${state.model.id}/ref/matched.png?az=${a.azimuth}&el=${a.elevation}`;
  const design = [tile("[PHOTO]", photo.source?.url), tile("[PLACEMENT GUIDE]", photo.guideUrl), tile("[MATCHED VIEW]", matched), tile("[TOP VIEW]", ref("top"))].join("");
  $("#photo-analysis").textContent = !a
    ? "Photo analysis: shown after Add Deck."
    : a.analyzed
      ? `Photo analysis: {wall} = ${a.wall}; {viewpoint} = ${a.viewpoint}${a.wall_length_ft ? `; wall about ${Math.round(a.wall_length_ft)} ft` : ""}.`
      : "Photo analysis failed, so the camera is assumed to face the wall straight on.";
  const master = latestMaster();
  const restyle = tile(master ? `Input: ${styleTitle(master.style)} result` : "Input: the first result", master?.url);
  // Only touch the DOM when an image changes, so thumbnails don't reload on every render
  if ($("#design-inputs").dataset.html !== design) $("#design-inputs").innerHTML = $("#design-inputs").dataset.html = design;
  if ($("#restyle-inputs").dataset.html !== restyle) $("#restyle-inputs").innerHTML = $("#restyle-inputs").dataset.html = restyle;
}

function renderPhoto() {
  renderPromptInputs();
  const placing = photo.mode !== "results";
  const selected = placing ? null : selectedResult();
  const busy = photo.mode === "analyzing" || (!placing && photo.pending.has(photo.style));
  if ($("#photo-original").src !== (photo.source?.url || "")) $("#photo-original").src = photo.source?.url || "";
  $("#placement").toggleAttribute("hidden", photo.mode !== "placing"); // SVG elements have no .hidden property
  if (photo.mode === "placing") renderPlacement();
  $("#placing-ui").hidden = !placing;
  $("#add-deck").disabled = photo.mode !== "placing";
  $("#results-ui").hidden = placing;
  if (selected && $("#photo-result").dataset.id !== String(selected.id)) {
    $("#photo-result").src = selected.url;
    $("#photo-result").dataset.id = selected.id;
  }
  $("#photo-result").hidden = !selected || photo.showingOriginal;
  $("#photo-tag").hidden = !(selected && photo.showingOriginal);
  $("#photo-busy").hidden = !busy;
  $("#photo-busy-text").textContent = photo.mode === "analyzing" ? "Finding walls…" : `Adding ${styleTitle(photo.style)} deck…`;

  $("#photo-styles").innerHTML = photo.styles
    .map((s) => {
      const cls = ["chip", s.id === photo.style && "selected", photo.pending.has(s.id) && "pending"].filter(Boolean).join(" ");
      return `<button type="button" class="${cls}" data-style="${s.id}">${s.title}</button>`;
    })
    .join("");
  $$("#photo-styles .chip").forEach((b) => b.addEventListener("click", () => chooseStyle(b.dataset.style)));

  const thumbs = $("#photo-thumbs");
  thumbs.hidden = photo.results.length + photo.pending.size < 2;
  thumbs.innerHTML =
    photo.results
      .map((r) => `<button type="button" class="thumb${r.id === photo.selectedId ? " selected" : ""}" data-id="${r.id}"><img src="${r.url}" alt=""><span>${styleTitle(r.style)}</span></button>`)
      .join("") +
    [...photo.pending.keys()].map((s) => `<span class="thumb"><span class="thumb-box"><span class="spinner small"></span></span><span>${styleTitle(s)}</span></span>`).join("");
  $$("#photo-thumbs .thumb[data-id]").forEach((b) =>
    b.addEventListener("click", () => {
      const r = photo.results.find((x) => x.id === Number(b.dataset.id));
      Object.assign(photo, { selectedId: r.id, style: r.style, showingOriginal: false });
      renderPhoto();
    })
  );

  $("#photo-error").hidden = !photo.error;
  $("#photo-error").textContent = photo.error || "";
  $("#photo-compare").disabled = !selected;
  $("#photo-compare").textContent = photo.showingOriginal ? "Result" : "Original";
  $("#photo-retry").disabled = !photo.source || busy;
  $("#photo-move").disabled = !candidates().length || busy;
  const unverified = selected?.check?.verified === false;
  $("#photo-check").hidden = !unverified;
  if (unverified) {
    const problem = selected.check.problem ? ` (${selected.check.problem.replace(/\.$/, "")})` : "";
    $("#photo-check").textContent = `This may not fully match your design${problem}. Try Again for another take.`;
  }
  const noFit = photo.mode === "placing" && !photo.place;
  $(".placing-hint").textContent = PLACING_HINT[noFit ? "noCorner" : photo.kind || snapKind()];
  $("#photo-save").disabled = !selected;
}

$("#photo-compare").addEventListener("click", () => {
  photo.showingOriginal = !photo.showingOriginal;
  renderPhoto();
});

// Try Again designs the current style from scratch; earlier results stay in the strip
$("#photo-retry").addEventListener("click", () => design(photo.style));

$("#photo-save").addEventListener("click", async () => {
  const { blob, url, style } = selectedResult();
  const name = `deck-${style}.${blob.type === "image/png" ? "png" : "jpg"}`;
  const file = new File([blob], name, { type: blob.type });
  // The share sheet ("Save Image") needs HTTPS; over plain HTTP fall back to a download
  if (navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file] });
    } catch {
      /* dismissed */
    }
    return;
  }
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
});

// ---------- desktop: point to the phone ----------

async function setupShare() {
  if (arSupported || isAndroid) return;
  try {
    const { url } = await (await fetch("/api/share")).json();
    $("#share-url").textContent = url;
    $("#share-qr").src = `/api/qr.svg?url=${encodeURIComponent(url)}`;
    $("#no-ar").hidden = false;
  } catch {
    /* sharing is a convenience; ignore */
  }
}

renderShapes();
setupShare();
go(1);
