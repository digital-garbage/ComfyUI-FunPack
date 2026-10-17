// Line chart: a loss curve, a metric over time.
//
// The lines are an SVG stretched over the box (preserveAspectRatio none, strokes
// that do not scale), the numbers around it are HTML, so text stays crisp at any
// width. Data is cut down to a few hundred drawn points by keeping each bucket's
// lowest and highest, which is what keeps a spike visible.

import { define } from "../internals/register.js";
import { el, svg } from "../internals/el.js";
import { setText } from "../internals/text.js";

const W = 600, H = 200;               // the drawing's own units; the box it is stretched to is CSS
const MAX_POINTS = 600;
const TONES = ["accent", "good", "warn", "muted"];
const HEIGHTS = ["sm", "md", "lg"];

/** Finite points only, as [{x, y}]. `values` are spaced 0, 1, 2…; logY drops what a log cannot show. */
function pointsOf(series, logY) {
  const raw = Array.isArray(series.points) ? series.points : (series.values || []).map((y, x) => ({ x, y }));
  return raw.filter((p) => p && Number.isFinite(p.x) && Number.isFinite(p.y) && (!logY || p.y > 0)).sort((a, b) => a.x - b.x);
}

/** At most MAX_POINTS points: each bucket keeps its lowest and highest point, in x order. */
function thin(points) {
  if (points.length <= MAX_POINTS) return points;
  const size = Math.ceil(points.length / (MAX_POINTS / 2)), out = [];
  for (let i = 0; i < points.length; i += size) {
    const bucket = points.slice(i, i + size);
    const lo = bucket.reduce((a, p) => (p.y < a.y ? p : a)), hi = bucket.reduce((a, p) => (p.y > a.y ? p : a));
    out.push(...(lo === hi ? [lo] : lo.x < hi.x ? [lo, hi] : [hi, lo]));
  }
  return out;
}

/** Round, readable tick values covering [lo, hi]. */
function niceTicks(lo, hi, want = 4) {
  const span = hi - lo, raw = span / want, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

/** 1, 2 and 5 times a power of ten inside [lo, hi] (a log axis); only the powers when that is too many. */
function logTicks(lo, hi) {
  const out = [];
  for (let k = Math.floor(Math.log10(lo)); k <= Math.ceil(Math.log10(hi)); k++) {
    for (const m of [1, 2, 5]) { const v = m * 10 ** k; if (v >= lo && v <= hi) out.push(v); }
  }
  return out.length > 8 ? out.filter((v) => Math.log10(v) % 1 === 0) : out;
}

export function format(n) {
  if (n === 0) return "0";
  const a = Math.abs(n);
  return a >= 10000 || a < 0.01 ? n.toExponential(1).replace("e+", "e") : String(Number(n.toPrecision(3)));
}

/** The point at or just before x, found by bisection (points are in x order). */
function nearest(points, x) {
  let lo = 0, hi = points.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (points[mid].x < x) lo = mid + 1; else hi = mid; }
  return lo > 0 && Math.abs(points[lo - 1].x - x) < Math.abs(points[lo].x - x) ? points[lo - 1] : points[lo];
}

define("chart", "line", ({ series = [], xLabel, yLabel, height = "md", empty = "No data yet", label, logY = false } = {}) => {
  if (!HEIGHTS.includes(height)) throw new TypeError(`chart.line height must be one of ${HEIGHTS.join(", ")}.`);
  let model = [];                                   // [{ spec, points, drawn }]

  const lines = svg("g");
  const cursor = svg("line", { class: "cx-chart-cursor", "vector-effect": "non-scaling-stroke" });
  const plot = svg("svg", { class: "cx-chart-svg", viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", "aria-hidden": "true" });
  plot.append(lines, cursor);
  const yAxis = el("div", { cls: "cx-chart-yaxis" });
  const xAxis = el("div", { cls: "cx-chart-xaxis" });
  const grid = el("div", { cls: "cx-chart-grid" });                    // horizontal rules, as boxes so they stay hairlines
  const field = el("div", { cls: "cx-chart-field", children: [grid, plot] });
  const legend = el("div", { cls: "cx-chart-legend" });
  const readout = el("p", { cls: "cx-chart-readout" });
  const note = el("p", { cls: "cx-chart-empty", text: empty });
  const axes = [yLabel ? el("span", { cls: "cx-chart-axis-name", text: yLabel }) : null, xLabel ? el("span", { cls: "cx-chart-axis-name cx-chart-axis-x", text: xLabel }) : null];
  const body = el("div", { cls: "cx-chart-body", children: [axes[0], yAxis, field, xAxis, axes[1]] });
  const node = el("div", { cls: ["cx-chart", `cx-chart-${height}`], attrs: { role: "group", "aria-label": label }, children: [body, legend, readout, note] });

  const lookup = (v, lo, hi) => (logY ? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (v - lo) / (hi - lo));
  let extent = null;

  const tickBox = (cls, value, fraction, axis) => {
    const t = el("span", { cls, text: format(value) });
    t.style.setProperty(axis === "y" ? "bottom" : "left", `${(fraction * 100).toFixed(2)}%`);
    return t;
  };

  const latest = () => model.filter((m) => m.points.length).map((m) => `${m.spec.name || "value"} ${format(m.points[m.points.length - 1].y)}`).join(" · ");
  const reading = (x) => {
    const parts = model.filter((m) => m.points.length && !m.spec.faint).map((m) => `${m.spec.name || "value"} ${format(nearest(m.points, x).y)}`);
    const at = model.find((m) => m.points.length);
    return `${xLabel || "x"} ${format(at ? nearest(at.points, x).x : x)} · ${parts.join(" · ")}`;
  };

  function draw() {
    lines.replaceChildren(); yAxis.replaceChildren(); xAxis.replaceChildren(); grid.replaceChildren(); legend.replaceChildren();
    const live = model.filter((m) => m.points.length);
    node.classList.toggle("cx-chart-none", !live.length);
    if (!live.length) { extent = null; setText(readout, ""); cursor.setAttribute("x1", "-1"); cursor.setAttribute("x2", "-1"); return; }

    const all = live.flatMap((m) => m.points);
    const range = (key) => all.reduce(([lo, hi], p) => [Math.min(lo, p[key]), Math.max(hi, p[key])], [Infinity, -Infinity]);
    let [x0, x1] = range("x");
    let [y0, y1] = range("y");
    if (x0 === x1) { x0 -= 1; x1 += 1; }
    if (y0 === y1) { const pad = logY ? y0 * 0.5 : Math.abs(y0) * 0.1 || 1; y0 -= pad; y1 += pad; }
    const yTicks = logY ? logTicks(y0, y1) : niceTicks(y0, y1);
    if (!logY && yTicks.length) { y0 = Math.min(y0, yTicks[0]); y1 = Math.max(y1, yTicks[yTicks.length - 1]); }
    const xTicks = niceTicks(x0, x1, 5);
    extent = { x0, x1 };

    const px = (x) => ((x - x0) / (x1 - x0)) * W, py = (y) => H - lookup(y, y0, y1) * H;
    for (const m of model.filter((q) => q.points.length)) {
      const cls = ["cx-chart-line", `cx-chart-${m.spec.tone || "accent"}`, m.spec.faint ? "cx-chart-faint" : null, m.spec.dashed ? "cx-chart-dashed" : null].filter(Boolean).join(" ");
      const d = m.drawn.length === 1 ? `M${px(m.drawn[0].x).toFixed(1)} ${py(m.drawn[0].y).toFixed(1)}h0`        // a lone point is a round dot
        : m.drawn.map((p, i) => `${i ? "L" : "M"}${px(p.x).toFixed(1)} ${py(p.y).toFixed(1)}`).join("");
      lines.append(svg("path", { d, class: cls, "vector-effect": "non-scaling-stroke", fill: "none" }));
    }
    for (const v of yTicks) {
      const f = lookup(v, y0, y1);
      if (f < -1e-9 || f > 1 + 1e-9) continue;
      yAxis.append(tickBox("cx-chart-tick", v, f, "y"));
      const rule = el("span", { cls: "cx-chart-rule" }); rule.style.setProperty("bottom", `${(f * 100).toFixed(2)}%`); grid.append(rule);
    }
    for (const v of xTicks) xAxis.append(tickBox("cx-chart-tick", v, (v - x0) / (x1 - x0), "x"));
    for (const m of model) {
      if (!m.spec.name) continue;
      legend.append(el("span", { cls: "cx-chart-key", children: [el("span", { cls: ["cx-chart-swatch", `cx-chart-${m.spec.tone || "accent"}`, m.spec.faint ? "cx-chart-faint" : null], attrs: { "aria-hidden": "true" } }), el("span", { text: m.spec.name })] }));
    }
    setText(readout, latest());
  }

  const at = (e) => {
    const box = plot.getBoundingClientRect();
    if (!extent || !box.width) return null;
    return extent.x0 + Math.min(1, Math.max(0, (e.clientX - box.left) / box.width)) * (extent.x1 - extent.x0);
  };
  plot.addEventListener("pointermove", (e) => {
    const x = at(e);
    if (x === null) return;
    const c = ((x - extent.x0) / (extent.x1 - extent.x0)) * W;
    cursor.setAttribute("x1", String(c)); cursor.setAttribute("x2", String(c)); cursor.setAttribute("y1", "0"); cursor.setAttribute("y2", String(H));
    setText(readout, reading(x));
  });
  plot.addEventListener("pointerleave", () => {
    cursor.setAttribute("x1", "-1"); cursor.setAttribute("x2", "-1");
    setText(readout, latest());
  });

  function setSeries(next) {
    model = (next || []).map((spec) => {
      if (spec.tone && !TONES.includes(spec.tone)) throw new TypeError(`chart.line tone must be one of ${TONES.join(", ")}.`);
      const points = pointsOf(spec, logY);
      return { spec, points, drawn: thin(points) };
    });
    draw();
  }
  setSeries(series);
  return { node, setSeries, setEmpty: (text) => setText(note, text), destroy: () => node.remove() };
});
