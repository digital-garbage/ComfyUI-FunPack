// Sliders.
//
// The track and thumb are drawn. v4 left them fully native with only
// accent-color, so a slider looked like the operating system rather than like
// the app, and its hit target was whatever the browser decided.

import { define } from "../internals/register.js";
import { el } from "../internals/el.js";
import { uid } from "../internals/ids.js";

const pct = (v, min, max) => (max === min ? 0 : ((v - min) / (max - min)) * 100);

function makeRange({ value, min, max, step, disabled, label, describedBy }) {
  const input = el("input", {
    cls: ["cx-range", "cx-focusable"],
    attrs: { type: "range", min, max, step, disabled, "aria-label": label, "aria-describedby": describedBy },
  });
  input.value = String(value);
  // The filled portion is drawn from a variable rather than a background image,
  // so it follows the theme's accent without the JS knowing a colour.
  const paint = () => input.style.setProperty("--fill", `${pct(Number(input.value), min, max)}%`);
  paint();
  input.addEventListener("input", paint);
  return { input, paint };
}

function sliderBase(cls, { value = 0, min = 0, max = 1, step = 0.01, onChange, onCommit, disabled, label } = {}) {
  const { input, paint } = makeRange({ value, min, max, step, disabled, label });
  if (onChange) input.addEventListener("input", () => onChange(Number(input.value)));
  if (onCommit) input.addEventListener("change", () => onCommit(Number(input.value)));
  const node = el("div", { cls, children: input });
  return {
    node, input, paint,
    get value() { return Number(input.value); },
    setValue(v) { input.value = String(v); paint(); },
    destroy: () => node.remove(),
  };
}

for (const size of ["md", "sm"]) {
  define("slider", size, (props = {}) => sliderBase(`cx-slider cx-slider-${size}`, props));
}

define("slider", "readout", ({ unit = "", precision = 2, ...props } = {}) => {
  const base = sliderBase("cx-slider cx-slider-md cx-slider-readout", props);
  const format = (v) => `${Number(v).toFixed(precision)}${unit}`;
  const out = el("output", { cls: "cx-readout", text: format(base.value) });
  base.input.addEventListener("input", () => { out.textContent = format(base.input.value); });
  base.node.append(out);
  const setValue = base.setValue;
  return {
    ...base,
    setValue(v) { setValue(v); out.textContent = format(v); },
    node: base.node,
  };
});

/**
 * A slider with named landing points. The presets are the point: most people
 * want "Chill" or "Chaos", not 0.62, and the number is there for the ones who do.
 */
define("slider", "macro", ({ presets = [], unit = "", precision = 2, ...props } = {}) => {
  const base = sliderBase("cx-slider cx-slider-md cx-slider-macro", props);
  const format = (v) => `${Number(v).toFixed(precision)}${unit}`;
  const out = el("output", { cls: "cx-readout", text: format(base.value) });
  base.node.append(out);

  const row = el("div", { cls: "cx-macro-presets" });
  const buttons = presets.map((preset) => {
    const b = el("button", {
      cls: ["cx-btn", "cx-btn-sm", "cx-btn-ghost", "cx-focusable"],
      text: preset.label,
      attrs: { type: "button" },
      on: { click: () => {
        base.setValue(preset.value);
        out.textContent = format(preset.value);
        sync();
        if (props.onChange) props.onChange(preset.value);
        if (props.onCommit) props.onCommit(preset.value);
      } },
    });
    row.append(b);
    return { preset, b };
  });

  function sync() {
    for (const { preset, b } of buttons) b.classList.toggle("cx-on", Number(base.value) === preset.value);
  }
  base.input.addEventListener("input", () => { out.textContent = format(base.input.value); sync(); });
  sync();

  const node = el("div", { cls: "cx-macro", children: [base.node, row] });
  return {
    node,
    get value() { return base.value; },
    setValue(v) { base.setValue(v); out.textContent = format(v); sync(); },
    destroy: () => node.remove(),
  };
});

/** Two thumbs. Trim handles and any "between x and y" setting are built on this. */
define("range", "md", ({ from = 0, to = 1, min = 0, max = 1, step = 0.01, onChange, label } = {}) => {
  const lo = makeRange({ value: from, min, max, step, label: label ? `${label} from` : "from" });
  const hi = makeRange({ value: to, min, max, step, label: label ? `${label} to` : "to" });

  // Thumbs must not pass each other: a range whose start is after its end is a
  // state every consumer would then have to defend against.
  const clampPair = (moved) => {
    let a = Number(lo.input.value);
    let b = Number(hi.input.value);
    if (a > b) {
      if (moved === "lo") a = b; else b = a;
      lo.input.value = String(a);
      hi.input.value = String(b);
      lo.paint(); hi.paint();
    }
    if (onChange) onChange({ from: a, to: b });
  };
  lo.input.addEventListener("input", () => clampPair("lo"));
  hi.input.addEventListener("input", () => clampPair("hi"));

  const node = el("div", { cls: "cx-range-pair", attrs: { role: "group", "aria-label": label },
    children: [lo.input, hi.input] });

  return {
    node,
    get value() { return { from: Number(lo.input.value), to: Number(hi.input.value) }; },
    setValue({ from: a, to: b }) {
      lo.input.value = String(a); hi.input.value = String(b);
      lo.paint(); hi.paint();
    },
    destroy: () => node.remove(),
  };
});
