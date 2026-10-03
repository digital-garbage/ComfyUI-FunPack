// A module's settings declaration, rendered.
//
// This is the bridge the whole architecture turns on: one declaration renders
// the panel, carries the defaults and names the values that come back. A module
// writes no DOM and reads no DOM -- it says what it needs and gets values.
//
// The panel builds into a DETACHED fragment. If any part of it throws -- an
// unknown renderer, a malformed option -- nothing has been appended, so the
// module is absent rather than half-rendered. That is "hide, don't warn" at the
// level where it actually has to hold.

import { composer } from "./internals/register.js";
import { el, frag } from "./internals/el.js";

/** The renderer a type gets when the declaration does not ask for one. */
export const DEFAULT_UI = {
  bool: "checkboxRow",
  int: "number",
  float: "number",
  enum: "select",
  text: "input",
  multiline: "textarea",
  path: "filterList",
  color: "swatch",
};

// Renderers that carry their own visible label, so wrapping them in a settings
// row would print the label twice. Exported because anything else rendering a
// control from a setting has to know the same thing, and a second copy of this
// list is a second chance to forget one.
export const SELF_LABELLING = new Set(["checkboxRow", "toggle"]);

/** The renderer name a setting resolves to, whether or not it named one. */
export const rendererNameFor = (setting) => setting.ui || DEFAULT_UI[setting.type];

const RENDERERS = {
  checkboxRow: (s, v, on) => composer.checkboxRow.default({ label: s.label, hint: s.hint, checked: v, onChange: on }),
  toggle: (s, v, on) => composer.toggle.default({ label: s.label, hint: s.hint, checked: v, onChange: on }),

  number: (s, v, on) => composer.number.md({
    value: v, min: s.min, max: s.max, step: s.step, unit: s.unit,
    precision: s.type === "int" ? 0 : s.precision, label: s.label, onChange: on }),
  stepper: (s, v, on) => composer.stepper.md({
    value: v, min: s.min, max: s.max, step: s.step, label: s.label, onChange: on }),
  slider: (s, v, on) => composer.slider.readout({
    value: v, min: s.min ?? 0, max: s.max ?? 1, step: s.step ?? 0.01,
    unit: s.unit ?? "", precision: s.type === "int" ? 0 : 2, label: s.label, onCommit: on, onChange: on }),
  macroSlider: (s, v, on) => composer.slider.macro({
    value: v, min: s.min ?? 0, max: s.max ?? 1, step: s.step ?? 0.01,
    unit: s.unit ?? "", presets: s.presets ?? [], label: s.label, onCommit: on, onChange: on }),

  select: (s, v, on) => composer.select.md({ options: s.options, value: v, label: s.label, onChange: on }),
  segmented: (s, v, on) => composer.segmented.md({ options: s.options, value: v, label: s.label, onChange: on }),
  radioGroup: (s, v, on) => composer.radioGroup.default({ options: s.options, value: v, label: s.label, onChange: on }),
  swatch: (s, v, on) => composer.color.swatch({ label: s.label, value: v, onChange: on }),

  filterList: (s, v, on) => composer.filterList.md({
    items: (s.options || []).map((o) => ({ id: o.value, label: o.label, hint: o.hint })),
    value: v, placeholder: s.placeholder, onChange: on }),

  // An overlay, so the row holds a button showing the current choice. Opening a
  // wheel from a settings panel is unusual; being able to is what lets a module
  // ask for one without writing any JavaScript.
  wheel: (s, v, on) => {
    const labelFor = (value) => (s.options.find((o) => o.value === value) || {}).label ?? String(value);
    const button = composer.button.md({
      label: labelFor(v),
      onClick: () => composer.wheel.picker({
        items: s.options.map((o) => ({ label: o.label, icon: o.icon, value: o.value })),
        onPick: (item) => { button.setLabel(labelFor(item.value)); on(item.value); },
      }),
    });
    return button;
  },

  input: (s, v, on) => composer.input.md({ value: v, placeholder: s.placeholder, label: s.label, onCommit: on }),
  search: (s, v, on) => composer.search.md({ value: v, placeholder: s.placeholder, label: s.label, onCommit: on }),
  textarea: (s, v, on) => composer.textarea.md({ value: v, placeholder: s.placeholder, label: s.label, onCommit: on }),
  autoTextarea: (s, v, on) => composer.textarea.md({
    value: v, placeholder: s.placeholder, label: s.label, autoGrow: true, onCommit: on }),
};

/** Is this setting's `when` satisfied by the current values? */
export function visible(setting, values) {
  const when = setting.when;
  if (!when) return true;
  return Object.entries(when).every(([key, expected]) =>
    (Array.isArray(expected) ? expected : [expected]).includes(values[key]));
}

export function rendererFor(setting) {
  // Core refuses a setting with no default long before the browser sees it, so
  // this is belt and braces -- but the panel is the layer that cannot proceed
  // without one: it would render a control holding `undefined` and report that
  // as the value the moment anything read it.
  if (!Object.prototype.hasOwnProperty.call(setting, "default")) {
    throw new TypeError(`Setting has no default, so there is nothing to render it with.`);
  }
  const name = setting.ui || DEFAULT_UI[setting.type];
  const render = RENDERERS[name];
  if (!render) {
    // Refusing here rather than falling back to something plausible: a panel
    // that silently renders a different control is worse than one that is
    // absent, because nobody notices it is wrong.
    throw new TypeError(`No renderer for "${name}" (setting type "${setting.type}").`);
  }
  return render;
}

/**
 * renderPanel(spec, { values, onChange }) -> { node, values, destroy }
 *
 * `values` starts from the module's declared defaults, so a panel and a
 * headless run begin from the same numbers.
 */
export function renderPanel(spec, { values: initial, onChange } = {}) {
  const settings = spec.settings || {};
  const values = { ...defaultsOf(spec), ...(initial || {}) };

  const rows = el("div", { cls: "cx-group-rows" });
  let handles = new Map();

  function build() {
    // Into a fragment, and into a map of its OWN, so a row that throws leaves
    // nothing half-made behind and touches nothing already on screen.
    const built = frag();
    const made = new Map();
    for (const [key, setting] of Object.entries(settings)) {
      if (!visible(setting, values)) continue;
      const render = rendererFor(setting);
      const control = render(setting, values[key], (next) => {
        values[key] = next;
        if (onChange) onChange(key, next, { ...values });
        // Another setting's visibility may depend on this one.
        if (dependents.has(key)) redraw();
      });
      made.set(key, control);
      built.appendChild(
        SELF_LABELLING.has(rendererNameFor(setting))
          ? control.node
          : composer.settingsRow.default({ label: setting.label, hint: setting.hint, control }).node
      );
    }
    return { built, made };
  }

  // Which keys other rows watch, so a change only redraws when it can matter.
  const dependents = new Set();
  for (const setting of Object.values(settings)) {
    for (const key of Object.keys(setting.when || {})) dependents.add(key);
  }

  function redraw() {
    // Built BEFORE anything is torn down. Destroying first and building second
    // meant a row that only appears once another value is set -- so a row the
    // first build never rendered and never checked -- emptied the panel and
    // then threw, leaving it permanently blank with no way back.
    const { built, made } = build();
    for (const handle of handles.values()) if (handle.destroy) handle.destroy();
    handles = made;
    rows.replaceChildren(built);
  }

  {
    const first = build();
    handles = first.made;
    rows.appendChild(first.built);
  }

  const node = el("section", { cls: "cx-group", children: [
    el("div", { cls: "cx-eyebrow", text: spec.title }),
    rows,
  ] });

  return {
    node,
    get values() { return { ...values }; },
    setValue(key, next) { values[key] = next; redraw(); },
    destroy() {
      for (const handle of handles.values()) if (handle.destroy) handle.destroy();
      node.remove();
    },
  };
}

/** The same defaults core derives, computed in the browser from the same source. */
export function defaultsOf(spec) {
  const out = {};
  for (const [key, setting] of Object.entries(spec.settings || {})) out[key] = setting.default;
  return out;
}
