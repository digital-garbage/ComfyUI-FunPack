// One field-widget builder for a module setting spec, shared by every panel
// that edits PipelineState's module-values tree (Engine Settings, the
// sampler quick-access bar). Extracted so a fix to how a slider or an enum
// renders lands once, not once per panel that happens to render one.
(function () {
  const { el } = window.dom;

  function field(labelText, control, hint) {
    const row = el("div", "sw-row eng-field");
    const main = el("div", "sw-row-main");
    main.append(el("div", "sw-row-title", labelText));
    if (hint) main.append(el("div", "sw-row-hint", hint));
    row.append(main, control);
    return row;
  }
  function toggleField(labelText, checkbox, hint) {
    checkbox.style.width = "auto";
    return field(labelText, checkbox, hint);
  }
  function group(parent, label) {
    if (label) parent.append(el("div", "sw-rows-label", label));
    const g = el("div", "sw-rows");
    parent.append(g);
    return g;
  }
  function hintEl(text) { return el("div", "sw-hint", text); }

  // A `when` clause names sibling settings on the SAME module that must hold
  // their given value for this row to show -- e.g. a strength slider only
  // matters once its own "enabled" checkbox is on.
  function whenSatisfied(values, moduleId, when) {
    if (!when) return true;
    const current = values[moduleId] || {};
    return Object.entries(when).every(([k, v]) => current[k] === v);
  }

  // `values` is the full current tree (PipelineState.currentValues()).
  // `onChange(moduleId, name, newValue)` is the caller's own write path --
  // always PipelineState.setModuleValue, but the caller decides what to do
  // once the save resolves (re-render its own panel, not any other one).
  function controlFor(moduleId, name, spec, values, onChange) {
    const label = spec.label || name;
    const hint = spec.hint ? spec.hint + (spec.unit ? ` (${spec.unit})` : "") : (spec.unit || "");
    const current = values[moduleId] && values[moduleId][name];
    const value = current !== undefined ? current : spec.default;

    if (spec.type === "bool") {
      const cb = el("input", "");
      cb.type = "checkbox";
      cb.checked = !!value;
      cb.onchange = () => onChange(moduleId, name, cb.checked);
      return toggleField(label, cb, spec.hint);
    }
    if (spec.type === "int" || spec.type === "float") {
      const useSlider = (spec.ui || "slider") === "slider" && spec.min != null && spec.max != null;
      const input = el("input", "eng-num-input");
      input.type = useSlider ? "range" : "number";
      if (spec.min != null) input.min = spec.min;
      if (spec.max != null) input.max = spec.max;
      if (spec.step != null) input.step = spec.step;
      input.value = value;
      input.onchange = () => {
        const n = spec.type === "int" ? parseInt(input.value, 10) : parseFloat(input.value);
        if (!Number.isNaN(n)) onChange(moduleId, name, n);
      };
      return field(label, input, hint);
    }
    if (spec.type === "enum") {
      const select = el("select", "eng-select");
      (spec.options || []).forEach((opt) => {
        const o = el("option", "", opt.label || opt.value);
        o.value = opt.value;
        if (opt.value === value) o.selected = true;
        select.append(o);
      });
      select.onchange = () => onChange(moduleId, name, select.value);
      return field(label, select, hint);
    }
    if (spec.type === "color") {
      const input = el("input", "");
      input.type = "color";
      input.value = value || "#000000";
      input.onchange = () => onChange(moduleId, name, input.value);
      return field(label, input, hint);
    }
    // text / multiline / path: a plain box. Committed on blur, not on every
    // keystroke -- a POST per character would be a request storm.
    const input = el(spec.type === "multiline" ? "textarea" : "input", "eng-text-input");
    if (spec.type !== "multiline") input.type = "text";
    input.value = value || "";
    input.onblur = () => onChange(moduleId, name, input.value);
    return field(label, input, hint);
  }

  window.SettingsField = { field, toggleField, group, hintEl, whenSatisfied, controlFor };
})();
