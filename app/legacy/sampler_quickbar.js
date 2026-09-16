// Sampler quick-access: steps/scheduler/sampler reachable in one click,
// instead of Settings ▸ Engine ▸ Sampling being several clicks deep.
//
// [[project_v5_ui_requirements]] asks for this specific shape: "a thin
// hovering window above the timeline", not a modal -- the whole point is
// staying visible without spending timeline height, unlike Engine Settings'
// full window (which this bar deliberately does not replace; it edits the
// exact same tree through the exact same functions, see settings_field.js
// and PipelineState.currentValues/setModuleValue).
//
// Pass-level, not KSampler-branch: this renders whatever the "sampling"
// category holds, same as Engine Settings' Sampling tab -- it does not know
// or care which node ends up wired, only that some installed module
// declared these settings under that category.
(function () {
  const { el, clear } = window.dom;
  const SF = window.SettingsField;
  const PS = window.PipelineState;

  const host = document.querySelector("#timeline-zone .zone-head-right");
  if (!host || !PS || !SF) return;

  let panel = null;
  let cleanup = null;
  let btn = null;

  function onChange(moduleId, name, value) {
    PS.setModuleValue(moduleId, name, value).then(render);
  }

  function samplingModules() {
    return Object.values(PS.modulesById())
      .filter((m) => (m.category || "") === "sampling" && Object.keys(m.settings || {}).length);
  }

  function renderInto(body) {
    clear(body);
    if (PS.loading()) { body.append(SF.hintEl("Loading…")); return; }
    if (PS.loadError()) { body.append(SF.hintEl(`Could not load: ${PS.loadError()}`)); return; }
    const modules = samplingModules();
    if (!modules.length) { body.append(SF.hintEl("No installed module exposes a sampling setting yet.")); return; }
    const values = PS.currentValues();
    modules.forEach((m) => {
      const g = SF.group(body, modules.length > 1 ? (m.title || m.id) : "");
      Object.entries(m.settings || {}).forEach(([name, spec]) => {
        if (!SF.whenSatisfied(values, m.id, spec.when)) return;
        g.append(SF.controlFor(m.id, name, spec, values, onChange));
      });
    });
  }

  function render() {
    if (!panel) return;
    renderInto(panel.querySelector(".sq-body"));
  }

  function position() {
    if (!panel || !btn) return;
    const bar = btn.getBoundingClientRect();
    const zone = document.getElementById("timeline-zone");
    const top = zone ? zone.getBoundingClientRect().top : bar.top;
    const pw = panel.offsetWidth || 320;
    panel.style.left = Math.max(8, Math.min(bar.right - pw, window.innerWidth - pw - 8)) + "px";
    // Above the timeline's own top edge, not inside it -- the whole point is
    // not spending timeline height on this.
    panel.style.top = Math.max(8, top - panel.offsetHeight - 8) + "px";
  }

  function close() {
    if (!panel) return;
    panel.remove();
    panel = null;
    btn.classList.remove("on");
    btn.setAttribute("aria-pressed", "false");
    if (cleanup) { cleanup(); cleanup = null; }
  }

  function open() {
    if (panel) { close(); return; }
    panel = el("div", "sq-panel");
    const head = el("div", "sq-head");
    head.append(el("span", "", "Sampler"));
    const closeBtn = el("button", "btn ghost tiny", "✕");
    closeBtn.type = "button";
    closeBtn.onclick = close;
    head.append(closeBtn);
    panel.append(head);
    const body = el("div", "sq-body");
    panel.append(body);
    document.body.append(panel);
    btn.classList.add("on");
    btn.setAttribute("aria-pressed", "true");
    renderInto(body);
    position();

    const onOutside = (e) => { if (!panel.contains(e.target) && e.target !== btn) close(); };
    const onResize = () => position();
    setTimeout(() => document.addEventListener("mousedown", onOutside, true), 0);
    window.addEventListener("resize", onResize);
    cleanup = () => {
      document.removeEventListener("mousedown", onOutside, true);
      window.removeEventListener("resize", onResize);
    };

    PS.ensureLoaded().then(render);
  }

  function toggle() { open(); } // open() itself closes when already open

  function refreshVisibility() {
    // Absent, not greyed, when nothing in the current pipeline has a
    // sampling-category setting to show -- same convention mounts.js
    // already uses for a role/module that does not apply.
    const has = samplingModules().length > 0;
    btn.hidden = !has;
    if (!has) close();
  }

  btn = el("button", "dock-tab", "⏱ Sampler");
  btn.type = "button";
  btn.title = "Steps, scheduler and sampler — one click, no modal.";
  btn.setAttribute("aria-pressed", "false");
  btn.hidden = true; // hidden until PipelineState confirms there's something to show
  btn.onclick = toggle;
  host.insertBefore(btn, host.firstChild);

  PS.ensureLoaded().then(refreshVisibility);

  window.SamplerQuickbar = { open, close, toggle };
})();
