// Settings ▸ Modules: switch a module off for THIS project (it vanishes from Engine settings and does
// not run), and see the ones the server turned off after they failed, with a way to turn them back on.
// Only modules that modify a run are listed: loaders, the sampler and the like are structure.
(function () {
  const { el, clear } = window.dom;
  const PS = window.PipelineState;
  const API = window.MovieEditorAPI;
  const SF = window.SettingsField;

  const CATEGORY_LABELS = {
    continuity: "Continuity", guidance: "Guidance", conditioning: "Conditioning",
    sampling: "Sampling", post: "Post", system: "System", "": "Other",
  };
  let _mounted = null;

  function render() {
    if (!_mounted) return;
    const { content } = _mounted;
    clear(content);
    const pane = el("div", "models-pane eng-pane");
    content.append(pane);
    if (PS.loading()) { pane.append(SF.hintEl("Loading…")); return; }
    if (PS.loadError()) { pane.append(SF.hintEl(`Could not load: ${PS.loadError()}`)); return; }
    const control = PS.control();
    const mine = Object.values(PS.modulesById()).filter((m) => control[m.id] && control[m.id].controllable);
    if (!mine.length) { pane.append(SF.hintEl("No installed module can be switched off.")); return; }
    pane.append(SF.hintEl("A module you turn off here is off for this project only, and disappears from Engine "
      + "settings. One that failed while generating is turned off for every project until its code is "
      + "repaired or you turn it back on."));
    const byCat = {};
    mine.forEach((m) => { (byCat[m.category || ""] = byCat[m.category || ""] || []).push(m); });
    Object.keys(CATEGORY_LABELS).filter((c) => byCat[c]).forEach((c) => {
      const g = SF.group(pane, CATEGORY_LABELS[c]);
      byCat[c].forEach((m) => {
        const held = control[m.id].quarantine;
        if (held) {
          const back = el("button", "btn ghost tiny", "Turn back on");
          back.onclick = async () => {
            try { await API.releaseModule(m.id); } catch (e) { alert(`Could not turn ${m.title || m.id} back on: ${e.message || e}`); }
            await PS.refreshControl(); render();
          };
          g.append(SF.field(`${m.title || m.id} — failed`, back,
            `Off since ${held.when || "a failure"}: ${held.reason || "an error"}`));
          return;
        }
        const cb = el("input"); cb.type = "checkbox"; cb.checked = !PS.isOff(m.id);
        cb.onchange = () => { PS.setOff(m.id, !cb.checked).then(render); };
        g.append(SF.toggleField(m.title || m.id, cb, PS.isOff(m.id) ? "Off for this project" : undefined));
      });
    });
  }

  function mount(body) {
    const content = el("div", "models-mount eng-mount");
    body.append(content);
    _mounted = { content };
    render();
    PS.ensureLoaded().then(() => PS.refreshControl()).then(render);
    return () => { _mounted = null; };
  }

  window.SettingsWindow.register({
    id: "modules", group: "Generation", order: 3, title: "Modules",
    subtitle: "Switch a module off for this project; see and re-enable ones that failed.",
    keywords: "modules enable disable off quarantine failed repair switch",
    iconBg: "linear-gradient(180deg,#6fd08c,#2f9e5a)",
    icon: '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="#fff" stroke-width="1.5" stroke-linecap="round"><path d="M3 4h10M3 8h10M3 12h10"/><circle cx="6" cy="4" r="1.4" fill="#fff"/><circle cx="10.5" cy="8" r="1.4" fill="#fff"/><circle cx="5" cy="12" r="1.4" fill="#fff"/></svg>',
    mount,
  });
})();
