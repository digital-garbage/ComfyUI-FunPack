// Easy Gen's "Shortcuts" settings section: prompt autocomplete + anchor toggles
// (same per-browser settings/localStorage key as the Editor's editor_settings.js,
// via Store.getEditorSetting/setEditorSetting) plus the shortcut revolver
// (server-side, shared with the Editor and with generation regardless of which
// UI queues the run). Deliberately NOT the whole of editor_settings.js — this
// skips "Shortcut ideas" (not wanted here) and the "Anchor as guide" custom-node
// bypass section (Easy Gen never creates anchor_guide scenes, so it would be
// dead UI). revolverSection() below is intentionally a copy of
// editor_settings.js's — it's small and purely API-driven (no Store deps beyond
// dom/API), so duplicating it here keeps Easy Gen's settings free of the
// Editor-only sections above without forcing a shared-module refactor.
(function () {
  const { el } = window.dom;
  const S = window.Store;
  const API = window.MovieEditorAPI;

  function toggleRow(label, hint, key) {
    const row = el("div", "es-row");
    const lbl = el("label", "chk es-toggle");
    const cb = el("input"); cb.type = "checkbox"; cb.checked = !!S.getEditorSetting(key);
    cb.onchange = () => S.setEditorSetting(key, cb.checked);
    lbl.append(cb, el("span", null, label));
    row.append(lbl);
    if (hint) row.append(el("div", "es-hint", hint));
    return row;
  }

  function revolverSection() {
    const wrap = el("div", "es-section");
    wrap.append(el("div", "es-section-title", "Shortcut revolver"));
    wrap.append(el("div", "es-hint",
      "For shortcuts with several replacements: remember which replacement fired and don't reuse it "
      + "until the WHOLE set has fired. Off = seeded random pick (may repeat). Saved with your shortcut "
      + "library — applies to every browser and to generation."));

    const main = el("label", "chk es-toggle");
    const mainCb = el("input"); mainCb.type = "checkbox"; mainCb.disabled = true;
    main.append(mainCb, el("span", null, "Shortcut revolver"));
    wrap.append(main);

    const sub = el("div", "es-subfields");
    const rand = el("label", "chk es-toggle");
    const randCb = el("input"); randCb.type = "checkbox"; randCb.disabled = true;
    rand.append(randCb, el("span", null, "Random"));
    sub.append(rand);
    sub.append(el("div", "es-hint",
      "Unchecked: replacements fire in order — first, second, … last, then the cycle restarts. "
      + "Checked: any order, but still no repeats until the set is exhausted. Changing either "
      + "option restarts all cycles."));
    wrap.append(sub);

    const apply = (s) => {
      mainCb.checked = !!(s && s.enabled);
      randCb.checked = !!(s && s.random);
      mainCb.disabled = randCb.disabled = false;
      sub.style.display = mainCb.checked ? "" : "none";
    };

    const save = () => {
      mainCb.disabled = randCb.disabled = true;
      API.setRevolverSettings({ enabled: mainCb.checked, random: randCb.checked })
        .then(apply)
        .catch(() => { API.revolverSettings().then(apply).catch(() => {}); });
    };
    mainCb.onchange = save;
    randCb.onchange = save;

    sub.style.display = "none";
    API.revolverSettings().then(apply).catch(() => {
      wrap.append(el("div", "es-hint", "Could not load revolver settings."));
    });
    return wrap;
  }

  function mount(body) {
    const content = el("div", "es-content");
    content.append(toggleRow(
      "Prompt autocomplete",
      "Suggest matching shortcuts while you type. Shows the trigger, its prompt, and category.",
      "autocomplete"));
    content.append(toggleRow(
      "Use anchor",
      "Text before the first split trigger (\"cut\", etc.) is a shared anchor prepended to every "
      + "scene. Turn off to make that leading text the first scene instead.",
      "anchorEnabled"));
    content.append(revolverSection());
    body.append(content);
  }

  window.SettingsWindow.register({
    id: "shortcuts", group: "Generation", order: 2, title: "Shortcuts",
    subtitle: "Autocomplete, anchor text, and the shortcut revolver — shared with the Editor.",
    keywords: "autocomplete anchor shortcuts split cut revolver random replacements cycle",
    iconBg: "linear-gradient(180deg,#6aa9ff,#3b6fd9)",
    icon: '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="#fff" stroke-width="1.5" stroke-linecap="round"><path d="M2 4.5h12M2 8h12M2 11.5h12"/><circle cx="6" cy="4.5" r="1.7" fill="#fff" stroke="none"/><circle cx="11" cy="8" r="1.7" fill="#fff" stroke="none"/><circle cx="5" cy="11.5" r="1.7" fill="#fff" stroke="none"/></svg>',
    mount,
  });
})();
