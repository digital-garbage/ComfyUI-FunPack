// Editor settings: per-browser editor preferences (autocomplete, anchor).
// A section of the unified Settings window. Distinct from Engine settings
// (which configures Studio / Chain Sampler behavior on the project).
(function () {
  const { el } = window.dom;
  const S = window.Store;
  const API = window.MovieEditorAPI;

  function labeled(text, control) {
    const l = el("label", "field"); l.append(el("span", null, text)); l.append(control); return l;
  }

  // "Anchor as guide" i2v-node bypass: declare a custom i2v node + the input/state to force
  // on anchor_guide runs so the latent stays empty. Built-in pipeline needs none of this.
  function anchorGuideSection() {
    const wrap = el("div", "es-section");
    wrap.append(el("div", "es-section-title", "Anchor as guide"));
    wrap.append(el("div", "es-hint",
      "The “Anchor as guide” scene source steers from a frame-0 guide while the latent stays empty "
      + "(text-to-video). The built-in FunPack pipeline needs nothing extra. If you run a CUSTOM i2v node, "
      + "declare it here so it’s forced to a pass-through state on those runs."));

    const has = el("label", "chk es-toggle");
    const cb = el("input"); cb.type = "checkbox"; cb.checked = !!S.getEditorSetting("anchorGuideHasI2v");
    has.append(cb, el("span", null, "I have a custom i2v node to bypass"));
    wrap.append(has);

    const detail = el("div", "es-subfields");
    detail.style.display = cb.checked ? "" : "none";
    wrap.append(detail);

    let slots = [];
    let cur = { ...(S.getEditorSetting("anchorGuideBypass") || {}) };
    const persist = () => S.setEditorSetting("anchorGuideBypass", { ...cur });

    cb.onchange = () => {
      S.setEditorSetting("anchorGuideHasI2v", cb.checked);
      detail.style.display = cb.checked ? "" : "none";
      if (cb.checked) loadSlots();
    };

    function loadSlots() {
      detail.replaceChildren(el("div", "es-hint", "Loading pipeline nodes…"));
      const pid = (S.get().project || {}).id || null;
      API.getModels(pid)
        .then((m) => { slots = (m && m.slots) || []; renderNode(); })
        .catch(() => detail.replaceChildren(el("div", "es-hint", "Could not load pipeline nodes.")));
    }

    function renderNode() {
      detail.replaceChildren();
      if (!slots.length) {
        detail.append(el("div", "es-hint",
          "No custom nodes in your pipeline yet — add your i2v node in Models, then come back here."));
        return;
      }
      const sel = el("select");
      sel.append(new Option("— pick i2v node —", ""));
      slots.forEach((s) => {
        const o = new Option(s.label || s.node_class || s.id, s.id);
        if (cur.node === s.id) o.selected = true;
        sel.append(o);
      });
      sel.onchange = () => {
        const s = slots.find((x) => x.id === sel.value);
        cur = { node: sel.value, nodeClass: s ? s.node_class : null,
                label: s ? (s.label || s.node_class) : null, input: "", value: null };
        persist(); renderNode();
      };
      detail.append(labeled("i2v node", sel));
      if (!cur.node) return;
      const inputsBox = el("div");
      inputsBox.append(el("div", "es-hint", "Loading inputs…"));
      detail.append(inputsBox);
      API.nodeSpec(cur.nodeClass)
        .then((spec) => renderInputs(inputsBox, (spec && spec.inputs) || []))
        .catch(() => inputsBox.replaceChildren(el("div", "es-hint", "Could not load node inputs.")));
    }

    function defaultFor(f) {
      if (!f) return null;
      if (f.kind === "boolean") return false;
      if (f.kind === "combo") return f.default != null ? f.default : (f.choices && f.choices[0]);
      return f.default != null ? f.default : null;
    }

    function renderInputs(box, fields) {
      box.replaceChildren();
      if (!fields.length) {
        box.append(el("div", "es-hint", "This node has no settable widget inputs to force.")); return;
      }
      const sel = el("select");
      sel.append(new Option("— pick input to force —", ""));
      fields.forEach((f) => {
        const o = new Option(f.name + (f.kind ? ` · ${f.kind}` : ""), f.name);
        if (cur.input === f.name) o.selected = true;
        sel.append(o);
      });
      sel.onchange = () => {
        const f = fields.find((x) => x.name === sel.value);
        cur = { ...cur, input: sel.value, value: defaultFor(f) };
        persist(); renderInputs(box, fields);
      };
      box.append(labeled("Input to force", sel));
      const f = fields.find((x) => x.name === cur.input);
      if (f) {
        box.append(labeled("State (forced value)", valueControl(f)));
        box.append(el("div", "es-hint",
          `On Anchor-as-guide runs, ${cur.label || "this node"} · ${f.name} is forced to this value.`));
      }
    }

    function valueControl(f) {
      let ctrl;
      if (f.kind === "boolean") {
        ctrl = el("input"); ctrl.type = "checkbox"; ctrl.checked = !!cur.value;
        ctrl.onchange = () => { cur = { ...cur, value: ctrl.checked }; persist(); };
      } else if (f.kind === "combo") {
        ctrl = el("select");
        (f.choices || []).forEach((c) => {
          const o = new Option(String(c), String(c));
          if (String(cur.value) === String(c)) o.selected = true;
          ctrl.append(o);
        });
        ctrl.onchange = () => { cur = { ...cur, value: ctrl.value }; persist(); };
      } else if (f.kind === "int" || f.kind === "float") {
        ctrl = el("input"); ctrl.type = "number"; if (f.kind === "float") ctrl.step = "0.01";
        ctrl.value = cur.value != null ? cur.value : (f.default != null ? f.default : 0);
        ctrl.onchange = () => {
          cur = { ...cur, value: f.kind === "int" ? parseInt(ctrl.value || "0", 10) : parseFloat(ctrl.value || "0") };
          persist();
        };
      } else {
        ctrl = el("input"); ctrl.type = "text"; ctrl.value = cur.value != null ? cur.value : "";
        ctrl.onchange = () => { cur = { ...cur, value: ctrl.value }; persist(); };
      }
      return ctrl;
    }

    if (cb.checked) loadSlots();
    return wrap;
  }

  // Shortcut revolver: server-side library setting (not per-browser) — the no-repeat cycle
  // is shared by previews and generation, so the toggle lives with the shortcut DB.
  function revolverSection() {
    const wrap = el("div", "es-section");
    wrap.append(el("div", "es-section-title", "Shortcut revolver"));
    wrap.append(el("div", "es-hint",
      "For shortcuts with several replacements: remember which replacement fired and don’t reuse it "
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
      mainCb.disabled = false;
      randCb.disabled = !mainCb.checked;
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

  function mount(body) {
    const content = el("div", "es-content");
    content.append(toggleRow(
      "Prompt autocomplete",
      "Suggest matching shortcuts while you type in the global prompt and scene prompts. Shows the trigger, its prompt, and category.",
      "autocomplete"));
    content.append(toggleRow(
      "Shortcut ideas",
      "A 💡 next to prompt fields lights up when shortcut categories in your library (camera moves, sub-actions…) aren't used in the prompt. Click it for insertable ideas — it never pops up on its own.",
      "suggestions"));
    content.append(toggleRow(
      "Use anchor",
      "Text before the first split trigger is a shared anchor prepended to every scene. Turn off to make that leading text Scene 1 instead.",
      "anchorEnabled"));
    content.append(revolverSection());
    content.append(anchorGuideSection());
    body.append(content);
  }

  window.SettingsWindow.register({
    id: "editor", group: "", title: "Editor",
    subtitle: "Per-browser preferences — they apply to this browser, not the project.",
    keywords: "autocomplete anchor prompt shortcuts ideas suggestions i2v bypass guide preferences revolver random replacements cycle",
    iconBg: "linear-gradient(180deg,#6aa9ff,#3b6fd9)",
    icon: '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="#fff" stroke-width="1.5" stroke-linecap="round"><path d="M2 4.5h12M2 8h12M2 11.5h12"/><circle cx="6" cy="4.5" r="1.7" fill="#fff" stroke="none"/><circle cx="11" cy="8" r="1.7" fill="#fff" stroke="none"/><circle cx="5" cy="11.5" r="1.7" fill="#fff" stroke="none"/></svg>',
    mount,
  });

  window.EditorSettingsModal = {
    open: () => window.SettingsWindow.open("editor"),
    close: () => window.SettingsWindow.close(),
  };
})();
