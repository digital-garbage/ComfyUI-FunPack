// "Models" settings modal: configure pluggable node slots for the fixed pipeline.
// Pick a Model Type -> a Loader Node -> the editor exposes that node's inputs.
(function () {
  const { el, clear } = window.dom;
  const API = window.MovieEditorAPI;

  let roles = [];                 // [{key,label,category}]
  const candCache = {};           // role -> [candidate]
  let config = { slots: [] };     // {slots:[{id,role,node_class,inputs:{}}]}
  let overlay = null;

  function uid() { return Math.random().toString(36).slice(2, 9); }

  async function ensureRoles() { if (!roles.length) roles = (await API.nodeRoles()).roles || []; }
  async function candidates(role, refresh) {
    if (!candCache[role] || refresh) candCache[role] = (await API.nodeCandidates(role, refresh)).candidates || [];
    return candCache[role];
  }
  function candidate(role, cls) { return (candCache[role] || []).find((c) => c.class === cls); }

  async function persist() { try { config = await API.saveModels(config); } catch (e) { console.error(e); } }

  // ── widget field rendering from object_info spec ─────────────────────────────
  function widgetField(spec, value, onChange) {
    const wrap = el("label", "field");
    wrap.append(el("span", null, spec.name + (spec.required ? "" : "  ·opt")));
    let ctrl;
    if (spec.kind === "combo") {
      ctrl = el("select");
      (spec.choices || []).forEach((c) => { const o = el("option", null, String(c)); o.value = c; if (c === value) o.selected = true; ctrl.append(o); });
      if (!spec.choices || !spec.choices.length) { ctrl.append(el("option", null, "(none installed)")); ctrl.disabled = true; }
      ctrl.onchange = () => onChange(ctrl.value);
    } else if (spec.kind === "boolean") {
      ctrl = el("input"); ctrl.type = "checkbox"; ctrl.checked = !!value; ctrl.style.width = "auto";
      ctrl.onchange = () => onChange(ctrl.checked);
    } else if (spec.kind === "int" || spec.kind === "float") {
      ctrl = el("input"); ctrl.type = "number"; if (spec.kind === "float") ctrl.step = "any";
      ctrl.value = value != null ? value : (spec.default != null ? spec.default : "");
      ctrl.oninput = () => onChange(spec.kind === "int" ? parseInt(ctrl.value || "0", 10) : parseFloat(ctrl.value || "0"));
    } else {
      ctrl = el("input"); ctrl.type = "text"; ctrl.value = value != null ? value : (spec.default || "");
      ctrl.oninput = () => onChange(ctrl.value);
    }
    wrap.append(ctrl);
    return wrap;
  }

  function defaultsFor(cand) {
    const o = {};
    (cand.inputs || []).forEach((s) => { if (s.default != null) o[s.name] = s.default; else if (s.kind === "combo" && s.choices?.length) o[s.name] = s.choices[0]; });
    return o;
  }

  // ── slot row (configured) ────────────────────────────────────────────────────
  function slotRow(slot) {
    const role = roles.find((r) => r.key === slot.role);
    const card = el("div", "slot-card");
    const head = el("div", "slot-head");
    head.append(el("span", "slot-role", role ? role.label : slot.role));
    head.append(el("span", "slot-node", slot.node_class));
    const rm = el("button", "btn ghost tiny danger", "remove");
    rm.onclick = async () => { config.slots = config.slots.filter((s) => s.id !== slot.id); await persist(); render(); };
    head.append(rm);
    card.append(head);

    const cand = candidate(slot.role, slot.node_class);
    if (cand && cand.inputs.length) {
      const grid = el("div", "slot-fields");
      cand.inputs.forEach((spec) => {
        grid.append(widgetField(spec, slot.inputs[spec.name], async (v) => { slot.inputs[spec.name] = v; await persist(); }));
      });
      card.append(grid);
    } else {
      card.append(el("div", "pj-meta", "No configurable inputs."));
    }
    return card;
  }

  // ── add-model composer ───────────────────────────────────────────────────────
  function composer() {
    const box = el("div", "composer");
    box.append(el("div", "composer-title", "Add Model / Node"));

    const row = el("div", "composer-row");
    const typeSel = el("select");
    const ph = el("option", null, "Model Type…"); ph.value = ""; typeSel.append(ph);
    const cats = {};
    roles.forEach((r) => { (cats[r.category] = cats[r.category] || []).push(r); });
    Object.keys(cats).forEach((cat) => {
      const og = document.createElement("optgroup"); og.label = cat;
      cats[cat].forEach((r) => { const o = el("option", null, r.label); o.value = r.key; og.append(o); });
      typeSel.append(og);
    });

    const nodeSel = el("select"); nodeSel.disabled = true; nodeSel.append(new Option("Loader Node…", ""));
    const addBtn = el("button", "btn primary", "Add"); addBtn.disabled = true;
    const preview = el("div", "composer-preview");

    let curRole = "", curCand = null;

    typeSel.onchange = async () => {
      curRole = typeSel.value; curCand = null; addBtn.disabled = true; clear(preview);
      nodeSel.innerHTML = ""; nodeSel.append(new Option("Loading…", "")); nodeSel.disabled = true;
      if (!curRole) { nodeSel.innerHTML = ""; nodeSel.append(new Option("Loader Node…", "")); return; }
      let list = [];
      try { list = await candidates(curRole); } catch (e) { nodeSel.innerHTML = ""; nodeSel.append(new Option("ComfyUI offline", "")); return; }
      nodeSel.innerHTML = ""; nodeSel.append(new Option(list.length ? "Loader Node…" : "No matching nodes", ""));
      list.forEach((c) => nodeSel.append(new Option(c.display_name + "  ·  " + c.class, c.class)));
      nodeSel.disabled = !list.length;
    };
    nodeSel.onchange = () => {
      curCand = candidate(curRole, nodeSel.value); addBtn.disabled = !curCand; clear(preview);
      if (curCand) {
        const d = defaultsFor(curCand);
        (curCand.inputs || []).forEach((spec) => preview.append(widgetField(spec, d[spec.name], (v) => { d[spec.name] = v; })));
        preview._draft = d;
      }
    };
    addBtn.onclick = async () => {
      if (!curCand) return;
      config.slots.push({ id: uid(), role: curRole, node_class: curCand.class, inputs: preview._draft || defaultsFor(curCand) });
      await persist();
      typeSel.value = ""; nodeSel.innerHTML = ""; nodeSel.append(new Option("Loader Node…", "")); nodeSel.disabled = true;
      addBtn.disabled = true; clear(preview); render();
    };

    row.append(typeSel); row.append(nodeSel); row.append(addBtn);
    box.append(row); box.append(preview);
    return box;
  }

  function body() {
    const b = el("div", "models-body");
    b.append(composer());
    const list = el("div", "slot-list");
    if (!config.slots.length) list.append(el("div", "empty-stage", "No models configured yet. Add Unet, VAEs, CLIP, and pipeline nodes above."));
    else config.slots.forEach((s) => list.append(slotRow(s)));
    b.append(list);
    return b;
  }

  function render() {
    if (!overlay) return;
    const content = overlay.querySelector(".modal-content");
    clear(content);
    content.append(body());
  }

  async function refreshList() {
    Object.keys(candCache).forEach((k) => delete candCache[k]);
    try { await API.refreshModels(); } catch (_) {}
    // re-pull candidates for roles already configured so combos show fresh files
    for (const role of new Set(config.slots.map((s) => s.role))) { try { await candidates(role, true); } catch (_) {} }
    render();
  }

  async function open() {
    await ensureRoles();
    try { config = await API.getModels(); } catch (_) { config = { slots: [] }; }
    // pre-warm candidate cache for configured roles so their fields render
    for (const role of new Set(config.slots.map((s) => s.role))) { try { await candidates(role); } catch (_) {} }

    overlay = el("div", "modal-overlay");
    const modal = el("div", "modal");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", "Models & Pipeline Nodes"));
    const refresh = el("button", "btn ghost", "↻ Refresh model list"); refresh.onclick = refreshList;
    const close = el("button", "btn ghost", "✕"); close.onclick = () => { overlay.remove(); overlay = null; };
    const heRight = el("div", "modal-head-right"); heRight.append(refresh); heRight.append(close);
    head.append(heRight);
    modal.append(head);
    modal.append(el("div", "modal-content"));
    overlay.append(modal);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) { overlay.remove(); overlay = null; } });
    document.body.append(overlay);
    render();
  }

  window.ModelsModal = { open, refresh: async () => { await ensureRoles().catch(() => {}); await (window.MovieEditorAPI.refreshModels().catch(() => {})); } };
})();
