// "Models" settings modal: configure pluggable node slots for the fixed pipeline.
// Pick a Model Type -> a Loader Node -> the editor exposes that node's inputs.
(function () {
  const { el, clear } = window.dom;
  const API = window.MovieEditorAPI;

  let roles = [];                 // [{key,label,category}]
  let ports = [];                 // pipeline connection points [{id,type,label}]
  let allNodes = null;            // [{class,display_name,category}] for "any node" picker
  const candCache = {};           // role -> [candidate]
  const specByClass = {};         // class -> full node spec (inputs/outputs/connection_inputs)
  let config = { slots: [] };     // {slots:[{id,role,node_class,inputs:{},wires:{outName:value}}]}
  let overlay = null;

  function roleLabel(key) { const r = roles.find((x) => x.key === key); return r ? r.label : (key === "custom" ? "Node" : key); }

  function uid() { return Math.random().toString(36).slice(2, 9); }

  async function ensureRoles() { if (!roles.length) roles = (await API.nodeRoles()).roles || []; }
  async function ensureAllNodes() { if (!allNodes) allNodes = (await API.allNodes()).nodes || []; return allNodes; }

  function cache(spec) { if (spec && spec.class) specByClass[spec.class] = spec; return spec; }
  async function candidates(role, refresh) {
    if (!candCache[role] || refresh) {
      candCache[role] = (await API.nodeCandidates(role, refresh)).candidates || [];
      candCache[role].forEach(cache);
    }
    return candCache[role];
  }
  async function loadSpec(cls) { if (!specByClass[cls]) cache(await API.nodeSpec(cls)); return specByClass[cls]; }
  function specFor(slot) { return specByClass[slot.node_class] || null; }

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

    const cand = specFor(slot);
    if (cand && cand.inputs.length) {
      const grid = el("div", "slot-fields");
      cand.inputs.forEach((spec) => {
        grid.append(widgetField(spec, slot.inputs[spec.name], async (v) => { slot.inputs[spec.name] = v; await persist(); }));
      });
      card.append(grid);
    }

    // Wiring: each output -> a destination (pipeline port or another node's input).
    if (cand && (cand.outputs || []).length) {
      slot.wires = slot.wires || {};
      const wbox = el("div", "wire-box");
      wbox.append(el("div", "wire-title", "Wire to"));
      cand.outputs.forEach((out) => {
        const row = el("div", "wire-row");
        row.append(el("span", "wire-out", `${out.name} (${out.type})`));
        row.append(el("span", "wire-arrow", "→"));
        row.append(destSelect(slot, out));
        wbox.append(row);
      });
      card.append(wbox);
    }
    return card;
  }

  function destinations(slot, type) {
    const out = [{ value: "", label: "— unwired —" }];
    ports.filter((p) => p.type === type).forEach((p) => out.push({ value: "port:" + p.id, label: p.label }));
    config.slots.filter((s) => s.id !== slot.id).forEach((s2) => {
      const c2 = specFor(s2);
      (c2?.connection_inputs || []).filter((ci) => ci.type === type).forEach((ci) =>
        out.push({ value: `node:${s2.id}:${ci.name}`, label: `${roleLabel(s2.role)} · ${ci.name}` }));
    });
    return out;
  }

  function destSelect(slot, out) {
    const sel = el("select", "wire-select");
    const cur = (slot.wires || {})[out.name] || "";
    destinations(slot, out.type).forEach((d) => { const o = el("option", null, d.label); o.value = d.value; if (d.value === cur) o.selected = true; sel.append(o); });
    if (cur && ![...sel.options].some((o) => o.value === cur)) { const o = el("option", null, cur + " (missing)"); o.value = cur; o.selected = true; sel.append(o); }
    sel.onchange = async () => { slot.wires = slot.wires || {}; slot.wires[out.name] = sel.value; await persist(); };
    return sel;
  }

  // ── add-model composer ───────────────────────────────────────────────────────
  function composer() {
    const box = el("div", "composer");
    box.append(el("div", "composer-title", "Add Model / Node"));

    const row = el("div", "composer-row");
    const typeSel = el("select");
    typeSel.append(new Option("Model Type…", ""));
    const cats = {};
    roles.forEach((r) => { (cats[r.category] = cats[r.category] || []).push(r); });
    Object.keys(cats).forEach((cat) => {
      const og = document.createElement("optgroup"); og.label = cat;
      cats[cat].forEach((r) => { og.append(new Option(r.label, r.key)); });
      typeSel.append(og);
    });
    const ogAny = document.createElement("optgroup"); ogAny.label = "Advanced";
    ogAny.append(new Option("Any node…", "__any__")); typeSel.append(ogAny);

    const nodeSel = el("select"); nodeSel.disabled = true; nodeSel.append(new Option("Node…", ""));
    const addBtn = el("button", "btn primary", "Add"); addBtn.disabled = true;
    const preview = el("div", "composer-preview");

    let curRole = "", curCand = null;

    function showPreview() {
      clear(preview);
      if (!curCand) return;
      const d = defaultsFor(curCand);
      (curCand.inputs || []).forEach((spec) => preview.append(widgetField(spec, d[spec.name], (v) => { d[spec.name] = v; })));
      preview._draft = d;
    }

    typeSel.onchange = async () => {
      curRole = typeSel.value; curCand = null; addBtn.disabled = true; clear(preview);
      nodeSel.innerHTML = ""; nodeSel.append(new Option("Loading…", "")); nodeSel.disabled = true;
      if (!curRole) { nodeSel.innerHTML = ""; nodeSel.append(new Option("Node…", "")); return; }
      try {
        let list;
        if (curRole === "__any__") {
          list = await ensureAllNodes();
          nodeSel.innerHTML = ""; nodeSel.append(new Option("Search a node…", ""));
          list.forEach((c) => nodeSel.append(new Option(`${c.display_name}  ·  ${c.class}`, c.class)));
        } else {
          list = await candidates(curRole);
          nodeSel.innerHTML = ""; nodeSel.append(new Option(list.length ? "Loader Node…" : "No matching nodes", ""));
          list.forEach((c) => nodeSel.append(new Option(`${c.display_name}  ·  ${c.class}`, c.class)));
        }
        nodeSel.disabled = !list.length;
      } catch (e) { nodeSel.innerHTML = ""; nodeSel.append(new Option("ComfyUI offline", "")); }
    };
    nodeSel.onchange = async () => {
      addBtn.disabled = true; curCand = null; clear(preview);
      if (!nodeSel.value) return;
      curCand = specByClass[nodeSel.value];
      if (!curCand) { try { curCand = await loadSpec(nodeSel.value); } catch (e) { return; } }
      addBtn.disabled = !curCand; showPreview();
    };
    addBtn.onclick = async () => {
      if (!curCand) return;
      const role = curRole === "__any__" ? "custom" : curRole;
      config.slots.push({ id: uid(), role, node_class: curCand.class, inputs: preview._draft || defaultsFor(curCand), wires: {} });
      await persist();
      typeSel.value = ""; nodeSel.innerHTML = ""; nodeSel.append(new Option("Node…", "")); nodeSel.disabled = true;
      addBtn.disabled = true; curCand = null; clear(preview); render();
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

  async function prewarmSpecs() {
    // resolve the spec for every configured node (by class) so fields + wiring render
    for (const cls of new Set(config.slots.map((s) => s.node_class))) { try { await loadSpec(cls); } catch (_) {} }
  }

  async function refreshList() {
    Object.keys(candCache).forEach((k) => delete candCache[k]);
    Object.keys(specByClass).forEach((k) => delete specByClass[k]);
    allNodes = null;
    try { await API.refreshModels(); } catch (_) {}
    await prewarmSpecs();
    render();
  }

  async function open() {
    await ensureRoles();
    try { ports = (await API.pipelinePorts()).ports || []; } catch (_) { ports = []; }
    try { config = await API.getModels(); } catch (_) { config = { slots: [] }; }
    await prewarmSpecs();

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
