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
  const expanded = new Set();     // slot ids currently expanded (collapsed by default)

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

  async function persist() {
    try { config = await API.saveModels(config); window.dispatchEvent(new Event("funpack-models-changed")); }
    catch (e) { console.error(e); }
  }

  // ── "expose to main editor" (eye toggle) ─────────────────────────────────────
  function isExposed(slot, name) { return (slot.exposed || []).some((e) => e.name === name); }
  function toggleExpose(slot, spec) {
    slot.exposed = slot.exposed || [];
    if (isExposed(slot, spec.name)) slot.exposed = slot.exposed.filter((e) => e.name !== spec.name);
    else slot.exposed.push({ name: spec.name, kind: spec.kind,
                             choices: spec.kind === "combo" ? (spec.choices || []) : undefined, label: spec.name });
  }
  function eyeButton(slot, spec) {
    const b = el("button", "eye-btn" + (isExposed(slot, spec.name) ? " on" : ""), "◉");
    b.type = "button";
    b.title = isExposed(slot, spec.name) ? "Hide from main editor window" : "Show in main editor window";
    b.onclick = async (e) => { e.preventDefault(); e.stopPropagation(); toggleExpose(slot, spec); await persist(); render(); };
    return b;
  }

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

  // ── validation ───────────────────────────────────────────────────────────────
  // Roles the fixed FunPack path cannot generate without (at least one slot each).
  const ESSENTIAL = [["unet", "Unet / Diffusion Model"], ["clip", "CLIP / Text Encoder"], ["video_vae", "Video VAE"]];

  function validateSlot(slot) {
    const issues = [];
    const spec = specFor(slot);
    if (!spec) {
      issues.push({ level: "error", msg: "Node spec unavailable — node not installed, or ComfyUI offline." });
      return issues;
    }
    (spec.inputs || []).forEach((w) => {
      if (!w.required) return;
      const v = slot.inputs[w.name];
      if (w.kind === "combo" && (!w.choices || !w.choices.length)) {
        issues.push({ level: "error", msg: `“${w.name}” has no installed options to pick from.` });
      } else if (v == null || v === "") {
        issues.push({ level: "error", msg: `Required field “${w.name}” is empty.` });
      }
    });
    const outs = spec.outputs || [];
    const wires = slot.wires || {};
    if (outs.length && !outs.some((o) => wires[o.name])) {
      issues.push({ level: "warn", msg: "No outputs wired — this node feeds nothing." });
    }
    outs.forEach((o) => {
      const cur = wires[o.name];
      if (cur && !destinations(slot, o.type).some((d) => d.value === cur)) {
        issues.push({ level: "error", msg: `“${o.name}” is wired to a destination that no longer exists.` });
      }
    });
    return issues;
  }

  function missingEssentials() {
    return ESSENTIAL.filter(([role]) => !config.slots.some((s) => s.role === role)).map(([, label]) => label);
  }

  function validation() {
    const perSlot = {};
    let errors = 0, warns = 0;
    config.slots.forEach((s) => {
      const list = validateSlot(s);
      perSlot[s.id] = list;
      list.forEach((i) => (i.level === "error" ? errors++ : warns++));
    });
    return { perSlot, errors, warns, missing: missingEssentials() };
  }

  function issuesBox(list) {
    if (!list || !list.length) return null;
    const box = el("div", "issue-box");
    list.forEach((i) => {
      const row = el("div", "issue issue-" + i.level);
      row.append(el("span", "issue-dot", i.level === "error" ? "✕" : "▲"));
      row.append(el("span", "issue-msg", i.msg));
      box.append(row);
    });
    return box;
  }

  // ── slot row (configured) ────────────────────────────────────────────────────
  function slotRow(slot, issues) {
    const role = roles.find((r) => r.key === slot.role);
    const isExp = expanded.has(slot.id);
    const card = el("div", "slot-card" + (isExp ? " open" : ""));
    const errs = (issues || []).filter((i) => i.level === "error").length;
    const warns = (issues || []).filter((i) => i.level === "warn").length;
    if (errs) card.classList.add("slot-bad");
    else if (warns) card.classList.add("slot-warn");

    const head = el("div", "slot-head");
    head.append(el("span", "slot-chev", isExp ? "▾" : "▸"));
    head.append(el("span", "slot-role", role ? role.label : slot.role));
    head.append(el("span", "slot-node", slot.node_class));
    const nExp = (slot.exposed || []).length;
    if (nExp) head.append(el("span", "slot-badge exposed", `◉ ${nExp}`));
    if (errs) head.append(el("span", "slot-badge bad", `${errs} error${errs > 1 ? "s" : ""}`));
    else if (warns) head.append(el("span", "slot-badge warn", `${warns} warning${warns > 1 ? "s" : ""}`));
    else head.append(el("span", "slot-badge ok", "ready"));
    const rm = el("button", "btn ghost tiny danger", "remove");
    rm.onclick = async (e) => { e.stopPropagation(); config.slots = config.slots.filter((s) => s.id !== slot.id); expanded.delete(slot.id); await persist(); render(); };
    head.append(rm);
    head.onclick = () => { isExp ? expanded.delete(slot.id) : expanded.add(slot.id); render(); };
    card.append(head);

    if (!isExp) return card;  // collapsed: header only

    const ib = issuesBox(issues);
    if (ib) card.append(ib);

    const cand = specFor(slot);
    if (cand && cand.inputs.length) {
      const grid = el("div", "slot-fields");
      cand.inputs.forEach((spec) => {
        const f = widgetField(spec, slot.inputs[spec.name], async (v) => { slot.inputs[spec.name] = v; await persist(); });
        f.classList.add("with-eye");
        f.append(eyeButton(slot, spec));
        grid.append(f);
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

    const search = el("input", "composer-search"); search.type = "text"; search.placeholder = "Filter nodes…"; search.style.display = "none";
    const nodeSel = el("select"); nodeSel.disabled = true; nodeSel.append(new Option("Node…", ""));
    const addBtn = el("button", "btn primary", "Add"); addBtn.disabled = true;
    const preview = el("div", "composer-preview");

    let curRole = "", curCand = null, curList = [];

    function showPreview() {
      clear(preview);
      if (!curCand) return;
      const d = defaultsFor(curCand);
      (curCand.inputs || []).forEach((spec) => preview.append(widgetField(spec, d[spec.name], (v) => { d[spec.name] = v; })));
      preview._draft = d;
    }

    function fill(list, placeholder) {
      nodeSel.innerHTML = ""; nodeSel.append(new Option(placeholder, ""));
      list.forEach((c) => nodeSel.append(new Option(`${c.display_name}  ·  ${c.class}`, c.class)));
      nodeSel.disabled = !list.length;
    }

    function applyFilter() {
      const q = search.value.trim().toLowerCase();
      const f = q ? curList.filter((c) => (c.display_name + " " + c.class + " " + (c.category || "")).toLowerCase().includes(q)) : curList;
      const shown = f.slice(0, 300);
      fill(shown, f.length ? `${f.length} node${f.length > 1 ? "s" : ""} — pick one…${f.length > 300 ? " (showing first 300)" : ""}` : "No match");
    }
    search.oninput = applyFilter;

    typeSel.onchange = async () => {
      curRole = typeSel.value; curCand = null; curList = []; addBtn.disabled = true; clear(preview);
      search.style.display = curRole === "__any__" ? "" : "none"; search.value = "";
      nodeSel.innerHTML = ""; nodeSel.append(new Option("Loading…", "")); nodeSel.disabled = true;
      if (!curRole) { nodeSel.innerHTML = ""; nodeSel.append(new Option("Node…", "")); return; }
      try {
        if (curRole === "__any__") {
          curList = await ensureAllNodes();
          applyFilter();
        } else {
          curList = await candidates(curRole);
          fill(curList, curList.length ? "Loader Node…" : "No matching nodes");
        }
      } catch (e) {
        nodeSel.innerHTML = ""; nodeSel.append(new Option("Error: " + (e && e.message ? e.message : "node list unavailable"), ""));
        console.error("[Models] node list failed:", e);
      }
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
      search.style.display = "none"; search.value = "";
      addBtn.disabled = true; curCand = null; curList = []; clear(preview); render();
    };

    row.append(typeSel); row.append(search); row.append(nodeSel); row.append(addBtn);
    box.append(row); box.append(preview);
    return box;
  }

  function summaryBanner(v) {
    if (!config.slots.length) return null;
    const bits = [];
    if (v.errors) bits.push(`${v.errors} error${v.errors > 1 ? "s" : ""}`);
    if (v.warns) bits.push(`${v.warns} warning${v.warns > 1 ? "s" : ""}`);
    if (v.missing.length) bits.push(`missing: ${v.missing.join(", ")}`);
    const ok = !v.errors && !v.missing.length;
    const bar = el("div", "valid-banner " + (v.errors || v.missing.length ? "bad" : (v.warns ? "warn" : "ok")));
    bar.append(el("span", "valid-dot", ok ? "✓" : "!"));
    bar.append(el("span", null, ok ? (v.warns ? `Config usable — ${bits.join(", ")}.` : "Config complete — all slots ready.")
                                   : `Config incomplete — ${bits.join(", ")}.`));
    return bar;
  }

  function body() {
    const b = el("div", "models-body");
    b.append(composer());
    const v = validation();
    const banner = summaryBanner(v);
    if (banner) b.append(banner);
    const list = el("div", "slot-list");
    if (!config.slots.length) list.append(el("div", "empty-stage", "No models configured yet. Add Unet, VAEs, CLIP, and pipeline nodes above."));
    else config.slots.forEach((s) => list.append(slotRow(s, v.perSlot[s.id])));
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
