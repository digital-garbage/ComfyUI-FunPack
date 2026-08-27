// "Models" settings section: configure pluggable node slots for the fixed pipeline.
// Pick a Model Type -> a Loader Node -> the editor exposes that node's inputs.
// Lives in the unified Settings window; ModelsModal.open() deep-links to it.
(function () {
  const { el, clear } = window.dom;
  const API = window.MovieEditorAPI;
  // Easy Gen has no per-scene inspector / main editor window for an exposed widget or
  // bypass toggle to actually surface in — the eye buttons below would just be inert
  // clutter (or worse, look actionable and do nothing visible), so they're hidden there.
  const EASY = () => !!window.FunPackMode?.isSimple();

  let roles = [];                 // [{key,label,category}]
  let ports = [];                 // pipeline connection points [{id,type,label}]
  let allNodes = null;            // [{class,display_name,category}] for "any node" picker
  const candCache = {};           // role -> [candidate]
  const specByClass = {};         // class -> full node spec (inputs/outputs/connection_inputs)
  let config = { slots: [] };     // {slots:[{id,role,role_label,node_class,inputs:{},wires:{},input_sources:{}}]}
  let coreNodes = [];             // built-in pipeline description (from /core-graph)
  let coreOpen = false;           // built-in pipeline panel expanded?
  let coreProducers = [];          // [{id,type,label}]
  let requirements = [];           // [{id,type,label,required,role_hint,hint}]
  let wiringRules = {};            // guided wiring rules from /pipeline-ports
  let defaultSlots = [];           // what a project with no pipeline yet starts as
  let container = null;           // mounted content root inside the Settings window
  let linkMode = false;           // selecting inputs to bind into one shared control
  let linkSel = [];               // [{slotId, input, kind, choices, label}]

  function roleLabel(key) { const r = roles.find((x) => x.key === key); return r ? r.label : (key === "custom" ? "Node" : (key || "?")); }
  function slotName(slot) { return slot.label || slot.node_class; }

  // FunPack's own nodes say what they ARE, not what class they are. Everything else keeps
  // its class name — that difference is the point: these are the built-in, wired-for-you
  // pieces, and a third-party node is an addition that does not need to look native.
  const BUILT_IN = {
    FunPackDiffusionModelLoader: { icon: "🧠", label: "Diffusion model" },
    FunPackCLIPLoader:           { icon: "🔤", label: "CLIP model" },
    FunPackVAELoader:            { icon: "🎞️", label: "VAE" },
    FunPackLoraLoader:           { icon: "🧩", label: "LoRA" },
    FunPackRefinementKeyLoader:  { icon: "🔑", label: "Refinement key" },
  };
  function builtIn(slot) { return BUILT_IN[slot?.node_class] || null; }
  // Full display name for use in pickers: "RoleLabel · NodeName"
  function slotFullLabel(slot) {
    const rl = slot.role_label || roleLabel(slot.role);
    const nn = slotName(slot);
    return rl && rl !== nn ? `${rl} · ${nn}` : nn;
  }
  function slotById(id) { return config.slots.find((s) => s.id === id); }

  function pipelineLocked() {
    return !config.disable_core && !config.full_control;
  }

  function defaultExtrasForRole(role, cand) {
    if (!pipelineLocked() || !cand) return {};
    const extras = {};
    const wires = wiringRules.default_wires?.[role];
    if (wires) {
      extras.wires = {};
      (cand.outputs || []).forEach((o) => {
        const t = wires[o.name] || wires[o.type];
        if (t) extras.wires[o.name] = t;
      });
    }
    const srcs = wiringRules.default_input_sources?.[role];
    if (srcs) extras.input_sources = { ...srcs };
    return extras;
  }

  function allowedDestinations(slot, out) {
    const all = destinations(slot, out.type);
    if (!pipelineLocked()) return all;
    const role = slot.role || "custom";
    const rules = wiringRules.role_targets?.[role] || [];
    const hidden = new Set(wiringRules.guided_hidden_ports || []);
    // Must match pipeline_wiring.allowed_port_ids exactly: a role's own rules when it has any
    // for this type, otherwise every port of the type. Filtering harder than the builder
    // validates is what made saved wires read back as "(not allowed)".
    const explicit = rules
      .filter((r) => r.type === out.type && (r.output_name == null || r.output_name === out.name))
      .map((r) => "port:" + r.port);
    const fallback = rules.some((r) => r.type === out.type)
      ? []
      : (wiringRules.type_fallback_ports?.[out.type] || []).map((p) => "port:" + p);
    const allowedPorts = [...explicit, ...fallback].filter((p, i, a) => a.indexOf(p) === i);
    return all.filter((d) => {
      if (!d.value.startsWith("port:")) return true;
      const id = d.value.slice(5);
      if (hidden.has(id)) return false;
      return allowedPorts.includes(d.value);
    });
  }

  function portToOpenCore(portId) {
    const hit = (wiringRules.open_core_ports || []).find((p) => p.port === portId);
    return hit ? [hit.core_id, hit.input] : null;
  }

  function _clearPortWires(portId, exceptSlotId, exceptOut) {
    const portTarget = "port:" + portId;
    for (const slot of config.slots) {
      slot.wires = slot.wires || {};
      for (const outName of Object.keys(slot.wires)) {
        if (slot.id === exceptSlotId && outName === exceptOut) continue;
        slot.wires[outName] = wireTargets(slot.wires[outName]).filter((t) => t !== portTarget);
      }
    }
  }

  // Keep core_overrides and slot port wires in sync so loader wiring survives reload.
  function reconcileOpenPortWiring() {
    config.core_overrides = config.core_overrides || {};
    for (const slot of config.slots) {
      for (const [outName, raw] of Object.entries(slot.wires || {})) {
        for (const t of wireTargets(raw)) {
          if (!t.startsWith("port:")) continue;
          const portId = t.slice(5);
          const map = portToOpenCore(portId);
          if (!map) continue;
          const [cid, inp] = map;
          config.core_overrides[cid] = config.core_overrides[cid] || {};
          config.core_overrides[cid][inp] = `out:${slot.id}:${outName}`;
        }
      }
    }
    for (const entry of wiringRules.open_core_ports || []) {
      const portId = entry.port;
      const cid = entry.core_id;
      const inp = entry.input;
      const src = config.core_overrides?.[cid]?.[inp];
      const parsed = _parseOutSource(src);
      if (!parsed) continue;
      // An override naming a slot that no longer exists is stale — _addWire below would
      // no-op on it while _clearPortWires still stripped everyone ELSE's wire to that port,
      // so the surviving loader lost its wire on load. Drop the override, keep the wire.
      if (!slotById(parsed.slotId)) { delete config.core_overrides[cid][inp]; continue; }
      _clearPortWires(portId, parsed.slotId, parsed.out);
      _addWire(parsed.slotId, parsed.out, "port:" + portId);
    }
  }

  function clearInternalCoreOverrides() {
    if (!config.core_overrides) return;
    const open = new Set(
      (wiringRules.open_core_ports || []).map((p) => `${p.core_id}.${p.input}`),
    );
    for (const cid of Object.keys(config.core_overrides)) {
      for (const inp of Object.keys(config.core_overrides[cid] || {})) {
        if (!open.has(`${cid}.${inp}`)) delete config.core_overrides[cid][inp];
      }
      if (!Object.keys(config.core_overrides[cid] || {}).length) delete config.core_overrides[cid];
    }
  }

  // Overrides are saved per CORE ID, not per node class, so a family switch that replaces a
  // node leaves the old node's input names sitting on the new one — audiodec.audio_vae
  // (LTXVAudioVAEDecode) surviving onto VAEDecodeAudio, which calls its VAE input `vae`. The
  // panel below lists the NEW node's inputs, so the leftover is invisible here; the builder
  // refuses to emit it, but it should not linger in the saved project either.
  function dropOverridesForMissingInputs() {
    if (!config.core_overrides) return;
    const byId = new Map((coreNodes || []).map((n) => [n.id, n]));
    for (const cid of Object.keys(config.core_overrides)) {
      const node = byId.get(cid);
      if (!node) continue;          // not a core node this family has an opinion about
      const names = new Set((node.inputs || []).map((i) => i.name));
      if (!names.size) continue;    // node spec unavailable — leave it alone
      for (const inp of Object.keys(config.core_overrides[cid] || {})) {
        if (!names.has(inp)) delete config.core_overrides[cid][inp];
      }
      if (!Object.keys(config.core_overrides[cid] || {}).length) delete config.core_overrides[cid];
    }
  }

  function allowedSources(slot, ci) {
    const all = sources(slot, ci.type);
    if (!pipelineLocked()) return all;
    // Guided wiring hides core internals, but marked media is user-supplied input, not
    // pipeline plumbing — it stays offered in both modes.
    if (slot.role === "image_processing" && ci.name === "image" && typeAccepts(ci.type, "IMAGE")) {
      return all.filter((s) => !s.value || s.value === "timeline"
        || s.value.startsWith("out:") || s.value.startsWith("ref:"));
    }
    return all.filter((s) => !s.value || s.value.startsWith("out:")
      || s.value === "timeline" || s.value.startsWith("ref:"));
  }

  // ── linked inputs (one control drives several node inputs) ────────────────────
  function ensureLinks() { if (!config.links) config.links = []; return config.links; }
  function linkOf(slotId, input) { return (config.links || []).find((l) => (l.members || []).some((m) => m.slotId === slotId && m.input === input)); }
  function linkSelHas(slotId, input) { return linkSel.some((s) => s.slotId === slotId && s.input === input); }
  function applyLinkValue(link, value) {
    link.value = value;
    (link.members || []).forEach((m) => { const s = slotById(m.slotId); if (s) { s.inputs = s.inputs || {}; s.inputs[m.input] = value; } });
  }

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
  function inputSpecFor(slot, name) { return ((specFor(slot) || {}).inputs || []).find((w) => w.name === name) || null; }

  // Numeric widget bounds/step/default straight from the node's object_info opts, so an
  // exposed control (e.g. a LoRA strength) gets its real min/max/step instead of step=1.
  function numMeta(spec) {
    const o = (spec && spec.options) || {};
    const m = {};
    if (o.min != null) m.min = o.min;
    if (o.max != null) m.max = o.max;
    if (o.step != null) m.step = o.step;
    const def = o.default != null ? o.default : (spec && spec.default);
    if (def != null) m.default = def;
    return m;
  }
  function applyNumMeta(ctrl, meta, isFloat) {
    if (meta.min != null) ctrl.min = String(meta.min);
    if (meta.max != null) ctrl.max = String(meta.max);
    if (meta.step != null) ctrl.step = String(meta.step);
    else if (isFloat) ctrl.step = "any";
  }

  // Exposed controls and shared links snapshot their combo `choices` at expose time (so the main
  // editor can render dropdowns without re-fetching object_info). Those snapshots go stale after
  // a model-list refresh — a newly installed LoRA/checkpoint shows up inside the Models menu (live
  // spec) but not in the exposed project-settings dropdown. Re-pull each exposed/link combo's
  // choices from the freshly loaded spec. Returns true if anything changed (caller persists).
  function refreshExposedChoices() {
    let changed = false;
    const syncMeta = (target, w) => {
      if (!w) return;
      if (target.kind === "combo") {
        if (Array.isArray(w.choices) && JSON.stringify(w.choices) !== JSON.stringify(target.choices || [])) {
          target.choices = w.choices.slice(); changed = true;
        }
      } else if (target.kind === "int" || target.kind === "float") {
        const meta = numMeta(w);
        ["min", "max", "step", "default"].forEach((k) => {
          if (meta[k] != null && target[k] !== meta[k]) { target[k] = meta[k]; changed = true; }
        });
      }
    };
    (config.slots || []).forEach((slot) => {
      (slot.exposed || []).forEach((e) => syncMeta(e, inputSpecFor(slot, e.name)));
    });
    (config.links || []).forEach((link) => {
      const first = (link.members || [])[0];
      const slot = first && slotById(first.slotId);
      syncMeta(link, slot && inputSpecFor(slot, first.input));
    });
    return changed;
  }

  // Shared full refresh used by the modal button, the public ModelsModal.refresh() and the menubar:
  // drop caches, refresh the backend model list, reload specs, then re-sync exposed/link choices and
  // persist so the main-editor dropdowns update too. Loads config only when it isn't already held
  // (keeps the open modal's live `config` object so its edit closures stay attached — see persist()).
  async function doFullRefresh() {
    Object.keys(candCache).forEach((k) => delete candCache[k]);
    Object.keys(specByClass).forEach((k) => delete specByClass[k]);
    allNodes = null;
    try { await API.refreshModels(); } catch (_) {}
    if (!config || !config.slots) {
      try { config = await API.getModels(window.Store?.get().project?.id); } catch (_) { config = { slots: [] }; }
    }
    try { coreNodes = (await API.coreGraph(window.Store?.get().project?.id)).nodes || []; } catch (_) {}
    await prewarmSpecs();
    if (refreshExposedChoices()) { try { await persist(); } catch (_) {} }
  }

  async function persistNow() {
    // Keep the SAME `config` object — every wire/widget handler closes over it, so
    // reassigning it to the server's returned object detaches those closures and the
    // next edit wouldn't persist until the node is re-rendered. Save in place instead.
    const familyBefore = config.model_family;
    try {
      // The server detects the family from the checkpoint and answers with it, so a model
      // swap can change the family without anyone touching a control. Copy the answer back
      // onto the SAME object (see above) and migrate the project's frame geometry, which
      // otherwise stays on the previous model's grid and fails the run.
      const saved = await API.saveModels(window.Store?.get().project?.id, config);
      if (saved && typeof saved === "object") {
        if (saved.model_family) config.model_family = saved.model_family;
        config.model_family_probe = saved.model_family_probe;
      }
      window.dispatchEvent(new Event("funpack-models-changed"));
      if (config.model_family && config.model_family !== familyBefore) {
        window.PipelineSetup?.applyFamilyGeometry?.();
      }
    } catch (e) { console.error(e); }
  }

  // ── deferred saving while a node is open ─────────────────────────────────────
  // Editing a node used to save on every change. A save fired between two keystrokes
  // stores what was on screen at that instant — start on 1.0, type 0.5, and the value
  // that reaches disk can be 0.58. Nothing about that is recoverable, because the
  // half-typed number is now the saved number.
  //
  // So the node editor buffers instead: `config` is still mutated in place (wiring edits
  // legitimately touch OTHER slots' wires, so a single-slot draft could not hold them),
  // but persist() is a no-op until Save. Cancel restores the snapshot taken on open.
  let deferSave = false;
  let deferDirty = false;
  let baseline = null;      // config as it was when the node page opened
  let footerEls = null;     // {save, hint} — updated without a re-render while typing

  async function persist() {
    if (deferSave) { deferDirty = true; paintFooter(); return; }
    return persistNow();
  }

  function beginEdit() {
    baseline = JSON.parse(JSON.stringify(config));
    deferSave = true;
    deferDirty = false;
  }

  function endEdit() {
    deferSave = false;
    deferDirty = false;
    baseline = null;
    footerEls = null;
  }

  function restoreBaseline() {
    if (!baseline) return;
    // In place, for the same reason persistNow() saves in place: the open page's
    // handlers all close over this object.
    Object.keys(config).forEach((k) => delete config[k]);
    Object.assign(config, JSON.parse(JSON.stringify(baseline)));
  }

  function paintFooter() {
    if (!footerEls) return;
    footerEls.save.classList.toggle("disabled", !deferDirty);
    footerEls.save.disabled = !deferDirty;
    footerEls.hint.textContent = deferDirty
      ? "Unsaved changes — nothing is written until you press Save."
      : "No changes yet.";
  }

  // ── "expose to main editor" (eye toggle) ─────────────────────────────────────
  function isExposed(slot, name) { return (slot.exposed || []).some((e) => e.name === name); }
  function toggleExpose(slot, spec) {
    slot.exposed = slot.exposed || [];
    if (isExposed(slot, spec.name)) { slot.exposed = slot.exposed.filter((e) => e.name !== spec.name); return; }
    const e = { name: spec.name, kind: spec.kind, label: spec.name };
    if (spec.kind === "combo") e.choices = spec.choices || [];
    else if (spec.kind === "int" || spec.kind === "float") Object.assign(e, numMeta(spec));
    slot.exposed.push(e);
  }
  function eyeButton(slot, spec) {
    const b = el("button", "eye-btn" + (isExposed(slot, spec.name) ? " on" : ""), "◉");
    b.type = "button";
    b.title = isExposed(slot, spec.name) ? "Hide from main editor window" : "Show in main editor window";
    b.onclick = async (e) => { e.preventDefault(); e.stopPropagation(); toggleExpose(slot, spec); await persist(); render(); };
    return b;
  }

  // ── bypass: skip this node's effect (pass its inputs straight through) without
  // losing its configuration — same "expose to main editor" pattern as a widget value.
  const BYPASS_NAME = "__bypass";
  function toggleBypassExpose(slot) {
    slot.exposed = slot.exposed || [];
    if (isExposed(slot, BYPASS_NAME)) { slot.exposed = slot.exposed.filter((e) => e.name !== BYPASS_NAME); return; }
    slot.exposed.push({ name: BYPASS_NAME, kind: "boolean", label: `Bypass ${slotName(slot)}`, isBypass: true });
  }
  function bypassEyeButton(slot) {
    const b = el("button", "eye-btn" + (isExposed(slot, BYPASS_NAME) ? " on" : ""), "◉");
    b.type = "button";
    b.title = isExposed(slot, BYPASS_NAME) ? "Hide bypass toggle from main editor window" : "Show bypass toggle in main editor window";
    b.onclick = async (e) => { e.preventDefault(); e.stopPropagation(); toggleBypassExpose(slot); await persist(); render(); };
    return b;
  }

  // ── widget field rendering from object_info spec ─────────────────────────────
  // ── list inputs ───────────────────────────────────────────────────────────────
  // A FunPack list input is ONE string widget holding a JSON array of rows, with the row
  // shape declared next to it (see widgets.py). ComfyUI has no repeatable input, so this is
  // how a node asks for N text encoders or N LoRAs — and rendering it as raw JSON, which is
  // what a plain STRING widget got, made the node unusable here.
  function parseListValue(value) {
    if (Array.isArray(value)) return value.map((r) => (r && typeof r === "object" ? { ...r } : {}));
    try {
      const parsed = JSON.parse(value || "[]");
      return Array.isArray(parsed) ? parsed.map((r) => (r && typeof r === "object" ? { ...r } : {})) : [];
    } catch (_) { return []; }
  }

  function listRowCell(f, row, onEdit) {
    const cell = el("div", "list-cell");
    cell.title = f.tooltip || f.label || f.name;
    let cur = row[f.name] != null ? row[f.name] : f.default;
    let ctrl;
    if (f.kind === "combo") {
      const choices = f.choices || [];
      // A row the user never touched still has to save what it SHOWS. Without this the
      // first option is displayed, nothing is stored, and the node loads an empty row.
      if (choices.length && !choices.map(String).includes(String(cur))) {
        cur = choices[0];
        row[f.name] = cur;
        onEdit();
      }
      ctrl = el("select");
      choices.forEach((c) => {
        const o = el("option", null, String(c)); o.value = c;
        if (String(c) === String(cur)) o.selected = true;
        ctrl.append(o);
      });
      if (!choices.length) { ctrl.append(el("option", null, "(none installed)")); ctrl.disabled = true; }
      ctrl.onchange = () => { row[f.name] = ctrl.value; onEdit(); };
    } else if (f.kind === "boolean") {
      if (row[f.name] == null) { row[f.name] = cur !== false; onEdit(); }
      ctrl = el("input"); ctrl.type = "checkbox"; ctrl.checked = cur !== false; ctrl.style.width = "auto";
      ctrl.onchange = () => { row[f.name] = ctrl.checked; onEdit(); };
    } else if (f.kind === "int" || f.kind === "float") {
      if (row[f.name] == null && cur != null) { row[f.name] = cur; onEdit(); }
      ctrl = el("input"); ctrl.type = "number";
      if (f.min != null) ctrl.min = f.min;
      if (f.max != null) ctrl.max = f.max;
      ctrl.step = f.step != null ? f.step : (f.kind === "int" ? 1 : 0.01);
      ctrl.value = cur != null ? cur : "";
      ctrl.oninput = () => {
        row[f.name] = f.kind === "int" ? parseInt(ctrl.value || "0", 10) : parseFloat(ctrl.value || "0");
        onEdit();
      };
    } else {
      ctrl = el("input"); ctrl.type = "text"; ctrl.value = cur != null ? cur : "";
      ctrl.oninput = () => { row[f.name] = ctrl.value; onEdit(); };
    }
    cell.append(ctrl);
    return cell;
  }

  // Pick a file from what is installed, by name, with a filter — the alternative is a
  // combo holding hundreds of LoRAs, where finding one means scrolling a dropdown.
  function openChoicePicker(anchor, choices, taken, onPick) {
    document.querySelectorAll(".mn-role-pop").forEach((n) => n.remove());
    const pop = el("div", "mn-role-pop mn-pick-pop");
    const search = el("input", "mn-pick-search");
    search.type = "search";
    search.placeholder = "Search…";
    pop.append(search);
    const list = el("div", "mn-pick-list");
    pop.append(list);
    const paint = () => {
      clear(list);
      const q = search.value.trim().toLowerCase();
      const hits = choices.filter((c) => !q || String(c).toLowerCase().includes(q));
      if (!hits.length) { list.append(el("div", "mn-pick-empty", "Nothing matches.")); return; }
      hits.slice(0, 300).forEach((c) => {
        const it = el("div", "mn-role-item", String(c));
        if (taken.includes(c)) it.append(el("span", "mn-pick-used", "in use"));
        it.onclick = () => { pop.remove(); onPick(c); };
        list.append(it);
      });
    };
    search.oninput = paint;
    paint();

    pop.style.visibility = "hidden";
    document.body.append(pop);
    const r = anchor.getBoundingClientRect();
    const box = pop.getBoundingClientRect();
    const gap = 6, edge = 8;
    const below = r.bottom + gap;
    pop.style.top = (below + box.height <= window.innerHeight - edge
      ? below : Math.max(edge, r.top - gap - box.height)) + "px";
    pop.style.left = Math.max(edge, Math.min(r.left, window.innerWidth - box.width - edge)) + "px";
    pop.style.visibility = "";
    search.focus();
    const away = (e) => { if (!pop.contains(e.target)) { pop.remove(); document.removeEventListener("mousedown", away, true); } };
    document.addEventListener("mousedown", away, true);
  }

  function listField(spec, value, onChange) {
    const meta = spec.list || {};
    const fields = (meta.fields || []).filter((f) => f && f.name);
    const rows = parseListValue(value != null ? value : spec.default);
    // Picker mode: one field names a file and the rest are its settings, so the row reads
    // "this LoRA, at this strength" instead of a table of dropdowns.
    const pickField = meta.picker ? fields.find((f) => f.name === meta.picker) : null;
    const restFields = pickField ? fields.filter((f) => f !== pickField) : fields;
    const wrap = el("div", "field list-field" + (pickField ? " list-picker" : ""));
    wrap.append(el("span", "list-field-name", meta.item ? `${spec.name} · ${meta.item}s` : spec.name));
    const box = el("div", "list-rows");
    wrap.append(box);

    // Rows are edited in place, so keys this frontend does not know about survive a save.
    const commit = () => onChange(JSON.stringify(rows));

    function draw() {
      clear(box);
      if (!rows.length) box.append(el("div", "list-empty", `No ${meta.item || "entries"} yet.`));
      if (rows.length && !pickField && fields.length > 1) {
        const head = el("div", "list-row list-head");
        head.append(el("span", "list-ord", ""));
        fields.forEach((f) => head.append(el("div", "list-cell", f.label || f.name)));
        head.append(el("span", "list-rm-head", ""));   // matches the row's remove button
        box.append(head);
      }
      rows.forEach((row, i) => {
        const r = el("div", "list-row");
        // Order is load order — the text encoder before its connector, LoRAs top to bottom.
        const ord = el("div", "list-ord");
        const up = el("button", "btn ghost tiny", "▲");
        up.title = "Move up"; up.disabled = i === 0;
        up.onclick = () => { rows.splice(i - 1, 0, rows.splice(i, 1)[0]); commit(); draw(); };
        const down = el("button", "btn ghost tiny", "▼");
        down.title = "Move down"; down.disabled = i === rows.length - 1;
        down.onclick = () => { rows.splice(i + 1, 0, rows.splice(i, 1)[0]); commit(); draw(); };
        ord.append(up, down);
        r.append(ord);
        if (pickField) {
          const name = el("button", "list-pick-name", String(row[pickField.name] || "Choose a file…"));
          name.title = `${row[pickField.name] || "Nothing picked"} — click to change`;
          name.onclick = (e) => openChoicePicker(e.currentTarget,
            (pickField.choices || []).filter((c) => c !== "None"),
            rows.map((r2) => r2[pickField.name]).filter(Boolean),
            (chosen) => { row[pickField.name] = chosen; commit(); draw(); });
          r.append(name);
        }
        restFields.forEach((f) => r.append(listRowCell(f, row, commit)));
        const rm = el("button", "btn ghost tiny wire-rm", "×");
        rm.title = `Remove this ${meta.item || "entry"}`;
        rm.onclick = () => { rows.splice(i, 1); commit(); draw(); };
        r.append(rm);
        box.append(r);
      });
      const max = meta.max_rows || 0;
      if (!max || rows.length < max) {
        const add = el("button", "btn ghost tiny wire-add", meta.add_label || "+ Add");
        const blank = () => {
          const row = {};
          fields.forEach((f) => { if (f.default != null) row[f.name] = f.default; });
          return row;
        };
        add.onclick = (e) => {
          if (!pickField) { rows.push(blank()); commit(); draw(); return; }
          const choices = (pickField.choices || []).filter((c) => c !== "None");
          if (!choices.length) return;
          openChoicePicker(e.currentTarget, choices,
            rows.map((r2) => r2[pickField.name]).filter(Boolean),
            (chosen) => {
              const row = blank();
              row[pickField.name] = chosen;
              rows.push(row); commit(); draw();
            });
        };
        box.append(add);
      }
    }

    draw();
    return wrap;
  }

  function widgetField(spec, value, onChange) {
    if (spec.kind === "list") return listField(spec, value, onChange);
    const wrap = el("label", "field");
    wrap.append(el("span", null, spec.name + (spec.required ? "" : "  ·opt")));
    let ctrl;
    if (spec.kind === "combo") {
      ctrl = el("select");
      (spec.choices || []).forEach((c) => { const o = el("option", null, String(c)); o.value = c; if (c === value) o.selected = true; ctrl.append(o); });
      if (!spec.choices || !spec.choices.length) { ctrl.append(el("option", null, "(none installed)")); ctrl.disabled = true; }
      // A saved value that no longer exists (renamed/removed file) leaves nothing
      // selected — the browser then shows the first option without persisting it,
      // so the stale value silently goes to generation. Persist what's displayed.
      else if (value != null && !spec.choices.includes(value)) onChange(spec.choices[0]);
      ctrl.onchange = () => onChange(ctrl.value);
    } else if (spec.kind === "boolean") {
      ctrl = el("input"); ctrl.type = "checkbox"; ctrl.checked = !!value; ctrl.style.width = "auto";
      ctrl.onchange = () => onChange(ctrl.checked);
    } else if (spec.kind === "int" || spec.kind === "float") {
      ctrl = el("input"); ctrl.type = "number";
      applyNumMeta(ctrl, numMeta(spec), spec.kind === "float");
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

  // A slot is "active" (part of the pipeline) if it wires an output somewhere or a core
  // override sources from it. Inert slots are left alone (no required-input errors).
  function slotIsActive(slot) {
    if (Object.values(slot.wires || {}).some((tg) => wireTargets(tg).some(Boolean))) return true;
    const ov = config.core_overrides || {};
    return Object.values(ov).some((ins) =>
      Object.values(ins || {}).some((src) => typeof src === "string" && src.startsWith(`out:${slot.id}:`)));
  }

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
        issues.push({ level: "error", msg: `"${w.name}" has no installed options to pick from.` });
      } else if (w.kind === "list") {
        // "[]" passes the emptiness check below but means the node has nothing to load —
        // unless the node declares empty a working state, where it passes its input through.
        if (!(w.list || {}).allow_empty && parseListValue(v != null ? v : w.default).length === 0)
          issues.push({ level: "error", msg: `"${w.name}" is empty — add at least one entry.` });
      } else if (v == null || v === "") {
        // A linked input is FILLED at generate time from a project value, so the stored
        // blank is the normal state for one — warning about it would train the user to
        // ignore the warning that matters.
        if (linkOf(slot.id, w.name)) return;
        // Not blocking: the builder emits a type-appropriate empty for any widget left
        // blank (matching ComfyUI's frontend), so an empty field won't stall generation.
        issues.push({ level: "warn", msg: `"${w.name}" is empty — using its default.` });
      }
    });
    const outs = spec.outputs || [];
    const wires = slot.wires || {};
    if (outs.length && !outs.some((o) => wireTargets(wires[o.name]).some(Boolean))) {
      issues.push({ level: "warn", msg: "No outputs wired — this node feeds nothing." });
    }
    const dests = (o) => allowedDestinations(slot, o);
    outs.forEach((o) => {
      wireTargets(wires[o.name]).filter(Boolean).forEach((t) => {
        if (!dests(o).some((d) => d.value === t))
          issues.push({ level: "error", msg: `"${o.name}" is wired to a destination that no longer exists.` });
      });
    });

    // Required connection INPUTS must actually be fed (matches the builder's blocking):
    // explicit input source, an incoming wire from another node, or a unique auto-wire
    // producer. Only checked for ACTIVE slots (wired into the pipeline) — an unused node
    // that feeds nothing must not be flagged or block generation.
    // A bypassed slot is on its way out of the graph, so its own inputs are not something
    // to fix — matches the builder, which auto-wires it if it can and otherwise stays quiet.
    if (slotIsActive(slot) && !slot.bypassed) (spec.connection_inputs || []).forEach((ci) => {
      if (!ci.required) return;
      if (typeAccepts(ci.type, "IMAGE")) return;  // a scene/timeline image can always feed an IMAGE input
      const src = (slot.input_sources || {})[ci.name];
      if (src && src !== "auto") return;  // explicitly sourced (incl. "timeline")
      const incoming = (config.slots || []).some((s2) =>
        s2.id !== slot.id && Object.values(s2.wires || {}).some((tg) => wireTargets(tg).includes(`node:${slot.id}:${ci.name}`)));
      if (incoming) return;
      const prod = sources(slot, ci.type)
        .filter((o) => o.value && o.value !== "timeline" && producerIsLive(o.value)).length;
      if (prod === 1) return;  // a single producer auto-wires
      if (prod > 1)
        issues.push({ level: "error", msg: `Input "${ci.name}" (${ci.type}): ${prod} possible sources — set its Input source.` });
      else
        issues.push({ level: "error", msg: `Input "${ci.name}" (${ci.type}) has no source — set its Input source or add a node that outputs ${ci.type}.` });
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

  // ── rename (pencil): role label shown in pickers, headers, and the sidebar ────
  async function renameSlot(slot) {
    const role = roles.find((r) => r.key === slot.role);
    const cur = slot.role_label || (role ? role.label : slot.role) || "";
    const n = prompt("Node label (shown in pickers, headers, and the sidebar):", cur);
    if (n != null) { slot.role_label = n.trim() || undefined; await persist(); render(); }
  }

  // Which widget on a node names the file it loads. Checked in order, then any
  // remaining combo input whose name ends in _name — enough to cover core loaders and
  // most third-party ones without a per-class table to keep up to date.
  const FILE_INPUTS = [
    "lora_name", "unet_name", "ckpt_name", "checkpoint_name", "model_name", "clip_name",
    "vae_name", "control_net_name", "style_model_name", "gguf_name", "upscale_model_name",
  ];

  // Five LoRAs all called "LoRA" are indistinguishable, and renaming each by hand is the
  // tax this avoids: derive the name from the file the node actually loads.
  function autoLabel(slot) {
    const inputs = slot.inputs || {};
    const spec = specFor(slot);
    let key = FILE_INPUTS.find((k) => typeof inputs[k] === "string" && inputs[k]);
    if (!key) {
      const combos = (spec?.inputs || []).filter((i) => i.kind === "combo" && /_name$/.test(i.name));
      key = combos.map((i) => i.name).find((k) => typeof inputs[k] === "string" && inputs[k]);
    }
    if (!key) return null;
    const raw = String(inputs[key]);
    // "SDXL/detail/add-detail-xl.safetensors" -> "add-detail-xl"
    const base = raw.split(/[\\/]/).pop().replace(/\.[a-z0-9]+$/i, "");
    return base || null;
  }

  function slotDisplayLabel(slot) {
    const role = roles.find((r) => r.key === slot.role);
    // An explicit rename always wins; the parsed filename only fills the gap where the
    // label would otherwise be the role, which every node of that role shares.
    return slot.role_label || autoLabel(slot) || (role ? role.label : slot.role) || slotName(slot) || "?";
  }

  // ── node page (one configured slot: values, wiring, sources) ──────────────────
  function nodePage(slot, issues) {
    const errs = (issues || []).filter((i) => i.level === "error").length;
    const warns = (issues || []).filter((i) => i.level === "warn").length;
    const card = el("div", "slot-card open node-page");
    if (errs) card.classList.add("slot-bad");
    else if (warns) card.classList.add("slot-warn");

    const head = el("div", "slot-head node-page-head");
    const back = el("button", "ic-btn node-back", "‹");
    back.title = "Back to the pipeline";
    back.onclick = () => setView("pipeline");
    head.append(back);
    const roleSpan = el("span", "slot-role", slotDisplayLabel(slot));
    head.append(roleSpan);
    const ren = el("button", "ic-btn", "✎"); ren.title = "Rename node label";
    ren.onclick = (e) => { e.stopPropagation(); renameSlot(slot); };
    head.append(ren);
    const biHead = builtIn(slot);
    head.append(el("span", "slot-node" + (biHead ? " built-in" : ""),
                   biHead ? `${biHead.icon} ${biHead.label}` : slotName(slot)));
    const nExp = (slot.exposed || []).length;
    if (nExp) head.append(el("span", "slot-badge exposed", `◉ ${nExp}`));
    if (slot.bypassed) head.append(el("span", "slot-badge warn", "bypassed"));
    if (errs) head.append(el("span", "slot-badge bad", `${errs} error${errs > 1 ? "s" : ""}`));
    else if (warns) head.append(el("span", "slot-badge warn", `${warns} warning${warns > 1 ? "s" : ""}`));
    else head.append(el("span", "slot-badge ok", "ready"));
    const acts = el("span", "slot-actions");
    const byp = el("button", "btn ghost tiny" + (slot.bypassed ? " on" : ""), slot.bypassed ? "bypassed" : "bypass");
    byp.title = "Skip this node's effect (pass its inputs straight through) without losing its configuration.";
    byp.onclick = async (e) => { e.stopPropagation(); slot.bypassed = !slot.bypassed; await persist(); render(); };
    acts.append(byp);
    if (!EASY()) acts.append(bypassEyeButton(slot));
    const rm = el("button", "btn ghost tiny danger", "remove");
    rm.onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Remove "${slotDisplayLabel(slot)}" (${slotName(slot)}) from the pipeline?`)) return;
      // Removal is its own confirmed decision, not part of the edit buffer — and there
      // would be nothing left to press Save on.
      removeSlot(slot);
      deferSave = false;
      await persistNow();
      _setView("pipeline");
    };
    acts.append(rm);
    head.append(acts);
    card.append(head);

    const ib = issuesBox(issues);
    if (ib) card.append(ib);

    // Which section this node is filed under. Purely a heading — moving a node between
    // groups changes nothing about how it is wired or built.
    const grpRow = el("div", "field node-group-field");
    grpRow.append(el("span", null, "Group"));
    const grpSel = el("select");
    const cur = (slot.group || "").trim();
    const opts = groupNames();
    if (cur && !opts.includes(cur)) opts.push(cur);
    [["", "— none —"], ...opts.map((g) => [g, g]), ["__new__", "New group…"]]
      .forEach(([val, label]) => {
        const o = el("option", null, label); o.value = val;
        if (val === cur) o.selected = true;
        grpSel.append(o);
      });
    grpSel.onchange = async () => {
      let next = grpSel.value;
      if (next === "__new__") {
        next = (prompt("Group name:", cur) || "").trim();
        if (!next) { render(); return; }
      }
      if (next) slot.group = next; else delete slot.group;
      await persist();
      render();
    };
    grpRow.append(grpSel);
    card.append(grpRow);

    const cand = specFor(slot);

    // Card reads top-down in signal order: what comes IN, what the node is set to,
    // then what goes OUT.
    // Input sources: explicitly choose where each connection input comes from.
    if (cand && (cand.connection_inputs || []).length) {
      slot.input_sources = slot.input_sources || {};
      const sbox = el("div", "wire-box");
      sbox.append(el("div", "wire-title", "Input sources"));
      // An input the node calls advanced is folded away UNLESS something feeds it: a wire
      // that exists has to stay visible, or the page would hide part of the pipeline.
      const visible = visibleConnectionInputs(slot, cand.connection_inputs);
      const advanced = visible.filter((ci) => ci.advanced && !inputIsFed(slot, ci.name));
      let advBox = null;
      if (advanced.length) {
        advBox = el("details", "wire-advanced");
        advBox.append(el("summary", null, `Advanced · ${advanced.length}`));
      }
      visible.forEach((ci) => {
        const row = el("div", "wire-row");
        const lbl = el("span", "wire-out", `${ci.display || ci.name} (${ci.type})`);
        if (ci.autogrow)
          lbl.title = `Entry ${ci.autogrow.index + 1} of the node's "${ci.autogrow.parent}" list — fill one and the next appears.`;
        row.append(lbl);
        row.append(el("span", "wire-arrow", "←"));
        const srcs = allowedSources(slot, ci);
        const sel = el("select", "wire-select");
        const cur = slot.input_sources[ci.name] || "";
        srcs.forEach((s) => { const o = el("option", null, s.label); o.value = s.value; if (s.value === cur) o.selected = true; sel.append(o); });
        if (cur && !srcs.some((s) => s.value === cur)) { const o = el("option", null, cur + " (missing)"); o.value = cur; o.selected = true; sel.append(o); }
        if (pipelineLocked() && slot.role === "image_processing" && ci.name === "image" && !cur) {
          slot.input_sources[ci.name] = "timeline";
          sel.value = "timeline";
        }
        sel.onchange = async () => { _setInputSource(slot, ci.name, sel.value); await persist(); render(); };
        row.append(sel);
        (advanced.includes(ci) ? advBox : sbox).append(row);
      });
      if (advBox) sbox.append(advBox);
      card.append(sbox);
    }

    if (cand && cand.inputs.length) {
      const grid = el("div", "slot-fields");
      // Widgets the node calls advanced are tweakable, but validated at their defaults —
      // folded away so they do not sit between the user and the file picker.
      const advWidgets = cand.inputs.filter((w) => w.advanced);
      let advGrid = null, advWrap = null;
      if (advWidgets.length) {
        advWrap = el("details", "wire-advanced");
        advWrap.append(el("summary", null, `Advanced · ${advWidgets.length}`));
        advGrid = el("div", "slot-fields");
        advWrap.append(advGrid);
      }
      cand.inputs.forEach((spec) => {
        const lk = linkOf(slot.id, spec.name);
        const iw = incomingWidgetWire(slot, spec.name);
        const f = widgetField(spec, slot.inputs[spec.name], async (v) => {
          slot.inputs[spec.name] = v;
          const l2 = linkOf(slot.id, spec.name); if (l2) applyLinkValue(l2, v);  // keep group in sync
          await persist();
        });
        const isList = spec.kind === "list";
        if (!isList && (!EASY() || iw || lk || linkMode)) f.classList.add("with-eye");
        if (iw) {
          // Wired from another node's output: the widget value is replaced by the
          // connection at generation — lock the field and say where it comes from
          // (same treatment as a linked control; the eye button is skipped because an
          // exposed control would edit a value generation ignores).
          f.classList.add("linked");
          const ctrl = f.querySelector("input,select"); if (ctrl) ctrl.disabled = true;
          const tag = el("span", "link-tag", `⇐ ${slotName(iw.slot)} · ${iw.out}`);
          tag.title = `This value comes from ${slotFullLabel(iw.slot)}'s "${iw.out}" output at generation — the local field is ignored while wired.`;
          f.append(tag);
        } else if (lk) {
          f.classList.add("linked");
          const ctrl = f.querySelector("input,select"); if (ctrl) ctrl.disabled = true;
          f.append(el("span", "link-tag", "🔗 " + lk.name));
        } else if (linkMode && !isList) {
          const chk = el("button", "eye-btn link-pick" + (linkSelHas(slot.id, spec.name) ? " on" : ""), "+");
          chk.type = "button"; chk.title = "Add to link selection";
          chk.onclick = (e) => {
            e.preventDefault(); e.stopPropagation();
            if (linkSelHas(slot.id, spec.name)) linkSel = linkSel.filter((s) => !(s.slotId === slot.id && s.input === spec.name));
            else linkSel.push({ slotId: slot.id, input: spec.name, kind: spec.kind, choices: spec.choices, label: `${slotName(slot)} · ${spec.name}` });
            render();
          };
          f.append(chk);
        } else if (!EASY() && !isList) {
          // A list holds many values; there is no single control to expose to the editor.
          f.append(eyeButton(slot, spec));
        }
        (spec.advanced ? advGrid : grid).append(f);
      });
      card.append(grid);
      if (advWrap) card.append(advWrap);
    }

    // Wiring: each output -> one or more destinations.
    if (cand && (cand.outputs || []).length) {
      slot.wires = slot.wires || {};
      const wbox = el("div", "wire-box");
      wbox.append(el("div", "wire-title", "Wire outputs to"));
      cand.outputs.forEach((out) => {
        const row = el("div", "wire-row");
        row.append(el("span", "wire-out", `${out.name} (${out.type})`));
        row.append(el("span", "wire-arrow", "→"));
        row.append(destMulti(slot, out));
        wbox.append(row);
      });
      card.append(wbox);
    }

    card.append(editFooter());
    return card;
  }

  // Nothing on the node page is written until Save. See the deferSave block above for why.
  function editFooter() {
    const foot = el("div", "node-foot");
    const hint = el("div", "node-foot-hint");
    const acts = el("div", "node-foot-acts");

    const cancel = el("button", "btn ghost", "Cancel");
    cancel.type = "button";
    cancel.onclick = () => {
      if (deferDirty && !confirm("Discard the changes to this node?")) return;
      restoreBaseline();
      finishNodeEdit();
    };

    const save = el("button", "btn primary", "Save");
    save.type = "button";
    save.onclick = async () => {
      save.disabled = true;
      deferSave = false;
      await persistNow();
      endEdit();
      finishNodeEdit();
    };

    acts.append(cancel, save);
    foot.append(hint, acts);
    footerEls = { save, hint };
    paintFooter();
    return foot;
  }

  const _KIND2T = { int: "INT", float: "FLOAT", string: "STRING", boolean: "BOOLEAN" };
  // A ComfyUI type string can name several types at once — a V3 MultiType socket serializes
  // its members comma-joined ("IMAGE,MASK"). An output feeds an input when they share one.
  function typeAccepts(inputType, outputType) {
    if (inputType === outputType) return true;
    const parts = (t) => String(t || "").split(",").map((p) => p.trim()).filter(Boolean);
    const a = parts(inputType), b = parts(outputType);
    return a.some((x) => b.includes(x));
  }
  function destinations(slot, type) {
    const out = [{ value: "", label: "— unwired —" }];
    // Global editor outputs: wire your final video/audio producer here and the editor shows
    // it (works with or without the built-in pipeline). IMAGE -> video, AUDIO -> audio.
    if (typeAccepts(type, "IMAGE")) out.push({ value: "global:video", label: "🌐 Global video output (shown in editor)" });
    if (typeAccepts(type, "AUDIO")) out.push({ value: "global:audio", label: "🌐 Global audio output (shown in editor)" });
    ports.filter((p) => typeAccepts(p.type, type)).forEach((p) => out.push({ value: "port:" + p.id, label: p.label }));
    config.slots.filter((s) => s.id !== slot.id).forEach((s2) => {
      const c2 = specFor(s2);
      visibleConnectionInputs(s2, c2?.connection_inputs).filter((ci) => typeAccepts(ci.type, type)).forEach((ci) =>
        out.push({ value: `node:${s2.id}:${ci.name}`, label: `${slotFullLabel(s2)} · ${ci.display || ci.name}` }));
      // Widget inputs can also receive a connection (ComfyUI converts a widget to an
      // input when wired) — e.g. EmptyLatentVideo.width/height. Offer them as targets.
      (c2?.inputs || []).filter((w) => _KIND2T[w.kind] === type).forEach((w) =>
        out.push({ value: `node:${s2.id}:${w.name}`, label: `${slotFullLabel(s2)} · ${w.name} (widget)` }));
    });
    return out;
  }

  // Whether a connection input already has something feeding it — an explicitly chosen
  // source, or an output wired in from another node.
  function inputIsFed(slot, name) {
    const src = (slot.input_sources || {})[name];
    if (src && src !== "auto") return true;
    return (config.slots || []).some((s2) => s2.id !== slot.id &&
      Object.values(s2.wires || {}).some((tg) => wireTargets(tg).includes(`node:${slot.id}:${name}`)));
  }

  // Autogrow list inputs (MiniMax H3's ref_images, ref_videos, …) reach us already expanded
  // into one socket per index, addressed by their dotted path (ref_images.ref_image_0).
  // Showing all ten of each list would bury the node, so the list renders what's filled plus
  // one empty slot and grows as it is used, the way ComfyUI's own canvas grows it.
  // Autogrow entries were first saved under the bare template name (ref_image_0) before the
  // dotted socket id turned out to be what ComfyUI accepts. Rename them in place on load —
  // the edge is the same edge, and leaving the old key in a config keeps failing the run.
  function migrateAutogrowNames() {
    let changed = false;
    const canonical = (slot, name) => {
      if (!name || name.includes(".")) return name;
      const ci = ((specFor(slot) || {}).connection_inputs || [])
        .find((c) => c.autogrow && c.display === name);
      return ci ? ci.name : name;
    };
    (config.slots || []).forEach((slot) => {
      const srcs = slot.input_sources || {};
      Object.keys(srcs).forEach((k) => {
        const nu = canonical(slot, k);
        if (nu !== k) { srcs[nu] = srcs[k]; delete srcs[k]; changed = true; }
      });
    });
    (config.slots || []).forEach((slot) => {
      Object.entries(slot.wires || {}).forEach(([out, raw]) => {
        const next = wireTargets(raw).map((t) => {
          const parsed = _parseNodeTarget(t);
          if (!parsed) return t;
          const dest = slotById(parsed.slotId);
          if (!dest) return t;
          const nu = canonical(dest, parsed.input);
          return nu === parsed.input ? t : `node:${parsed.slotId}:${nu}`;
        });
        if (next.some((t, i) => t !== wireTargets(raw)[i])) { slot.wires[out] = next; changed = true; }
      });
    });
    return changed;
  }

  function visibleConnectionInputs(slot, cis) {
    const lastFed = {};
    (cis || []).forEach((ci) => {
      const ag = ci.autogrow;
      if (ag && inputIsFed(slot, ci.name))
        lastFed[ag.parent] = Math.max(lastFed[ag.parent] == null ? -1 : lastFed[ag.parent], ag.index);
    });
    return (cis || []).filter((ci) => !ci.autogrow ||
      ci.autogrow.index <= (lastFed[ci.autogrow.parent] == null ? -1 : lastFed[ci.autogrow.parent]) + 1);
  }

  // What a marked reference can feed, by media kind. A video reference can drive either a
  // VIDEO socket or a frames (IMAGE) one — the builder picks a loader to match whichever
  // the destination actually asks for.
  const REF_KIND_TYPES = { image: ["IMAGE"], audio: ["AUDIO"], video: ["VIDEO", "IMAGE"] };
  const REF_KIND_LABEL = { image: "image", audio: "audio", video: "video" };

  // Media marked "R" in the Media Bin / gallery, in mark order — R1, R2, R3 — offered to any
  // socket whose type that kind of media can fill.
  function referenceSources(type) {
    const st = window.Store?.get() || {};
    const marks = st.project?.references || [];
    const bin = st.mediaBin || [];
    const out = [];
    marks.forEach((id, i) => {
      const m = bin.find((x) => x.id === id);
      if (!m) return;
      const kinds = REF_KIND_TYPES[m.kind] || [];
      if (!kinds.some((t) => typeAccepts(type, t))) return;
      out.push({ value: `ref:${id}`, label: `R${i + 1} · ${m.name} (${REF_KIND_LABEL[m.kind] || m.kind})` });
    });
    return out;
  }

  // Available sources for a slot connection input of a given type.
  // Source IDs: "" = auto, "out:<slotId>:<outName>", "core:<coreId>:<outIdx>",
  // "timeline" (IMAGE only), "ref:<mediaId>" (media marked R in the bin).
  // A pass-through output is only a real source once its own input is fed: the LoRA loader
  // hands CLIP straight back, so an unwired one emits nothing while still making every
  // other CLIP consumer read as ambiguous. Matches builder._producers, which is what
  // actually decides whether generation is blocked.
  function producerIsLive(value) {
    const m = /^out:([^:]+):(.+)$/.exec(String(value));
    if (!m) return true;
    const slot = (config.slots || []).find((s) => s.id === m[1]);
    if (!slot) return true;
    const spec = specFor(slot);
    if (!spec) return true;
    const out = (spec.outputs || []).find((o) => o.name === m[2]);
    if (!out) return true;
    return !(spec.connection_inputs || []).some(
      (ci) => ci.type === out.type && !inputIsFed(slot, ci.name));
  }

  // Numbered reference SLOTS — "Reference image 1" is whatever is marked first among the
  // image references, not a particular file. Wire a socket to a slot once and re-ordering
  // marks in the bin re-points it, with no node page involved. Numbered per kind so marking
  // an audio file never shifts the image slots.
  //
  // Offered up to one past the highest slot this node already uses, which is what makes the
  // next free number the obvious pick: with 1 taken you see 1 (in use) and 2.
  function referenceSlots(slot, type) {
    const used = {};
    Object.values(slot?.input_sources || {}).forEach((src) => {
      const m = /^ref#([a-z]+):(\d+)$/.exec(String(src || ""));
      if (m) used[m[1]] = Math.max(used[m[1]] || 0, +m[2]);
    });
    const out = [];
    Object.keys(REF_KIND_TYPES).forEach((kind) => {
      if (!REF_KIND_TYPES[kind].some((t) => typeAccepts(type, t))) return;
      const top = Math.max(1, (used[kind] || 0) + 1);
      for (let n = 1; n <= top; n++) {
        out.push({ value: `ref#${kind}:${n}`,
                   label: `Reference ${REF_KIND_LABEL[kind] || kind} ${n}`
                        + ((used[kind] || 0) >= n ? " · in use" : "") });
      }
    });
    return out;
  }

  function sources(slot, type) {
    const out = [{ value: "", label: "(auto-wire)" }];
    if (typeAccepts(type, "IMAGE")) out.push({ value: "timeline", label: "Timeline (scene image)" });
    referenceSlots(slot, type).forEach((r) => out.push(r));
    referenceSources(type).forEach((r) => out.push(r));
    coreProducers.filter((p) => typeAccepts(type, p.type)).forEach((p) =>
      out.push({ value: p.id, label: p.label }));
    config.slots.filter((s) => s.id !== slot.id).forEach((s2) => {
      const c2 = specFor(s2);
      (c2?.outputs || []).filter((o) => typeAccepts(type, o.type)).forEach((o) =>
        out.push({ value: `out:${s2.id}:${o.name}`, label: `${slotFullLabel(s2)} → ${o.name}` }));
    });
    return out;
  }

  // Normalize wires[outName] to always be an array (supports legacy string format).
  function wireTargets(raw) {
    if (!raw) return [];
    return Array.isArray(raw) ? raw : [raw];
  }

  // The wire (if any) feeding a WIDGET input of `slot` from another slot's output —
  // e.g. ImageTransform.width → EmptyLatent.width. The builder replaces the widget's
  // value with that connection at generation, so the local field is dead while wired
  // and must render locked (same treatment as a linked control).
  function incomingWidgetWire(slot, inputName) {
    const target = `node:${slot.id}:${inputName}`;
    for (const s2 of config.slots) {
      if (s2.id === slot.id) continue;
      for (const [outName, raw] of Object.entries(s2.wires || {})) {
        if (wireTargets(raw).includes(target)) return { slot: s2, out: outName };
      }
    }
    return null;
  }

  // ── edge mirroring ────────────────────────────────────────────────────────────
  // An output→input edge can be authored from either side: a slot's wires
  // ("node:<destId>:<input>") OR the destination's input_sources ("out:<srcId>:<out>").
  // They describe the SAME connection, so we keep both in sync — wiring an output shows
  // the input as connected on the other node, and vice-versa.
  function _parseNodeTarget(t) {
    if (typeof t !== "string" || !t.startsWith("node:")) return null;
    const parts = t.split(":");
    return { slotId: parts[1], input: parts.slice(2).join(":") };
  }
  function _parseOutSource(s) {
    if (typeof s !== "string" || !s.startsWith("out:")) return null;
    const parts = s.split(":");
    return { slotId: parts[1], out: parts.slice(2).join(":") };
  }
  function _addWire(slotId, outName, target) {
    const s = slotById(slotId); if (!s) return;
    s.wires = s.wires || {};
    const arr = wireTargets(s.wires[outName]);
    if (!arr.includes(target)) arr.push(target);
    s.wires[outName] = arr;
  }
  function _removeWire(slotId, outName, target) {
    const s = slotById(slotId); if (!s || !s.wires) return;
    s.wires[outName] = wireTargets(s.wires[outName]).filter((t) => t !== target);
  }
  // Set a destination's input source + mirror it as a wire on the source slot.
  function _setInputSource(destSlot, inp, value) {
    const prev = _parseOutSource(destSlot.input_sources[inp]);
    if (prev) _removeWire(prev.slotId, prev.out, `node:${destSlot.id}:${inp}`);
    destSlot.input_sources[inp] = value;
    const next = _parseOutSource(value);
    if (next) _addWire(next.slotId, next.out, `node:${destSlot.id}:${inp}`);
  }
  // Replace a wire target + mirror it as an input source on the destination slot.
  function _setWireTarget(srcSlot, outName, oldTarget, newTarget) {
    // A wire onto an open core port is recorded TWICE: as the slot's wire, and as a
    // core_override. reconcileOpenPortWiring() rebuilds each from the other, so dropping
    // only the wire left the override behind and the next reconcile put the wire straight
    // back — which is why "Studio · model" could not be unwired without deleting the node.
    if (String(oldTarget || "").startsWith("port:")) {
      const map = portToOpenCore(oldTarget.slice(5));
      if (map) {
        const [cid, inp] = map;
        // Only if the override still names THIS wire; another output may own it now.
        if (config.core_overrides?.[cid]?.[inp] === `out:${srcSlot.id}:${outName}`)
          delete config.core_overrides[cid][inp];
      }
    }
    const old = _parseNodeTarget(oldTarget);
    if (old) {
      const ds = slotById(old.slotId);
      if (ds && ds.input_sources && ds.input_sources[old.input] === `out:${srcSlot.id}:${outName}`)
        ds.input_sources[old.input] = "";
    }
    const nu = _parseNodeTarget(newTarget);
    if (nu) {
      const ds = slotById(nu.slotId);
      if (ds) { ds.input_sources = ds.input_sources || {}; ds.input_sources[nu.input] = `out:${srcSlot.id}:${outName}`; }
    }
  }

  function destMulti(slot, out) {
    slot.wires = slot.wires || {};
    // normalize to array in-place
    const raw = slot.wires[out.name];
    let targets = wireTargets(raw);
    slot.wires[out.name] = targets;

    const wrap = el("div", "wire-multi");

    function renderRows() {
      clear(wrap);
      const dests = allowedDestinations(slot, out);
      targets.forEach((t, i) => {
        const row = el("div", "wire-multi-row");
        const sel = el("select", "wire-select");
        dests.forEach((d) => { const o = el("option", null, d.label); o.value = d.value; if (d.value === t) o.selected = true; sel.append(o); });
        if (t && !dests.some((d) => d.value === t)) { const o = el("option", null, t + " (not allowed)"); o.value = t; o.selected = true; sel.append(o); }
        sel.onchange = async () => { _setWireTarget(slot, out.name, targets[i], sel.value); targets[i] = sel.value; reconcileOpenPortWiring(); await persist(); render(); };
        const rm = el("button", "btn ghost tiny wire-rm", "×");
        rm.title = "Remove this wire";
        rm.onclick = async () => { _setWireTarget(slot, out.name, targets[i], ""); targets.splice(i, 1); reconcileOpenPortWiring(); await persist(); render(); };
        row.append(sel, rm);
        wrap.append(row);
      });
      const portOpts = dests.filter((d) => d.value.startsWith("port:"));
      if (!pipelineLocked() || portOpts.length > 1 || (portOpts.length === 1 && !targets.some((t) => t.startsWith("port:")))) {
        const add = el("button", "btn ghost tiny wire-add", "+ Add");
        add.onclick = () => { targets.push(""); renderRows(); };
        wrap.append(add);
      }
    }

    renderRows();
    return wrap;
  }

  // Same multi-destination editor as the node page, for a DRAFT slot: it has no id, is not
  // in config yet, and nothing about it is persisted until "Add node". So this writes only
  // into the draft's own wires map — no _setWireTarget mirroring, no persist, no re-render.
  // The targets are replayed onto the real slot once it exists.
  function draftDestMulti(draftSlot, out, wires) {
    const targets = wireTargets(wires[out.name]);
    if (!targets.length) targets.push("");  // always show one row, as the old single select did
    wires[out.name] = targets;
    const wrap = el("div", "wire-multi");

    function renderRows() {
      clear(wrap);
      const dests = allowedDestinations(draftSlot, out);
      targets.forEach((t, i) => {
        const row = el("div", "wire-multi-row");
        const sel = el("select", "wire-select");
        dests.forEach((d) => {
          const o = el("option", null, d.label); o.value = d.value;
          if (d.value === t) o.selected = true;
          sel.append(o);
        });
        sel.onchange = () => { targets[i] = sel.value; };
        const rm = el("button", "btn ghost tiny wire-rm", "×");
        rm.type = "button";
        rm.title = "Remove this wire";
        rm.onclick = () => { targets.splice(i, 1); renderRows(); };
        row.append(sel, rm);
        wrap.append(row);
      });
      // Same gate as the node page: with the built-in pipeline in charge a node normally
      // feeds exactly one open core port, so offering more wires there would be offering
      // something the pipeline will not honour.
      const portOpts = dests.filter((d) => d.value.startsWith("port:"));
      if (!pipelineLocked() || portOpts.length > 1
          || (portOpts.length === 1 && !targets.some((t) => t.startsWith("port:")))) {
        const add = el("button", "btn ghost tiny wire-add", "+ Add");
        add.type = "button";
        add.onclick = () => { targets.push(""); renderRows(); };
        wrap.append(add);
      }
    }

    renderRows();
    return wrap;
  }

  // ── "+ New node": role dropdown → "Setup node" modal (search → values + wiring) ──
  function openRoleMenu(anchor, onlyCategory) {
    document.querySelectorAll(".mn-role-pop").forEach((n) => n.remove());
    const pop = el("div", "mn-role-pop");
    const cats = {};
    roles.filter((r) => !onlyCategory || r.category === onlyCategory)
      .forEach((r) => { (cats[r.category] = cats[r.category] || []).push(r); });
    Object.keys(cats).forEach((cat) => {
      pop.append(el("div", "mn-role-cat", cat));
      cats[cat].forEach((r) => {
        const it = el("div", "mn-role-item", r.label);
        it.onclick = () => { pop.remove(); openNodeSetup(r.key); };
        pop.append(it);
      });
    });
    if (!onlyCategory) {
      pop.append(el("div", "mn-role-cat", "Advanced"));
      const any = el("div", "mn-role-item", "Any node…");
      any.onclick = () => { pop.remove(); openNodeSetup("__any__"); };
      pop.append(any);
    }

    // Measure before placing: below the card is the default, but the shelf is usually
    // near the bottom of a scrolled pane, where "below" is off-screen.
    pop.style.visibility = "hidden";
    document.body.append(pop);
    const r = anchor.getBoundingClientRect();
    const box = pop.getBoundingClientRect();
    const gap = 6, edge = 8;
    const below = r.bottom + gap;
    const top = below + box.height <= window.innerHeight - edge
      ? below
      : Math.max(edge, r.top - gap - box.height);
    pop.style.top = top + "px";
    pop.style.left = Math.max(edge, Math.min(r.left, window.innerWidth - box.width - edge)) + "px";
    pop.style.visibility = "";
    const away = (e) => { if (!pop.contains(e.target)) { pop.remove(); document.removeEventListener("mousedown", away, true); } };
    document.addEventListener("mousedown", away, true);
  }

  // ── export settings as a picture ──────────────────────────────────────────────
  // "Which model was that?" outlives the session that could answer it. The card is built
  // on the server (it knows torch/CUDA/GPU; the browser only knows the laptop showing the
  // page) and comes back as one PNG that is shown, saved or copied from here.
  function openSettingsCard() {
    document.querySelectorAll(".sc-overlay").forEach((n) => n.remove());
    const overlay = el("div", "modal-overlay sc-overlay");
    const modal = el("div", "modal sc-modal");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", "Export settings"));
    const hr = el("div", "modal-head-right");
    const x = el("button", "btn ghost tiny", "✕");
    const close = () => { if (url) URL.revokeObjectURL(url); overlay.remove(); };
    x.onclick = close;
    hr.append(x); head.append(hr);
    const content = el("div", "modal-content sc-content");
    const foot = el("div", "modal-foot sc-foot");
    modal.append(head, content, foot);
    overlay.append(modal);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.body.append(overlay);

    let url = null, blob = null;
    content.append(el("div", "pj-meta", "Rendering…"));

    const status = el("div", "sc-status");
    const dl = el("button", "btn primary tiny", "⤓ Download");
    const cp = el("button", "btn ghost tiny", "⧉ Copy image");
    const cl = el("button", "btn ghost tiny", "Close");
    cl.onclick = close;
    dl.disabled = true; cp.disabled = true;
    foot.append(status, dl, cp, cl);

    // The card is rendered in whatever theme the app is showing, because it is a document
    // about this install and it should look like this install.
    const theme = window.FunPackTheme?.resolved?.()
      || document.documentElement.getAttribute("data-theme") || "dark";
    const project = window.Store?.get().project;
    API.settingsCard(project?.id, theme)
      .then((b) => {
        if (!overlay.isConnected) return;
        blob = b; url = URL.createObjectURL(b);
        clear(content);
        const img = el("img", "sc-img");
        img.src = url;
        img.alt = "FunPack settings card";
        content.append(img);
        dl.disabled = false; cp.disabled = false;
      })
      .catch((e) => {
        if (!overlay.isConnected) return;
        clear(content);
        content.append(el("div", "pj-meta", "Could not render the card: " + (e?.message || e)));
      });

    dl.onclick = () => {
      if (!url) return;
      const a = document.createElement("a");
      a.href = url;
      const base = (project?.name || "funpack-settings").replace(/[^\w.-]+/g, "_");
      a.download = `${base}-settings.png`;
      document.body.append(a); a.click(); a.remove();
    };

    cp.onclick = async () => {
      // Clipboard image writes need a secure context, which a rental reached over plain
      // http://<ip> is not. Say that, rather than failing silently — Download still works.
      status.textContent = "";
      try {
        if (!navigator.clipboard || typeof window.ClipboardItem !== "function") {
          throw new Error("this browser/connection has no image clipboard");
        }
        await navigator.clipboard.write([new window.ClipboardItem({ "image/png": blob })]);
        status.textContent = "Copied.";
      } catch (e) {
        status.textContent = "Could not copy (" + (e?.message || e) + "). Use Download.";
      }
    };
  }

  // Setup-node modal: pick the node by search, then set widget values AND wire its
  // outputs / input sources before it lands in the pipeline. On Add the new node's
  // page opens (multi-wire, expose, bypass live there).
  function openNodeSetup(roleKey) {
    document.querySelectorAll(".ns-overlay").forEach((n) => n.remove());
    const role = roleKey === "__any__" ? null : roles.find((r) => r.key === roleKey);
    const roleFinal = roleKey === "__any__" ? "custom" : roleKey;

    const overlay2 = el("div", "modal-overlay ns-overlay");
    const modal = el("div", "modal ns-modal");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", "Setup node — " + (role ? role.label : "Any node")));
    const hr = el("div", "modal-head-right");
    const closeBtn = el("button", "btn ghost tiny", "✕");
    const closeSetup = () => overlay2.remove();
    closeBtn.onclick = closeSetup;
    hr.append(closeBtn); head.append(hr);
    modal.append(head);
    const content = el("div", "modal-content ns-content");
    modal.append(content);
    overlay2.append(modal);
    overlay2.addEventListener("click", (e) => { if (e.target === overlay2) closeSetup(); });
    document.body.append(overlay2);

    const search = el("input", "ns-search");
    search.type = "search"; search.placeholder = "Search nodes…"; search.autocomplete = "off";
    const listBox = el("div", "ns-list");
    const detail = el("div", "ns-detail");
    content.append(search, listBox, detail);
    setTimeout(() => search.focus(), 0);

    let curList = [], curCand = null, draft = null;
    // Collapsed once a node is picked — the full list is clutter once Values/Wiring are
    // showing below it. Re-expands on "Change" or as soon as the user edits the search again.
    let listOpen = true;

    function renderList() {
      clear(listBox);
      if (curCand && !listOpen) {
        const row = el("div", "ns-row ns-row-selected active");
        row.append(el("span", "ns-row-name", curCand.display_name));
        row.append(el("span", "ns-row-class", curCand.class));
        const change = el("button", "btn ghost tiny ns-change", "Change");
        change.type = "button";
        change.onclick = (e) => { e.stopPropagation(); listOpen = true; renderList(); search.focus(); };
        row.append(change);
        listBox.append(row);
        return;
      }
      const q = search.value.trim().toLowerCase();
      const f = q ? curList.filter((c) => (c.display_name + " " + c.class + " " + (c.category || "")).toLowerCase().includes(q)) : curList;
      if (!f.length) { listBox.append(el("div", "ns-empty", curList.length ? "No match." : "No matching nodes.")); return; }
      f.slice(0, 200).forEach((c) => {
        const row = el("div", "ns-row" + (curCand && curCand.class === c.class ? " active" : ""));
        row.append(el("span", "ns-row-name", c.display_name));
        row.append(el("span", "ns-row-class", c.class));
        row.onclick = () => pick(c.class);
        listBox.append(row);
      });
      if (f.length > 200) listBox.append(el("div", "ns-empty", `…${f.length - 200} more — refine the search.`));
    }
    search.oninput = () => { listOpen = true; renderList(); };

    // Opening "Add loader → Text encoder" on FunPack's own loader is the difference between
    // choosing a file and choosing a node first. Any other node is still one search away.
    function preferredClassFor(role) {
      const recipe = defaultSlots.find((r) => r.role === role);
      return recipe ? recipe.node_class : null;
    }

    async function pick(cls) {
      curCand = specByClass[cls];
      if (!curCand) { try { curCand = await loadSpec(cls); } catch (_) { return; } }
      const extras = defaultExtrasForRole(roleFinal, curCand);
      draft = {
        inputs: defaultsFor(curCand),
        wires: { ...(extras.wires || {}) },              // outName -> single target
        input_sources: { ...(extras.input_sources || {}) },
      };
      listOpen = false;
      renderList(); renderDetail();
    }

    function renderDetail() {
      clear(detail);
      if (!curCand) return;
      // Shares the draft's input_sources so a filled autogrow entry reveals the next one.
      const draftSlot = { id: "__draft__", role: roleFinal, wires: {}, input_sources: draft.input_sources };

      // Same top-down signal order as the node card: sources in, values, wires out.
      if ((curCand.connection_inputs || []).length) {
        const sbox = el("div", "wire-box");
        sbox.append(el("div", "wire-title", "Input sources"));
        visibleConnectionInputs(draftSlot, curCand.connection_inputs).forEach((ci) => {
          const row = el("div", "wire-row");
          row.append(el("span", "wire-out", `${ci.display || ci.name} (${ci.type})`));
          row.append(el("span", "wire-arrow", "←"));
          const sel = el("select", "wire-select");
          const cur = draft.input_sources[ci.name] || "";
          allowedSources(draftSlot, ci).forEach((s) => {
            const o = el("option", null, s.label); o.value = s.value; if (s.value === cur) o.selected = true; sel.append(o);
          });
          sel.onchange = () => {
            if (sel.value) draft.input_sources[ci.name] = sel.value; else delete draft.input_sources[ci.name];
            if (ci.autogrow) renderDetail();  // filling a list entry reveals the next one
          };
          row.append(sel);
          sbox.append(row);
        });
        detail.append(sbox);
      }

      if ((curCand.inputs || []).length) {
        detail.append(el("div", "wire-title", "Values"));
        const grid = el("div", "slot-fields");
        curCand.inputs.forEach((spec) =>
          grid.append(widgetField(spec, draft.inputs[spec.name], (v) => { draft.inputs[spec.name] = v; })));
        detail.append(grid);
      }

      if ((curCand.outputs || []).length) {
        const wbox = el("div", "wire-box");
        wbox.append(el("div", "wire-title", "Wire outputs to"));
        curCand.outputs.forEach((out) => {
          const row = el("div", "wire-row");
          row.append(el("span", "wire-out", `${out.name} (${out.type})`));
          row.append(el("span", "wire-arrow", "→"));
          row.append(draftDestMulti(draftSlot, out, draft.wires));
          wbox.append(row);
        });
        detail.append(wbox);
      }

      const foot = el("div", "ns-foot");
      const add = el("button", "btn primary", "Add node");
      add.onclick = async () => {
        const slot = {
          id: uid(), role: roleFinal, node_class: curCand.class,
          inputs: draft.inputs,
          wires: {}, input_sources: {},
        };
        config.slots.push(slot);
        Object.entries(draft.wires).forEach(([o, raw]) => {
          wireTargets(raw).forEach((t) => {
            if (!t) return;
            _addWire(slot.id, o, t);
            _setWireTarget(slot, o, "", t); // mirror node: targets onto the destination's input source
          });
        });
        Object.entries(draft.input_sources).forEach(([inp, v]) => { if (v) _setInputSource(slot, inp, v); });
        feedMatchingInputs(slot);
        reconcileOpenPortWiring();
        await persist();
        closeSetup();
        setView("node:" + slot.id);
      };
      const cancel = el("button", "btn ghost", "Cancel");
      cancel.onclick = closeSetup;
      foot.append(cancel, add);
      detail.append(foot);
    }

    listBox.append(el("div", "ns-empty", "Loading nodes…"));
    (roleKey === "__any__" ? ensureAllNodes() : candidates(roleKey))
      .then((list) => {
        curList = list || [];
        const preferred = preferredClassFor(roleFinal);
        if (preferred && curList.some((c) => c.class === preferred)) { pick(preferred); return; }
        renderList();
      })
      .catch((e) => {
        clear(listBox);
        listBox.append(el("div", "ns-empty", "Node list unavailable: " + (e && e.message ? e.message : e)));
      });
  }

  // Does any configured slot produce the given type (optionally filtered by role)?
  function slotProducesType(type, roleHint) {
    return config.slots.some((s) => {
      if (roleHint && s.role !== roleHint) return false;
      const spec = specFor(s);
      return (spec?.outputs || []).some((o) => o.type === type);
    });
  }

  // Does this slot SOURCE the type — output it without consuming one? A loader or a
  // generator does; a patcher that only passes the type through (a LoRA on MODEL/CLIP)
  // does not, and can't satisfy a requirement on its own.
  function slotSourcesType(type, slot) {
    const spec = specFor(slot);
    if (!(spec?.outputs || []).some((o) => o.type === type)) return false;
    return !(spec?.connection_inputs || []).some((ci) => ci.type === type);
  }

  // VAE is asked for twice — video and audio — and only the role tells the two apart, so
  // those stay role-bound (two separate producers required). Every other requirement is
  // about whether the pipeline can GET that type from somewhere, so any node that really
  // sources it counts, whatever role the user filed it under: MiniMax H3's fl2va / ref2va
  // nodes emit their own AV latent, which is exactly what an Empty AV Latent would provide.
  function requirementSatisfied(req) {
    if (req.type === "VAE") return slotProducesType(req.type, req.role_hint);
    if (config.slots.some((s) => slotSourcesType(req.type, s))) return true;
    if (req.role_hint) return slotProducesType(req.type, req.role_hint);
    return slotProducesType(req.type, null);
  }

  // Dismiss state for the requirements list. Persisted, because a pipeline someone is
  // deliberately leaving incomplete stays incomplete across reloads, and re-nagging every
  // time the panel opens is exactly what the dismiss is for.
  const REQ_HIDE_KEY = "funpack_models_req_hidden";
  function reqPanelHidden(next) {
    try {
      if (next === undefined) return localStorage.getItem(REQ_HIDE_KEY) === "1";
      if (next) localStorage.setItem(REQ_HIDE_KEY, "1");
      else localStorage.removeItem(REQ_HIDE_KEY);
    } catch (_) { /* private mode: the list just stays visible */ }
    return !!next;
  }

  function requirementsPanel() {
    const sec = el("div", "req-panel");
    if (config.disable_core) {
      // No FunPack core means no FunPack requirements — say so once, briefly, and get out
      // of the way. Everything else in this panel still works on the user's own nodes.
      sec.append(el("div", "req-empty",
        "Custom workflow — the built-in pipeline is off, so FunPack's loader requirements "
        + "do not apply. Wire your own nodes below."));
      return sec;
    }
    if (reqPanelHidden()) {
      const missingNow = requirements.filter((r) => r.required && !requirementSatisfied(r));
      if (!missingNow.length) return sec;          // nothing to say; stay quiet
      const line = el("div", "req-collapsed");
      line.append(el("span", "req-dot", "✕"));
      line.append(el("span", null, `${missingNow.length} missing in pipeline`));
      const show = el("button", "btn ghost tiny", "Show");
      show.onclick = () => { reqPanelHidden(false); render(); };
      line.append(show);
      sec.append(line);
      return sec;
    }
    // The wiring rules are reference material, not a warning — folded away so the first
    // thing in this panel is the short list of what is actually missing.
    function guidedRulesFold() {
      // The latent and audio path differ by family — H3 makes both streams in one node
      // that feeds the sampler directly, so pointing at Studio · latent here would send
      // the user to a port its graph does not even have.
      const h3 = familyKey() === "minimax_h3";
      const det = el("details", "req-fold");
      det.append(el("summary", "req-fold-sum", "Guided wiring rules"));
      det.append(el("div", "req-hint guided-hint",
        (h3
          ? "The AV LATENT → Chain Sampler · latent_template (Empty MiniMax H3 "
            + "AV Latent makes both streams). Audio VAE → VAE Decode Audio · vae, and optionally "
            + "also Chain Sampler · audio_vae to encode audio references. "
          : "LATENT → Studio · latent (core forwards to Concat · video_latent). "
            + "Audio latent → Concat · audio_latent only. ")
        + "IMAGE → Studio · source_image (Input Image Processing defaults to Timeline scene image). "
        + "MODEL/CLIP may chain through LoRA. "
        + "Enable Full control for manual rewiring."));
      return det;
    }
    if (!requirements.length) {
      sec.append(el("div", "req-title", "Pipeline requirements"));
      sec.append(el("div", "req-empty", "Requirements not loaded — open Models to refresh."));
      return sec;
    }

    const required = requirements.filter((r) => r.required);
    const optional = requirements.filter((r) => !r.required);
    const missing = required.filter((r) => !requirementSatisfied(r));
    const allOk = !missing.length;

    // One line per requirement: dot, name, type, and — only when it is missing and we know
    // which role fills it — a + on the right. The hint used to be a paragraph inside the
    // row, which made every missing item a block three times the height of a satisfied one.
    function reqRow(req, cls) {
      const ok = requirementSatisfied(req);
      const row = el("div", "req-row" + cls + (ok ? " ok" : (req.required ? " missing" : "")));
      row.append(el("span", "req-dot", ok ? "✓" : (req.required ? "✕" : "·")));
      row.append(el("span", "req-label", req.label));
      row.append(el("span", "req-type", req.type));
      if (req.hint) row.title = req.hint;
      if (!ok && req.required && req.role_hint) {
        const addBtn = el("button", "btn ghost tiny req-add", "+");
        addBtn.title = `Add a ${req.label} — ${req.hint || "picks the node for you"}`;
        addBtn.onclick = () => openNodeSetup(req.role_hint);
        row.append(addBtn);
      }
      return row;
    }

    const head = el("div", "req-head");
    head.append(el("span", "req-title", missing.length
      ? `Missing in pipeline · ${missing.length}` : "Pipeline requirements"));
    // Dismissable, because a half-built pipeline the user is deliberately leaving half-built
    // should not keep shouting. It comes back as a one-line link, never silently.
    const hide = el("button", "req-dismiss", "✕");
    hide.title = "Hide this list";
    hide.onclick = () => { reqPanelHidden(true); render(); };
    head.append(hide);
    sec.append(head);

    // Missing first: it is the only part that needs doing.
    missing.forEach((req) => sec.append(reqRow(req, "")));
    const satisfied = required.filter(requirementSatisfied);
    if (satisfied.length || optional.length) {
      const det = el("details", "req-fold");
      det.open = false;
      det.append(el("summary", "req-fold-sum",
        `Satisfied${optional.length ? " and optional" : ""} · ${satisfied.length + optional.length}`));
      satisfied.forEach((req) => det.append(reqRow(req, "")));
      optional.forEach((req) => det.append(reqRow(req, " opt")));
      sec.append(det);
    }

    if (allOk && config.slots.length) {
      const v = validation();
      const bits = [];
      if (v.errors) bits.push(`${v.errors} slot error${v.errors > 1 ? "s" : ""}`);
      if (v.warns) bits.push(`${v.warns} warning${v.warns > 1 ? "s" : ""}`);
      const bar = el("div", "valid-banner " + (v.errors ? "bad" : (v.warns ? "warn" : "ok")));
      bar.append(el("span", "valid-dot", v.errors ? "!" : "✓"));
      bar.append(el("span", null, v.errors ? `Not ready — ${bits.join(", ")} to fix.`
        : (v.warns ? `Pipeline ready — ${bits.join(", ")}.` : "Pipeline ready.")));
      sec.append(bar);
    }

    if (pipelineLocked()) sec.append(guidedRulesFold());
    return sec;
  }

  function summaryBanner(v) {
    // Kept for any callers — but requirementsPanel() is now shown instead.
    return null;
  }

  // ── linked-inputs section ──────────────────────────────────────────────────────
  // Project values a linked input can be driven by at generation time. `kinds` is the
  // widget kinds each one can legitimately fill — the prompt bindings are what let a
  // custom node's own text field (e.g. MiniMaxH3ReferenceToVideo.prompt) receive the
  // same prompt the built-in encoder gets, instead of being typed twice.
  const EDITOR_SOURCES = [
    { key: "", label: "Manual value" },
    { key: "prompt", label: "Project · Prompt (global)", kinds: ["string"] },
    { key: "negative_prompt", label: "Project · Negative prompt", kinds: ["string"] },
    // The texts Studio wraps every scene in. A node encoding on its own knows nothing about
    // them, so it can take them apart (anchor / postfix) or take the whole thing at once.
    // The global prompt already carries the anchor — the postfix is the only piece outside it.
    { key: "anchor", label: "Project · Anchor text (prepended)", kinds: ["string"] },
    { key: "postfix", label: "Project · Postfix (appended)", kinds: ["string"] },
    { key: "full_prompt", label: "Project · Prompt + postfix (what Studio encodes)", kinds: ["string"] },
    { key: "seed", label: "Project · Seed", kinds: ["int", "float"] },
    { key: "frame_rate", label: "Project · FPS", kinds: ["int", "float"] },
    { key: "num_frames_per_scene", label: "Project · Frames", kinds: ["int", "float"] },
    { key: "width", label: "Project · Width", kinds: ["int", "float"] },
    { key: "height", label: "Project · Height", kinds: ["int", "float"] },
  ];

  function linkExposeBtn(link) {
    const b = el("button", "eye-btn" + (link.exposed ? " on" : ""), "◉"); b.type = "button";
    b.title = link.exposed ? "Hide from main editor window" : "Show in main editor window";
    b.onclick = async () => { link.exposed = !link.exposed; await persist(); render(); };
    return b;
  }

  function linkCard(link) {
    const card = el("div", "link-card");
    const head = el("div", "link-head");
    head.append(el("span", "link-ico", "🔗"));
    const name = el("input", "link-name"); name.value = link.name || "";
    name.onchange = async () => { link.name = name.value.trim() || link.name; await persist(); render(); };
    head.append(name);
    head.append(linkExposeBtn(link));
    const rm = el("button", "btn ghost tiny danger", "unlink all");
    rm.onclick = async () => { config.links = (config.links || []).filter((l) => l.id !== link.id); await persist(); render(); };
    head.append(rm);
    card.append(head);

    // source: a manual value, or driven by a project/editor setting
    const srcRow = el("label", "field link-source");
    srcRow.append(el("span", null, "Driven by"));
    const srcSel = el("select");
    const cur = link.source === "editor" ? link.editor_key : "";
    EDITOR_SOURCES.forEach(({ key, label, kinds }) => {
      // Only offer a project value a field of this kind can actually hold — feeding a
      // number into a text field (or a prompt into a width) silently breaks generation.
      if (key && kinds && link.kind && !kinds.includes(link.kind) && key !== cur) return;
      const o = new Option(label, key); if (key === cur) o.selected = true; srcSel.append(o);
    });
    srcSel.onchange = async () => {
      if (srcSel.value) { link.source = "editor"; link.editor_key = srcSel.value; }
      else { link.source = "manual"; delete link.editor_key; }
      await persist(); render();
    };
    srcRow.append(srcSel);
    card.append(srcRow);

    if (link.source === "editor") {
      const srcLbl = (EDITOR_SOURCES.find((s) => s.key === link.editor_key) || {}).label || link.editor_key;
      const isText = ["prompt", "negative_prompt", "anchor", "postfix", "full_prompt"].includes(link.editor_key);
      // "the same text Studio would" is only true of full_prompt. Studio appends the postfix
      // to every scene itself, so a node linked to the global prompt encodes strictly less --
      // silently, and the postfix is where audio and style directions usually live.
      let note = "";
      if (isText) {
        note = " Shortcuts and $variables are expanded first";
        if (link.editor_key === "full_prompt") {
          note += ", so the node encodes the same text Studio would.";
        } else if (link.editor_key === "prompt") {
          const pj = window.Store?.get().project || {};
          const hasPostfix = pj.postfix_enabled !== false && String(pj.postfix || "").trim();
          note += hasPostfix
            ? ". The postfix is NOT included — Studio appends it to every scene. Pick Project · Prompt + postfix to match."
            : ".";
        } else {
          note += ".";
        }
      }
      card.append(el("div", "link-bound", `Value comes from ${srcLbl} at generate — the fields below are ignored.` + note));
    } else {
      const spec = { name: "shared value", kind: link.kind, choices: link.choices, required: false };
      const vf = widgetField(spec, link.value, async (v) => { applyLinkValue(link, v); await persist(); });
      vf.classList.add("link-value");
      card.append(vf);
    }

    const mem = el("div", "link-members");
    (link.members || []).forEach((m) => {
      const s = slotById(m.slotId);
      const chip = el("span", "link-chip", `${s ? slotName(s) : "?"} · ${m.input}`);
      const x = el("button", "chip-x", "✕"); x.title = "Remove from link";
      x.onclick = async () => {
        link.members = (link.members || []).filter((mm) => !(mm.slotId === m.slotId && mm.input === m.input));
        if (link.members.length < 1) config.links = (config.links || []).filter((l) => l.id !== link.id);
        await persist(); render();
      };
      chip.append(x); mem.append(chip);
    });
    card.append(mem);
    return card;
  }

  async function saveLinkSelection() {
    if (linkSel.length < 1) return;
    const def = "size " + ((config.links || []).length + 1);
    const nm = prompt("Link name:", def); if (nm == null) return;
    const first = linkSel[0]; const s0 = slotById(first.slotId);
    const val = s0 ? (s0.inputs || {})[first.input] : undefined;
    const link = { id: uid(), name: nm.trim() || def, kind: first.kind,
      choices: first.kind === "combo" ? (first.choices || []) : undefined, value: val,
      members: linkSel.map((s) => ({ slotId: s.slotId, input: s.input })), exposed: false };
    if (first.kind === "int" || first.kind === "float") {
      const w = s0 && inputSpecFor(s0, first.input);
      if (w) Object.assign(link, numMeta(w));
    }
    ensureLinks().push(link);
    applyLinkValue(link, val);
    linkMode = false; linkSel = [];
    // Members are picked ON node pages, where saving is deferred until the node's Save
    // button. Going through that buffer meant "Save link" only marked the page dirty, and
    // the jump to the links view then offered to DISCARD the link it had just made — so a
    // link could never be created at all. Pressing Save link is the explicit save.
    if (deferSave) { await persistNow(); deferDirty = false; }
    else await persist();
    setView("links");
  }

  // Persistent bar while picking link members — stays visible as the user moves
  // between node pages, replacing the old everything-expanded scrollable list.
  function linkModeBar() {
    const bar = el("div", "models-linkbar");
    bar.append(el("span", "models-linkbar-txt",
      linkSel.length
        ? `Linking ${linkSel.length} input${linkSel.length > 1 ? "s" : ""}: ${linkSel.map((s) => s.label).join(", ")}`
        : "Pick inputs to link: open a node in the sidebar and click ＋ next to its inputs."));
    const save = el("button", "btn primary tiny", `Save link (${linkSel.length})`);
    save.disabled = linkSel.length < 1;
    save.onclick = saveLinkSelection;
    const cancel = el("button", "btn ghost tiny", "Cancel");
    cancel.onclick = () => { linkMode = false; linkSel = []; render(); };
    bar.append(save, cancel);
    return bar;
  }

  function linksView() {
    const sec = el("div", "links-section");
    const head = el("div", "links-head");
    head.append(el("div", "composer-title", "Linked inputs"));
    const right = el("div", "links-head-right");
    if (!linkMode) {
      const b = el("button", "btn ghost tiny", "＋ New link");
      b.onclick = () => { linkMode = true; linkSel = []; render(); };
      right.append(b);
    }
    head.append(right);
    sec.append(head);
    sec.append(el("div", "links-hint",
      "A link drives several node inputs from one control, or ties an input to a project value (e.g. a loader's FPS to Project · FPS)."));
    (config.links || []).forEach((l) => sec.append(linkCard(l)));
    if (!linkMode && !(config.links || []).length)
      sec.append(el("div", "links-empty", "No links yet. Click ＋ New link, then pick inputs on the node pages."));
    return sec;
  }

  // ── built-in pipeline (core) — visible + re-wireable ──────────────────────────
  function setCoreOverride(cid, inp, value) {
    config.core_overrides = config.core_overrides || {};
    config.core_overrides[cid] = config.core_overrides[cid] || {};
    const prev = config.core_overrides[cid][inp];
    if (value) config.core_overrides[cid][inp] = value;
    else delete config.core_overrides[cid][inp];
    const parsed = _parseOutSource(value);
    if (parsed) {
      for (const entry of wiringRules.open_core_ports || []) {
        if (entry.core_id === cid && entry.input === inp) {
          _clearPortWires(entry.port, parsed.slotId, parsed.out);
          _addWire(parsed.slotId, parsed.out, "port:" + entry.port);
          break;
        }
      }
    } else if (prev) {
      const old = _parseOutSource(prev);
      if (old) {
        for (const entry of wiringRules.open_core_ports || []) {
          if (entry.core_id === cid && entry.input === inp) {
            _removeWire(old.slotId, old.out, "port:" + entry.port);
            break;
          }
        }
      }
    }
  }
  function coreCard(n) {
    const card = el("div", "slot-card open" + (n.installed ? "" : " slot-bad"));
    const head = el("div", "slot-head");
    head.append(el("span", "slot-role", n.display_name));
    head.append(el("span", "slot-node", n.class));
    if (!n.installed) head.append(el("span", "slot-badge bad", "not installed"));
    card.append(head);
    if ((n.inputs || []).length) {
      const box = el("div", "wire-box");
      box.append(el("div", "wire-title", "Inputs ← source"));
      n.inputs.forEach((inp) => {
        const row = el("div", "wire-row");
        row.append(el("span", "wire-out", `${inp.name} (${inp.type})`));
        row.append(el("span", "wire-arrow", "←"));
        if (inp.locked) {
          const lbl = el("span", "wire-locked", inp.detail || "Fixed built-in link");
          lbl.title = "Guided wiring — enable Full control to override this input";
          row.append(lbl);
        } else {
          const sel = el("select", "wire-select");
          (inp.options || []).forEach((o) => { const op = el("option", null, o.label); op.value = o.value; if (o.value === (inp.value || "")) op.selected = true; sel.append(op); });
          if (inp.value && !(inp.options || []).some((o) => o.value === inp.value)) { const op = el("option", null, inp.value + " (missing)"); op.value = inp.value; op.selected = true; sel.append(op); }
          sel.onchange = async () => { setCoreOverride(n.id, inp.name, sel.value); await persist(); coreNodes = (await API.coreGraph(window.Store?.get().project?.id).catch(() => ({}))).nodes || coreNodes; render(); };
          row.append(sel);
        }
        box.append(row);
      });
      card.append(box);
    }
    // Show every core output. In Full control these are tappable as slot input sources
    // (pick "<node> → <output>" in a slot's Input sources), so the user can, e.g., feed a
    // replacement sampler with Studio's conditioning. In guided mode they're informational.
    const outs = n.outputs || [];
    if (outs.length) {
      const box = el("div", "wire-box");
      box.append(el("div", "wire-title", "Outputs → destinations"));
      outs.forEach((o) => {
        const row = el("div", "wire-row");
        row.append(el("span", "wire-out", `${o.name} (${o.type})`));
        row.append(el("span", "wire-arrow", "→"));
        const dest = (o.to || []).length
          ? o.to.join(", ")
          : (config.full_control ? "available — pick in a slot's Input sources" : "—");
        const destEl = el("span", "wire-dest", dest);
        if (!(o.to || []).length && !config.full_control) destEl.classList.add("wire-muted");
        row.append(destEl);
        box.append(row);
      });
      card.append(box);
    }
    return card;
  }
  // Which model family the built-in graph is built for. LTXAV and MiniMax H3 need
  // different node classes (H3 has no LTXVConditioning, makes both latent streams in one
  // node, and decodes audio with core's generic node), so this is asked, never guessed
  // from a checkpoint filename — a wrong guess fails deep inside ComfyUI at generate time
  // instead of here, where it can be fixed.
  const FAMILIES = [
    { key: "ltxav", label: "LTX2 / LTX2.3 / LTX2.5",
      sub: "Gemma3 text encoder (Gemma4 on 2.5) · 8k+1 frames · separate video and audio latents" },
    { key: "minimax_h3", label: "MiniMax H3 (Hailuo)",
      sub: "Qwen3-VL text encoder · 17k+5 frames at 24 fps · one AV latent node · ref2va references" },
  ];

  function familyKey() {
    const f = String(config.model_family || "ltxav").toLowerCase();
    return FAMILIES.some((x) => x.key === f) ? f : "ltxav";
  }

  function familySection() {
    const sec = el("div", "links-section");
    const head = el("div", "links-head");
    head.append(el("span", "lib-sub", "Model family"));
    sec.append(head);
    // Detected from the checkpoint, never chosen. A radio button here could contradict the
    // file that is actually loaded — pick LTX, load H3 — and the whole graph would be wired
    // for the wrong model, surfacing as a stray port instead of as a family error.
    const probe = config.model_family_probe || {};
    const known = FAMILIES.find((x) => x.key === familyKey());
    const row = el("div", "links-row");
    if (probe.detected) {
      row.append(el("span", "lib-sub", (known ? known.label : familyKey())));
      sec.append(row);
      sec.append(el("div", "links-hint", "Detected from " + probe.reason));
    } else {
      row.append(el("span", "lib-sub", known ? known.label : familyKey()));
      sec.append(row);
      sec.append(el("div", "links-hint",
        (probe.reason ? "Could not read the model: " + probe.reason + ". " : "")
        + "Showing the last known family. Pick a .safetensors diffusion model and the "
        + "pipeline will wire itself for whatever it is."));
    }
    const setup = el("div", "links-row");
    const btn = el("button", "btn ghost tiny", "Setup\u2026");
    btn.title = "What this model family needs — nodes and model files, including anything "
      + "not released yet.";
    btn.onclick = () => window.PipelineSetup?.open();
    setup.append(btn);
    sec.append(setup);
    return sec;
  }


  function coreSection() {
    const sec = el("div", "links-section");
    const disabled = !!config.disable_core;
    const head = el("div", "links-head");
    const toggle = el("button", "btn ghost tiny", (coreOpen ? "▾ " : "▸ ") + "Built-in pipeline");
    toggle.onclick = () => { coreOpen = !coreOpen; render(); };
    head.append(toggle);
    head.append(el("span", "lib-sub", disabled ? "disabled" : `${coreNodes.length} fixed nodes`));
    if (!disabled) {
      const fc = el("button", "btn ghost tiny" + (config.full_control ? " active" : ""), config.full_control ? "Full control" : "Guided wiring");
      fc.title = config.full_control
        ? "Full control ON — every core socket can be rewired manually (advanced)."
        : "Guided wiring ON — loaders wire only to their built-in ports. Click for full manual control.";
      fc.onclick = async () => {
        config.full_control = !config.full_control;
        if (!config.full_control) clearInternalCoreOverrides();
        await persist();
        try { coreNodes = (await API.coreGraph(window.Store?.get().project?.id)).nodes || coreNodes; } catch (_) {}
        render();
      };
      head.append(fc);
    }
    const dis = el("button", "btn ghost tiny" + (disabled ? " danger" : ""), disabled ? "↺ Enable built-in pipeline" : "⏻ Disable built-in pipeline");
    dis.title = disabled
      ? "Re-enable the built-in FunPack Studio → Chain Sampler → decode → combine pipeline."
      : "Drop the built-in pipeline entirely and run only your wired nodes. Wire your final image to 🌐 Global video output (and audio to 🌐 Global audio output) so the editor shows the result.";
    dis.onclick = async () => { config.disable_core = !disabled; await persist(); render(); };
    head.append(dis);
    sec.append(head);
    if (disabled) {
      sec.append(el("div", "links-hint", "Built-in pipeline is OFF — generation runs only your configured nodes. Wire your final IMAGE output to 🌐 Global video output (and AUDIO to 🌐 Global audio output) so the result shows in the editor."));
      return sec;
    }
    if (coreOpen) {
      const h3core = familyKey() === "minimax_h3";
      const hint = pipelineLocked()
        ? (h3core
            ? "Fixed FunPack path (Studio → Chain Sampler → separate AV → decode). H3 has no "
              + "conditioning node between Studio and the sampler. Wire the AV LATENT to "
              + "Chain Sampler · latent_template. "
            : "Fixed FunPack path (Studio → Conditioning → Chain Sampler → decode). "
              + "Wire video LATENT to Studio · latent (not Concat · video_latent — that link is internal). ")
          + "Wire IMAGE via Input Image Processing → Studio · source_image (Timeline default). "
          + "MODEL/CLIP may chain through patchers. Enable Full control to override core links."
        : "The fixed FunPack nodes and their wiring. Each input defaults to its built-in source — pick another to re-wire it.";
      sec.append(el("div", "links-hint", hint));
      coreNodes.forEach((n) => sec.append(coreCard(n)));
    }
    return sec;
  }

  // ── shell: inner sidebar (Links · Pipeline · + New node · nodes) + content pane ──
  let view = "pipeline"; // "pipeline" | "links" | "node:<slotId>"

  // Internal: no unsaved-changes check. Used by Cancel/Save, which have already dealt
  // with the buffer, and by paths that removed the node they were editing.
  function _setView(v) {
    if (v !== "node:" + _directNodeId) _directNodeId = null;
    if (view.startsWith("node:")) endEdit();
    view = v;
    if (view.startsWith("node:")) beginEdit();
    render();
  }

  function setView(v) {
    if (v === view) return;
    if (deferDirty && !confirm(
      "This node has unsaved changes.\n\nLeaving now discards them. Continue?"
    )) return;
    if (deferDirty) restoreBaseline();
    _setView(v);
  }

  const mnItem = (opts) => window.SettingsWindow.navItem(opts);

  function sideBar(v) {
    const side = el("div", "models-side");

    side.append(mnItem({
      icon: "🔗", label: "Linked inputs",
      badge: (config.links || []).length || null,
      active: view === "links", onClick: () => setView("links"),
    }));
    const pipeBad = !config.disable_core && missingEssentials().length > 0;
    side.append(mnItem({
      icon: "▦", label: "Pipeline",
      sub: config.disable_core ? "built-in off" : (pipeBad ? "missing loaders" : null),
      active: view === "pipeline", onClick: () => setView("pipeline"),
    }));

    // Nodes themselves live in the Pipeline grid, not here — the sidebar is for the two
    // places you can be, the way the reference app splits categories from the shelf.
    return side;
  }

  // ── the node shelf ───────────────────────────────────────────────────────────
  function nodeCard(slot, issues) {
    const errs = (issues || []).filter((i) => i.level === "error").length;
    const warns = (issues || []).filter((i) => i.level === "warn").length;
    const card = el("button", "node-card");
    card.type = "button";
    if (errs) card.classList.add("bad");
    else if (warns || slot.bypassed) card.classList.add("warn");
    card.onclick = () => setView("node:" + slot.id);

    const art = el("div", "node-card-art");
    // For FunPack's own nodes the art is an icon and a plain-language name. For anything
    // else the class name IS the art — the label below it is the node's own name, so
    // showing the class twice would say nothing new.
    const bi = builtIn(slot);
    if (bi) {
      card.classList.add("built-in");
      const mark = el("div", "node-card-builtin");
      mark.append(el("div", "node-card-icon", bi.icon));
      mark.append(el("div", "node-card-kind", bi.label));
      art.append(mark);
    } else {
      art.append(el("span", "node-card-mark", slotName(slot) || "?"));
    }
    const flags = el("div", "node-card-flags");
    if (slot.bypassed) flags.append(el("span", "node-flag warn", "bypassed"));
    const nExp = (slot.exposed || []).length;
    if (nExp) flags.append(el("span", "node-flag exposed", `◉ ${nExp}`));
    if (errs) flags.append(el("span", "node-flag bad", errs + (errs > 1 ? " errors" : " error")));
    else if (warns) flags.append(el("span", "node-flag warn", warns + (warns > 1 ? " warnings" : " warning")));
    art.append(flags);

    // On the tile, not under it: the name and what it feeds are the card, and a caption
    // floating below left the shelf reading as a grid of unlabelled squares.
    const meta = el("div", "node-card-meta");
    meta.append(el("div", "node-card-title", slotDisplayLabel(slot)));
    const outs = wireDestinationLabels(slot);
    const flow = el("div", "node-card-flow");
    if (!outs.length) {
      flow.classList.add("none");
      flow.append(el("span", null, "not wired"));
    } else {
      flow.append(el("span", null, "→ " + outs[0]));
      if (outs.length > 1) flow.append(el("span", "node-card-more", `+${outs.length - 1}`));
    }
    meta.append(flow);
    art.append(meta);
    card.append(art);
    return card;
  }

  // Human label for one wire target ("port:…", "node:<id>:<input>", "global:…").
  function targetLabel(value) {
    if (!value) return null;
    if (value === "global:video") return "Global video output";
    if (value === "global:audio") return "Global audio output";
    if (value.startsWith("port:")) {
      const p = ports.find((x) => "port:" + x.id === value);
      return p ? p.label : value.slice(5);
    }
    const n = _parseNodeTarget(value);
    if (n) {
      const dest = slotById(n.slotId);
      return dest ? `${slotDisplayLabel(dest)} · ${n.input}` : n.input;
    }
    return value;
  }

  function wireDestinationLabels(slot) {
    const out = [];
    Object.values(slot.wires || {}).forEach((raw) => {
      wireTargets(raw).forEach((t) => {
        const l = targetLabel(t);
        if (l && !out.includes(l)) out.push(l);
      });
    });
    return out;
  }

  // ── search ────────────────────────────────────────────────────────────────────
  // One box over everything a node can be recognised by: its class, its label, the group
  // it is filed under, the names AND values of its widgets (so a model filename matches),
  // and the names and TYPES of its sockets. "model" therefore finds UNETLoader by class,
  // anything with a MODEL input or output by type, and every node holding a file with
  // "model" in its name.
  let searchQ = "";

  function slotHaystack(slot) {
    const out = [slot.node_class || "", slot.label || "", slot.role_label || "", slot.group || ""];
    Object.entries(slot.inputs || {}).forEach(([k, v]) => {
      out.push(k);
      if (v != null && typeof v !== "object") out.push(String(v));
    });
    const cand = specFor(slot);
    (cand?.connection_inputs || []).forEach((ci) => out.push(ci.name || "", ci.type || ""));
    (cand?.outputs || []).forEach((o) => out.push(o.name || "", o.type || ""));
    (cand?.inputs || []).forEach((i) => out.push(i.name || ""));
    return out.join("\n").toLowerCase();
  }

  function slotMatches(slot) {
    if (!searchQ) return true;
    const hay = slotHaystack(slot);
    // Every word must appear somewhere, so "lora model" narrows instead of widening.
    return searchQ.split(/\s+/).filter(Boolean).every((w) => hay.includes(w));
  }

  function searchBar(shown, total) {
    const row = el("div", "node-search");
    const inp = el("input", "node-search-input");
    inp.type = "search";
    inp.placeholder = "Search nodes — name, model file, or socket type (e.g. model, vae, lora)";
    inp.value = searchQ;
    inp.dataset.k = "models-search";
    inp.oninput = () => { searchQ = inp.value.trim().toLowerCase(); render(); };
    row.append(inp);
    if (searchQ) {
      row.append(el("span", "node-search-count", `${shown} of ${total}`));
      const clear = el("button", "btn ghost tiny", "Clear");
      clear.onclick = () => { searchQ = ""; render(); };
      row.append(clear);
    }
    return row;
  }

  // Groups a node can be filed under, in the order they first appear.
  function groupNames() {
    const seen = [];
    config.slots.forEach((s) => {
      const g = (s.group || "").trim();
      if (g && !seen.includes(g)) seen.push(g);
    });
    return seen;
  }

  // Collapsed groups, per project. A group someone folded away should stay folded.
  const GROUP_FOLD_KEY = "funpack_models_groups_closed";
  function closedGroups() {
    try { return new Set(JSON.parse(localStorage.getItem(GROUP_FOLD_KEY) || "[]")); }
    catch (_) { return new Set(); }
  }
  function toggleGroup(name) {
    const set = closedGroups();
    if (set.has(name)) set.delete(name); else set.add(name);
    try { localStorage.setItem(GROUP_FOLD_KEY, JSON.stringify([...set])); } catch (_) {}
  }

  // Removing a node from the middle of a chain joins its neighbours, the way deleting a node
  // in ComfyUI does: loader → LoRA → sampler, drop the LoRA and the loader feeds the sampler.
  // Same rule as bypass — for each output, exactly one input of a matching type is the one
  // that carried it — except this one is permanent, so it runs before the slot goes.
  function rewireAround(slot) {
    const cand = specFor(slot);
    if (!cand) return;
    const cis = cand.connection_inputs || [];
    const outs = cand.outputs || [];
    // output name -> the source feeding the one input that can carry it through
    const passthrough = {};
    outs.forEach((out) => {
      const matching = cis.filter((ci) => typeAccepts(ci.type, out.type));
      if (matching.length !== 1) return;          // ambiguous: nothing to promote
      const upstream = (slot.input_sources || {})[matching[0].name];
      if (upstream && upstream !== "auto") passthrough[out.name] = upstream;
    });
    if (!Object.keys(passthrough).length) return;
    // An imported workflow records output names as its own graph spelled them, which need
    // not match this node class's output_name. With a single output there is no ambiguity.
    const only = outs.length === 1 ? passthrough[outs[0].name] : undefined;
    const resolve = (outName) => (passthrough[outName] !== undefined ? passthrough[outName] : only);

    // An edge can be authored from either end. Downstream nodes usually hold it as their own
    // input source ("out:<this>:<output>") and carry no wire on this side at all — which is
    // exactly how an imported workflow arrives, so both directions have to be walked.
    config.slots.forEach((s2) => {
      if (s2.id === slot.id) return;
      Object.entries(s2.input_sources || {}).forEach(([inp, value]) => {
        const p = _parseOutSource(value);
        if (!p || p.slotId !== slot.id) return;
        const up = resolve(p.out);
        if (up) _setInputSource(s2, inp, up);
      });
    });
    Object.entries(slot.wires || {}).forEach(([outName, raw]) => {
      const up = resolve(outName);
      if (!up) return;
      const src = _parseOutSource(up);
      wireTargets(raw).filter(Boolean).forEach((target) => {
        const dest = _parseNodeTarget(target);
        if (dest) {
          const ds = slotById(dest.slotId);
          if (ds && ds.id !== slot.id) {
            ds.input_sources = ds.input_sources || {};
            _setInputSource(ds, dest.input, up);
          }
          return;
        }
        // A core port or a global output can only be re-fed from another node's output —
        // the timeline image and the core primitives have no wire to inherit.
        if (src) _addWire(src.slotId, src.out, target);
      });
    });
  }

  // Everything still pointing at a node that is gone. Left behind, these render as
  // "(missing)" sources and dead wires that the builder then has to reject.
  function purgeSlotReferences(id) {
    config.slots.forEach((s) => {
      Object.keys(s.wires || {}).forEach((out) => {
        s.wires[out] = wireTargets(s.wires[out])
          .filter((t) => { const n = _parseNodeTarget(t); return !n || n.slotId !== id; });
      });
      Object.entries(s.input_sources || {}).forEach(([inp, src]) => {
        const p = _parseOutSource(src);
        if (p && p.slotId === id) s.input_sources[inp] = "";
      });
    });
    Object.values(config.core_overrides || {}).forEach((ins) => {
      Object.keys(ins || {}).forEach((k) => {
        if (String(ins[k]).startsWith(`out:${id}:`)) delete ins[k];
      });
    });
  }

  function removeSlot(slot) {
    rewireAround(slot);
    config.slots = config.slots.filter((s) => s.id !== slot.id);
    purgeSlotReferences(slot.id);
    reconcileOpenPortWiring();
  }

  function _grid(slots, v) {
    const wrap = el("div", "node-grid");
    slots.forEach((slot) => wrap.append(nodeCard(slot, v.perSlot[slot.id] || [])));
    return wrap;
  }

  // Why this node cannot be bypassed, or [] when it can. Mirrors builder._apply_bypass: for
  // every output something CONSUMES there must be exactly one input that can carry it
  // through. Checked here so a group's bypass button is never offered when pressing it
  // would stop the run instead — the builder still has the last word, and still explains.
  // Whether a wire target lands on an OPTIONAL input, and so needs no pass-through when the
  // node feeding it goes away. Core ports are never treated as optional here — the editor
  // does not model their requiredness, and guessing wrong would hide a real block.
  function targetIsOptional(t) {
    if (!t || !t.startsWith("node:")) return false;
    const [, sid, inName] = t.split(":");
    const s2 = slotById(sid);
    const ci = s2 && (specFor(s2)?.connection_inputs || []).find((c) => c.name === inName);
    return !!ci && !ci.required;
  }

  function bypassBlockers(slot, alsoGoing) {
    const cand = specFor(slot);
    if (!cand) return [];                            // spec unknown: let the builder decide
    // `alsoGoing` are the other nodes leaving with it. An output read only by them needs no
    // pass-through, because after the bypass nothing reads it — which is the whole
    // difference between bypassing a node and bypassing the group it lives in.
    const leaving = new Set([slot.id, ...(alsoGoing || [])]);
    // Targets an ACTIVE slot also feeds. Wiring two alternatives at one port and bypassing
    // the one you are not using is the point of bypass here, so a target somebody else
    // still drives is not a consumer this node has to satisfy.
    const covered = new Set();
    (config.slots || []).forEach((s2) => {
      if (leaving.has(s2.id) || s2.bypassed) return;
      Object.values(s2.wires || {}).forEach((tg) => wireTargets(tg).filter(Boolean)
        .forEach((t) => covered.add(t)));
    });
    const consumed = new Set();
    Object.keys(slot.wires || {}).forEach((outName) => {
      const targets = wireTargets(slot.wires[outName]).filter(Boolean);
      // A wire onto a core port or a global output is always an outside consumer; a wire
      // onto another node counts only when that node is staying — and never when something
      // active feeds the same place, or when the input it lands on is optional.
      if (targets.some((t) => !covered.has(t) && !targetIsOptional(t)
                            && (!t.startsWith("node:") || !leaving.has(t.split(":")[1]))))
        consumed.add(outName);
    });
    config.slots.forEach((s2) => {
      if (leaving.has(s2.id) || s2.bypassed) return;
      Object.entries(s2.input_sources || {}).forEach(([inName, src]) => {
        const hit = _parseOutSource(src);
        if (!hit || hit.slotId !== slot.id) return;
        const ci = (specFor(s2)?.connection_inputs || []).find((c) => c.name === inName);
        if (ci && !ci.required) return;   // optional: the consumer runs without it
        consumed.add(hit.out);
      });
    });
    const out = [];
    (cand.outputs || []).forEach((o) => {
      if (!consumed.has(o.name)) return;             // feeds nothing: nothing to pass through
      const n = (cand.connection_inputs || []).filter((ci) => typeAccepts(ci.type, o.type)).length;
      if (n !== 1) {
        out.push(`${slotName(slot)}: ${o.name} (${o.type}) has `
          + `${n ? "more than one" : "no"} matching input to pass through`);
      }
    });
    return out;
  }

  async function setGroupBypassed(slots, on) {
    slots.forEach((s) => { if (on) s.bypassed = true; else delete s.bypassed; });
    await persist();
    render();
  }

  // A group is a VIEW, never a container: the nodes stay flat, every wire keeps its real
  // endpoints, and anything can still wire to anything. That is the whole difference from a
  // subgraph — there is no inside to navigate into, only a heading you can fold.
  // Whether a slot is a model loader. Loaders are pinned to the top of the shelf: picking
  // model files is the one thing every project has to do, and on a busy pipeline the
  // loaders were scattered among nodes nobody needs to touch.
  const LOADER_OUTPUT_TYPES = ["MODEL", "CLIP", "VAE", "CLIP_VISION"];
  function isLoaderSlot(slot) {
    const r = roles.find((x) => x.key === slot.role);
    if (r) return r.category === "Loaders";
    // An imported workflow has no roles — its nodes are all "custom" — so fall back to what
    // the node actually produces. Anything emitting a model, encoder or VAE is model-side
    // and belongs up here, including LoRA patchers.
    const cand = specFor(slot);
    return !!cand && (cand.outputs || []).some((o) => LOADER_OUTPUT_TYPES.includes(o.type));
  }

  function addCard(title, hint, onClick) {
    const add = el("button", "node-card node-card-add");
    add.type = "button";
    add.title = hint;
    const art = el("div", "node-card-art");
    art.append(el("span", "node-card-plus", "+"));
    const meta = el("div", "node-card-meta");
    meta.append(el("div", "node-card-title", title));
    art.append(meta);
    add.append(art);
    add.onclick = onClick;
    return add;
  }

  // Recipes whose role nothing fills yet. A project made before FunPack had its own
  // loaders — or one inherited from a global config — never gets seeded, so the offer has
  // to be reachable by hand too. Only MISSING roles are ever added: an existing loader is
  // someone's choice, and in guided mode a second producer on the same port would take it.
  function missingDefaultLoaders() {
    if (!pipelineLocked()) return [];
    return defaultSlots.filter((r) => !(config.slots || []).some((s) => s.role === r.role));
  }

  async function addMissingDefaultLoaders() {
    const missing = missingDefaultLoaders();
    if (!missing.length) return;
    for (const recipe of missing) {
      const id = (config.slots || []).some((s) => s.id === recipe.id)
        ? recipe.id + "_" + Math.random().toString(36).slice(2, 7) : recipe.id;
      const slot = { ...recipe, id, inputs: { ...recipe.inputs },
                     input_sources: { ...(recipe.input_sources || {}) } };
      graftRecipeSources(slot);
      config.slots.push(slot);
    }
    reconcileOpenPortWiring();
    await persistNow();
    await prewarmSpecs();
    render();
  }

  // A recipe's sources name the OTHER seeded slots by their fixed ids. Dropped into a
  // pipeline that was not seeded, those ids may not be there — and where they are, the
  // producer is already wired to the port this pass-through now takes over, which would
  // leave the port with two sources.
  function graftRecipeSources(slot) {
    const outTypes = new Set(((specFor(slot) || {}).outputs || []).map((o) => o.type));
    Object.entries(slot.input_sources || {}).forEach(([input, src]) => {
      const m = /^out:([^:]+):(.+)$/.exec(String(src));
      if (!m) return;
      const producer = (config.slots || []).find((s) => s.id === m[1]);
      if (!producer) { delete slot.input_sources[input]; return; }
      const out = ((specFor(producer) || {}).outputs || []).find((o) => o.name === m[2]);
      if (out && outTypes.has(out.type))
        (producer.wires = producer.wires || {})[m[2]] = [`node:${slot.id}:${input}`];
    });
  }

  // A role whose output more than one node needs, and the input name each of them uses.
  // The audio VAE is the case that bites: it decodes the sound AND encodes the empty audio
  // latent, so wiring it to one of the two leaves the pipeline unable to generate — and
  // with two VAE loaders present neither input can auto-resolve by type, so nothing filled
  // the gap and nothing said which node was still waiting.
  const ROLE_EXTRA_INPUTS = { audio_vae: ["audio_vae"], video_vae: ["vae"] };

  function feedMatchingInputs(slot) {
    const names = ROLE_EXTRA_INPUTS[slot.role];
    if (!names) return;
    const out = ((specFor(slot) || {}).outputs || [])[0];
    if (!out) return;
    (config.slots || []).forEach((other) => {
      if (other.id === slot.id) return;
      ((specFor(other) || {}).connection_inputs || []).forEach((ci) => {
        if (!names.includes(ci.name) || !typeAccepts(ci.type, out.type)) return;
        const cur = (other.input_sources || {})[ci.name];
        if (cur && cur !== "auto") return;
        _setInputSource(other, ci.name, `out:${slot.id}:${out.name}`);
      });
    });
  }

  function loaderSection(slots, v) {
    const sec = el("div", "node-group node-group-pinned");
    const head = el("div", "node-group-head");
    const title = el("div", "node-group-fold static");
    title.append(el("span", "node-group-name", "Loaders"));
    title.append(el("span", "node-group-count", String(slots.length)));
    const bad = slots.reduce(
      (n, s) => n + ((v.perSlot[s.id] || []).some((m) => m.level === "error") ? 1 : 0), 0);
    if (bad) title.append(el("span", "node-group-bad", `${bad} to fix`));
    head.append(title);
    const missing = missingDefaultLoaders();
    if (missing.length && !searchQ) {
      const use = el("button", "btn ghost tiny", `Use FunPack loaders (${missing.length})`);
      use.title = "Add FunPack's own loaders for the roles nothing fills yet, already wired "
        + "to the pipeline. Loaders you have set up are left exactly as they are.";
      use.onclick = addMissingDefaultLoaders;
      head.append(use);
    }
    sec.append(head);
    sec.append(el("div", "node-group-note", "Your model files. Pick them here — the rest of the pipeline wires itself."));
    const grid = _grid(slots, v);
    if (!searchQ)
      grid.append(addCard("Add loader", "Add a model loader to the pipeline",
        (e) => openRoleMenu(e.currentTarget, "Loaders")));
    sec.append(grid);
    return sec;
  }

  function nodeGrid(v) {
    const host = el("div", "node-groups");
    const visible = config.slots.filter(slotMatches);
    host.append(searchBar(visible.length, config.slots.length));

    const loaders = visible.filter(isLoaderSlot);
    const rest = visible.filter((s) => !isLoaderSlot(s));
    if (loaders.length || !searchQ) host.append(loaderSection(loaders, v));

    if (!searchQ) {                            // the add card is an action, not a result
      const addWrap = el("div", "node-grid");
      addWrap.append(addCard("New node", "Add a loader or custom node to the pipeline",
        (e) => openRoleMenu(e.currentTarget)));
      host.append(addWrap);
    } else if (!visible.length) {
      host.append(el("div", "req-empty", `Nothing matches "${searchQ}".`));
      return host;
    }

    const names = groupNames().filter((n) => rest.some((s) => (s.group || "").trim() === n));
    if (!names.length) {                       // nothing grouped: the plain grid, as before
      if (rest.length) host.append(_grid(rest, v));
      return host;
    }
    // A match inside a folded group has to be reachable, so search overrides the folds.
    const closed = searchQ ? new Set() : closedGroups();
    const sections = names.map((n) => [n, rest.filter((s) => (s.group || "").trim() === n)]);
    const loose = rest.filter((s) => !(s.group || "").trim());
    if (loose.length) sections.push(["Ungrouped", loose]);

    sections.forEach(([name, slots]) => {
      if (!slots.length) return;
      const sec = el("div", "node-group");
      const head = el("div", "node-group-head");
      const isClosed = closed.has(name);
      const fold = el("button", "node-group-fold");
      fold.type = "button";
      fold.append(el("span", "node-group-caret", isClosed ? "▸" : "▾"));
      fold.append(el("span", "node-group-name", name));
      fold.append(el("span", "node-group-count", String(slots.length)));
      // Errors only: a folded group must still say when something inside it is broken.
      const bad = slots.reduce(
        (n, s) => n + ((v.perSlot[s.id] || []).some((m) => m.level === "error") ? 1 : 0), 0);
      if (bad) fold.append(el("span", "node-group-bad", `${bad} to fix`));
      fold.onclick = () => { toggleGroup(name); render(); };
      head.append(fold);

      // Bypass the whole group. No new graph concept: it sets the per-node bypass every
      // card already has, so the pass-through rules and the reporting are the same ones.
      const allOff = slots.every((s) => s.bypassed);
      const ids = slots.map((s) => s.id);
      const blockers = allOff ? [] : slots.flatMap((s) => bypassBlockers(s, ids));
      const byp = el("button", "btn ghost tiny node-group-byp" + (allOff ? " on" : ""),
        allOff ? "bypassed" : "bypass");
      // Advisory, never a gate. What we can see here is only the explicit wiring; the
      // builder also auto-wires, so this predicate is neither sufficient nor necessary —
      // it would refuse bypasses that work and allow ones that don't. So warn with what we
      // do know and let the builder, which resolves the real graph, have the last word.
      const warn = blockers.length
        ? "Some nodes here may not be bypassable:\n· " + blockers.slice(0, 4).join("\n· ")
          + (blockers.length > 4 ? `\n· …and ${blockers.length - 4} more` : "")
        : "";
      byp.title = (allOff
        ? `Put all ${slots.length} nodes in "${name}" back in the graph.`
        : `Skip all ${slots.length} nodes in "${name}", passing their inputs straight through.`)
        + (warn ? "\n\n" + warn : "");
      if (warn) byp.classList.add("risky");
      byp.onclick = () => {
        if (warn && !confirm(`Bypass "${name}"?\n\n${warn}\n\n`
            + "Generation will say exactly which node and input it could not pass through.")) return;
        setGroupBypassed(slots, !allOff);
      };
      head.append(byp);
      sec.append(head);
      if (!isClosed) sec.append(_grid(slots, v));
      host.append(sec);
    });
    return host;
  }

  function paneContent(v) {
    const pane = el("div", "models-pane");
    if (view.startsWith("node:")) {
      const slot = slotById(view.slice(5));
      if (slot) { pane.append(nodePage(slot, v.perSlot[slot.id])); return pane; }
      view = "pipeline"; // node was removed — fall back
    }
    if (view === "links") { pane.append(linksView()); return pane; }
    if (config.workflow_import?.name) {
      const banner = el("div", "wf-import-banner");
      banner.textContent = `Imported workflow: ${config.workflow_import.name} (${config.workflow_import.node_count || config.slots.length} nodes) · built-in pipeline disabled`;
      pane.append(banner);
    }
    pane.append(familySection());
    // Above the node grid: what is missing is what the user came here to fix, and it used
    // to sit below a grid tall enough to push it off screen.
    pane.append(requirementsPanel());
    pane.append(nodeGrid(v));
    pane.append(coreSection());
    return pane;
  }

  function body() {
    const b = el("div", "models-shell");
    if (linkMode) b.append(linkModeBar());
    const v = validation();
    const cols = el("div", "models-cols");
    cols.append(sideBar(v), paneContent(v));
    b.append(cols);
    return b;
  }

  function render() {
    if (!container) return;
    // Preserve the content pane's scroll across the full re-render every edit triggers.
    const prevPane = container.querySelector(".models-pane");
    const scrollTop = prevPane ? prevPane.scrollTop : 0;
    const hadSearchFocus = document.activeElement
      && document.activeElement.dataset && document.activeElement.dataset.k === "models-search";
    clear(container);
    container.append(body());
    const pane = container.querySelector(".models-pane");
    if (pane) pane.scrollTop = scrollTop;
    // Typing in the search box re-renders on every keystroke, which would otherwise throw
    // the caret away after the first character.
    if (hadSearchFocus) {
      const box = container.querySelector('[data-k="models-search"]');
      if (box) { box.focus(); box.setSelectionRange(box.value.length, box.value.length); }
    }
  }

  async function prewarmSpecs() {
    // resolve the spec for every configured node (by class) so fields + wiring render
    for (const cls of new Set(config.slots.map((s) => s.node_class))) { try { await loadSpec(cls); } catch (_) {} }
  }

  async function refreshList() {
    await doFullRefresh();
    render();
  }

  async function loadAll() {
    await ensureRoles();
    const pid = window.Store?.get().project?.id;
    // Config FIRST: the ports and wiring rules depend on the project's model family, so
    // asking for them before we know it answered for whatever family was saved globally.
    try { config = await API.getModels(pid); } catch (_) { config = { slots: [] }; }
    try {
      const pp = await API.pipelinePorts(pid);
      ports = pp.ports || [];
      coreProducers = pp.core_producers || [];
      requirements = pp.requirements || [];
      wiringRules = pp.wiring || {};
      defaultSlots = pp.default_slots || [];
    } catch (_) { ports = []; coreProducers = []; requirements = []; wiringRules = {}; defaultSlots = []; }
    reconcileOpenPortWiring();
    try { coreNodes = (await API.coreGraph(window.Store?.get().project?.id)).nodes || []; } catch (_) { coreNodes = []; }
    await prewarmSpecs();
    // Backfill numeric bounds/step (and combo choices) onto controls exposed before this
    // metadata was captured, so opening Models once upgrades existing projects. persist()
    // dispatches funpack-models-changed → the store reloads and the inspector re-renders.
    let dirty = refreshExposedChoices();
    if (migrateAutogrowNames()) dirty = true;
    if (dirty) { try { await persist(); } catch (_) {} }
  }

  // A pinned button can ask for a node before the section has ever mounted, and the slots
  // only exist after loadAll(). Held here and applied once they do.
  let _openNodeOnMount = null;
  // The node currently being shown as a DESTINATION rather than as a stop inside Settings.
  // Set when a pinned button opens it; cleared as soon as the user navigates elsewhere,
  // because from then on they are browsing Settings normally.
  let _directNodeId = null;

  /** Leave a node page: back to the pipeline normally, but a node opened straight from a
   *  pinned button is somewhere the user went ON PURPOSE — dropping them into the Models
   *  list afterwards would undo the point of the shortcut. Finishing there closes Settings
   *  and returns them to the editor, saved or discarded alike. */
  function finishNodeEdit() {
    if (_directNodeId && view === "node:" + _directNodeId) {
      _directNodeId = null;
      endEdit();
      window.SettingsWindow.close();
      return;
    }
    _setView("pipeline");
  }

  function mount(body, ctx) {
    container = el("div", "models-mount");
    container.append(el("div", "pj-meta models-loading", "Loading models & pipeline…"));
    body.append(container);
    view = "pipeline";
    linkMode = false; linkSel = [];
    const imp = el("button", "btn ghost tiny", "⇪ Import workflow…");
    imp.title = "Import a ComfyUI workflow (API format) as the pipeline";
    imp.disabled = !window.Store?.get().project;
    imp.onclick = () => window.WorkflowImportWizard?.open();
    const refresh = el("button", "btn ghost tiny", "↻ Refresh model list");
    refresh.onclick = refreshList;
    const card = el("button", "btn ghost tiny", "🖼 Export settings…");
    card.title = "Render this pipeline as a PNG — loaders, LoRAs, typed-in node values, "
               + "and the host's torch / CUDA / attention";
    card.onclick = openSettingsCard;
    ctx.setActions([imp, refresh, card]);
    loadAll()
      .then(() => {
        if (!container || !container.isConnected) return;
        const pending = _openNodeOnMount; _openNodeOnMount = null;
        if (pending && slotById(pending)) {
          _setView("node:" + pending);
          _directNodeId = pending;   // after _setView, which clears it
          return;
        }
        render();
        if (pending) {
          // Landing on the pipeline instead, without saying why, would look like the
          // button simply did the wrong thing.
          alert("That node is not in this project's pipeline.\n\n"
                + "The pinned button still points at it — open the node you want and use "
                + "Pin to a button to repoint the slot.");
        }
      })
      .catch(() => {
        if (container && container.isConnected) {
          clear(container);
          container.append(el("div", "pj-meta models-loading", "Could not load models & pipeline."));
        }
      });
    return () => {
      container = null;
      // The section can be torn down mid-edit — Escape, the ✕, switching sections — and
      // loadAll() replaces `config` on the next mount anyway. Leaving the edit buffer
      // behind would carry deferDirty and a baseline for an object that no longer exists
      // into that mount: a spurious "unsaved changes" prompt, and a Cancel that splices
      // stale slots into freshly loaded ones.
      endEdit();
      _directNodeId = null;
      _openNodeOnMount = null;
      document.querySelectorAll(".mn-role-pop, .ns-overlay, .sc-overlay").forEach((n) => n.remove());
    };
  }

  window.SettingsWindow.register({
    id: "models", group: "Generation", order: 2, title: "Models & Pipeline", flush: true,
    subtitle: "Loaders and custom nodes wired into the fixed FunPack pipeline.",
    keywords: "models loaders unet vae clip lora nodes pipeline wiring workflow import "
      + "bypass disable core full control links",
    iconBg: "linear-gradient(180deg,#b18cff,#7a4fd0)",
    icon: '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="#fff" stroke-width="1.4" stroke-linejoin="round"><path d="M8 1.8 14 5v6l-6 3.2L2 11V5l6-3.2z"/><path d="M2 5l6 3 6-3M8 8v6.2"/></svg>',
    mount,
    // Pinning while a node is open should pin THAT node, not the section it lives in —
    // the node page is the thing that takes several clicks to reach.
    pinTarget: () => {
      if (!view.startsWith("node:")) return null;
      const slot = slotById(view.slice(5));
      if (!slot) return null;
      return { kind: "node", id: slot.id, label: slotDisplayLabel(slot) };
    },
  });

  window.ModelsModal = {
    open: () => window.SettingsWindow.open("models"),
    // Open Models & Pipeline showing ONE node. Used by pinned buttons; safe to call
    // whether or not the section has been mounted before.
    openNode: (slotId) => {
      if (!slotId) return;
      // One route, whatever is on screen: open() mounts (or REMOUNTS) the section, and the
      // request is armed AFTER that call — a remount tears the old section down, which
      // clears pending state, so arming first would lose it. mount()'s load is async, so
      // this still lands before the node view is chosen. The remount costs one refetch and
      // buys freshness, which is what a shortcut into a quick edit wants anyway.
      window.SettingsWindow.open("models");
      _openNodeOnMount = slotId;
    },
    refresh: async () => {
      await ensureRoles().catch(() => {});
      await doFullRefresh().catch(() => {});
      if (container) render();
    },
  };
})();
