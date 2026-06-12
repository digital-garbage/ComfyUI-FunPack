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
  let config = { slots: [] };     // {slots:[{id,role,role_label,node_class,inputs:{},wires:{},input_sources:{}}]}
  let coreNodes = [];             // built-in pipeline description (from /core-graph)
  let coreOpen = false;           // built-in pipeline panel expanded?
  let coreProducers = [];          // [{id,type,label}]
  let requirements = [];           // [{id,type,label,required,role_hint,hint}]
  let wiringRules = {};            // guided wiring rules from /pipeline-ports
  let overlay = null;
  const expanded = new Set();     // slot ids currently expanded (collapsed by default)
  let linkMode = false;           // selecting inputs to bind into one shared control
  let linkSel = [];               // [{slotId, input, kind, choices, label}]
  const LIVE_PREVIEW_PORT = "port:FunPackLTXAVSceneChainSampler.preview_vae";

  function _clearLivePreviewWires(exceptSlotId, exceptOut) {
    for (const s of config.slots) {
      s.wires = s.wires || {};
      for (const outName of Object.keys(s.wires)) {
        if (s.id === exceptSlotId && outName === exceptOut) continue;
        const kept = wireTargets(s.wires[outName]).filter((t) => t !== LIVE_PREVIEW_PORT);
        if (kept.length) s.wires[outName] = kept;
        else delete s.wires[outName];
      }
    }
  }

  function _syncLivePreviewConfig(slotId, outName, wired) {
    config.live_preview = {
      enabled: false,
      slot_id: null,
      vae_output: "VAE",
      ...(config.live_preview || {}),
    };
    if (wired && slotId && outName) {
      config.live_preview.enabled = true;
      config.live_preview.slot_id = slotId;
      config.live_preview.vae_output = outName;
    } else if (!wired && config.live_preview.slot_id === slotId
        && (config.live_preview.vae_output || "VAE") === outName) {
      config.live_preview.enabled = false;
      config.live_preview.slot_id = null;
    }
  }

  function _findLivePreviewWire() {
    for (const s of config.slots) {
      for (const [outName, raw] of Object.entries(s.wires || {})) {
        if (wireTargets(raw).includes(LIVE_PREVIEW_PORT)) {
          return { slotId: s.id, outName };
        }
      }
    }
    return null;
  }

  function _scanLivePreviewFromWires() {
    const hit = _findLivePreviewWire();
    if (hit) _syncLivePreviewConfig(hit.slotId, hit.outName, true);
  }

  function _afterLivePreviewWireChange(slot, outName) {
    const targets = wireTargets(slot.wires?.[outName]);
    if (targets.includes(LIVE_PREVIEW_PORT)) {
      _clearLivePreviewWires(slot.id, outName);
      if (!wireTargets(slot.wires[outName]).includes(LIVE_PREVIEW_PORT)) {
        _addWire(slot.id, outName, LIVE_PREVIEW_PORT);
      }
      _syncLivePreviewConfig(slot.id, outName, true);
    } else if (config.live_preview?.slot_id === slot.id
        && (config.live_preview?.vae_output || "VAE") === outName) {
      _syncLivePreviewConfig(slot.id, outName, false);
    }
  }

  function slotDisplayLabel(slot) {
    const cls = slot.node_class || "?";
    const name = (slot.label || "").trim();
    if (name && name !== cls) return `${name} — ${cls}`;
    return cls;
  }

  function roleLabel(key) { const r = roles.find((x) => x.key === key); return r ? r.label : (key === "custom" ? "Node" : (key || "?")); }
  function slotName(slot) { return slot.label || slot.node_class; }
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

  function _guidedPortAllowlist(role, out) {
    const rules = wiringRules.role_targets?.[role] || [];
    const chainPorts = wiringRules.type_chain_terminals?.[out.type] || [];
    let allowedPorts = [...rules
      .filter((r) => r.type === out.type && (r.output_name == null || r.output_name === out.name))
      .map((r) => "port:" + r.port),
    ...chainPorts.map((p) => "port:" + p),
    ].filter((p, i, a) => a.indexOf(p) === i);
    if (role === "custom" && !allowedPorts.length) {
      for (const rlist of Object.values(wiringRules.role_targets || {})) {
        for (const r of rlist) {
          if (r.type !== out.type) continue;
          if (r.output_name != null && r.output_name !== out.name) continue;
          const v = "port:" + r.port;
          if (!allowedPorts.includes(v)) allowedPorts.push(v);
        }
      }
      for (const p of chainPorts) {
        const v = "port:" + p;
        if (!allowedPorts.includes(v)) allowedPorts.push(v);
      }
    }
    if (out.type === "VAE") {
      for (const p of wiringRules.optional_vae_ports || []) {
        const v = "port:" + p;
        if (!allowedPorts.includes(v)) allowedPorts.push(v);
      }
    }
    return allowedPorts;
  }

  function allowedDestinations(slot, out) {
    const all = destinations(slot, out.type);
    if (!pipelineLocked()) return all;
    const role = slot.role || "custom";
    const hidden = new Set(wiringRules.guided_hidden_ports || []);
    const allowedPorts = _guidedPortAllowlist(role, out);
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
      _clearPortWires(portId, parsed.slotId, parsed.out);
      _addWire(parsed.slotId, parsed.out, "port:" + portId);
    }
    _scanLivePreviewFromWires();
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

  function allowedSources(slot, ci) {
    const all = sources(slot, ci.type);
    if (!pipelineLocked()) return all;
    if (slot.role === "image_processing" && ci.name === "image" && ci.type === "IMAGE") {
      return all.filter((s) => !s.value || s.value === "timeline" || s.value.startsWith("out:"));
    }
    return all.filter((s) => !s.value || s.value.startsWith("out:") || s.value === "timeline");
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

  async function persist() {
    // Keep the SAME `config` object — every wire/widget handler closes over it, so
    // reassigning it to the server's returned object detaches those closures and the
    // next edit wouldn't persist until the node is re-rendered. Save in place instead.
    try { await API.saveModels(window.Store?.get().project?.id, config); window.dispatchEvent(new Event("funpack-models-changed")); }
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
      } else if (v == null || v === "") {
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
    if (slotIsActive(slot)) (spec.connection_inputs || []).forEach((ci) => {
      if (!ci.required) return;
      if (ci.type === "IMAGE") return;  // a scene/timeline image can always feed an IMAGE input
      const src = (slot.input_sources || {})[ci.name];
      if (src && src !== "auto") return;  // explicitly sourced (incl. "timeline")
      const incoming = (config.slots || []).some((s2) =>
        s2.id !== slot.id && Object.values(s2.wires || {}).some((tg) => wireTargets(tg).includes(`node:${slot.id}:${ci.name}`)));
      if (incoming) return;
      const prod = sources(slot, ci.type).filter((o) => o.value && o.value !== "timeline").length;
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
    const roleText = slot.role_label || (role ? role.label : slot.role) || "?";
    const roleSpan = el("span", "slot-role", roleText);
    roleSpan.title = "Click ✎ to rename this label";
    head.append(roleSpan);
    head.append(el("span", "slot-node", slotDisplayLabel(slot)));
    const ren = el("button", "ic-btn", "✎"); ren.title = "Rename role label";
    ren.onclick = async (e) => {
      e.stopPropagation();
      const cur = slot.role_label || (role ? role.label : slot.role) || "";
      const n = prompt("Role label (shown in pickers and headers):", cur);
      if (n != null) { slot.role_label = n.trim() || undefined; await persist(); render(); }
    };
    head.append(ren);
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
        const lk = linkOf(slot.id, spec.name);
        const f = widgetField(spec, slot.inputs[spec.name], async (v) => {
          slot.inputs[spec.name] = v;
          const l2 = linkOf(slot.id, spec.name); if (l2) applyLinkValue(l2, v);  // keep group in sync
          await persist();
        });
        f.classList.add("with-eye");
        if (lk) {
          f.classList.add("linked");
          const ctrl = f.querySelector("input,select"); if (ctrl) ctrl.disabled = true;
          f.append(el("span", "link-tag", "🔗 " + lk.name));
        } else if (linkMode) {
          const chk = el("button", "eye-btn link-pick" + (linkSelHas(slot.id, spec.name) ? " on" : ""), "+");
          chk.type = "button"; chk.title = "Add to link selection";
          chk.onclick = (e) => {
            e.preventDefault(); e.stopPropagation();
            if (linkSelHas(slot.id, spec.name)) linkSel = linkSel.filter((s) => !(s.slotId === slot.id && s.input === spec.name));
            else linkSel.push({ slotId: slot.id, input: spec.name, kind: spec.kind, choices: spec.choices, label: `${slotName(slot)} · ${spec.name}` });
            render();
          };
          f.append(chk);
        } else {
          f.append(eyeButton(slot, spec));
        }
        grid.append(f);
      });
      card.append(grid);
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

    // Input sources: explicitly choose where each connection input comes from.
    if (cand && (cand.connection_inputs || []).length) {
      slot.input_sources = slot.input_sources || {};
      const sbox = el("div", "wire-box");
      sbox.append(el("div", "wire-title", "Input sources"));
      cand.connection_inputs.forEach((ci) => {
        const row = el("div", "wire-row");
        row.append(el("span", "wire-out", `${ci.name} (${ci.type})`));
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
        sbox.append(row);
      });
      card.append(sbox);
    }

    return card;
  }

  const _KIND2T = { int: "INT", float: "FLOAT", string: "STRING", boolean: "BOOLEAN" };
  function destinations(slot, type) {
    const out = [{ value: "", label: "— unwired —" }];
    // Global editor outputs: wire your final video/audio producer here and the editor shows
    // it (works with or without the built-in pipeline). IMAGE -> video, AUDIO -> audio.
    if (type === "IMAGE") out.push({ value: "global:video", label: "🌐 Global video output (shown in editor)" });
    if (type === "AUDIO") out.push({ value: "global:audio", label: "🌐 Global audio output (shown in editor)" });
    ports.filter((p) => p.type === type).forEach((p) => out.push({ value: "port:" + p.id, label: p.label }));
    config.slots.filter((s) => s.id !== slot.id).forEach((s2) => {
      const c2 = specFor(s2);
      (c2?.connection_inputs || []).filter((ci) => ci.type === type).forEach((ci) =>
        out.push({ value: `node:${s2.id}:${ci.name}`, label: `${slotFullLabel(s2)} · ${ci.name}` }));
      // Widget inputs can also receive a connection (ComfyUI converts a widget to an
      // input when wired) — e.g. EmptyLatentVideo.width/height. Offer them as targets.
      (c2?.inputs || []).filter((w) => _KIND2T[w.kind] === type).forEach((w) =>
        out.push({ value: `node:${s2.id}:${w.name}`, label: `${slotFullLabel(s2)} · ${w.name} (widget)` }));
    });
    return out;
  }

  // Available sources for a slot connection input of a given type.
  // Source IDs: "" = auto, "out:<slotId>:<outName>", "core:<coreId>:<outIdx>", "timeline" (IMAGE only).
  function sources(slot, type) {
    const out = [{ value: "", label: "(auto-wire)" }];
    if (type === "IMAGE") out.push({ value: "timeline", label: "Timeline (scene image)" });
    coreProducers.filter((p) => p.type === type).forEach((p) =>
      out.push({ value: p.id, label: p.label }));
    config.slots.filter((s) => s.id !== slot.id).forEach((s2) => {
      const c2 = specFor(s2);
      (c2?.outputs || []).filter((o) => o.type === type).forEach((o) =>
        out.push({ value: `out:${s2.id}:${o.name}`, label: `${slotFullLabel(s2)} → ${o.name}` }));
    });
    return out;
  }

  // Normalize wires[outName] to always be an array (supports legacy string format).
  function wireTargets(raw) {
    if (!raw) return [];
    return Array.isArray(raw) ? raw : [raw];
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
        sel.onchange = async () => {
          _setWireTarget(slot, out.name, targets[i], sel.value);
          targets[i] = sel.value;
          slot.wires[out.name] = targets;
          _afterLivePreviewWireChange(slot, out.name);
          reconcileOpenPortWiring();
          await persist();
          render();
        };
        const rm = el("button", "btn ghost tiny wire-rm", "×");
        rm.title = "Remove this wire";
        rm.onclick = async () => {
          _setWireTarget(slot, out.name, targets[i], "");
          targets.splice(i, 1);
          slot.wires[out.name] = targets;
          _afterLivePreviewWireChange(slot, out.name);
          reconcileOpenPortWiring();
          await persist();
          render();
        };
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
      const extras = defaultExtrasForRole(role, curCand);
      config.slots.push({
        id: uid(), role, node_class: curCand.class,
        inputs: preview._draft || defaultsFor(curCand),
        wires: extras.wires || {},
        input_sources: extras.input_sources || {},
      });
      await persist();
      typeSel.value = ""; nodeSel.innerHTML = ""; nodeSel.append(new Option("Node…", "")); nodeSel.disabled = true;
      search.style.display = "none"; search.value = "";
      addBtn.disabled = true; curCand = null; curList = []; clear(preview); render();
    };

    row.append(typeSel); row.append(search); row.append(nodeSel); row.append(addBtn);
    box.append(row); box.append(preview);
    return box;
  }

  // Does any configured slot produce the given type (optionally filtered by role)?
  function slotProducesType(type, roleHint) {
    return config.slots.some((s) => {
      if (roleHint && s.role !== roleHint) return false;
      const spec = specFor(s);
      return (spec?.outputs || []).some((o) => o.type === type);
    });
  }

  // For VAE: need two separate producers (video_vae + audio_vae roles).
  // For others: at least one producer of the type (regardless of role).
  function requirementSatisfied(req) {
    if (req.role_hint) return slotProducesType(req.type, req.role_hint);
    return slotProducesType(req.type, null);
  }

  function requirementsPanel() {
    const sec = el("div", "req-panel");
    if (config.disable_core) {
      sec.append(el("div", "req-title", "Custom workflow"));
      sec.append(el("div", "req-empty",
        "Built-in FunPack pipeline is off. Wire your workflow nodes below; FunPack loader requirements do not apply."));
      return sec;
    }
    if (pipelineLocked()) {
      sec.append(el("div", "req-hint guided-hint",
        "Guided wiring: LATENT → Studio · latent (core forwards to Concat · video_latent). "
        + "IMAGE → Studio · source_image (Input Image Processing defaults to Timeline scene image). "
        + "MODEL/CLIP may chain through LoRA. Audio latent → Concat · audio_latent only. "
        + "Enable Full control for manual rewiring."));
    }
    sec.append(el("div", "req-title", "Pipeline requirements"));

    if (!requirements.length) {
      sec.append(el("div", "req-empty", "Requirements not loaded — open Models to refresh."));
      return sec;
    }

    const required = requirements.filter((r) => r.required);
    const optional = requirements.filter((r) => !r.required);
    const allOk = required.every(requirementSatisfied);

    required.forEach((req) => {
      const ok = requirementSatisfied(req);
      const row = el("div", "req-row" + (ok ? " ok" : " missing"));
      row.append(el("span", "req-dot", ok ? "✓" : "✕"));
      const lbl = el("span", "req-label", req.label);
      const badge = el("span", "req-type", req.type);
      row.append(lbl); row.append(badge);
      if (!ok) {
        const hint = el("div", "req-hint", req.hint);
        row.append(hint);
        if (req.role_hint) {
          const addBtn = el("button", "btn ghost tiny req-add", "+ Add");
          addBtn.onclick = () => {
            // pre-scroll the composer to the right role and expand it
            const typeSelEl = document.querySelector(".composer select");
            if (typeSelEl) { typeSelEl.value = req.role_hint; typeSelEl.dispatchEvent(new Event("change")); }
          };
          row.append(addBtn);
        }
      }
      sec.append(row);
    });

    if (optional.length) {
      const optHead = el("div", "req-opt-head", "Optional");
      sec.append(optHead);
      optional.forEach((req) => {
        const ok = requirementSatisfied(req);
        const row = el("div", "req-row opt" + (ok ? " ok" : ""));
        row.append(el("span", "req-dot", ok ? "✓" : "·"));
        row.append(el("span", "req-label", req.label));
        row.append(el("span", "req-type", req.type));
        sec.append(row);
      });
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

    return sec;
  }

  function summaryBanner(v) {
    // Kept for any callers — but requirementsPanel() is now shown instead.
    return null;
  }

  // ── linked-inputs section ──────────────────────────────────────────────────────
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
    [["", "Manual value"], ["frame_rate", "Project · FPS"], ["num_frames_per_scene", "Project · Frames"],
     ["width", "Project · Width"], ["height", "Project · Height"]].forEach(([v, label]) => {
      const o = new Option(label, v); if ((link.source === "editor" ? link.editor_key : "") === v) o.selected = true; srcSel.append(o);
    });
    srcSel.onchange = async () => {
      if (srcSel.value) { link.source = "editor"; link.editor_key = srcSel.value; }
      else { link.source = "manual"; delete link.editor_key; }
      await persist(); render();
    };
    srcRow.append(srcSel);
    card.append(srcRow);

    if (link.source === "editor") {
      card.append(el("div", "link-bound", `Value comes from the project setting at generate.`));
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

  function linksSection() {
    const sec = el("div", "links-section");
    const head = el("div", "links-head");
    head.append(el("div", "composer-title", "Linked inputs"));
    const right = el("div", "links-head-right");
    if (!linkMode) {
      const b = el("button", "btn ghost tiny", "🔗 Link inputs");
      b.onclick = () => { linkMode = true; linkSel = []; config.slots.forEach((s) => expanded.add(s.id)); render(); };
      right.append(b);
    } else {
      const save = el("button", "btn primary tiny", `Save link (${linkSel.length})`);
      save.disabled = linkSel.length < 1;
      save.onclick = async () => {
        if (linkSel.length < 1) return;
        const def = "size " + ((config.links || []).length + 1);
        const nm = prompt("Link name:", def); if (nm == null) return;
        const first = linkSel[0]; const s0 = slotById(first.slotId);
        const val = s0 ? (s0.inputs || {})[first.input] : undefined;
        const link = { id: uid(), name: nm.trim() || def, kind: first.kind,
          choices: first.kind === "combo" ? (first.choices || []) : undefined, value: val,
          members: linkSel.map((s) => ({ slotId: s.slotId, input: s.input })), exposed: false };
        ensureLinks().push(link);
        applyLinkValue(link, val);
        linkMode = false; linkSel = []; await persist(); render();
      };
      const cancel = el("button", "btn ghost tiny", "Cancel");
      cancel.onclick = () => { linkMode = false; linkSel = []; render(); };
      right.append(save); right.append(cancel);
    }
    head.append(right);
    sec.append(head);
    if (linkMode) sec.append(el("div", "links-hint", "Click ＋ next to any input below to add it to the link, then Save. Pick a single input to tie it to a project value (e.g. FPS), or several to drive them together."));
    (config.links || []).forEach((l) => sec.append(linkCard(l)));
    if (!linkMode && !(config.links || []).length)
      sec.append(el("div", "links-empty", "No links yet. Use 'Link inputs' to bind a node input to a project value (e.g. tie a loader's FPS to Project · FPS), or to drive several inputs from one control."));
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
      if (cid === "sampler" && inp === "preview_vae") {
        _clearLivePreviewWires(parsed.slotId, parsed.out);
        _syncLivePreviewConfig(parsed.slotId, parsed.out, true);
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
      if (cid === "sampler" && inp === "preview_vae") {
        _clearLivePreviewWires(null, null);
        config.live_preview = { ...(config.live_preview || {}), enabled: false, slot_id: null };
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
    const outs = (n.outputs || []).filter((o) => (o.to || []).length);
    if (outs.length) {
      const box = el("div", "wire-box");
      box.append(el("div", "wire-title", "Outputs → destinations"));
      outs.forEach((o) => {
        const row = el("div", "wire-row");
        row.append(el("span", "wire-out", `${o.name} (${o.type})`));
        row.append(el("span", "wire-arrow", "→"));
        row.append(el("span", "wire-dest", o.to.join(", ")));
        box.append(row);
      });
      card.append(box);
    }
    return card;
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
      const hint = pipelineLocked()
        ? "Fixed FunPack path (Studio → Conditioning → Chain Sampler → decode). "
          + "Wire video LATENT to Studio · latent (not Concat · video_latent — that link is internal). "
          + "Wire IMAGE via Input Image Processing → Studio · source_image (Timeline default). "
          + "Wire a fast VAE (e.g. taehv) to Chain Sampler · live preview for per-scene previews. "
          + "MODEL/CLIP may chain through patchers. Enable Full control to override core links."
        : "The fixed FunPack nodes and their wiring. Each input defaults to its built-in source — pick another to re-wire it.";
      sec.append(el("div", "links-hint", hint));
      coreNodes.forEach((n) => sec.append(coreCard(n)));
    }
    return sec;
  }

  function body() {
    const b = el("div", "models-body");
    if (config.workflow_import?.name) {
      const banner = el("div", "wf-import-banner");
      banner.textContent = `Imported workflow: ${config.workflow_import.name} (${config.workflow_import.node_count || config.slots.length} nodes) · built-in pipeline disabled`;
      b.append(banner);
    }
    b.append(requirementsPanel());
    b.append(coreSection());
    b.append(composer());
    b.append(linksSection());
    const v = validation();
    const list = el("div", "slot-list");
    if (!config.slots.length) list.append(el("div", "empty-stage", "No models configured yet — use the form above to add loaders."));
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
    try { coreNodes = (await API.coreGraph(window.Store?.get().project?.id)).nodes || []; } catch (_) {}
    await prewarmSpecs();
    render();
  }

  async function open() {
    if (overlay) return;  // already open — don't stack a second modal (orphans handlers)
    await ensureRoles();
    try {
      const pp = await API.pipelinePorts();
      ports = pp.ports || [];
      coreProducers = pp.core_producers || [];
      requirements = pp.requirements || [];
      wiringRules = pp.wiring || {};
    } catch (_) { ports = []; coreProducers = []; requirements = []; wiringRules = {}; }
    try { config = await API.getModels(window.Store?.get().project?.id); } catch (_) { config = { slots: [] }; }
    reconcileOpenPortWiring();
    try { coreNodes = (await API.coreGraph(window.Store?.get().project?.id)).nodes || []; } catch (_) { coreNodes = []; }
    await prewarmSpecs();

    overlay = el("div", "modal-overlay");
    const modal = el("div", "modal");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", "Models & Pipeline Nodes"));
    const refresh = el("button", "btn ghost", "↻ Refresh model list"); refresh.onclick = refreshList;
    const closeModal = () => { if (overlay) overlay.remove(); overlay = null; };
    const close = el("button", "btn ghost", "✕"); close.onclick = closeModal;
    const heRight = el("div", "modal-head-right"); heRight.append(refresh); heRight.append(close);
    head.append(heRight);
    modal.append(head);
    modal.append(el("div", "modal-content"));
    overlay.append(modal);
    overlay.addEventListener("click", (e) => { if (overlay && e.target === overlay) closeModal(); });
    document.body.append(overlay);
    render();
  }

  window.ModelsModal = { open, refresh: async () => { await ensureRoles().catch(() => {}); await (window.MovieEditorAPI.refreshModels().catch(() => {})); } };
})();
