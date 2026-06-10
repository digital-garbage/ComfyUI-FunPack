// Engine settings modal: Studio, Chain Sampler, continuity (moved out of Project inspector).
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;

  let overlay = null;
  let unsub = null;
  let _editing = false;
  let _onKey = null;

  const FOLD_KEY = "funpack_engine_fold_";
  function foldOpen(id, def) {
    const v = localStorage.getItem(FOLD_KEY + id);
    return v === null ? def : v === "1";
  }
  function setFoldOpen(id, open) { localStorage.setItem(FOLD_KEY + id, open ? "1" : "0"); }

  function field(labelText, control) {
    const l = el("label", "field"); l.append(el("span", null, labelText)); l.append(control); return l;
  }

  function collapsibleSection(parent, id, title, defaultOpen) {
    const det = el("details", "insp-fold");
    det.open = foldOpen(id, defaultOpen);
    det.addEventListener("toggle", () => setFoldOpen(id, det.open));
    det.append(el("summary", "insp-fold-sum", title));
    const inner = el("div", "insp-fold-body");
    det.append(inner);
    parent.append(det);
    return inner;
  }

  function engineCard(parent, id, title, defaultOpen) {
    const card = el("div", "insp-engine");
    const head = el("button", "insp-engine-head");
    head.type = "button";
    let open = foldOpen(id, defaultOpen);
    const chev = el("span", "insp-engine-chev", open ? "▾" : "▸");
    const ttl = el("span", "insp-engine-title", title);
    const badge = el("span", "insp-engine-badge");
    badge.hidden = true;
    head.append(chev, ttl, badge);
    const cardBody = el("div", "insp-engine-body");
    cardBody.hidden = !open;
    head.onclick = () => {
      open = !open;
      cardBody.hidden = !open;
      chev.textContent = open ? "▾" : "▸";
      setFoldOpen(id, open);
    };
    card.append(head, cardBody);
    parent.append(card);
    return { body: cardBody, setBadge: (n) => { badge.textContent = n > 0 ? `${n} changed` : ""; badge.hidden = n <= 0; } };
  }

  function slotLabelFor(st, slotId) {
    if (!slotId || slotId === "funpack") return null;
    const slot = (st.models?.slots || []).find((s) => s.id === slotId);
    return slot ? (slot.label || slot.node_class || slot.id) : slotId;
  }

  function activeSceneCount(p) {
    return (p.scenes || []).filter((s) => !s.excluded).length;
  }

  const STUDIO_REFINER_ESSENTIALS = [
    { name: "vision_conditioning", label: "Vision conditioning", default: true },
    { name: "reference_injection", label: "Reference injection", default: false },
    { name: "prompt_repair", label: "Prompt repair", default: false },
  ];
  const STUDIO_REFINER_ADVANCED = [
    { name: "value_guidance", label: "Value guidance", kind: "bool", default: true },
    { name: "steer_mode", label: "Steer mode", kind: "combo", choices: ["relative", "absolute", "both"], default: "relative" },
    { name: "absolute_strength", label: "Absolute strength", kind: "float", default: 0.6, min: 0, max: 1, step: 0.05,
      dependsOn: "steer_mode", dependsVals: ["absolute", "both"] },
    { name: "temporal_style", label: "Temporal style", kind: "combo",
      choices: ["natural", "auto", "accelerate", "decelerate", "loop", "freeze", "pulse"], default: "natural" },
    { name: "split_transition_placement", label: "Transition placement", kind: "combo",
      choices: ["start", "end", "silent"], default: "start" },
  ];

  function parseStudioSettings(p) {
    const si = p.studio_inputs || {};
    let cur = {};
    try { cur = JSON.parse(si.studio_settings || "{}"); } catch (_) {}
    const rf = (cur.refiner && typeof cur.refiner === "object") ? cur.refiner : {};
    return { si, cur, rf };
  }

  function countStudioChanges(p) {
    const { rf } = parseStudioSettings(p);
    let n = 0;
    [...STUDIO_REFINER_ESSENTIALS, ...STUDIO_REFINER_ADVANCED].forEach((f) => {
      const cur = rf[f.name] != null ? rf[f.name] : f.default;
      if (cur !== f.default) n++;
    });
    return n;
  }

  function persistStudioRefiner(patch, now) {
    const { cur } = parseStudioSettings(S.get().project);
    const rf = (cur.refiner && typeof cur.refiner === "object") ? cur.refiner : {};
    const next = JSON.stringify({ ...cur, refiner: { ...rf, ...patch } });
    if (now) S.setStudioInputNow("studio_settings", next);
    else S.setStudioInput("studio_settings", next);
  }

  function renderStudioRefinerBool(parent, rf, f) {
    const cur = rf[f.name] != null ? rf[f.name] : f.default;
    const ctrl = el("input"); ctrl.type = "checkbox"; ctrl.checked = !!cur; ctrl.style.width = "auto";
    ctrl.dataset.k = "rf-" + f.name;
    ctrl.onchange = () => persistStudioRefiner({ [f.name]: ctrl.checked }, true);
    parent.append(field(f.label, ctrl));
  }

  function renderStudioRefinerField(parent, rf, f) {
    if (f.dependsOn) {
      const depVal = rf[f.dependsOn] != null ? rf[f.dependsOn] : STUDIO_REFINER_ADVANCED.find((x) => x.name === f.dependsOn)?.default;
      if (!(f.dependsVals || []).includes(depVal)) return;
    }
    const val = rf[f.name] != null ? rf[f.name] : f.default;
    let ctrl;
    if (f.kind === "bool") {
      ctrl = el("input"); ctrl.type = "checkbox"; ctrl.checked = !!val; ctrl.style.width = "auto";
      ctrl.dataset.k = "rf-" + f.name;
      ctrl.onchange = () => persistStudioRefiner({ [f.name]: ctrl.checked }, true);
    } else if (f.kind === "combo") {
      ctrl = el("select"); ctrl.dataset.k = "rf-" + f.name;
      (f.choices || []).forEach((c) => { const o = new Option(c, c); if (c === val) o.selected = true; ctrl.append(o); });
      ctrl.onchange = () => persistStudioRefiner({ [f.name]: ctrl.value }, true);
    } else {
      ctrl = el("input"); ctrl.type = "number";
      if (f.step != null) ctrl.step = String(f.step);
      if (f.min != null) ctrl.min = String(f.min);
      if (f.max != null) ctrl.max = String(f.max);
      ctrl.value = val; ctrl.dataset.k = "rf-" + f.name;
      ctrl.oninput = () => persistStudioRefiner({ [f.name]: parseFloat(ctrl.value || "0") }, false);
    }
    parent.append(field(f.label, ctrl));
  }

  function renderStudioCard(parent, st) {
    const p = st.project;
    const { cur: curSettings, rf } = parseStudioSettings(p);
    const samplers = curSettings.samplers || null;
    const card = engineCard(parent, "studio_card", "FunPack Studio", true);
    card.setBadge(countStudioChanges(p));

    const ess = collapsibleSection(card.body, "studio_ess", "Essentials", true);
    STUDIO_REFINER_ESSENTIALS.forEach((f) => renderStudioRefinerBool(ess, rf, f));

    const refSec = collapsibleSection(card.body, "studio_ref", "Refinement", false);
    STUDIO_REFINER_ADVANCED.forEach((f) => renderStudioRefinerField(refSec, rf, f));

    const sampSec = collapsibleSection(card.body, "studio_samp", "Sampler algorithm", false);
    function persistSamplers(updatedSamplers, quiet) {
      const { cur } = parseStudioSettings(S.get().project);
      const next = JSON.stringify({ ...cur, samplers: updatedSamplers });
      if (quiet) S.setStudioInput("studio_settings", next);
      else S.setStudioInputNow("studio_settings", next);
    }
    try {
      window.SamplerPanel.render(sampSec, samplers,
        (s) => persistSamplers(s, true),
        (s) => persistSamplers(s, false));
    } catch (e) {
      const err = el("div", "insp-hint"); err.style.color = "var(--err,#e55)";
      err.textContent = "Studio sampler panel failed to render: " + e.message;
      sampSec.append(err);
    }

    card.body.append(el("div", "insp-hint",
      "Scene text and transitions come from the timeline. Advisor, LoRA, batch training, and adjustments remain in the ComfyUI Studio popup on the graph."));
  }

  const SAMPLER_KNOBS = [
    { name: "frame_overlap",         label: "Frame overlap",         kind: "int",   default: 16,    min: 0, max: 512, step: 8 },
    { name: "transition_duration",   label: "Transition duration",   kind: "int",   default: 16,    min: 0, max: 128, step: 2 },
    { name: "use_same_seed",         label: "Same seed per scene",   kind: "bool",  default: false },
    { name: "carry_i2v_guides",      label: "Carry i2v guides",      kind: "bool",  default: false, lockMulti: true },
    { name: "cfg",                   label: "CFG",                   kind: "float", default: 1.0,   min: 0, max: 20,  step: 0.1 },
    { name: "embed_guidance",        label: "Embed guidance",        kind: "bool",  default: false },
    { name: "embed_guidance_source", label: "Embed mode",            kind: "combo", choices: ["relative", "absolute"], default: "relative", dependsOn: "embed_guidance" },
    { name: "embed_guidance_strength", label: "Embed strength",      kind: "float", default: 0.02,  min: 0.005, max: 0.1, step: 0.005, dependsOn: "embed_guidance" },
    { name: "decode_noise_scale",    label: "Decode noise scale",    kind: "float", default: 0.0,   min: 0, max: 1,   step: 0.01 },
    { name: "decode_timestep",       label: "Decode timestep",       kind: "float", default: 0.05,  min: 0, max: 1,   step: 0.01 },
    { name: "decode_tile_size",      label: "Decode tile size",      kind: "int",   default: 0,     min: 0, max: 4096, step: 64 },
    { name: "mid_scene_guide",       label: "Mid-scene guide",       kind: "bool",  default: false },
    { name: "mid_scene_guide_strength", label: "Guide strength",   kind: "float", default: 0.25,  min: 0.25, max: 0.5, step: 0.05, dependsOn: "mid_scene_guide" },
  ];

  const CONTINUITY_DEFAULTS = {
    auto_enabled: true,
    identity_pin_ref: null,
    identity_pin_strength: 0.35,
    prior_scene_guides: true,
    prior_scene_strength: 0.35,
    mid_scene_guide: true,
    mid_scene_guide_strength: 0.3,
    guide_decay: 0.85,
    solo_scene_guides: true,
  };

  function normContinuitySettings(p) {
    const cs = (p && p.continuity_settings) || {};
    return {
      auto_enabled: cs.auto_enabled !== false,
      identity_pin_ref: cs.identity_pin_ref || null,
      identity_pin_strength: cs.identity_pin_strength != null ? +cs.identity_pin_strength : CONTINUITY_DEFAULTS.identity_pin_strength,
      prior_scene_guides: cs.prior_scene_guides !== false,
      prior_scene_strength: cs.prior_scene_strength != null ? +cs.prior_scene_strength : CONTINUITY_DEFAULTS.prior_scene_strength,
      mid_scene_guide: cs.mid_scene_guide !== false,
      mid_scene_guide_strength: cs.mid_scene_guide_strength != null ? +cs.mid_scene_guide_strength : CONTINUITY_DEFAULTS.mid_scene_guide_strength,
      guide_decay: cs.guide_decay != null ? +cs.guide_decay : CONTINUITY_DEFAULTS.guide_decay,
      solo_scene_guides: cs.solo_scene_guides !== false,
    };
  }

  function patchContinuitySettings(patch) {
    const p = S.get().project;
    S.patchProject({ continuity_settings: { ...(p.continuity_settings || {}), ...patch } });
  }

  function normGuideSettings(p) {
    const gs = (p && p.guide_settings) || {};
    return { stack_enabled: !!gs.stack_enabled, accumulate_prior: !!gs.accumulate_prior };
  }

  function patchGuideSettings(patch) {
    const p = S.get().project;
    S.patchProject({ guide_settings: { ...(p.guide_settings || {}), ...patch } });
  }

  function renderContinuitySettings(parent, st, multiScene) {
    const cs = normContinuitySettings(st.project);
    const gs = normGuideSettings(st.project);
    const hint = el("div", "insp-hint");
    hint.textContent = cs.auto_enabled
      ? (gs.stack_enabled
        ? "Auto continuity: mid-scene guide only — custom guide stack overrides auto guide lists."
        : "Auto continuity builds hidden guides per run: identity pin (all modes), prior-scene guides on carry chains and solo mixed runs, mid-scene anchor on multi-scene carry. Image / empty / generated_frame solo runs use their anchor only.")
      : "Auto continuity off — use manual Chain Sampler knobs and optional custom guide stack below.";
    parent.append(hint);

    const autoLbl = el("label", "chk");
    const autoCb = el("input"); autoCb.type = "checkbox"; autoCb.checked = cs.auto_enabled;
    autoCb.dataset.k = "cs-auto";
    autoCb.onchange = () => patchContinuitySettings({ auto_enabled: autoCb.checked });
    autoLbl.append(autoCb, el("span", null, "Auto continuity (recommended)"));
    parent.append(autoLbl);

    const pin = el("select");
    pin.append(new Option("— no identity pin —", ""));
    (st.mediaBin || []).filter((m) => m.kind === "image").forEach((m) => {
      const o = new Option(m.name, m.id);
      if (cs.identity_pin_ref === m.id) o.selected = true;
      pin.append(o);
    });
    pin.disabled = !cs.auto_enabled;
    pin.onchange = () => patchContinuitySettings({ identity_pin_ref: pin.value || null });
    parent.append(field("Identity pin (all scenes)", pin));

    const adv = collapsibleSection(parent, "continuity_adv", "Advanced", false);
    const priorLbl = el("label", "chk");
    const priorCb = el("input"); priorCb.type = "checkbox"; priorCb.checked = cs.prior_scene_guides;
    priorCb.disabled = !cs.auto_enabled;
    priorCb.onchange = () => patchContinuitySettings({ prior_scene_guides: priorCb.checked });
    priorLbl.append(priorCb, el("span", null, "Borrow prior-scene guides"));
    adv.append(priorLbl);
    const soloLbl = el("label", "chk");
    const soloCb = el("input"); soloCb.type = "checkbox"; soloCb.checked = cs.solo_scene_guides;
    soloCb.disabled = !cs.auto_enabled;
    soloCb.onchange = () => patchContinuitySettings({ solo_scene_guides: soloCb.checked });
    soloLbl.title = "Mixed mode only — image/empty/generated_frame solo runs use their anchor only";
    soloLbl.append(soloCb, el("span", null, "Prior guides on solo mixed runs"));
    adv.append(soloLbl);
    const midLbl = el("label", "chk");
    const midCb = el("input"); midCb.type = "checkbox"; midCb.checked = cs.mid_scene_guide;
    midCb.disabled = !cs.auto_enabled || !multiScene;
    midCb.title = multiScene ? "" : "Only applies to multi-scene carry chains";
    midCb.onchange = () => patchContinuitySettings({ mid_scene_guide: midCb.checked });
    midLbl.append(midCb, el("span", null, "Mid-scene layout guide (carry chains)"));
    adv.append(midLbl);

    const num = (label, val, key, min, max, step) => {
      const i = el("input"); i.type = "number"; i.min = String(min); i.max = String(max); i.step = String(step);
      i.value = val; i.disabled = !cs.auto_enabled;
      i.oninput = () => patchContinuitySettings({ [key]: parseFloat(i.value || "0") });
      adv.append(field(label, i));
    };
    num("Pin strength", cs.identity_pin_strength, "identity_pin_strength", 0.25, 0.5, 0.05);
    num("Prior guide strength", cs.prior_scene_strength, "prior_scene_strength", 0.25, 0.5, 0.05);
    num("Mid-scene strength", cs.mid_scene_guide_strength, "mid_scene_guide_strength", 0.25, 0.5, 0.05);
    num("Guide decay / scene", cs.guide_decay, "guide_decay", 0.5, 1, 0.05);
  }

  function renderGuideStackSettings(parent, st, multiScene) {
    const gs = normGuideSettings(st.project);
    const cs = normContinuitySettings(st.project);
    const hint = el("div", "insp-hint");
    hint.textContent = gs.stack_enabled
      ? "Custom guide stack — per-scene lists in the Scene inspector. Scenes without entries use the Studio default (scene 1 template · frame 0 · apply 0)."
      : (cs.auto_enabled
        ? "Auto continuity supplies guides at generation time (see Auto continuity above)."
        : "Studio default: one i2v guide from scene 1's template on multi-scene carry runs.");
    parent.append(hint);
    const stackLbl = el("label", "chk");
    const stackCb = el("input"); stackCb.type = "checkbox"; stackCb.checked = gs.stack_enabled;
    stackCb.dataset.k = "gs-stack";
    stackCb.onchange = () => patchGuideSettings({ stack_enabled: stackCb.checked });
    stackLbl.append(stackCb, el("span", null, "Custom guide stack"));
    parent.append(stackLbl);
    if (gs.stack_enabled) {
      const accLbl = el("label", "chk");
      const accCb = el("input"); accCb.type = "checkbox"; accCb.checked = gs.accumulate_prior;
      accCb.dataset.k = "gs-accum";
      accCb.onchange = () => patchGuideSettings({ accumulate_prior: accCb.checked });
      accLbl.append(accCb, el("span", null, "Stack guides from all prior scenes"));
      parent.append(accLbl);
      parent.append(el("div", "insp-hint", "Negative frame_idx / apply_at count from the end (e.g. −1 = last frame)."));
    } else if (multiScene) {
      parent.append(el("div", "insp-hint", "Carry i2v guides is auto-enabled for multi-scene runs."));
    }
  }

  const CHAIN_SECTIONS = [
    { id: "chain_auto", title: "Auto continuity", defaultOpen: true, knobs: [] },
    { id: "chain_timing", title: "Timing", defaultOpen: false, knobs: ["frame_overlap", "transition_duration", "use_same_seed"] },
    { id: "chain_cont", title: "Manual continuity", defaultOpen: false, knobs: ["carry_i2v_guides"] },
    { id: "chain_guid", title: "Guidance", defaultOpen: false, knobs: ["cfg", "embed_guidance", "embed_guidance_source", "embed_guidance_strength"] },
    { id: "chain_dec", title: "Decode", defaultOpen: false, knobs: ["decode_noise_scale", "decode_timestep", "decode_tile_size"] },
    { id: "chain_exp", title: "Experimental", defaultOpen: false, knobs: ["mid_scene_guide", "mid_scene_guide_strength"] },
  ];
  const SAMPLER_KNOB_MAP = Object.fromEntries(SAMPLER_KNOBS.map((k) => [k.name, k]));

  function countChainChanges(p) {
    const si = p.sampler_inputs || {};
    let n = 0;
    if (si.seed != null) n++;
    SAMPLER_KNOBS.forEach((k) => {
      if (si[k.name] != null && si[k.name] !== k.default) n++;
    });
    return n;
  }

  function renderSeedKnob(parent, si) {
    const ctrl = el("input");
    ctrl.type = "number";
    ctrl.min = "0";
    ctrl.placeholder = "Random each run";
    ctrl.value = si.seed != null ? String(si.seed) : "";
    ctrl.dataset.k = "si-seed";
    ctrl.oninput = () => {
      const raw = ctrl.value.trim();
      if (raw === "") S.unsetSamplerInput("seed");
      else S.setSamplerInput("seed", parseInt(raw, 10));
    };
    parent.append(field("Seed (optional)", ctrl));
    parent.append(el("div", "insp-hint",
      "Leave empty to randomize on every Generate. Set a fixed value here for reproducible runs (pair with Same seed per scene for multi-scene chains)."));
  }

  function knobVisible(k, si) {
    if (k.dependsOn) {
      const depVal = si[k.dependsOn] != null ? si[k.dependsOn] : SAMPLER_KNOB_MAP[k.dependsOn]?.default;
      if (!depVal) return false;
    }
    return true;
  }

  function renderSamplerKnob(parent, st, k, si, multiScene) {
    if (!knobVisible(k, si)) return;
    const val = si[k.name] != null ? si[k.name] : k.default;
    const gs = normGuideSettings(st.project);
    const cs = normContinuitySettings(st.project);
    const forced = k.lockMulti && multiScene && !gs.stack_enabled && cs.auto_enabled;
    let ctrl;
    if (k.kind === "bool") {
      ctrl = el("input"); ctrl.type = "checkbox";
      ctrl.checked = forced ? true : !!val;
      ctrl.style.width = "auto";
      ctrl.dataset.k = "si-" + k.name;
      if (forced) {
        ctrl.disabled = true;
        ctrl.title = "Auto-enabled by Auto continuity for multi-scene carry runs";
      } else {
        ctrl.onchange = () => S.setSamplerInputNow(k.name, ctrl.checked);
      }
    } else if (k.kind === "combo") {
      ctrl = el("select"); ctrl.dataset.k = "si-" + k.name;
      (k.choices || []).forEach((c) => { const o = el("option", null, c); o.value = c; if (c === val) o.selected = true; ctrl.append(o); });
      ctrl.onchange = () => S.setSamplerInputNow(k.name, ctrl.value);
    } else {
      ctrl = el("input"); ctrl.type = "number";
      if (k.step != null) ctrl.step = String(k.step);
      if (k.min != null) ctrl.min = String(k.min);
      if (k.max != null) ctrl.max = String(k.max);
      ctrl.value = val; ctrl.dataset.k = "si-" + k.name;
      ctrl.oninput = () => {
        const v = k.kind === "int" ? parseInt(ctrl.value || "0", 10) : parseFloat(ctrl.value || "0");
        S.setSamplerInput(k.name, v);
      };
    }
    parent.append(field(k.label + (forced ? " (auto)" : ""), ctrl));
  }

  function renderChainCard(parent, st) {
    const p = st.project;
    const si = p.sampler_inputs || {};
    const multiScene = activeSceneCount(p) > 1;
    const card = engineCard(parent, "chain_card", "Chain Sampler", true);
    card.setBadge(countChainChanges(p));

    CHAIN_SECTIONS.forEach((sec) => {
      const inner = collapsibleSection(card.body, sec.id, sec.title, sec.defaultOpen);
      sec.knobs.forEach((name) => {
        const k = SAMPLER_KNOB_MAP[name];
        if (k) renderSamplerKnob(inner, st, k, si, multiScene);
      });
      if (sec.id === "chain_auto") renderContinuitySettings(inner, st, multiScene);
      if (sec.id === "chain_cont") renderGuideStackSettings(inner, st, multiScene);
      if (sec.id === "chain_timing") {
        renderSeedKnob(inner, si);
        inner.append(el("div", "insp-hint",
          "Transition duration controls in-decode boundary fades (from transition library). Post-render crossfades are per-clip in the scene inspector."));
      }
    });
  }

  function renderContent(container, st) {
    if (!st.project) {
      container.append(el("div", "pj-meta", "No project open."));
      return;
    }
    const p = st.project;
    const slots = (st.models?.slots || []);

    const summary = el("div", "engine-modal-summary");
    const studioOn = !p.conditioning_slot || p.conditioning_slot === "funpack";
    const chainOn = !p.sampler_slot || p.sampler_slot === "funpack";
    const parts = [];
    parts.push(studioOn ? "FunPack Studio" : (slotLabelFor(st, p.conditioning_slot) || p.conditioning_slot));
    parts.push(chainOn ? "Chain Sampler" : (slotLabelFor(st, p.sampler_slot) || p.sampler_slot));
    summary.textContent = "Generating with " + parts.join(" + ");
    container.append(summary);

    const engTag = el("div", "insp-tag"); engTag.textContent = "Pipeline nodes"; container.append(engTag);
    const engRow = el("div", "fields-row insp-engine-pickers");
    const condSel = el("select"); condSel.dataset.k = "pj-cond";
    [["funpack", "FunPack Studio"], ...slots.map((s) => [s.id, s.label || s.node_class || s.id])]
      .forEach(([v, lbl]) => { const o = new Option(lbl, v); if ((p.conditioning_slot || "funpack") === v) o.selected = true; condSel.append(o); });
    condSel.onchange = () => S.setConditioningSlot(condSel.value);
    engRow.append(field("Conditioning", condSel));
    const sampSel = el("select"); sampSel.dataset.k = "pj-samp";
    [["funpack", "FunPack Chain Sampler"], ...slots.map((s) => [s.id, s.label || s.node_class || s.id])]
      .forEach(([v, lbl]) => { const o = new Option(lbl, v); if ((p.sampler_slot || "funpack") === v) o.selected = true; sampSel.append(o); });
    sampSel.onchange = () => S.setSamplerSlot(sampSel.value);
    engRow.append(field("Sampler", sampSel));
    container.append(engRow);

    if (!studioOn) {
      container.append(el("div", "insp-hint", "Custom conditioning node — wire and tune it in Models. FunPack Studio settings are hidden."));
    } else {
      renderStudioCard(container, st);
    }
    if (!chainOn) {
      container.append(el("div", "insp-hint", "Custom sampler node — wire and tune it in Models. Chain Sampler settings are hidden."));
    } else {
      renderChainCard(container, st);
    }

    const foot = el("div", "insp-hint");
    foot.append(el("span", null, "Unet / VAE / linked inputs: "));
    const modelsLink = el("button", "btn ghost tiny", "Models & Pipeline…");
    modelsLink.type = "button";
    modelsLink.onclick = () => window.ModelsModal.open();
    foot.append(modelsLink);
    container.append(foot);
  }

  function render() {
    if (!overlay) return;
    if (_editing) {
      const content = overlay.querySelector(".modal-content");
      const a = document.activeElement;
      if (a && content && content.contains(a) && (a.tagName === "INPUT" || a.tagName === "TEXTAREA" || a.tagName === "SELECT") && a.dataset.k) return;
      _editing = false;
    }
    const content = overlay.querySelector(".modal-content");
    const scrollTop = content.scrollTop;
    clear(content);
    renderContent(content, S.get());
    content.scrollTop = scrollTop;
  }

  function close() {
    if (unsub) { unsub(); unsub = null; }
    if (_onKey) { document.removeEventListener("keydown", _onKey); _onKey = null; }
    if (overlay) overlay.remove();
    overlay = null;
    _editing = false;
  }

  function open() {
    if (!S.get().project) return;
    if (overlay) { render(); return; }

    overlay = el("div", "modal-overlay");
    const modal = el("div", "modal modal-wide");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", "Engine Settings"));
    const closeBtn = el("button", "btn ghost", "✕");
    closeBtn.onclick = close;
    const heRight = el("div", "modal-head-right"); heRight.append(closeBtn);
    head.append(heRight);
    modal.append(head);
    const content = el("div", "modal-content engine-modal-content");
    modal.append(content);
    overlay.append(modal);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.body.append(overlay);

    content.addEventListener("focusin", (e) => {
      const t = e.target;
      if (t && t.dataset && t.dataset.k) _editing = true;
    });
    content.addEventListener("focusout", (e) => {
      const t = e.target;
      if (!(t && t.dataset && t.dataset.k)) return;
      _editing = false;
      setTimeout(() => { if (!_editing) render(); }, 60);
    });

    _onKey = (e) => { if (e.key === "Escape") close(); };
    document.addEventListener("keydown", _onKey);

    unsub = S.subscribe(() => render());
    render();
  }

  window.EngineSettingsModal = { open, close };
})();