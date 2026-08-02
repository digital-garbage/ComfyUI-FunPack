// Engine settings: Studio, Chain Sampler, continuity (moved out of Project inspector).
// A section of the unified Settings window with its OWN inner sidebar — categories are
// always visible (Overview · Studio: Refinement/Adjustments/Sampler · Chain Sampler:
// Continuity/Timing/Guidance/Decode/Experimental), no long scrolling card list.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const API = window.MovieEditorAPI;

  // Easy Gen has no rating UI at all (by design — see easy_gen/frontend/), so every
  // setting that is a no-op without a trained refinement key / rated history is hidden
  // there, not just made harder to find. window.FunPackAppName is the same discriminator
  // settings_window.js already uses to relabel the About section for a different frontend
  // sharing this file; Editor leaves it unset.
  const EASY = !!window.FunPackAppName;
  const RATING_GATED_KNOBS = new Set([
    "embed_guidance", "embed_guidance_source", "embed_guidance_strength",
    "score_slider", "score_slider_strength", "taste_nearest_prompt",
    "output_guidance", "output_guidance_strength",
    "dynashift", "dynashift_strength", "dynashift_threshold",
  ]);
  const RATING_GATED_STUDIO = new Set(["reference_injection", "value_guidance", "steer_mode", "absolute_strength"]);

  let _mounted = null; // { scroller (pane, set per render), content (shell root) }
  let unsub = null;
  let _editing = false;
  let view = "overview";
  function setView(v) { view = v; render(); }

  // macOS-style row: title on the left, control on the right. Append into a .sw-rows group.
  function field(labelText, control, hint) {
    const row = el("div", "sw-row eng-field");
    const main = el("div", "sw-row-main");
    main.append(el("div", "sw-row-title", labelText));
    if (hint) main.append(el("div", "sw-row-hint", hint));
    row.append(main, control);
    return row;
  }

  function toggleField(labelText, checkbox, hint) {
    checkbox.style.width = "auto";
    return field(labelText, checkbox, hint);
  }

  function group(parent, label) {
    if (label) parent.append(el("div", "sw-rows-label", label));
    const g = el("div", "sw-rows");
    parent.append(g);
    return g;
  }

  function hintEl(text) { return el("div", "sw-hint", text); }

  function slotLabelFor(st, slotId) {
    if (!slotId || slotId === "funpack") return null;
    const slot = (st.models?.slots || []).find((s) => s.id === slotId);
    return slot ? (slot.label || slot.node_class || slot.id) : slotId;
  }

  function activeSceneCount(p) {
    return (p.scenes || []).filter((s) => !s.excluded).length;
  }

  // ── FunPack Studio: refiner fields ─────────────────────────────────────────
  const STUDIO_REFINER_ESSENTIALS = [
    { name: "vision_conditioning", label: "Vision conditioning", default: true },
    { name: "reference_injection", label: "Reference injection", default: false },
  ];
  const STUDIO_REFINER_ADVANCED = [
    { name: "value_guidance", label: "Value guidance", kind: "bool", default: true },
    { name: "steer_mode", label: "Steer mode", kind: "combo", choices: ["relative", "absolute", "both"], default: "relative" },
    { name: "absolute_strength", label: "Absolute strength", kind: "float", default: 0.6, min: 0, max: 1, step: 0.05,
      dependsOn: "steer_mode", dependsVals: ["absolute", "both"] },
    { name: "temporal_style", label: "Temporal style", kind: "combo",
      choices: ["natural", "auto", "accelerate", "decelerate", "loop", "freeze", "pulse", "rapid_start", "rapid_end", "rapid_start_end"], default: "natural" },
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

  function countStudioRefine(p) {
    const { rf } = parseStudioSettings(p);
    let n = 0;
    [...STUDIO_REFINER_ESSENTIALS, ...STUDIO_REFINER_ADVANCED].forEach((f) => {
      if (EASY && RATING_GATED_STUDIO.has(f.name)) return;
      const cur = rf[f.name] != null ? rf[f.name] : f.default;
      if (cur !== f.default) n++;
    });
    if (!EASY && (p.refinement_key || "default") !== "default") n++;
    return n;
  }

  function persistStudioRefiner(patch, now) {
    const { cur } = parseStudioSettings(S.get().project);
    const rf = (cur.refiner && typeof cur.refiner === "object") ? cur.refiner : {};
    const next = JSON.stringify({ ...cur, refiner: { ...rf, ...patch } });
    if (now) S.setStudioInputNow("studio_settings", next);
    else S.setStudioInput("studio_settings", next);
  }

  // Conditioning adjustments — universal per-phrase steering. Stored as the FunPackStudio
  // `adjustments` widget (JSON [{phrase, strength}]), a top-level studio_input the builder
  // wires straight onto the node. Each phrase is CLIP-encoded and shifts conditioning toward
  // (+) / away (−) from it on EVERY generation, regardless of prompt. The backend ignores
  // blank/zero rows, so we persist the list as-is (no need to prune while editing).
  function parseAdjustments(p) {
    const si = (p && p.studio_inputs) || {};
    try {
      const v = JSON.parse(si.adjustments || "[]");
      return Array.isArray(v) ? v.map((i) => ({ phrase: String(i.phrase || ""), strength: +i.strength || 0 })) : [];
    } catch (_) { return []; }
  }

  function adjIsActive(i) {
    return !!String(i.phrase || "").trim() && Math.abs(+i.strength || 0) > 1e-6;
  }

  function persistAdjustments(items, now) {
    const json = JSON.stringify(items);
    if (now) S.setStudioInputNow("adjustments", json);
    else S.setStudioInput("adjustments", json);
  }

  function renderStudioRefinerBool(parentGroup, rf, f) {
    const cur = rf[f.name] != null ? rf[f.name] : f.default;
    const ctrl = el("input"); ctrl.type = "checkbox"; ctrl.checked = !!cur;
    ctrl.dataset.k = "rf-" + f.name;
    ctrl.onchange = () => persistStudioRefiner({ [f.name]: ctrl.checked }, true);
    parentGroup.append(toggleField(f.label, ctrl));
  }

  function renderStudioRefinerField(parentGroup, rf, f) {
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
    parentGroup.append(field(f.label, ctrl));
  }

  // ── Chain Sampler knobs ────────────────────────────────────────────────────
  const SAMPLER_KNOBS = [
    { name: "frame_overlap",         label: "Frame overlap",         kind: "int",   default: 16,    min: 0, max: 512, step: 8 },
    { name: "transition_duration",   label: "Transition duration",   kind: "int",   default: 16,    min: 0, max: 128, step: 2 },
    { name: "use_same_seed",         label: "Same seed per scene",   kind: "bool",  default: false },
    { name: "carry_i2v_guides",      label: "Carry i2v guides",      kind: "bool",  default: false, lockMulti: true },
    { name: "cfg",                   label: "CFG",                   kind: "float", default: 1.0,   min: 0, max: 20,  step: 0.1 },
    { name: "embed_guidance",        label: "Embed guidance",        kind: "bool",  default: false },
    { name: "embed_guidance_source", label: "Embed mode",            kind: "combo", choices: ["relative", "absolute"], default: "relative", dependsOn: "embed_guidance" },
    { name: "embed_guidance_strength", label: "Embed strength",      kind: "float", default: 0.02,  min: 0.005, max: 0.1, step: 0.005, dependsOn: "embed_guidance" },
    { name: "score_slider",          label: "Score slider",          kind: "bool",  default: false },
    { name: "score_slider_strength", label: "Slider strength (eta)", kind: "float", default: 1.0,   min: 0, max: 3, step: 0.25, dependsOn: "score_slider" },
    { name: "taste_nearest_prompt",  label: "Per-prompt taste direction", kind: "bool", default: false,
      hint: "EXPERIMENTAL: source Embed guidance / Score slider from the taste direction learned on the prompts NEAREST this scene's prompt, instead of one global liked-direction average. Each liked rating records (prompt → its liked direction); this retrieves the closest matches per scene (a forest prompt pulls what worked on forests). No extra model pass — a cosine lookup + vector mean. Falls back to the global direction when nothing rated is close. Needs Embed guidance or Score slider on. UNVALIDATED LIVE." },
    { name: "output_guidance",       label: "Output guidance",       kind: "bool",  default: false },
    { name: "output_guidance_strength", label: "Output guidance strength", kind: "float", default: 0.02, min: 0.005, max: 0.1, step: 0.005, dependsOn: "output_guidance" },
    { name: "decode_noise_scale",    label: "Decode noise scale",    kind: "float", default: 0.0,   min: 0, max: 1,   step: 0.01 },
    { name: "decode_timestep",       label: "Decode timestep",       kind: "float", default: 0.05,  min: 0, max: 1,   step: 0.01 },
    { name: "decode_tile_size",      label: "Decode tile size",      kind: "int",   default: 0,     min: 0, max: 4096, step: 64 },
    { name: "mid_scene_guide",       label: "Mid-scene guide",       kind: "bool",  default: false },
    { name: "mid_scene_guide_strength", label: "Guide strength",   kind: "float", default: 0.25,  min: 0.25, max: 0.5, step: 0.05, dependsOn: "mid_scene_guide" },
    { name: "joyai_memory",          label: "JoyAI-Echo memory",     kind: "bool",  default: false },
    { name: "joyai_memory_size",     label: "Memory size",           kind: "int",   default: 7,     min: 1, max: 32, step: 1, dependsOn: "joyai_memory" },
    { name: "joyai_fix_frames",      label: "Pinned anchors",        kind: "int",   default: 3,     min: 0, max: 16, step: 1, dependsOn: "joyai_memory" },
    { name: "joyai_frame_select",    label: "Frame select",          kind: "combo", choices: ["center", "first", "random"], default: "center", dependsOn: "joyai_memory" },
    { name: "joyai_memory_strength", label: "Memory strength",       kind: "float", default: 0.3,   min: 0.25, max: 10.0, step: 0.05, dependsOn: "joyai_memory" },
    { name: "joyai_audio_memory",    label: "Paired audio memory",   kind: "bool",  default: false, dependsOn: "joyai_memory" },
    { name: "v2a_grad_scale",        label: "Video→audio coupling", kind: "float", default: 1.0, min: 0.0, max: 4.0, step: 0.25, dependsOn: "joyai_audio_memory" },
    { name: "alg_blur_guides",       label: "Blur i2v guides and JoyAI memory", kind: "bool", default: false },
    { name: "alg_guide_blur_strength", label: "Guide blur strength", kind: "float", default: 2.0, min: 1.0, max: 4.0, step: 0.1, dependsOn: "alg_blur_guides" },
    { name: "alg_guide_blur_sigma_threshold", label: "Guide blur sigma threshold", kind: "float", default: 0.975, min: 0.5, max: 0.999, step: 0.005, dependsOn: "alg_blur_guides" },
    { name: "bounded_attention_enabled", label: "Bounded attention (multi-subject)", kind: "bool", default: false },
    { name: "dynashift",             label: "DynaShift (steer off bad gens)", kind: "bool", default: false },
    { name: "dynashift_strength",    label: "DynaShift strength",    kind: "float", default: 0.3, min: 0.05, max: 1.0, step: 0.05, dependsOn: "dynashift" },
    { name: "dynashift_threshold",   label: "DynaShift match threshold", kind: "float", default: 0.6, min: 0.3, max: 0.95, step: 0.05, dependsOn: "dynashift" },
    { name: "identity_transfer_enabled", label: "Best-FaceID compatibility", kind: "bool", default: false,
      hint: "Full native port of the overlap+source_phase+ArcFace conditioning Best-FaceID-style identity LoRAs were trained on. Replaces Continuity's Identity pin guide (Engine → Continuity) with separate, non-rendered reference tokens plus an optional ArcFace projector below. Load the LoRA itself the normal way — Models → add a LoRA loader onto the model path. No effect without an Identity pin image set." },
    { name: "source_id", label: "Source-phase id", kind: "float", default: 2.0, min: 0.0, max: 8.0, step: 1.0, dependsOn: "identity_transfer_enabled",
      hint: "Matches the LoRA's training convention (ltx-trainer used 2). 0 disables the RoPE rotation." },
    { name: "phase_scale", label: "Phase scale", kind: "float", default: 1.0, min: 0.0, max: 4.0, step: 0.1, dependsOn: "identity_transfer_enabled" },
    { name: "id_strength", label: "ArcFace token strength", kind: "float", default: 1.0, min: 0.0, max: 50.0, step: 0.5, dependsOn: "identity_transfer_enabled",
      hint: "Only applies when an ArcFace projector is selected below — weak channel, push high (5-20) to test." },
    { name: "arcface_mode", label: "ArcFace detection mode", kind: "combo", choices: ["auto_adjust", "as_is", "disable"], default: "auto_adjust", dependsOn: "identity_transfer_enabled" },
    { name: "debug_log", label: "Debug log", kind: "bool", default: false, dependsOn: "identity_transfer_enabled",
      hint: "Print per-scene identity-transfer shape/status logs to the ComfyUI console." },
    { name: "plateau_cache", label: "Plateau step-cache (speed)", kind: "bool", default: false,
      hint: "Ignored while Context windows is on — the cache can't tell one window from another within a step, so it's skipped with a note in the scene report. EXPERIMENTAL speed: the near-pure-noise early steps carry almost no signal, so the transformer output barely changes across them. Computes it once at the top of the plateau and reuses it for the rest, skipping ~3-4 of 8 transformer passes. Deterministic given seed (safe in Batch Training) but an approximation — A/B it before trusting on finals. Much of wall-clock time is outside the sampler, so total speedup is smaller than the forward count suggests. UNVALIDATED LIVE." },
    { name: "plateau_cache_threshold", label: "Plateau threshold (sigma)", kind: "float", default: 0.975, min: 0.5, max: 0.999, step: 0.005, dependsOn: "plateau_cache",
      hint: "Steps with sigma at or above this count as the reusable plateau. Higher = fewer steps cached (safer); lower = more cached (faster, more approximation). 0.975 catches the documented noise plateau while leaving structure formation fully computed." },
    { name: "segmented_detailing", label: "Segmented detailing (region refine)", kind: "bool", default: false,
      hint: "EXPERIMENTAL ADetailer-for-video: after each scene renders, CLIPSeg finds the regions named below (hands, feet, …), cuts them out of the latent as a tube, refines them at 2× working resolution through Lightricks' latent upsampler (3 extra steps on the crop only), and pastes them back through a feathered silhouette. Final resolution never changes. Cost ≈ 4 × region area × 3 steps (hands ~+15%); regions over 35% of the frame are refused. The upsampler model below is found — or downloaded (~1 GB, once) — automatically; skips are reported in the scene report. UNVALIDATED LIVE." },
    { name: "detail_targets", label: "Detail targets", kind: "text", default: "hands", dependsOn: "segmented_detailing", placeholder: "hands, feet",
      hint: "Comma-separated regions to detail, in plain words. Each becomes a CLIPSeg text query — malformed anatomy still matches its name. Also editable from Composer ▸ Compose while detailing is on." },
    { name: "detail_strength", label: "Detail strength", kind: "float", default: 1.0, min: 0, max: 1, step: 0.05, dependsOn: "segmented_detailing",
      hint: "Blend of the refined region at paste-back. 1.0 = full replacement inside the silhouette; 0 disables the pass." },
    { name: "detail_threshold", label: "Detail match threshold", kind: "float", default: 0.35, min: 0.05, max: 0.9, step: 0.05, dependsOn: "segmented_detailing",
      hint: "CLIPSeg match confidence required before a region counts as found. Its raw score for a real region is often well under 0.5 — if the scene report shows 'no match: max CLIPSeg score X < threshold', lower this toward X rather than assuming nothing is there." },
    { name: "detail_max_area", label: "Detail max area", kind: "float", default: 0.35, min: 0.05, max: 1.0, step: 0.05, dependsOn: "segmented_detailing",
      hint: "Ceiling on how much of the frame the region may cover before it's refused, as a fraction of the frame. Cost-only guardrail (a bigger region costs more, roughly 4× its area × 3 steps) — never a judgment call about whether it's worth detailing. If the scene report shows a region refused at some %, raise this above that % to detail it anyway. 1.0 = no cap." },
    { name: "detail_mode", label: "Detail mode", kind: "combo", choices: ["repair", "sharpen"], default: "repair", dependsOn: "segmented_detailing",
      hint: "'repair' (default): upsamples the crop, then re-denoises it through the video model — can genuinely fix wrong structure (bad anatomy) but costs real compute (~4× region area × 3 steps). 'sharpen': stops after the upsampler's own pass — no video-model calls at all, close to free — good for a region that's blurry/under-resolved but already correctly shaped; it cannot fix wrong structure (an extra finger stays an extra finger, just sharper)." },
    { name: "detail_denoise", label: "Detail re-noise strength", kind: "float", default: 0.85, min: 0.3, max: 0.99, step: 0.05,
      deps: [{ name: "segmented_detailing" }, { name: "detail_mode", value: "repair" }],
      hint: "How much noise the crop gets re-noised to before its 3-step refine (0.85 is the official LTX 2.3 recipe's own value). Higher = more freedom to genuinely reconstruct the region (fix bad anatomy), risking drift from the surrounding frame; lower = closer to a plain upscale — looks 'detailed' as interpolation but doesn't actually repair it. If the result looks upscaled but not corrected, raise this. Only used in 'repair' mode." },
    { name: "cut_opening_frames", label: "Cut the opening (frames)", kind: "int", default: 0, min: 0, max: 512, step: 8,
      hint: "Let the i2v anchor do its work, then cut it out of the clip. The anchor is a pinned frame at position 0, so it carries character detail, style and composition better than anything that weakens it on the way in (ALG blurs it and loses detail; Best-FaceID tokens approximate it and lose some too) — but it is also literally the first frame you see. The scene is generated exactly as normal, with the anchor pinned at full strength the whole way and no extra sampling, and this many frames are then dropped off the FRONT of the finished clip: an i2v generation that reads as t2v. 0 = off. 8 (one latent frame) removes just the anchor itself, which is usually not enough — the anchor is followed by a settling-in stretch where the shot is still leaving the reference still and little is happening, and on a prompt asking for immediate action that dead time is exactly what you want gone. 48 was the value that worked on a 768×768×305@30 chain with a quick-cut prompt; treat it as a starting point for this pipeline, not a universal default. NOTHING IS REGROWN: the scene comes out that much SHORTER than the length you set, and the audio is cropped to match — every surviving frame was generated as part of one continuous shot, with no invented ending. Needs an anchor image; skipped (with the reason in the scene report) on continuation scenes and on scenes carrying guide frames or JoyAI audio memory." },
    { name: "second_pass_op", label: "Between-pass operation", kind: "combo", choices: ["none", "sharpen", "upscale_2x"], default: "none",
      hint: "OPTIONAL operation applied to the latent between the two passes — 'none' by default, nothing runs unless you pick one. Both operations need the LTX 2.3 spatial upsampler in models/latent_upscale_models — the same ~1 GB file segmented detailing uses, found automatically or downloaded once on first use (watch the ComfyUI console). If it can't be obtained the second pass still runs, with the operation skipped and the reason in the scene report. 'sharpen': one forward of Lightricks' trained 2x latent upsampler, resampled straight back to the original size. No video-model calls, so it costs a fraction of a step; pass 2 then re-denoises the sharpened latent, which is what makes it stick. It adds detail consistent with what's already there and CANNOT fix wrong structure (an extra finger stays an extra finger, just sharper). 'upscale_2x': the same upsampler, but kept at 2x — pass 2 runs at four times the pixels and the scene decodes at double resolution. That's 3-5x the cost of the second half, and it drops the i2v pin (the anchor and its mask are the old size, and rescaling them would be inventing an anchor), so the scene can drift from the reference image; the scene report says when that happens. Both use the same upsampler file as segmented detailing. Video only — audio is never reshaped." },
    { name: "context_windows", label: "Context windows (long scenes)", kind: "bool", default: false,
      hint: "EXPERIMENTAL: denoise a scene that's LONGER than the model's comfortable window as overlapping windows instead of one giant pass — ComfyUI core's own mechanism, audio-aware on LTX (it maps each video window to its audio window and re-slices anchors, guides and JoyAI memory per window). Engages only on scenes longer than the window length below; shorter scenes are untouched and pay nothing. Cost at the defaults (145/40) is about 1.45× the per-frame work, since each window re-does its 40-frame overlap — offset by attention getting cheaper the longer the scene is (quadratic in one pass, near-flat when windowed). Roughly break-even around 200 frames, a net win past ~300. Needs ComfyUI v0.29.0 or newer; on older builds it's skipped with a note in the scene report. UNVALIDATED LIVE." },
    { name: "context_window_length", label: "Window length (frames)", kind: "int", default: 145, min: 9, max: 2049, step: 8, dependsOn: "context_windows",
      hint: "Window size in real frames. A scene at or below this length skips windowing entirely, so this doubles as the engage threshold. Keep it at or under the length the model already generates well in one pass — the point is to stay inside that range while the scene as a whole goes past it." },
    { name: "context_window_overlap", label: "Window overlap (frames)", kind: "int", default: 40, min: 0, max: 512, step: 8, dependsOn: "context_windows",
      hint: "How many frames consecutive windows share. This is the only thing carrying motion and appearance across a window boundary, and also the only extra compute this costs. Too low shows as a seam or a motion hitch at the boundary; too high pays for frames you already have." },
    { name: "context_window_schedule", label: "Window schedule", kind: "combo", choices: ["standard_uniform", "standard_static", "looped_uniform", "batched"], default: "standard_uniform", dependsOn: "context_windows",
      legacy: { uniform_standard: "standard_uniform", static_standard: "standard_static", uniform_looped: "looped_uniform" },
      hint: "How windows are laid out each step (ComfyUI core's own schedule names). 'standard_uniform' (default) shifts the grid between steps so boundaries never bake in — safest. 'standard_static' keeps fixed cut points (cheapest, but a bad boundary stays bad). 'looped_uniform' wraps the end into the start for looping content. 'batched' uses disjoint chunks with no overlap logic (fastest, weakest continuity)." },
    { name: "context_window_fuse", label: "Window blend", kind: "combo", choices: ["pyramid", "relative", "flat", "overlap-linear"], default: "pyramid", dependsOn: "context_windows",
      hint: "How overlapping windows are weighted when merged. 'pyramid' (default) fades each window toward its edges so seams go soft. 'flat' averages equally (can smear). Change this if boundaries look ghosted rather than merely misaligned." },
    { name: "context_window_freenoise", label: "FreeNoise blending", kind: "bool", default: true, dependsOn: "context_windows",
      hint: "Shuffle rather than redraw the starting noise between windows, so overlapping regions begin from correlated noise. Free (a one-time permutation) and core's own default because it measurably improves how windows blend. Turn off only to A/B whether it's helping." },
    { name: "context_window_retain_first", label: "Pin anchor in every window", kind: "bool", default: false, dependsOn: "context_windows",
      hint: "Keep latent frame 0 (the i2v anchor) inside every window instead of just the first. Turn on if later windows drift away from the reference image. Off by default because on a continuation scene frame 0 is the carried tail of the previous scene, not the anchor — pinning it everywhere can make the scene read as static. Turn it off again if motion stalls." },
  ];
  const SAMPLER_KNOB_MAP = Object.fromEntries(SAMPLER_KNOBS.map((k) => [k.name, k]));

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

  function _depSatisfied(name, value, si) {
    const depVal = si[name] != null ? si[name] : SAMPLER_KNOB_MAP[name]?.default;
    return value !== undefined ? depVal === value : !!depVal;
  }

  function knobVisible(k, si) {
    if (EASY && RATING_GATED_KNOBS.has(k.name)) return false;
    // dependsOn/dependsValue: single condition (dependsValue absent = plain truthy
    // gate, as every existing boolean dependsOn already relies on).
    if (k.dependsOn && !_depSatisfied(k.dependsOn, k.dependsValue, si)) return false;
    // deps: an AND'd list, for knobs gated by more than one condition (e.g. the
    // feature toggle AND a mode combo equaling a specific option) — dependsOn
    // alone can't express two conditions without one silently overriding the other.
    if (Array.isArray(k.deps)) {
      for (const d of k.deps) {
        if (!_depSatisfied(d.name, d.value, si)) return false;
      }
    }
    return true;
  }

  function renderSamplerKnob(parentGroup, st, k, si, multiScene) {
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
      // A renamed choice: show what the stored value MEANS, and write the new name back, so
      // the dropdown never quietly displays a different setting from the one that will run.
      const shown = (k.legacy && k.legacy[val]) || val;
      if (shown !== val) S.setSamplerInput(k.name, shown);
      (k.choices || []).forEach((c) => { const o = el("option", null, c); o.value = c; if (c === shown) o.selected = true; ctrl.append(o); });
      ctrl.onchange = () => S.setSamplerInputNow(k.name, ctrl.value);
    } else if (k.kind === "text") {
      ctrl = el("input"); ctrl.type = "text";
      ctrl.value = val != null ? String(val) : "";
      if (k.placeholder) ctrl.placeholder = k.placeholder;
      ctrl.dataset.k = "si-" + k.name;
      // Quiet while typing (no repaint under the caret), commit on blur/Enter.
      ctrl.oninput = () => S.setSamplerInput(k.name, ctrl.value);
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
    parentGroup.append(field(k.label + (forced ? " (auto)" : ""), ctrl, k.hint));
  }

  function renderKnobList(parentGroup, st, names) {
    const si = st.project.sampler_inputs || {};
    const multiScene = activeSceneCount(st.project) > 1;
    names.forEach((name) => {
      const k = SAMPLER_KNOB_MAP[name];
      if (k) renderSamplerKnob(parentGroup, st, k, si, multiScene);
    });
  }

  // ── views (inner-sidebar categories) ───────────────────────────────────────
  const CHAIN_VIEW_KNOBS = {
    chain_continuity: ["carry_i2v_guides"],
    chain_timing: ["frame_overlap", "transition_duration", "use_same_seed", "cut_opening_frames"],
    chain_guidance: ["cfg", "embed_guidance", "embed_guidance_source", "embed_guidance_strength", "score_slider", "score_slider_strength", "taste_nearest_prompt", "output_guidance", "output_guidance_strength", "dynashift", "dynashift_strength", "dynashift_threshold"],
    chain_decode: ["decode_noise_scale", "decode_timestep", "decode_tile_size"],
    chain_experimental: ["context_windows", "context_window_length", "context_window_overlap", "context_window_schedule", "context_window_fuse", "context_window_freenoise", "context_window_retain_first", "plateau_cache", "plateau_cache_threshold", "segmented_detailing", "detail_targets", "detail_strength", "detail_threshold", "detail_max_area", "detail_mode", "detail_denoise", "mid_scene_guide", "mid_scene_guide_strength", "joyai_memory", "joyai_memory_size", "joyai_fix_frames", "joyai_frame_select", "joyai_memory_strength", "joyai_audio_memory", "v2a_grad_scale", "alg_blur_guides", "alg_guide_blur_strength", "alg_guide_blur_sigma_threshold", "bounded_attention_enabled", "identity_transfer_enabled", "source_id", "phase_scale", "id_strength", "arcface_mode", "debug_log"],
  };

  function countChainView(p, id) {
    const si = p.sampler_inputs || {};
    let n = 0;
    (CHAIN_VIEW_KNOBS[id] || []).forEach((name) => {
      if (EASY && RATING_GATED_KNOBS.has(name)) return;
      const k = SAMPLER_KNOB_MAP[name];
      if (k && si[name] != null && si[name] !== k.default) n++;
    });
    if (id === "chain_timing" && si.seed != null) n++;
    return n;
  }

  function viewList(st) {
    const p = st.project;
    const PC = window.PipelineCaps;
    const studioOn = PC?.usesFunpackStudio(st);
    const chainOn = PC?.usesChainSampler(st);
    const out = [{ id: "overview", group: "", title: "Overview", icon: "◎" }];
    if (studioOn) {
      out.push(
        { id: "studio_refine", group: "FunPack Studio", title: "Refinement", icon: "✦", badge: countStudioRefine(p) || null },
        { id: "studio_adjust", group: "FunPack Studio", title: "Adjustments", icon: "±", badge: parseAdjustments(p).filter(adjIsActive).length || null },
        { id: "studio_sampler", group: "FunPack Studio", title: "Sampler algorithm", icon: "∿" },
      );
    }
    if (chainOn) {
      out.push(
        { id: "chain_continuity", group: "Chain Sampler", title: "Continuity", icon: "∞", badge: countChainView(p, "chain_continuity") || null },
        { id: "chain_timing", group: "Chain Sampler", title: "Timing & Seed", icon: "⏱", badge: countChainView(p, "chain_timing") || null },
        { id: "chain_guidance", group: "Chain Sampler", title: "Guidance", icon: "◇", badge: countChainView(p, "chain_guidance") || null },
        { id: "chain_decode", group: "Chain Sampler", title: "Decode", icon: "▣", badge: countChainView(p, "chain_decode") || null },
        { id: "chain_experimental", group: "Chain Sampler", title: "Experimental", icon: "⚗", badge: countChainView(p, "chain_experimental") || null },
      );
      // ref2va is H3's own conditioning mode — there is no LTX equivalent, so the view
      // only exists for the family that has it rather than sitting there greyed out.
      if (PC?.isH3(st)) {
        out.splice(out.findIndex((v) => v.id === "chain_timing"), 0, {
          id: "chain_references", group: "Chain Sampler", title: "Reference media", icon: "◉",
          badge: h3Refs(p).length || null,
        });
      }
    }
    return out;
  }

  function renderOverview(pane, st) {
    const p = st.project;
    const PC = window.PipelineCaps;
    const studioOn = PC?.usesFunpackStudio(st);
    const chainOn = PC?.usesChainSampler(st);
    const slots = (st.models?.slots || []);

    if (studioOn || chainOn) {
      const presetBar = el("div", "engine-preset-bar");
      presetBar.append(el("span", "engine-preset-label", "Preset"));
      Object.entries(S.ENGINE_PRESETS || {}).forEach(([key, pr]) => {
        const b = el("button", "btn ghost tiny", pr.label || key);
        b.onclick = () => S.applyEnginePreset(key);
        presetBar.append(b);
      });
      pane.append(presetBar);
    }

    const summary = el("div", "engine-modal-summary");
    const parts = [];
    parts.push(studioOn ? "FunPack Studio" : (slotLabelFor(st, p.conditioning_slot) || p.conditioning_slot));
    parts.push(chainOn ? "Chain Sampler" : (slotLabelFor(st, p.sampler_slot) || p.sampler_slot));
    summary.textContent = "Generating with " + parts.join(" + ");
    pane.append(summary);

    if (!studioOn && !chainOn) {
      pane.append(hintEl("Neither FunPack Studio nor Chain Sampler is active — pick them below or wire custom nodes in Models & Pipeline."));
    }

    const g = group(pane, "Pipeline nodes");
    const condSel = el("select"); condSel.dataset.k = "pj-cond";
    [["funpack", "FunPack Studio"], ...slots.map((s) => [s.id, s.label || s.node_class || s.id])]
      .forEach(([v, lbl]) => { const o = new Option(lbl, v); if ((p.conditioning_slot || "funpack") === v) o.selected = true; condSel.append(o); });
    condSel.onchange = () => S.setConditioningSlot(condSel.value);
    g.append(field("Conditioning", condSel));
    const sampSel = el("select"); sampSel.dataset.k = "pj-samp";
    [["funpack", "FunPack Chain Sampler"], ...slots.map((s) => [s.id, s.label || s.node_class || s.id])]
      .forEach(([v, lbl]) => { const o = new Option(lbl, v); if ((p.sampler_slot || "funpack") === v) o.selected = true; sampSel.append(o); });
    sampSel.onchange = () => S.setSamplerSlot(sampSel.value);
    g.append(field("Sampler", sampSel));

    if (!studioOn) pane.append(hintEl("Custom conditioning node — wire and tune it in Models & Pipeline. FunPack Studio categories are hidden."));
    if (!chainOn) pane.append(hintEl("Custom sampler node — wire and tune it in Models & Pipeline. Chain Sampler categories are hidden."));

    const foot = el("div", "sw-hint eng-foot");
    foot.append(el("span", null, "Unet / VAE / linked inputs: "));
    const modelsLink = el("button", "btn ghost tiny", "Models & Pipeline…");
    modelsLink.type = "button";
    modelsLink.onclick = () => window.ModelsModal.open();
    foot.append(modelsLink);
    pane.append(foot);
  }

  function renderStudioRefine(pane, st) {
    const p = st.project;
    const { rf } = parseStudioSettings(p);

    if (!EASY) {
      // Refinement key — project-level (feeds Studio / Chain Sampler / SaveRefinementLatent).
      // "default" uses the keyless store; a custom name trains/loads its own key. Shortcuts
      // bound to a non-default key layer per-scene training on top of this.
      const gKey = group(pane, "Session");
      const keyCtrl = el("input"); keyCtrl.type = "text"; keyCtrl.dataset.k = "refinement_key";
      keyCtrl.placeholder = "default"; keyCtrl.value = p.refinement_key || "default";
      keyCtrl.onchange = () => S.patchProject({ refinement_key: (keyCtrl.value || "").trim() || "default" });
      gKey.append(field("Refinement key", keyCtrl, "Named learning session — \"default\" is the keyless store."));
    }

    const gEss = group(pane, "Essentials");
    STUDIO_REFINER_ESSENTIALS.filter((f) => !EASY || !RATING_GATED_STUDIO.has(f.name))
      .forEach((f) => renderStudioRefinerBool(gEss, rf, f));

    const gAdv = group(pane, EASY ? "Prompt shaping" : "Refinement");
    STUDIO_REFINER_ADVANCED.filter((f) => !EASY || !RATING_GATED_STUDIO.has(f.name))
      .forEach((f) => renderStudioRefinerField(gAdv, rf, f));

    if (EASY) {
      pane.append(hintEl(
        "Studio runs in Prompt-only mode from Easy Gen — it shapes and splits the prompt "
        + "and passes conditioning through unchanged. Rating-dependent controls (refinement "
        + "key, value guidance, steer mode, reference injection) aren't shown here since "
        + "there's no rating UI in Easy Gen to feed them. For the full learned refiner, use "
        + "the Cutting Room (Movie Editor) or the ComfyUI node graph directly."));
    } else {
      pane.append(hintEl("Scene text and transitions come from the timeline. Advisor, LoRA, and batch training remain in the ComfyUI Studio popup on the graph."));
    }
  }

  function renderStudioAdjust(pane, st) {
    pane.append(hintEl(
      "Universal per-phrase steering: each phrase is encoded by CLIP and shifts conditioning "
      + "toward (+) or away (−) from it on every generation, regardless of the prompt. "
      + "Typical range −0.3 to +0.3."));
    const items = parseAdjustments(st.project);
    const g = group(pane, "Phrases");
    if (!items.length) g.append(el("div", "sw-row eng-empty-row", "No adjustments. Add a phrase below."));
    items.forEach((item, idx) => {
      const row = el("div", "sw-row studio-adjust-row");
      const phrase = el("input"); phrase.type = "text"; phrase.placeholder = "phrase or word";
      phrase.value = item.phrase; phrase.dataset.k = "adj-phrase-" + idx; phrase.style.flex = "1";
      phrase.oninput = () => { items[idx].phrase = phrase.value; persistAdjustments(items, false); };
      const strength = el("input"); strength.type = "number";
      strength.step = "0.05"; strength.min = "-1"; strength.max = "1";
      strength.value = item.strength; strength.dataset.k = "adj-str-" + idx; strength.style.width = "72px";
      strength.title = "Positive steers toward the phrase, negative away.";
      strength.oninput = () => { items[idx].strength = parseFloat(strength.value || "0"); persistAdjustments(items, false); };
      const del = el("button", "btn ghost tiny", "✕"); del.type = "button"; del.title = "Remove";
      del.onclick = () => { items.splice(idx, 1); persistAdjustments(items, true); };
      row.append(phrase, strength, del);
      g.append(row);
    });
    const add = el("button", "btn ghost tiny eng-add", "+ Add phrase"); add.type = "button";
    add.onclick = () => {
      items.push({ phrase: "", strength: 0.1 });
      persistAdjustments(items, true);
      setTimeout(() => {
        const inputs = document.querySelectorAll('[data-k^="adj-phrase-"]');
        inputs[inputs.length - 1]?.focus();
      }, 0);
    };
    pane.append(add);
  }

  function renderStudioSampler(pane, st) {
    const { cur: curSettings } = parseStudioSettings(st.project);
    const samplers = curSettings.samplers || null;
    const box = el("div", "eng-sampler-panel");
    pane.append(box);
    function persistSamplers(updatedSamplers, quiet) {
      const { cur } = parseStudioSettings(S.get().project);
      const next = JSON.stringify({ ...cur, samplers: updatedSamplers });
      if (quiet) S.setStudioInput("studio_settings", next);
      else S.setStudioInputNow("studio_settings", next);
      syncSecondPassFromSchedule(updatedSamplers, quiet);
    }
    // The second pass is driven by ONE thing: the schedule. Typing one turns it on; clearing
    // it turns it off. There is no cut and no re-entry point to configure — pass 1 always runs
    // the main schedule in full and pass 2 always runs this one in full.
    function syncSecondPassFromSchedule(samplers, quiet) {
      const raw = String(samplers?.low?.sigmas || "").replace(/;/g, ",");
      const vals = raw.split(",").map((v) => parseFloat(v.trim())).filter((v) => !isNaN(v));
      const set = quiet ? S.setSamplerInput : S.setSamplerInputNow;
      if (vals.length >= 2) set("second_pass", true);       // one number is not a schedule
      else S.unsetSamplerInput("second_pass");
    }

    try {
      window.SamplerPanel.render(box, samplers,
        (s) => persistSamplers(s, true),
        (s) => persistSamplers(s, false));
    } catch (e) {
      const err = hintEl("Studio sampler panel failed to render: " + e.message);
      err.style.color = "var(--danger)";
      box.append(err);
    }
    // Second pass lives here rather than under Experimental: at its defaults the split is
    // behaviour-neutral (pass 2 resumes from exactly the state pass 1 handed over), so it
    // is a sampler setting, not a gamble. The only control is the schedule field in the
    // panel above — enable and cut are derived from it (see syncSecondPassFromSchedule);
    // all that is left here is the optional between-pass operation.
    const g = group(pane, "Second pass");
    renderKnobList(g, st, ["second_pass_op"]);
    const si = st.project.sampler_inputs || {};
    if (si.second_pass_op && si.second_pass_op !== "none") {
      // Both operations are one forward of the same trained latent upsampler segmented
      // detailing uses, so the model choice belongs here too — otherwise picking 'sharpen'
      // silently depends on a file the user was never shown.
      renderDetailUpsampler(pane, si, "Between-pass operation: upsampler model");
    }
  }

  function renderChainContinuity(pane, st) {
    const p = st.project;
    const multiScene = activeSceneCount(p) > 1;
    const cs = normContinuitySettings(p);
    const gs = normGuideSettings(p);

    pane.append(hintEl(cs.auto_enabled
      ? (gs.stack_enabled
        ? "Auto continuity: mid-scene guide only — custom guide stack overrides auto guide lists."
        : "Auto continuity builds hidden guides per run: identity pin (all modes), prior-scene guides on carry chains and solo mixed runs, mid-scene anchor on multi-scene carry. Image / empty / generated_frame solo runs use their anchor only.")
      : "Auto continuity off — use manual Chain Sampler knobs and optional custom guide stack below."));

    const g = group(pane, "Auto continuity");
    const autoCb = el("input"); autoCb.type = "checkbox"; autoCb.checked = cs.auto_enabled;
    autoCb.dataset.k = "cs-auto";
    autoCb.onchange = () => patchContinuitySettings({ auto_enabled: autoCb.checked });
    g.append(toggleField("Auto continuity (recommended)", autoCb));

    const pinRow = el("div", "sw-row eng-field eng-stack");
    const pinMain = el("div", "sw-row-main");
    pinMain.append(el("div", "sw-row-title", "Identity pin (all scenes)"));
    pinRow.append(pinMain);
    const pin = window.MediaPicker.create({
      value: cs.identity_pin_ref,
      mediaBin: st.mediaBin,
      noneLabel: "— no identity pin —",
      onChange: (v) => patchContinuitySettings({ identity_pin_ref: v }),
    });
    if (!cs.auto_enabled) pin.classList.add("disabled");
    pinRow.append(pin);
    g.append(pinRow);

    const gAdv = group(pane, "Advanced");
    const mk = (label, checked, key, opts = {}) => {
      const cb = el("input"); cb.type = "checkbox"; cb.checked = checked;
      cb.disabled = !cs.auto_enabled || !!opts.disabled;
      if (opts.title) cb.title = opts.title;
      cb.onchange = () => patchContinuitySettings({ [key]: cb.checked });
      gAdv.append(toggleField(label, cb, opts.hint));
    };
    mk("Borrow prior-scene guides", cs.prior_scene_guides, "prior_scene_guides");
    mk("Prior guides on solo mixed runs", cs.solo_scene_guides, "solo_scene_guides",
      { hint: "Mixed mode only — image/empty/generated_frame solo runs use their anchor only" });
    mk("Mid-scene layout guide (carry chains)", cs.mid_scene_guide, "mid_scene_guide",
      { disabled: !multiScene, title: multiScene ? "" : "Only applies to multi-scene carry chains" });
    const num = (label, val, key, min, max, step) => {
      const i = el("input"); i.type = "number"; i.min = String(min); i.max = String(max); i.step = String(step);
      i.value = val; i.disabled = !cs.auto_enabled; i.dataset.k = "cs-" + key;
      i.oninput = () => patchContinuitySettings({ [key]: parseFloat(i.value || "0") });
      gAdv.append(field(label, i));
    };
    num("Pin strength", cs.identity_pin_strength, "identity_pin_strength", 0.25, 0.5, 0.05);
    num("Prior guide strength", cs.prior_scene_strength, "prior_scene_strength", 0.25, 0.5, 0.05);
    num("Mid-scene strength", cs.mid_scene_guide_strength, "mid_scene_guide_strength", 0.25, 0.5, 0.05);
    num("Guide decay / scene", cs.guide_decay, "guide_decay", 0.5, 1, 0.05);

    const gMan = group(pane, "Manual");
    renderKnobList(gMan, st, CHAIN_VIEW_KNOBS.chain_continuity);
    const stackCb = el("input"); stackCb.type = "checkbox"; stackCb.checked = gs.stack_enabled;
    stackCb.dataset.k = "gs-stack";
    stackCb.onchange = () => patchGuideSettings({ stack_enabled: stackCb.checked });
    gMan.append(toggleField("Custom guide stack", stackCb,
      gs.stack_enabled
        ? "Per-scene lists in the Scene inspector. Scenes without entries use the Studio default (scene 1 template · frame 0 · apply 0)."
        : (cs.auto_enabled
          ? "Auto continuity supplies guides at generation time."
          : "Studio default: one i2v guide from scene 1's template on multi-scene carry runs.")));
    if (gs.stack_enabled) {
      const accCb = el("input"); accCb.type = "checkbox"; accCb.checked = gs.accumulate_prior;
      accCb.dataset.k = "gs-accum";
      accCb.onchange = () => patchGuideSettings({ accumulate_prior: accCb.checked });
      gMan.append(toggleField("Stack guides from all prior scenes", accCb,
        "Negative frame_idx / apply_at count from the end (e.g. −1 = last frame)."));
    } else if (multiScene) {
      pane.append(hintEl("Carry i2v guides is auto-enabled for multi-scene runs."));
    }
  }

  // ── MiniMax H3 reference media (ref2va) ────────────────────────────────────
  // The order of this list is load-bearing, not cosmetic: Studio numbers the prompt
  // labels ("<Picture 1>", "<Video 2>") from it and the Chain Sampler encodes the same
  // list into packed blocks. Reordering here reorders both, together — which is exactly
  // why reordering has to happen HERE and not in two places.
  const H3_REF_KINDS = [
    { key: "image", label: "Image", tag: "Picture", hint: "A face, a character, a style plate." },
    { key: "video", label: "Video", tag: "Video", hint: "2–15s at 24 fps. Frames are decoded here; the model sees them at 2 fps." },
    { key: "audio", label: "Audio", tag: "Audio", hint: "A voice or ambience the generation should sound like. Needs an audio VAE." },
  ];

  function h3Refs(p) {
    return Array.isArray(p.h3_references) ? p.h3_references : [];
  }

  function patchH3Refs(list) {
    S.patchProject({ h3_references: list });
  }

  // 1-based ordinal per TYPE, exactly how the tokenizer numbers them — so the label shown
  // here is the label the user types in the prompt.
  function h3RefLabels(refs) {
    const counts = { image: 0, video: 0, audio: 0 };
    return refs.map((r) => {
      const kind = H3_REF_KINDS.find((k) => k.key === r.kind) || H3_REF_KINDS[0];
      counts[kind.key] = (counts[kind.key] || 0) + 1;
      return `<${kind.tag} ${counts[kind.key]}>`;
    });
  }

  function renderChainReferences(pane, st) {
    const p = st.project;
    const refs = h3Refs(p);
    const labels = h3RefLabels(refs);

    pane.append(hintEl(
      "MiniMax H3 packs reference media straight into the sequence — no projector, no extra "
      + "tokens on the prompt. Refer to each one in your prompt by the tag shown beside it "
      + "(e.g. \"the woman from <Picture 1> walks into frame\"). Order matters: it decides "
      + "the numbering, so moving an entry renumbers the tags."));

    const g = group(pane, "References");
    if (!refs.length) {
      g.append(hintEl("No references — the run is plain text-to-audio-video (or first-frame "
        + "anchored, if a scene has an anchor image)."));
    }

    refs.forEach((ref, i) => {
      const kind = H3_REF_KINDS.find((k) => k.key === ref.kind) || H3_REF_KINDS[0];
      const row = el("div", "sw-row eng-field eng-stack");
      const main = el("div", "sw-row-main");
      main.append(el("div", "sw-row-title", `${labels[i]} · ${kind.label}`));
      main.append(el("div", "sw-hint", kind.hint));
      row.append(main);

      const picker = window.MediaPicker.create({
        value: ref.filename,
        mediaBin: st.mediaBin,
        noneLabel: "— pick from the media bin —",
        onChange: (v) => {
          const next = refs.slice();
          next[i] = { ...ref, filename: v };
          patchH3Refs(next);
        },
      });
      row.append(picker);

      // a video's own soundtrack: presented as "<Audio j>" immediately before its
      // "<Video k>", which is what tells the model the two belong together
      if (ref.kind === "video") {
        const track = window.MediaPicker.create({
          value: ref.audio || "",
          mediaBin: st.mediaBin,
          noneLabel: "— no soundtrack —",
          onChange: (v) => {
            const next = refs.slice();
            next[i] = { ...ref, audio: v };
            patchH3Refs(next);
          },
        });
        row.append(track);
      }

      const tools = el("div", "sw-row-tools");
      const up = el("button", "btn ghost tiny", "↑");
      up.disabled = i === 0;
      up.title = "Move earlier — this renumbers the tags.";
      up.onclick = () => {
        const next = refs.slice();
        [next[i - 1], next[i]] = [next[i], next[i - 1]];
        patchH3Refs(next);
      };
      const down = el("button", "btn ghost tiny", "↓");
      down.disabled = i === refs.length - 1;
      down.title = "Move later — this renumbers the tags.";
      down.onclick = () => {
        const next = refs.slice();
        [next[i], next[i + 1]] = [next[i + 1], next[i]];
        patchH3Refs(next);
      };
      const del = el("button", "btn ghost tiny", "✕");
      del.title = "Remove this reference — the ones after it move up a number.";
      del.onclick = () => patchH3Refs(refs.filter((_, j) => j !== i));
      tools.append(up, down, del);
      row.append(tools);
      g.append(row);
    });

    const add = group(pane, "Add");
    const addRow = el("div", "sw-row");
    H3_REF_KINDS.forEach((k) => {
      const b = el("button", "btn ghost tiny", "＋ " + k.label);
      b.title = k.hint;
      b.onclick = () => patchH3Refs(refs.concat([{ kind: k.key, filename: "" }]));
      addRow.append(b);
    });
    add.append(addRow);

    const unset = refs.filter((r) => !r.filename).length;
    if (unset) {
      pane.append(hintEl(`${unset} reference${unset > 1 ? "s have" : " has"} no file picked yet — `
        + "those are skipped at generation, and everything after them renumbers."));
    }
    if (refs.some((r) => r.kind === "audio" || (r.kind === "video" && r.audio))) {
      pane.append(hintEl("Audio references need an Audio VAE wired to the Chain Sampler's "
        + "audio_vae input (Settings → Models). Without it the audio is dropped and the "
        + "<Audio> numbering shifts."));
    }
    pane.append(hintEl("Reference tokens ride through every sampling step, so each one you add "
      + "— and every extra second of a reference video — costs time on every step of every scene."));
  }

  function renderChainTiming(pane, st) {
    const si = st.project.sampler_inputs || {};
    const g = group(pane, "Timing");
    renderKnobList(g, st, CHAIN_VIEW_KNOBS.chain_timing);
    const gSeed = group(pane, "Seed");
    const ctrl = el("input");
    ctrl.type = "number"; ctrl.min = "0";
    ctrl.placeholder = "Random each run";
    ctrl.value = si.seed != null ? String(si.seed) : "";
    ctrl.dataset.k = "si-seed";
    ctrl.oninput = () => {
      const raw = ctrl.value.trim();
      if (raw === "") S.unsetSamplerInput("seed");
      else S.setSamplerInput("seed", parseInt(raw, 10));
    };
    gSeed.append(field("Seed (optional)", ctrl,
      "Empty = randomize every Generate. Fixed value = reproducible runs (pair with Same seed per scene)."));
    pane.append(hintEl(
      "Transition duration controls in-decode boundary fades (from transition library). Post-render crossfades are per-clip in the scene inspector."));
  }

  function renderChainKnobsView(pane, st, id, label, hint) {
    if (hint) pane.append(hintEl(hint));
    const g = group(pane, label);
    renderKnobList(g, st, CHAIN_VIEW_KNOBS[id]);
  }

  // ── Segmented detailing: the upsampler model list is live (models/latent_upscale_models
  // on the server), so like the ArcFace projector below it can't be a static SAMPLER_KNOBS
  // combo. The chain sampler node's own spec already carries the choices (with "None").
  let _detailUpsamplerChoices = null;

  async function ensureDetailUpsamplerChoices() {
    if (_detailUpsamplerChoices) return _detailUpsamplerChoices;
    try {
      const spec = await API.nodeSpec("FunPackLTXAVSceneChainSampler");
      const w = (spec?.inputs || []).find((i) => i.name === "detail_upsampler");
      _detailUpsamplerChoices = (w && w.choices) || ["None"];
    } catch (_) {
      _detailUpsamplerChoices = ["None"];
    }
    render();
    return _detailUpsamplerChoices;
  }

  // Two features load this same file: segmented detailing and the second pass's
  // between-pass operation (sharpen / upscale_2x). It used to render only under
  // Experimental and only while detailing was on, so someone using 'sharpen' alone got no
  // upsampler control and no warning that a ~1 GB download was involved.
  function renderDetailUpsampler(pane, si, title) {
    const g = group(pane, title || "Latent upsampler model");
    if (!_detailUpsamplerChoices) {
      g.append(hintEl("Loading upsampler list…"));
      ensureDetailUpsamplerChoices();
      return;
    }
    const sel = el("select"); sel.dataset.k = "si-detail_upsampler";
    const cur = si.detail_upsampler && si.detail_upsampler !== "None" ? si.detail_upsampler : "auto";
    _detailUpsamplerChoices.forEach((c) => {
      const o = el("option", null, c); o.value = c;
      if (c === cur) o.selected = true;
      sel.append(o);
    });
    sel.onchange = () => S.setSamplerInputNow("detail_upsampler", sel.value);
    g.append(field("Latent upsampler", sel,
      "The LTX 2.3 spatial upsampler from models/latent_upscale_models (the official two-stage workflows use the same file). 'auto' picks the newest installed spatial upscaler — or downloads the official one (~1 GB, once) when the folder is empty."));
    if (_detailUpsamplerChoices.length <= 1) {
      g.append(hintEl("Nothing installed in models/latent_upscale_models yet — the first detailed run downloads the official upsampler automatically (watch the ComfyUI console)."));
    }
  }

  // ── Best-FaceID ArcFace projector: the one identity-transfer field that needs a
  // live (server-fetched) choice list, so it can't live in the static SAMPLER_KNOBS combo
  // shape. Reuses the same loras folder listing LoraLoaderModelOnly exposes.
  let _loraChoices = null;

  async function ensureLoraChoices() {
    if (_loraChoices) return _loraChoices;
    try {
      const spec = await API.nodeSpec("LoraLoaderModelOnly");
      const w = (spec?.inputs || []).find((i) => i.name === "lora_name");
      _loraChoices = (w && w.choices) || [];
    } catch (_) {
      _loraChoices = [];
    }
    render();
    return _loraChoices;
  }

  function renderChainExperimental(pane, st) {
    renderChainKnobsView(pane, st, "chain_experimental", "Experimental",
      "Research techniques — off by default; expect quality/overhead trade-offs.");
    const si = st.project.sampler_inputs || {};
    if (si.segmented_detailing) {
      renderDetailUpsampler(pane, si, "Segmented detailing: upsampler model");
    }
    if (!si.identity_transfer_enabled) return;
    const g = group(pane, "Best-FaceID: ArcFace projector");
    if (!_loraChoices) {
      g.append(hintEl("Loading projector list…"));
      ensureLoraChoices();
      return;
    }
    const sel = el("select"); sel.dataset.k = "si-identity_projector";
    const noneOpt = el("option", null, "None"); noneOpt.value = "None";
    if (!si.identity_projector || si.identity_projector === "None") noneOpt.selected = true;
    sel.append(noneOpt);
    _loraChoices.forEach((c) => {
      const o = el("option", null, c); o.value = c;
      if (c === si.identity_projector) o.selected = true;
      sel.append(o);
    });
    sel.onchange = () => S.setSamplerInputNow("identity_projector", sel.value);
    g.append(field("ArcFace projector", sel,
      "Optional secondary identity channel — the overlap reference tokens above carry the bulk of identity even with this set to None."));
  }

  function renderPane(pane, st) {
    const p = st.project;
    if (!p) { pane.append(el("div", "pj-meta", "No project open.")); return; }
    if (st.models?.disable_core) {
      pane.append(hintEl("Built-in FunPack pipeline is disabled — Studio and Chain Sampler settings are unavailable. Tune your workflow in Models & Pipeline."));
      const link = el("button", "btn ghost tiny", "Models & Pipeline…");
      link.type = "button";
      link.onclick = () => window.ModelsModal.open();
      pane.append(link);
      return;
    }
    switch (view) {
      case "studio_refine": return renderStudioRefine(pane, st);
      case "studio_adjust": return renderStudioAdjust(pane, st);
      case "studio_sampler": return renderStudioSampler(pane, st);
      case "chain_continuity": return renderChainContinuity(pane, st);
      case "chain_references": return renderChainReferences(pane, st);
      case "chain_timing": return renderChainTiming(pane, st);
      case "chain_guidance": return renderChainKnobsView(pane, st, "chain_guidance", "Guidance",
        EASY ? "Rating-dependent guidance (embed guidance, score slider, output guidance, taste retrieval, "
          + "DynaShift) is hidden here — use the Cutting Room or ComfyUI graph for those." : null);
      case "chain_decode": return renderChainKnobsView(pane, st, "chain_decode", "Decode");
      case "chain_experimental": return renderChainExperimental(pane, st);
      default: return renderOverview(pane, st);
    }
  }

  function renderContent(container, st) {
    // No project / built-in pipeline off: single message pane, no category sidebar.
    if (!st.project || st.models?.disable_core) {
      const solo = el("div", "models-pane");
      renderPane(solo, st);
      container.append(solo);
      return;
    }
    const views = viewList(st);
    if (!views.some((v) => v.id === view)) view = "overview";
    const cols = el("div", "models-cols");
    const side = el("div", "models-side");
    let lastGroup = null;
    views.forEach((v) => {
      if (v.group && v.group !== lastGroup) side.append(el("div", "mn-group", v.group));
      lastGroup = v.group || lastGroup;
      side.append(window.SettingsWindow.navItem({
        icon: v.icon, label: v.title, badge: v.badge,
        active: view === v.id, onClick: () => setView(v.id),
      }));
    });
    const pane = el("div", "models-pane eng-pane");
    renderPane(pane, st);
    cols.append(side, pane);
    container.append(cols);
  }

  // Protect a text/number/range field being typed into from autosave rebuilds, but let
  // checkboxes and selects rebuild immediately so their dependent controls (absolute
  // strength, embed mode/strength, mid-scene strength, …) appear right after the toggle.
  function shouldProtect(a, scope) {
    if (!a || !a.dataset || !a.dataset.k || !scope || !scope.contains(a)) return false;
    if (a.tagName === "TEXTAREA") return true;
    if (a.tagName !== "INPUT") return false;
    const t = (a.type || "text").toLowerCase();
    return t !== "checkbox" && t !== "radio";
  }

  function render() {
    if (!_mounted) return;
    const { content } = _mounted;
    if (_editing) {
      if (shouldProtect(document.activeElement, content)) return;
      _editing = false;
    }
    const prevPane = content.querySelector(".models-pane");
    const scrollTop = prevPane ? prevPane.scrollTop : 0;
    clear(content);
    renderContent(content, S.get());
    const pane = content.querySelector(".models-pane");
    if (pane) pane.scrollTop = scrollTop;
  }

  function mount(body) {
    const content = el("div", "models-mount eng-mount");
    body.append(content);
    _mounted = { content };
    view = "overview";

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

    unsub = S.subscribe(() => render());
    render();
    return () => {
      if (unsub) { unsub(); unsub = null; }
      _mounted = null;
      _editing = false;
    };
  }

  window.SettingsWindow.register({
    id: "engine", group: "Generation", order: 1, title: "Engine", flush: true,
    subtitle: "FunPack Studio, Chain Sampler, and continuity for the open project.",
    keywords: "studio chain sampler refinement key seed cfg guidance continuity guides presets "
      + "embed dynashift joyai decode transition overlap steer adjustments temporal",
    iconBg: "linear-gradient(180deg,#ffb64d,#e07f1f)",
    icon: '<svg viewBox="0 0 16 16" width="13" height="13"><path d="M9.2 1.3 3 9h4.1l-1 5.7L12.9 7H8.5l.7-5.7z" fill="#fff"/></svg>',
    mount,
  });

  window.EngineSettingsModal = {
    open: () => window.SettingsWindow.open("engine"),
    close: () => window.SettingsWindow.close(),
  };
})();
