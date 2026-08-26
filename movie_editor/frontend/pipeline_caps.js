// Shared FunPack pipeline capability flags (Studio / Chain Sampler availability).
(function () {
  const CHAIN_ONLY = new Set(["carry", "mixed", "generated_frame", "v2v", "anchor_guide"]);

  // The Editor keeps the live model config at state.models (synced into project.models on
  // commit); Easy Gen has no separate copy and keeps it on the project. Read both, or every
  // family-dependent answer here silently reads as LTX in Easy Gen.
  function models(st) {
    if (!st) return {};
    return st.models || (st.project && st.project.models) || {};
  }

  function usesFunpackStudio(st) {
    if (models(st).disable_core) return false;
    const slot = (st && st.project && st.project.conditioning_slot) || "funpack";
    return slot === "funpack";
  }

  function usesChainSampler(st) {
    if (models(st).disable_core) return false;
    const slot = (st && st.project && st.project.sampler_slot) || "funpack";
    return slot === "funpack";
  }

  // Read the family from whichever config actually carries it. `models()` returns the FIRST
  // truthy of state.models / project.models, and state.models is {slots: []} both while the
  // live config is still loading and after a failed fetch — truthy, but with no family — so
  // taking it from there alone reports LTXAV for an H3 project during that window, and every
  // family-dependent answer here (frame grid, fps, inert-setting chips) is wrong with it.
  function modelFamily(st) {
    const live = st && st.models;
    const saved = st && st.project && st.project.models;
    const raw = (live && live.model_family) || (saved && saved.model_family) || "ltxav";
    const f = String(raw).toLowerCase();
    return f === "minimax_h3" ? "minimax_h3" : "ltxav";
  }

  function isH3(st) {
    return modelFamily(st) === "minimax_h3";
  }

  // Frame geometry per family. LTX generates on an 8k+1 pixel-frame grid at whatever fps
  // the project says; MiniMax H3 generates on a 17k+5 grid and ALWAYS at 24 fps (the rate
  // is baked into the model, not a conditioning field). Both frontends snap through here so
  // a project can never ask for a length the chosen model cannot produce.
  const FRAME_GRIDS = {
    ltxav: { step: 8, base: 1, fps: null, label: "8k+1" },
    minimax_h3: { step: 17, base: 5, fps: 24, label: "17k+5" },
  };

  function frameGrid(st) {
    return FRAME_GRIDS[modelFamily(st)] || FRAME_GRIDS.ltxav;
  }

  // round | floor | ceil onto the family's grid. Always >= one whole grid step.
  function snapFramesTo(n, st, mode) {
    const g = frameGrid(st);
    const min = g.step + g.base;
    const k = (Math.round(Number(n) || 0) - g.base) / g.step;
    const rounded = mode === "floor" ? Math.floor(k) : mode === "ceil" ? Math.ceil(k) : Math.round(k);
    return Math.max(min, rounded * g.step + g.base);
  }

  // What a frames INPUT should do while the user is using it. The two families are not
  // equally strict, and treating them as if they were is what made the field unusable:
  //   H3  — off-grid is a dead run (its latent node snaps its own length up while the
  //         sampler expects the number the project asked for), so the field snaps on
  //         commit and its arrows walk WHOLE grid steps (22 -> 39 -> 56). Before this the
  //         arrows moved by 1 and the snap put the value straight back.
  //   LTX — the builder rounds an off-grid length up to 8k+1 by itself, so nothing is at
  //         risk: the field behaves like a plain number input and only its arrows follow
  //         the grid (min 1 + step 8 lands them on 9 / 17 / 25 …).
  // `min` is the arrow BASE as well as the floor, which is what puts the arrows on the grid.
  function frameInputSpec(st) {
    const g = frameGrid(st);
    const hard = isH3(st);
    return { step: g.step, min: hard ? g.step + g.base : g.base, snap: hard, grid: g };
  }

  // Snap only where the model actually demands it (H3). On LTX the typed number is kept.
  function snapFramesIfRequired(n, st, mode) {
    if (isH3(st)) return snapFramesTo(n, st, mode);
    return Math.max(1, Math.round(Number(n) || 0));
  }

  // H3 renders at a fixed 24 fps. The project's frame rate still drives the container the
  // clip is muxed into, so anything else plays the generated frames at the wrong speed —
  // the builder pins it, and this is what says so before the run.
  function frameRateIssue(st) {
    const p = st && st.project;
    const g = frameGrid(st);
    if (!p || !g.fps) return null;
    const fps = Number(p.frame_rate);
    if (!fps || Math.abs(fps - g.fps) < 0.01) return null;
    return {
      short: "H3: frame rate is fixed at " + g.fps,
      detail: "This project is set to " + fps + " fps, but MiniMax H3 always generates at "
        + g.fps + " fps — the rate is part of the model, not a setting. The render is muxed at "
        + g.fps + " fps regardless, so set FPS to " + g.fps + " in the project settings to keep "
        + "the timeline honest.",
      // a project field, not an engine setting: the chip must not send the user to the
      // Engine window, where there is no frame rate to change
      target: "project",
    };
  }

  function caps(st) {
    const m = models(st);
    return {
      studio: usesFunpackStudio(st),
      chain_sampler: usesChainSampler(st),
      disable_core: !!m.disable_core,
      imported_workflow: !!m.workflow_import,
      family: modelFamily(st),
      h3: isH3(st),
      frame_grid: frameGrid(st),
    };
  }

  // Sampler toggles that depend on an LTX transformer structure MiniMax H3 does not have.
  // The Chain Sampler switches each of these off and says why on the console, but the user
  // only sees that AFTER spending a generation — so the main window says it first.
  // key -> why it cannot run on H3.
  const H3_DEAD_SAMPLER_INPUTS = {
    identity_transfer_enabled:
      "Best-FaceID needs LTX cross-attention and its trained ArcFace projector; H3 has neither. "
      + "Use reference media instead — H3 packs references into the sequence natively.",
    bounded_attention_enabled:
      "Bounded Attention masks text cross-attention; H3 puts text in the same self-attention "
      + "stream as the video, where the mask would be far too large to hold.",
    context_windows:
      "Core's context windowing unpacks the LTXAV stream specifically — mapping each video "
      + "window onto its audio window and re-slicing the guide entries. H3 packs its sequence "
      + "by a different layout, and the window length is measured on LTX's 8x latent ratio.",
    alg_blur_guides:
      "This blurs the trailing GUIDE frames appended to the latent. On H3 a guide is a "
      + "condition row rather than an appended frame, so that tail is always empty. "
      + "(alg_anchor is NOT hidden — a continuation scene does carry real latent frames.)",
    joyai_memory:
      "JoyAI-Echo is a LoRA-driven technique: the base weights were never trained to read the "
      + "injected memory frames as memory, the LoRA is what teaches that, and no JoyAI-Echo "
      + "LoRA exists for H3. There is nothing to load, so this is inaccessible on H3 by "
      + "design, not merely unwired. It would also actively hurt output if forced on: the "
      + "bank places memory frame i at sequence position i, and writing index 0 twice "
      + "replaces whatever was there — evicting the scene's own i2v anchor.",
  };

  // Sub-settings of the two groups above: they configure machinery that cannot run here.
  const CONTEXT_SUB_INPUTS = [
    "context_window_length", "context_window_overlap", "context_window_schedule",
    "context_window_fuse", "context_window_freenoise", "context_window_retain_first",
  ];
  const ALG_GUIDE_SUB_INPUTS = ["alg_guide_blur_strength", "alg_guide_blur_sigma_threshold"];

  // JoyAI's own sub-settings: they configure a bank that cannot be built here.
  const JOYAI_SUB_INPUTS = [
    "joyai_memory_size", "joyai_fix_frames", "joyai_frame_select", "joyai_memory_strength",
    "joyai_audio_memory",
  ];

  // Same idea for non-boolean knobs whose neutral value means "off".
  // The latent ops (second_pass_op, segmented detailing) are NOT listed here any more: what
  // they need is an upsampler whose latents are the same width as the model's, which depends
  // on what is installed rather than on the family, and the sampler reports a mismatch by
  // name. A chip here would be a guess about the user's models folder.
  function h3DeadValueIssues(_si) {
    return [];
  }

  // The mirror image of H3_DEAD_SAMPLER_INPUTS: settings that only mean something ON H3.
  // Left on against an LTX pipeline they are just as silently inert, so they get a chip
  // by the same rule — the user should not have to spend a generation to find out.
  // key -> why it cannot run off H3.
  const H3_CAUSAL_REASON =
    "The chunk cache is an H3 lane: it needs the chunk-causal DiT and the RAVEN LoRA, both "
    + "of which are MiniMax H3 only. LTX has no equivalent.";

  const H3_ONLY_SAMPLER_INPUTS = {
    h3_audio_clock:
      "The audio clock corrects for H3 denoising video and audio on two different flow "
      + "schedules. LTX puts both streams on one schedule, so there is nothing to correct.",
    // The render gains scale H3's per-modality AdaLN gates and its token refiner. LTX has
    // neither — one modulation path, cross-attention instead of a packed text span — so
    // there is nothing for these to reach on any other family.
    h3_causal_chunks: H3_CAUSAL_REASON,
    h3_causal_step_rule: H3_CAUSAL_REASON,
    h3_causal_sink: H3_CAUSAL_REASON,
    h3_causal_window: H3_CAUSAL_REASON,
  };

  // Sub-settings of identity transfer: meaningless wherever the feature itself cannot run,
  // so they travel with it rather than each needing its own entry above.
  const IDENTITY_SUB_INPUTS = ["source_id", "phase_scale", "id_strength", "arcface_mode"];

  // Non-boolean knobs H3 cannot use. v2a_grad_scale hooks LTXAV's video_to_audio_attn
  // submodule; H3 has no separate video->audio cross-attention to scale, and the sampler
  // forces it back to 1.0. Its parent feature (joyai_audio_memory) DOES work on H3, so only
  // the coupling knob goes.
  const H3_DEAD_VALUE_INPUTS = ["v2a_grad_scale"];

  // The sampler inputs the LOADED model cannot use. Engine Settings hides these outright
  // rather than offering a control and explaining once per run that it does nothing — a
  // toggle that is not offered needs no explanation. The stored value is left untouched, so
  // switching family back restores the user's setting instead of silently resetting it.
  //
  // Safe because the sampler already forces every one of these off for the family in
  // question, so hiding removes a control that was inert either way — it never makes a
  // working knob unreachable.
  function familyInertInputs(st) {
    if (!usesChainSampler(st)) return new Set();
    if (!isH3(st)) return new Set(Object.keys(H3_ONLY_SAMPLER_INPUTS));
    return new Set([...Object.keys(H3_DEAD_SAMPLER_INPUTS), ...IDENTITY_SUB_INPUTS,
                    ...JOYAI_SUB_INPUTS, ...CONTEXT_SUB_INPUTS, ...ALG_GUIDE_SUB_INPUTS,
                    ...H3_DEAD_VALUE_INPUTS]);
  }

  // Returns [{short, detail}] for settings that are ON but cannot do anything on H3.
  function h3InertSettings(st) {
    const p = st && st.project;
    if (!p || !usesChainSampler(st)) return [];
    // Everything family-inert is now HIDDEN from Engine Settings, so a chip here would
    // point at a control the user cannot find ("turn it off in Settings → Engine" names a
    // row that is not rendered). What is left is the one issue hiding cannot express: a
    // frame rate the model will not honour, which is a value the user must change.
    const out = [];
    const fps = frameRateIssue(st);
    if (fps) out.push(fps);
    return out;
  }

  function effectiveSourceType(scene, st) {
    const t = (scene && scene.source && scene.source.type) || "carry";
    if (usesChainSampler(st)) return t;
    if (t === "image" && scene.source && scene.source.media_ref) return "image";
    if (t === "v2v" && scene.source && scene.source.media_ref) return "v2v";
    return "empty";
  }

  function isChainOnlySource(type) {
    return CHAIN_ONLY.has(type);
  }

  // "t2v" = shots start from the prompt. Mirrors pipeline_caps.is_t2v on the backend.
  function isT2V(st) {
    return String(st?.project?.generation_mode || "i2v").toLowerCase() === "t2v";
  }

  function defaultSceneSourceType(st) {
    if (isT2V(st)) return usesChainSampler(st) ? "carry" : "empty";
    return usesChainSampler(st) ? "carry" : "image";
  }

  // Mirrors pipeline_caps.source_needs_anchor_media on the backend. Keep the two in
  // step: the backend decides what the run reports, this decides what the inspector
  // warns about, and they should never disagree about the same scene.
  const ANCHOR_MEDIA_SOURCES = new Set(["image", "mixed", "generated_frame", "v2v", "anchor_guide"]);

  function sourceNeedsAnchorMedia(scene, st) {
    if (isT2V(st)) return false;   // a t2v project expects no anchors
    return ANCHOR_MEDIA_SOURCES.has(effectiveSourceType(scene, st));
  }

  function isMissingAnchorMedia(scene, st) {
    if (!scene || !scene.source || scene.source.media_ref) return false;
    return sourceNeedsAnchorMedia(scene, st);
  }

  // Best-FaceID reads identity from an identity_pin-tagged guide, and ONLY the auto
  // continuity builder ever applies that tag (backend timeline._identity_pin_guide).
  // Every way of losing it is silent — the run succeeds carrying no identity at all —
  // so the main window says so before you spend a generation finding out.
  // Returns {short, detail} for a compact chip and a full-width strip respectively,
  // or null when the setting will actually do something.
  function identityTransferIssue(st) {
    const p = st && st.project;
    if (!p || !(p.sampler_inputs || {}).identity_transfer_enabled) return null;
    if (!usesChainSampler(st)) return null;   // knob is inert without the Chain Sampler
    if (isH3(st)) return null;                // h3InertSettings already reports this one
    const cs = p.continuity_settings || {};
    const gs = p.guide_settings || {};
    if (!cs.identity_pin_ref) {
      return {
        short: "Best-FaceID: no identity pin",
        detail: "You have Best-FaceID enabled but no identity pin is set — it has no face to transfer. "
          + "Set the identity pin in Settings → Engine → Continuity, or press 📌 on an image in the Media bin.",
      };
    }
    if (gs.stack_enabled) {
      return {
        short: "Best-FaceID: pin not reaching sampler",
        detail: "You have Best-FaceID enabled and an identity pin set, but the custom guide stack replaces "
          + "the auto-continuity guides that carry the pin — so the pin never reaches the sampler. "
          + "Turn the custom guide stack off in Settings → Engine → Continuity.",
      };
    }
    if (cs.auto_enabled === false) {
      return {
        short: "Best-FaceID: pin not reaching sampler",
        detail: "You have Best-FaceID enabled and an identity pin set, but auto continuity is off — the pin "
          + "only travels with auto-continuity guides. Turn auto continuity on in Settings → Engine → Continuity.",
      };
    }
    return null;
  }

  // Studio needs ONE positive-conditioning source: a CLIP it encodes the prompt with, or a
  // finished CONDITIONING wired into positive_conditioning (encoded somewhere else — a
  // deliberate setup, not a mistake, and nothing to warn about). Only having NEITHER is
  // broken: there is no way for any prompt to reach the model.
  const STUDIO_COND_PORT = "port:FunPackStudio.positive_conditioning";
  const STUDIO_CLIP_PORT = "port:FunPackStudio.clip";

  function anySlotWiresTo(m, port) {
    return (m.slots || []).some((s) => Object.values((s && s.wires) || {})
      .some((t) => (Array.isArray(t) ? t : [t]).includes(port)));
  }

  function promptSourceIssue(st) {
    const m = models(st);
    if (m.disable_core || !usesFunpackStudio(st)) return null;
    const ov = (m.core_overrides && m.core_overrides.studio) || {};
    if (ov.positive_conditioning || anySlotWiresTo(m, STUDIO_COND_PORT)) return null;
    if (ov.clip || anySlotWiresTo(m, STUDIO_CLIP_PORT)) return null;
    // No explicit wire either way: a slot in a CLIP-carrying role still gets one, through
    // its role default or auto-wire. The roles that carry CLIP are the text encoder and a
    // LoRA passing one through.
    if ((m.slots || []).some((s) => s.role === "clip" || s.role === "lora")) return null;
    return {
      short: "Nothing encodes your prompt",
      detail: "Studio has no text encoder on its clip input and no pre-encoded CONDITIONING on "
        + "positive_conditioning, so nothing turns your prompt into something the model reads. "
        + "Add a CLIP / Text Encoder in Models, or wire a node that outputs CONDITIONING into "
        + "Studio · positive_conditioning.",
      target: "models",
    };
  }

  function sourceLabel(type) {
    const map = {
      empty: "Empty · text-to-video",
      image: "Image · i2v anchor",
      generated_frame: "From generated frame",
      v2v: "Video · v2v source",
      carry: "Carry i2v guide · continue previous",
      mixed: "Mixed · Img2Video anchor + prior guides",
      anchor_guide: "Anchor as guide · image steers, empty latent",
    };
    return map[type] || type;
  }

  const api = {
    usesFunpackStudio,
    usesChainSampler,
    caps,
    modelFamily,
    isH3,
    frameGrid,
    snapFramesTo,
    frameInputSpec,
    snapFramesIfRequired,
    frameRateIssue,
    h3InertSettings,
    familyInertInputs,
    effectiveSourceType,
    isChainOnlySource,
    isT2V,
    defaultSceneSourceType,
    sourceNeedsAnchorMedia,
    isMissingAnchorMedia,
    identityTransferIssue,
    promptSourceIssue,
    sourceLabel,
  };
  // Node can require this for the pure predicates (familyInertInputs, snapFramesTo, …);
  // the browser keeps the window global. `window` does not exist under Node, so neither
  // assignment may assume the other's environment.
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.PipelineCaps = api;
})();
