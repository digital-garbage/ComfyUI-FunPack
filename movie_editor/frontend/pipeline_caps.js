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

  function modelFamily(st) {
    const f = String(models(st).model_family || "ltxav").toLowerCase();
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
    segmented_detailing:
      "Segmented detailing uses Lightricks' latent upsampler, trained on LTX's 128-channel "
      + "latent; H3's latent has 24 channels.",
  };

  // Same idea for non-boolean knobs whose neutral value means "off".
  function h3DeadValueIssues(si) {
    const out = [];
    if (si.second_pass && si.second_pass_op && si.second_pass_op !== "none") {
      out.push(["second_pass_op",
        "The second pass still runs on H3, but its latent op ('" + si.second_pass_op + "') uses the "
        + "same LTX-only upsampler, so the op is skipped."]);
    }
    return out;
  }

  // The mirror image of H3_DEAD_SAMPLER_INPUTS: settings that only mean something ON H3.
  // Left on against an LTX pipeline they are just as silently inert, so they get a chip
  // by the same rule — the user should not have to spend a generation to find out.
  // key -> why it cannot run off H3.
  const H3_ONLY_SAMPLER_INPUTS = {
    h3_audio_clock:
      "The audio clock corrects for H3 denoising video and audio on two different flow "
      + "schedules. LTX puts both streams on one schedule, so there is nothing to correct.",
  };

  // Returns [{short, detail}] for settings that are ON but cannot do anything on H3.
  function h3InertSettings(st) {
    const p = st && st.project;
    if (!p || !usesChainSampler(st)) return [];
    const si = p.sampler_inputs || {};
    if (!isH3(st)) {
      return Object.keys(H3_ONLY_SAMPLER_INPUTS)
        .filter((key) => si[key])
        .map((key) => ({
          short: key.replace(/_/g, " ") + " needs MiniMax H3",
          detail: H3_ONLY_SAMPLER_INPUTS[key]
            + " Turn it off in Settings → Engine to stop it showing here.",
        }));
    }
    const issues = [];
    Object.keys(H3_DEAD_SAMPLER_INPUTS).forEach((key) => {
      if (si[key]) issues.push([key, H3_DEAD_SAMPLER_INPUTS[key]]);
    });
    issues.push(...h3DeadValueIssues(si));
    const out = issues.map(([key, detail]) => ({
      short: "H3: " + key.replace(/_enabled$/, "").replace(/_/g, " ") + " can't run",
      detail: detail + " Turn it off in Settings → Engine to stop it showing here.",
    }));
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

  function defaultSceneSourceType(st) {
    // t2v ("empty") is no longer a user-facing mode — new scenes default to an i2v
    // anchor (image). Anchorless scenes still fall back to t2v in the engine.
    return usesChainSampler(st) ? "carry" : "image";
  }

  // Mirrors pipeline_caps.source_needs_anchor_media on the backend. Keep the two in
  // step: the backend decides what the run reports, this decides what the inspector
  // warns about, and they should never disagree about the same scene.
  const ANCHOR_MEDIA_SOURCES = new Set(["image", "mixed", "generated_frame", "v2v", "anchor_guide"]);

  function sourceNeedsAnchorMedia(scene, st) {
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

  window.PipelineCaps = {
    usesFunpackStudio,
    usesChainSampler,
    caps,
    modelFamily,
    isH3,
    frameGrid,
    snapFramesTo,
    frameRateIssue,
    h3InertSettings,
    effectiveSourceType,
    isChainOnlySource,
    defaultSceneSourceType,
    sourceNeedsAnchorMedia,
    isMissingAnchorMedia,
    identityTransferIssue,
    sourceLabel,
  };
})();
