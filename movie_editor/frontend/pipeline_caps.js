// Shared FunPack pipeline capability flags (Studio / Chain Sampler availability).
(function () {
  const CHAIN_ONLY = new Set(["carry", "mixed", "generated_frame", "v2v", "anchor_guide"]);

  function models(st) {
    return (st && st.models) || {};
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

  function caps(st) {
    const m = models(st);
    return {
      studio: usesFunpackStudio(st),
      chain_sampler: usesChainSampler(st),
      disable_core: !!m.disable_core,
      imported_workflow: !!m.workflow_import,
      family: modelFamily(st),
      h3: isH3(st),
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
    if (Math.abs(Number(si.v2a_grad_scale ?? 1) - 1) > 1e-6) {
      out.push(["v2a_grad_scale",
        "v2a_grad_scale scales LTXAV's video→audio cross-attention module; H3 has no separate "
        + "video→audio attention, only joint rows."]);
    }
    if (si.second_pass && si.second_pass_op && si.second_pass_op !== "none") {
      out.push(["second_pass_op",
        "The second pass still runs on H3, but its latent op ('" + si.second_pass_op + "') uses the "
        + "same LTX-only upsampler, so the op is skipped."]);
    }
    return out;
  }

  // Returns [{short, detail}] for settings that are ON but cannot do anything on H3.
  function h3InertSettings(st) {
    const p = st && st.project;
    if (!p || !isH3(st) || !usesChainSampler(st)) return [];
    const si = p.sampler_inputs || {};
    const issues = [];
    Object.keys(H3_DEAD_SAMPLER_INPUTS).forEach((key) => {
      if (si[key]) issues.push([key, H3_DEAD_SAMPLER_INPUTS[key]]);
    });
    issues.push(...h3DeadValueIssues(si));
    return issues.map(([key, detail]) => ({
      short: "H3: " + key.replace(/_enabled$/, "").replace(/_/g, " ") + " can't run",
      detail: detail + " Turn it off in Settings → Engine to stop it showing here.",
    }));
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
