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

  function caps(st) {
    const m = models(st);
    return {
      studio: usesFunpackStudio(st),
      chain_sampler: usesChainSampler(st),
      disable_core: !!m.disable_core,
      imported_workflow: !!m.workflow_import,
    };
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
    effectiveSourceType,
    isChainOnlySource,
    defaultSceneSourceType,
    sourceNeedsAnchorMedia,
    isMissingAnchorMedia,
    sourceLabel,
  };
})();
