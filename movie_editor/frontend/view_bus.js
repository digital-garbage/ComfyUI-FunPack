// Selective view updates: skip full zone rebuilds when relevant state is unchanged.
(function () {
  function hash(obj) {
    try { return JSON.stringify(obj); } catch (_) { return String(Date.now()); }
  }

  function fpTimelineData(st) {
    if (!st.project) return "none";
    const p = st.project;
    return hash({
      pid: p.id,
      scenes: (p.scenes || []).map((s) => [
        s.id, s.text, s.excluded, s.frames, s.fps, s.frames_mode, s.fps_mode,
        s.video_transition, s.transition_frames, s.transition_to_next, s.effects,
        s.gen_unit_id, s.cut_offset_frames, s.source?.type, s.source?.media_ref,
        s.source_in, s.source_dur, s.rating, s.audio_volume,
      ]),
      renders: st.sceneRenders,
      ghosts: st.sceneGhosts,
      audio: p.audio_tracks,
      keepAud: p.keep_original_audio,
      gen: st.gen?.state,
      labels: (st.ratingLabels || []).length,
      notice: st.notice || "",
    });
  }

  function fpTimelineSel(st) {
    return hash({ sel: st.selectedSceneId, sels: st.selectedSceneIds });
  }

  function fpTimeline(st) {
    return hash({ d: fpTimelineData(st), s: fpTimelineSel(st) });
  }

  function fpInspector(st) {
    if (!st.project) return "none";
    const p = st.project;
    const sc = st.selectedSceneId ? p.scenes.find((s) => s.id === st.selectedSceneId) : null;
    return hash({
      pid: p.id,
      sel: st.selectedSceneId,
      sels: st.selectedSceneIds,
      scene: sc,
      proj: sc ? null : {
        name: p.name, anchor: p.anchor, global_prompt: p.global_prompt,
        negative_prompt: p.negative_prompt, intro_transition: p.intro_transition,
        seed: p.seed, num_frames_per_scene: p.num_frames_per_scene,
        frame_rate: p.frame_rate, max_scenes: p.max_scenes, width: p.width, height: p.height,
        conditioning_slot: p.conditioning_slot, sampler_slot: p.sampler_slot,
        studio_inputs: p.studio_inputs, sampler_inputs: p.sampler_inputs,
        guide_settings: p.guide_settings, continuity_settings: p.continuity_settings,
      },
      preview: st.preview ? {
        warning: st.preview.warning, parse_error: st.preview.parse_error,
        scene_count: st.preview.parsed?.scenes?.length,
      } : null,
      models: st.models,
      chars: st.characters?.length,
      media: st.mediaBin?.length,
      targets: st.imageTargets?.length,
    });
  }

  function fpPlayer(st) {
    if (!st.project) return "none";
    return hash({
      pid: st.project.id,
      scenes: (st.project.scenes || []).map((s) => [s.id, s.excluded, s.frames, s.fps, s.frames_mode]),
      renders: st.sceneRenders,
      ghosts: st.sceneGhosts,
      gen: { state: st.gen?.state, media: st.gen?.media },
      mediaPreview: st.mediaPreviewId,
      fps: st.project.frame_rate,
    });
  }

  function fpActionbar(st) {
    return hash({
      pid: st.project?.id,
      sel: st.selectedSceneIds?.length,
      gen: st.gen?.state,
    });
  }

  function fpMediabrowser(st) {
    return hash({
      pid: st.project?.id,
      projects: st.projects?.length,
      media: st.mediaBin?.length,
      chars: st.characters?.length,
      shortcuts: st.shortcuts?.length,
      transitions: st.transitions?.length,
      nleEffects: st.nleEffects?.length,
      nleVideoTransitions: st.nleVideoTransitions?.length,
    });
  }

  function subscribeZone(name, fn, fpFn) {
    const S = window.Store;
    if (!S) return () => {};
    let last = null;
    return S.subscribe((st) => {
      const fp = fpFn(st);
      if (fp === last) return;
      last = fp;
      fn(st);
    });
  }

  window.ViewBus = {
    subscribeTimeline: (fn) => subscribeZone("timeline", fn, fpTimeline),
    subscribeInspector: (fn) => subscribeZone("inspector", fn, fpInspector),
    subscribePlayer: (fn) => subscribeZone("player", fn, fpPlayer),
    subscribeActionbar: (fn) => subscribeZone("actionbar", fn, fpActionbar),
    subscribeMediabrowser: (fn) => subscribeZone("mediabrowser", fn, fpMediabrowser),
    fingerprints: { fpTimeline, fpTimelineData, fpTimelineSel, fpInspector, fpPlayer, fpActionbar, fpMediabrowser },
  };
})();
