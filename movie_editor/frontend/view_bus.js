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
      timing: [p.num_frames_per_scene, p.frame_rate, p.width, p.height],
      tlorder: p.timeline_order,
      scenes: (p.scenes || []).map((s) => [
        s.id, s.text, s.excluded, s.frames, s.fps, s.frames_mode, s.fps_mode,
        s.video_transition, s.transition_frames, s.transition_to_next, s.effects,
        s.gen_unit_id, s.cut_offset_frames, s.source?.type, s.source?.media_ref,
        s.source_in, s.source_dur, s.rating, s.audio_volume, s.audio_separated, s.scene_archive,
        s.gap_after_sec,
      ]),
      renders: st.sceneRenders,
      ghosts: st.sceneGhosts,
      audio: p.audio_tracks,
      overlays: p.overlay_tracks,
      ovLanes: p.overlay_lanes,
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
      overlay: st.selectedOverlayId,
      overlays: p.overlay_tracks,
      ovLanes: p.overlay_lanes,
      timing: [p.num_frames_per_scene, p.frame_rate, p.width, p.height],
      scene: sc,
      proj: sc ? null : {
        name: p.name, anchor: p.anchor, global_prompt: p.global_prompt,
        negative_prompt: p.negative_prompt, postfix: p.postfix, postfix_enabled: p.postfix_enabled,
        intro_transition: p.intro_transition,
        seed: p.seed, num_frames_per_scene: p.num_frames_per_scene,
        frame_rate: p.frame_rate, max_scenes: p.max_scenes, width: p.width, height: p.height,
        conditioning_slot: p.conditioning_slot, sampler_slot: p.sampler_slot,
        studio_inputs: p.studio_inputs, sampler_inputs: p.sampler_inputs,
        guide_settings: p.guide_settings, continuity_settings: p.continuity_settings,
        // The Project panel warns when its Frames field reaches no shot, so it depends on
        // scene state even with no scene selected — without this the warning stays on screen
        // after "Use project length everywhere" has already cleared it.
        ownLength: (p.scenes || []).filter((s) => (s.frames_mode || "project") !== "project").length,
      },
      preview: st.preview ? {
        warning: st.preview.warning, parse_error: st.preview.parse_error,
        scene_count: st.preview.parsed?.scenes?.length,
      } : null,
      models: st.models,
      media: st.mediaBin?.length,
      targets: st.imageTargets?.length,
    });
  }

  function fpPlayer(st) {
    if (!st.project) return "none";
    const p = st.project;
    // While the media bin preview overlay is open, ignore timeline/render churn so the
    // preview video/image is not torn down on every generation tick.
    if (st.mediaPreviewId) {
      const m = (st.mediaBin || []).find((x) => x.id === st.mediaPreviewId);
      return hash({
        pid: p.id,
        mediaPreview: st.mediaPreviewId,
        previewAsset: m ? [m.id, m.kind, m.name] : null,
        gen: { state: st.gen?.state, msg: st.gen?.msg },
      });
    }
    return hash({
      pid: p.id,
      timing: [p.num_frames_per_scene, p.frame_rate],
      tlorder: p.timeline_order,
      scenes: (p.scenes || []).map((s) => [
        s.id, s.excluded, s.frames, s.fps, s.frames_mode, s.fps_mode,
        s.audio_volume, s.audio_separated, s.source_in, s.source_dur, s.effects,
        s.source?.type, s.source?.media_ref, s.gap_after_sec,
      ]),
      renders: st.sceneRenders,
      ghosts: st.sceneGhosts,
      audio: p.audio_tracks,
      keepAud: p.keep_original_audio,
      gen: { state: st.gen?.state, media: st.gen?.media },
      mediaPreview: st.mediaPreviewId,
      fps: p.frame_rate,
    });
  }

  function fpOverlays(st) {
    if (!st.project) return "none";
    return hash({
      pid: st.project.id,
      overlays: st.project.overlay_tracks,
      ovLanes: st.project.overlay_lanes,
      sel: st.selectedOverlayId,
      mediaPreview: st.mediaPreviewId,
    });
  }

  function fpActionbar(st) {
    return hash({
      pid: st.project?.id,
      sel: st.selectedSceneIds?.length,
      gen: st.gen?.state,
      // The Best-FaceID warning chip lives in this zone, so everything its message
      // depends on has to be part of the fingerprint — otherwise setting the pin
      // leaves a stale warning sitting next to Generate.
      idOn: st.project?.sampler_inputs?.identity_transfer_enabled,
      idPin: st.project?.continuity_settings?.identity_pin_ref,
      idAuto: st.project?.continuity_settings?.auto_enabled,
      idStack: st.project?.guide_settings?.stack_enabled,
      sampler: st.project?.sampler_slot,
      // The MiniMax H3 "this setting can't run" chips live in the same zone, so every
      // input they read belongs here too — switching the family or turning one of these
      // off must clear its chip, not leave it sitting next to Generate. That includes
      // the models config every chip is gated on, read the same way PipelineCaps reads
      // it (state.models in the Editor, project.models otherwise).
      family: (st.models || st.project?.models)?.model_family,
      core: (st.models || st.project?.models)?.disable_core,
      h3Bounded: st.project?.sampler_inputs?.bounded_attention_enabled,
      h3Detail: st.project?.sampler_inputs?.segmented_detailing,
      h3Pass2: st.project?.sampler_inputs?.second_pass,
      h3Pass2Op: st.project?.sampler_inputs?.second_pass_op,
      // H3-ONLY settings get a chip when the pipeline is NOT H3, so this belongs here for
      // exactly the same reason as the ones above — in the mirror direction.
      // ... including the project's frame rate, which the H3 fixed-24-fps chip reads.
      h3Fps: st.project?.frame_rate,
      // The "nothing encodes your prompt" chip reads the whole models WIRING, which is
      // far too big to hash field by field — track the verdict itself instead, so rewiring
      // Studio's clip / positive_conditioning in Models repaints this zone and nothing else
      // about the config does.
      promptChip: window.PipelineCaps?.promptSourceIssue?.(st)?.short,
    });
  }

  function fpMediabrowser(st) {
    return hash({
      pid: st.project?.id,
      projects: st.projects?.length,
      media: st.mediaBin?.length,
      mediaPreview: st.mediaPreviewId,
      // Continuity pin renders on gallery cards (📌 button state + thumb badge).
      pin: st.project?.continuity_settings?.identity_pin_ref,
      // Reference marks render the same way (R button + numbered badge), and their ORDER
      // is the numbering — so the fingerprint has to follow the list, not just its length.
      refs: (st.project?.references || []).join(","),
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
      // Every zone builds different content in Simple mode, so the mode belongs in every
      // fingerprint. Folded in here rather than in each fpFn — one place, no way to forget
      // it in the next one added.
      const fp = (window.FunPackMode ? window.FunPackMode.get() + "|" : "") + fpFn(st);
      if (fp === last) return;
      const ok = fn(st);
      if (ok === false) return; // handler declined to apply this update — retry next notify
      last = fp;
    });
  }

  window.ViewBus = {
    subscribeTimeline: (fn) => subscribeZone("timeline", fn, fpTimeline),
    subscribeInspector: (fn) => subscribeZone("inspector", fn, fpInspector),
    subscribePlayer: (fn) => subscribeZone("player", fn, fpPlayer),
    subscribeOverlays: (fn) => subscribeZone("overlays", fn, fpOverlays),
    subscribeActionbar: (fn) => subscribeZone("actionbar", fn, fpActionbar),
    subscribeMediabrowser: (fn) => subscribeZone("mediabrowser", fn, fpMediabrowser),
    fingerprints: { fpTimeline, fpTimelineData, fpTimelineSel, fpInspector, fpPlayer, fpOverlays, fpActionbar, fpMediabrowser },
  };
})();
