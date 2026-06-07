// Central state + data actions. Views subscribe and re-render their zone.
(function () {
  const API = window.MovieEditorAPI;

  const state = {
    projects: [],          // [{id,name,scene_count,updated_at}]
    project: null,         // full Project
    selectedSceneId: null,
    transitions: [],       // library [{trigger/name, visual_effect}]
    health: null,          // {ok, comfy_url, template_exists}
    preview: null,         // {combined_prompt, parsed, warning, parse_error}
    gen: { state: "idle", promptId: null, media: [], msg: "" },
    // Per-scene render mapping: { sceneId: { media, inSec } } — `inSec` is the IN-point
    // within that source video this clip starts at (so split halves / deleted clips play
    // the correct portion). Not persisted to disk; cleared on project switch.
    sceneRenders: {},
    saving: false,
    models: { slots: [] },   // pluggable node config (shared with Models modal)
    mediaBin: [],            // uploaded assets [{id,name,kind,...}]
    shortcuts: [],           // prompt shortcut library
    imageTargets: [],        // where an image asset can be wired [{value,label}]
    ratingLabels: [],        // FunPack Studio V2 rating options
  };

  const listeners = new Set();
  let saveTimer = null;

  function notify() { listeners.forEach((fn) => { try { fn(state); } catch (e) { console.error(e); } }); }
  function subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }
  function set(patch) { Object.assign(state, patch); notify(); }
  function get() { return state; }

  // ── project lifecycle ──────────────────────────────────────────────────────
  async function refreshProjectList() {
    const { projects } = await API.listProjects();
    state.projects = projects; notify();
  }

  async function loadProject(id) {
    state.project = await API.getProject(id);
    state.selectedSceneId = state.project.scenes[0]?.id || null;
    state.gen = { state: "idle", promptId: null, media: [], msg: "" };
    state.sceneRenders = {};  // per-session; not persisted
    notify();
    refreshPreview();
    loadModels();  // models config is per-project — reload for this project
  }

  async function newProject(name) {
    const p = await API.createProject(name || "Untitled montage");
    await refreshProjectList();
    await loadProject(p.id);
  }

  async function deleteProject(id) {
    await API.deleteProject(id);
    state.project = null; state.selectedSceneId = null;
    await refreshProjectList();
    if (state.projects[0]) await loadProject(state.projects[0].id);
    else notify();
  }

  function downloadProject() {
    if (!state.project) return;
    const a = document.createElement("a");
    a.href = API.downloadProjectUrl(state.project.id);
    a.download = "";
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
  }

  async function importProject(file) {
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      const p = await API.importProject(data);
      await refreshProjectList();
      await loadProject(p.id);
    } catch (e) { alert("Import failed: " + e.message); }
  }

  // ── conditioning / sampler role ────────────────────────────────────────────────
  function setConditioningSlot(slotId) {
    patchProject({ conditioning_slot: slotId || "funpack" });
  }
  function setSamplerSlot(slotId) {
    patchProject({ sampler_slot: slotId || "funpack" });
  }
  function setSamplerInput(name, value) {
    if (!state.project) return;
    const prev = state.project.sampler_inputs || {};
    patchProjectQuiet({ sampler_inputs: { ...prev, [name]: value } });
  }
  function setSamplerInputNow(name, value) {
    if (!state.project) return;
    const prev = state.project.sampler_inputs || {};
    patchProject({ sampler_inputs: { ...prev, [name]: value } });
  }
  function setStudioInput(name, value) {
    if (!state.project) return;
    const prev = state.project.studio_inputs || {};
    patchProjectQuiet({ studio_inputs: { ...prev, [name]: value } });
  }
  function setStudioInputNow(name, value) {
    if (!state.project) return;
    const prev = state.project.studio_inputs || {};
    patchProject({ studio_inputs: { ...prev, [name]: value } });
  }

  // ── editing ────────────────────────────────────────────────────────────────
  // Autosave just persists the project to disk so edits aren't lost — it does NOT need
  // to fire every keystroke. Discrete actions (dropdowns, checkboxes) save ~1s after the
  // last change; free typing waits longer and is flushed on blur (see flushSave).
  function scheduleSave() {
    state.saving = true; notify();
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => { saveTimer = null; commit(); }, 1200);
  }

  // For free-text / number fields: update state and queue a save WITHOUT re-rendering,
  // so typing isn't interrupted. Long debounce; blur flushes it (flushSave).
  function scheduleSaveSilent() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => { saveTimer = null; commit(); }, 6000);
  }

  // Commit any pending save immediately (field blur / before generate). Returns the
  // commit promise when it flushed, else null.
  function flushSave() {
    if (!saveTimer) return null;
    clearTimeout(saveTimer); saveTimer = null;
    return commit();
  }

  async function commit() {
    if (!state.project) return;
    try {
      state.project = await API.saveProject(state.project.id, state.project);
      state.saving = false;
      // Preserve selectedSceneId: null means "project settings" — keep it.
      // Only reset to first scene if a NON-null id is no longer in the scene list.
      if (state.selectedSceneId !== null &&
          !state.project.scenes.some((s) => s.id === state.selectedSceneId))
        state.selectedSceneId = state.project.scenes[0]?.id || null;
      notify();
      refreshProjectList();
      refreshPreview();
    } catch (e) {
      state.saving = false; notify();
      console.error("save failed", e);
    }
  }

  function patchProject(patch) { if (!state.project) return; Object.assign(state.project, patch); notify(); scheduleSave(); }
  function patchProjectQuiet(patch) { if (!state.project) return; Object.assign(state.project, patch); scheduleSaveSilent(); }

  function scene(id) { return state.project?.scenes.find((s) => s.id === id) || null; }

  function patchScene(id, patch) {
    const s = scene(id); if (!s) return;
    Object.assign(s, patch); notify(); scheduleSave();
  }
  function patchSceneQuiet(id, patch) {
    const s = scene(id); if (!s) return;
    Object.assign(s, patch); scheduleSaveSilent();
  }

  function selectScene(id) { state.selectedSceneId = id; notify(); }

  function addScene() {
    if (!state.project) return;
    // Default to "carry": a new scene continues the previous one (overlap) unless the
    // user picks an anchor/empty. Scene 1 carrying just starts a fresh run.
    const s = { text: "", transition_to_next: "", source: { type: "carry" }, excluded: false };
    state.project.scenes.push(s);
    notify(); scheduleSave(); // server assigns id; reselect after commit
  }

  function removeScene(id) {
    if (!state.project) return;
    state.project.scenes = state.project.scenes.filter((s) => s.id !== id);
    delete state.sceneRenders[id];  // drop its render mapping (preview/render skip it)
    if (state.selectedSceneId === id) state.selectedSceneId = state.project.scenes[0]?.id || null;
    notify(); scheduleSave();
  }

  // LTX-valid frame counts are 8k+1 (so (frames-1) % 8 === 0); snap to the nearest.
  function snapFrames(n) { return Math.max(9, Math.round((Math.round(n) - 1) / 8) * 8 + 1); }

  // A scene's duration in seconds (respecting per-scene frames/fps modes).
  function sceneDurationSec(sc) {
    const p = state.project; if (!p || !sc) return 0;
    const fps = (sc.fps_mode !== "project" && sc.fps != null ? sc.fps : p.frame_rate) || 25;
    const frames = (sc.frames_mode !== "project" && sc.frames != null ? sc.frames : p.num_frames_per_scene) || 1;
    return frames / fps;
  }

  // Resize a clip by setting its frame count from a target duration (seconds) × fps.
  // Uses silent save so the drag handle isn't interrupted by a re-render mid-drag.
  function resizeScene(id, durationSec) {
    if (!state.project) return;
    const s = scene(id); if (!s) return;
    if (s.frames_mode === "custom") return;  // custom length is locked against trim
    const fps = (s.fps_mode !== "project" && s.fps != null) ? s.fps : state.project.frame_rate;
    s.frames = snapFrames(Math.max(1, durationSec) * fps);
    // Dragging the trim handle opts the scene into timeline-driven length.
    if (s.frames_mode == null || s.frames_mode === "project") s.frames_mode = "timeline";
    scheduleSaveSilent();
  }

  // Split a clip in two at `atFrames` (defaults to the midpoint).
  function splitScene(id, atFrames) {
    if (!state.project) return;
    const arr = state.project.scenes;
    const i = arr.findIndex((s) => s.id === id); if (i < 0) return;
    const s = arr[i];
    const fps = s.fps != null ? s.fps : state.project.frame_rate;
    const frames = s.frames != null ? s.frames : state.project.num_frames_per_scene;
    const cut = snapFrames(atFrames != null ? atFrames : frames / 2);
    if (cut <= 9 || cut >= frames) return;
    const second = JSON.parse(JSON.stringify(s));
    // Assign a client id up-front (server honors it) so the rendered video can be mapped
    // to BOTH halves immediately — like an NLE, a split yields two clips of one source.
    second.id = "c" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
    second.frames = snapFrames(frames - cut);
    second.transition_to_next = s.transition_to_next || "";
    second.transition_frames = s.transition_frames || null;
    s.frames = cut; s.transition_to_next = ""; s.transition_frames = null;
    arr.splice(i + 1, 0, second);
    // Keep the render across the cut: the second half plays the SAME source video starting
    // at the first half's out-point (its in-point + first-half duration).
    const r = state.sceneRenders[id];
    if (r && r.media) {
      const fps = s.fps_mode !== "project" && s.fps != null ? s.fps : state.project.frame_rate;
      state.sceneRenders[second.id] = { media: r.media, inSec: (r.inSec || 0) + cut / (fps || 25) };
    }
    notify(); scheduleSave();
  }

  function moveScene(id, delta) {
    if (!state.project) return;
    const arr = state.project.scenes;
    const i = arr.findIndex((s) => s.id === id);
    const j = i + delta;
    if (i < 0 || j < 0 || j >= arr.length) return;
    [arr[i], arr[j]] = [arr[j], arr[i]];
    notify(); scheduleSave();
  }

  // Move a scene to an absolute index (post-removal index). Used by timeline drag-reorder.
  function moveSceneTo(id, toIndex) {
    if (!state.project) return;
    const arr = state.project.scenes;
    const from = arr.findIndex((s) => s.id === id);
    if (from < 0) return;
    toIndex = Math.max(0, Math.min(arr.length - 1, toIndex));
    if (toIndex === from) return;
    const [it] = arr.splice(from, 1);
    arr.splice(toIndex, 0, it);
    notify(); scheduleSave();
  }

  // ── audio editing ─────────────────────────────────────────────────────────────
  function _uid() { return "a" + Math.random().toString(36).slice(2, 9) + Date.now().toString(36); }

  function addAudioTrack(mediaId, startSec) {
    if (!state.project || !mediaId) return;
    const asset = (state.mediaBin || []).find((m) => m.id === mediaId);
    state.project.audio_tracks = state.project.audio_tracks || [];
    state.project.audio_tracks.push({
      id: _uid(), media_ref: mediaId, start_sec: +(startSec || 0),
      volume: 1.0, label: (asset && asset.name) || "track",
    });
    notify(); scheduleSave();
  }
  function updateAudioTrack(id, patch, quiet) {
    const t = (state.project?.audio_tracks || []).find((x) => x.id === id);
    if (!t) return;
    Object.assign(t, patch);
    notify(); quiet ? scheduleSaveSilent() : scheduleSave();
  }
  function removeAudioTrack(id) {
    if (!state.project) return;
    state.project.audio_tracks = (state.project.audio_tracks || []).filter((x) => x.id !== id);
    notify(); scheduleSave();
  }

  // ── sync scenes from preview (distribute parsed anchor/transitions back) ──────
  function syncFromPreview() {
    if (!state.project || !state.preview) return;
    // Authoritative lossless split (verbatim text, shortcut-aware boundaries).
    const v = state.preview.parsed_verbatim || state.preview.parsed_raw || state.preview.parsed || {};
    const parsedScenes = v.scenes || [];
    if (!parsedScenes.length) return;

    state.project.anchor = v.anchor || "";

    // Sync scene texts verbatim (only if count matches to avoid destructive mismatches)
    const activeScenes = state.project.scenes.filter((s) => !s.excluded);
    if (parsedScenes.length === activeScenes.length) {
      parsedScenes.forEach((ps, i) => { if (ps.text) activeScenes[i].text = ps.text; });
    }
    _applyDetectedTransitions(activeScenes, v.transitions);

    notify();
    scheduleSave();
  }

  // ── global prompt → distribute into anchor / scenes / transitions ──────────────
  // Reparses a master prompt (Studio combined syntax) and rebuilds the timeline:
  // anchor + one scene per parsed segment + transitions matched from the library.
  // Existing per-scene source / length settings are carried over by index.
  async function applyGlobalPrompt(text) {
    if (!state.project) return;
    state.project.global_prompt = text;
    let res;
    try { res = await API.parsePrompt(state.project.id, text); }
    catch (e) { alert("Could not parse the global prompt: " + e.message); return; }
    // The verbatim split is authoritative: correct (shortcut-aware) boundaries, scene
    // text kept exactly as typed, anchor + scenes reproduce the global prompt.
    const v = res.parsed_verbatim || res.parsed_raw || res.parsed || {};
    if (!(v.scenes || []).length) { alert("Nothing parsed — no scenes detected."); return; }

    state.project.anchor = v.anchor || "";
    const old = state.project.scenes || [];
    const next = (v.scenes || []).map((ps, i) => {
      const base = old[i] ? JSON.parse(JSON.stringify(old[i])) : { source: { type: "carry" }, excluded: false };
      base.text = ps.text || "";          // verbatim chunk of the global prompt
      base.transition_to_next = "";
      return base;
    });
    _applyDetectedTransitions(next, v.transitions);
    state.project.scenes = next;
    state.selectedSceneId = null;  // show project view so the result is visible
    notify(); scheduleSave();
  }

  // Map detected transitions (visual_effect) onto scene seams via the library (display
  // metadata; the trigger word itself already lives verbatim in the scene text).
  function _applyDetectedTransitions(scenes, transitions) {
    (transitions || []).forEach((t) => {
      const idx = t.after_scene;
      if (idx >= 0 && idx < scenes.length && t.visual_effect) {
        const lib = (state.transitions || []).find((tr) => (tr.visual_effect || "none") === t.visual_effect);
        if (lib) scenes[idx].transition_to_next = lib.trigger || lib.name || "";
      }
    });
  }

  // ── preview ──────────────────────────────────────────────────────────────────
  let previewTimer = null;
  function refreshPreview() {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(async () => {
      if (!state.project) return;
      try { state.preview = await API.preview(state.project.id, false); }
      catch (e) { state.preview = { parse_error: e.message }; }
      notify();
    }, 250);
  }

  // ── generation ───────────────────────────────────────────────────────────────
  const GEN_ERROR_MAP = [
    [/audio_vae.*no source/i,    "Missing Audio VAE — add an Audio VAE loader in Models (distinct from the video VAE)."],
    [/audio_latent.*no source/i, "Missing Audio latent — add an Audio Encoder node in Models (e.g. LTXVAudioEncoder)."],
    [/\.vae.*no source/i,        "Missing Video VAE — add a Video VAE loader in Models."],
    [/\.model.*no source/i,      "Missing diffusion model — add a Unet loader in Models."],
    [/\.clip.*no source/i,       "Missing text encoder — add a CLIP loader in Models."],
    [/not installed/i,           "A configured node class is not installed in ComfyUI — check Models for details."],
    [/Node registry unavailable/i, "ComfyUI is offline or still starting up — wait a moment and try again."],
  ];

  function _friendlyGenError(raw) {
    // "Generation blocked — …" already names the exact input(s); pass it through so the
    // message is actionable instead of a vague "a required input has no source".
    if (/Generation blocked/i.test(raw)) return raw;
    for (const [pat, msg] of GEN_ERROR_MAP) if (pat.test(raw)) return msg;
    return raw;
  }

  let pollTimer = null;
  let progressTimer = null;
  let pollStart = 0;
  let _interrupted = false;

  function _clearGenTimers() { clearInterval(pollTimer); clearInterval(progressTimer); }

  // Ask ComfyUI to stop the current generation. The running poll then resolves as
  // "Interrupted" and the run loop stops.
  async function interrupt() {
    _interrupted = true;
    set({ gen: { ...state.gen, msg: "Interrupting…" } });
    try { await API.interrupt(); } catch (_) {}
  }

  function _elapsed() {
    const s = Math.floor((Date.now() - pollStart) / 1000);
    return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
  }

  // Split the active scenes into chain runs. A run starts at an anchored scene
  // (empty / image / generated_frame); a "carry" scene appends to the current run
  // so it overlaps the previous scene (one chain-sampler request per run).
  // True if a scene's image anchor still exists in the media bin (deleted image → fall
  // back to carry / i2v guides, not a broken anchor).
  function _anchorAvailable(s) {
    const t = s.source && s.source.type;
    if (t !== "image" && t !== "generated_frame") return false;
    const ref = s.source.media_ref;
    return !!(ref && (state.mediaBin || []).some((m) => m.id === ref));
  }
  function _runs() {
    const active = state.project.scenes.filter((s) => !s.excluded);
    const runs = [];
    for (const s of active) {
      const t = (s.source && s.source.type) || "empty";
      // carry OR an anchor whose image is gone -> continue the previous run (i2v guides).
      const isCarry = t === "carry" || ((t === "image" || t === "generated_frame") && !_anchorAvailable(s));
      if (isCarry && runs.length) runs[runs.length - 1].push(s.id);
      else runs.push([s.id]);
    }
    return runs;
  }

  // Record a completed run's output: map each of the run's scenes to the one source
  // video at its cumulative in-point, so splits/deletes later play the right portions.
  function _recordSegment(mediaList, targetSceneIds) {
    if (!mediaList || !mediaList.length || !targetSceneIds || !targetSceneIds.length) return;
    const primary = mediaList.find((m) => m.kind === "videos" || m.kind === "gifs") || mediaList[0];
    let inAcc = 0;
    for (const id of targetSceneIds) {
      const sc = scene(id); if (!sc) continue;
      state.sceneRenders[id] = { media: primary, inSec: inAcc };
      inAcc += sceneDurationSec(sc);
    }
  }

  // Poll a single queued prompt to completion. Resolves true on success, false on error.
  function _pollPromise(promptId, targetSceneIds, prefix) {
    prefix = prefix || "Generating…";
    return new Promise((resolve) => {
      _clearGenTimers();
      let pendingStreak = 0;
      // Faster step-progress poll (sampler current/total steps).
      progressTimer = setInterval(async () => {
        if (_interrupted) return;
        try {
          const pr = await API.progress();
          if (pr && pr.max > 0) set({ gen: { ...state.gen, step: pr.value, maxStep: pr.max } });
        } catch (_) {}
      }, 700);
      pollTimer = setInterval(async () => {
        if (_interrupted) {
          _clearGenTimers();
          set({ gen: { state: "idle", promptId, media: [], msg: "Interrupted." } });
          resolve(false);
          return;
        }
        try {
          const s = await API.status(state.project.id, promptId);
          if (s.state === "error") {
            _clearGenTimers();
            const msg = s.error ? `ComfyUI error: ${s.error}` : "Generation failed inside ComfyUI — check the ComfyUI terminal for details.";
            set({ gen: { state: "error", promptId, media: [], msg } });
            resolve(false);
          } else if (s.state === "completed") {
            _clearGenTimers();
            _recordSegment(s.media, targetSceneIds);
            set({ gen: { state: "done", promptId, media: s.media, msg: s.media.length ? "" : "Completed but no output media found — check ComfyUI terminal." } });
            resolve(true);
          } else {
            // "pending" after "running" means the job left the queue without a history
            // entry — it likely crashed or was interrupted by ComfyUI.
            if (s.state === "pending") pendingStreak++; else pendingStreak = 0;
            if (pendingStreak >= 3) {
              _clearGenTimers();
              set({ gen: { state: "error", promptId, media: [], msg: "Job disappeared from ComfyUI queue — it may have crashed or been interrupted. Check the ComfyUI terminal." } });
              resolve(false);
              return;
            }
            const step = (state.gen.maxStep > 0) ? `  ·  step ${state.gen.step}/${state.gen.maxStep}` : "";
            set({ gen: { ...state.gen, state: s.state, msg: `${prefix} ${_elapsed()}${step}` } });
          }
        } catch (e) {
          _clearGenTimers();
          set({ gen: { ...state.gen, state: "error", msg: e.message } });
          resolve(false);
        }
      }, 2000);
    });
  }

  // Toggle a pending Studio session reset — applied to the FIRST run of the next
  // generation. Clicking again disarms it (in case of a mis-click).
  let _resetSessionPending = false;
  function resetStudioSession() {
    _resetSessionPending = !_resetSessionPending;
    set({ resetSessionArmed: _resetSessionPending });
  }

  // Generate one run (single scene, or an explicit list of scene ids). Returns success.
  async function _generateRun(sceneIds, onlyScene, prefix, resetSession) {
    _interrupted = false;
    set({ gen: { state: "queuing", promptId: null, media: [], msg: `${prefix}: queuing…`, step: 0, maxStep: 0 } });
    try {
      const r = await API.generate(state.project.id, onlyScene || null, onlyScene ? null : sceneIds, !!resetSession);
      if (!r.prompt_id) { set({ gen: { ...state.gen, state: "error", msg: "No prompt id returned." } }); return false; }
      pollStart = Date.now();
      set({ gen: { state: "running", promptId: r.prompt_id, media: [], msg: `${prefix}: generating…` } });
      return await _pollPromise(r.prompt_id, sceneIds, prefix);
    } catch (e) {
      set({ gen: { state: "error", promptId: null, media: [], msg: _friendlyGenError(e.message) } });
      return false;
    }
  }

  async function generate(onlyScene) {
    if (!state.project) return;
    await flushSave();  // ensure the server has the latest edits before generating
    if (!onlyScene) return generateMontage();
    const reset = _resetSessionPending; _resetSessionPending = false;
    if (reset) state.resetSessionArmed = false;
    await _generateRun([onlyScene], onlyScene, "Generating scene", reset);
  }

  // Generate the whole montage: one chain request per run, fired sequentially
  // (one GPU at a time). Each run's first scene supplies its i2v anchor. A pending
  // session reset applies to the FIRST run only.
  async function generateMontage() {
    if (!state.project) return;
    await flushSave();  // persist pending edits before reading scenes/runs
    const runs = _runs();
    if (!runs.length) { set({ gen: { state: "error", promptId: null, media: [], msg: "No active scenes to generate." } }); return; }
    const reset = _resetSessionPending; _resetSessionPending = false;
    if (reset) state.resetSessionArmed = false;
    for (let i = 0; i < runs.length; i++) {
      const ok = await _generateRun(runs[i], null, `Run ${i + 1}/${runs.length}`, reset && i === 0);
      if (!ok) return;  // error already surfaced
    }
    set({ gen: { state: "done", promptId: null, media: state.gen.media, msg: `${runs.length} run(s) generated — use Render Final Video to stitch them.` } });
  }

  // Local timestamp for unique export filenames: YYYYMMDD-HHMMSS.
  function _stamp() {
    const d = new Date(), p = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  }

  // Fetch a render and offer a Save dialog (File System Access API where available,
  // else a normal download). Renders live in ComfyUI's temp dir, so persisting one
  // is the user's job — this is how they do it.
  async function _saveBlobAs(url, name) {
    let blob;
    try { const r = await fetch(url); if (!r.ok) throw new Error(r.statusText); blob = await r.blob(); }
    catch (e) { alert("Could not fetch the render: " + e.message); return; }
    if (window.showSaveFilePicker) {
      try {
        const h = await window.showSaveFilePicker({
          suggestedName: name,
          types: [{ description: "MP4 video", accept: { "video/mp4": [".mp4"] } }],
        });
        const w = await h.createWritable(); await w.write(blob); await w.close();
        return;
      } catch (e) { if (e && e.name === "AbortError") return; }  // cancelled, or unsupported → fall through
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }

  // Ordered list of rendered clips in timeline order: {media, inSec, durationSec} for
  // each non-excluded scene that has a render. This is what plays/exports — so splits
  // and deletes are honoured (deleted scenes simply aren't here).
  function _renderClips() {
    const p = state.project;
    const out = [];
    for (const sc of (p.scenes || [])) {
      if (sc.excluded) continue;
      const r = state.sceneRenders[sc.id];
      if (!(r && r.media)) continue;
      const fps = (sc.fps_mode !== "project" && sc.fps != null ? sc.fps : p.frame_rate) || 25;
      const tFrames = sc.transition_frames || 0;
      out.push({
        media: r.media, inSec: r.inSec || 0, durationSec: sceneDurationSec(sc),
        fx: sc.effects || {}, fps,
        w: (sc.width != null ? sc.width : p.width) || null,
        h: (sc.height != null ? sc.height : p.height) || null,
        transition: sc.video_transition || "",
        tdur: tFrames > 0 ? tFrames / fps : 0,
        volume: sc.audio_volume != null ? sc.audio_volume : 1,
      });
    }
    return out;
  }

  // Export the render covering the selected clip via a Save dialog.
  async function exportSelected() {
    if (!state.project) return;
    const id = state.selectedSceneId;
    if (!id) { alert("Select a clip to export."); return; }
    const r = state.sceneRenders[id];
    if (!r || !r.media) { alert("That clip hasn't been generated yet — generate it first."); return; }
    const idx = state.project.scenes.findIndex((s) => s.id === id);
    const proj = (state.project.name || "montage").replace(/[^\w.-]+/g, "_");
    await _saveBlobAs(API.resultUrl(state.project.id, r.media), `${proj}_scene${idx >= 0 ? idx + 1 : "x"}_${_stamp()}.mp4`);
  }

  // Stitch the kept clips (in/out per clip, hard cut, video + audio) into one final file.
  async function renderFinal() {
    if (!state.project) return;
    const rc = _renderClips();
    if (!rc.length) { alert("No generated clips to stitch yet — generate first."); return; }
    const clips = rc.map((c) => ({
      filename: c.media.filename, subfolder: c.media.subfolder || "", type: c.media.type || "output",
      in: +c.inSec.toFixed(3), dur: +c.durationSec.toFixed(3),
      fx: c.fx, fps: c.fps, w: c.w, h: c.h, transition: c.transition, tdur: +(c.tdur || 0).toFixed(3),
      volume: c.volume,
    }));
    await flushSave();  // audio tracks / keep-original live on the project — render reads it from disk
    set({ gen: { state: "running", promptId: null, media: [], msg: `Stitching ${clips.length} clip(s)…` } });
    try {
      const r = await API.renderFinal(state.project.id, clips);
      // Play the stitched file across the whole timeline (single source, in-point 0).
      const order = state.project.scenes.filter((s) => !s.excluded);
      state.sceneRenders = {};
      let acc = 0;
      for (const sc of order) { state.sceneRenders[sc.id] = { media: r.media, inSec: acc }; acc += sceneDurationSec(sc); }
      set({ gen: { state: "done", promptId: null, media: [r.media], msg: `Final video rendered from ${r.clips} clip(s) — saving…` } });
      const name = (state.project.name || "montage").replace(/[^\w.-]+/g, "_") + `_final_${_stamp()}.mp4`;
      await _saveBlobAs(API.resultUrl(state.project.id, r.media), name);
    } catch (e) {
      set({ gen: { state: "error", promptId: null, media: [], msg: "Render failed: " + e.message } });
    }
  }

  // ── pluggable models / exposed controls ──────────────────────────────────────
  async function loadModels() {
    try { state.models = await API.getModels(state.project?.id); } catch (_) { state.models = { slots: [] }; }
    notify();
    loadImageTargets();
  }
  async function loadImageTargets() {
    try { state.imageTargets = (await API.imageTargets(state.project?.id)).targets || []; } catch (_) { state.imageTargets = []; }
    notify();
  }

  // Edit a configured node input from the main editor (an "exposed" control) and persist.
  async function setModelInput(slotId, name, value) {
    const slot = (state.models.slots || []).find((s) => s.id === slotId);
    if (!slot) return;
    slot.inputs = slot.inputs || {}; slot.inputs[name] = value;
    notify();
    try { state.models = await API.saveModels(state.project?.id, state.models); notify(); }
    catch (e) { console.error("saveModels failed", e); }
  }

  // Set a linked control's shared value (writes through to all member inputs) and persist.
  async function setModelLink(linkId, value) {
    const link = (state.models.links || []).find((l) => l.id === linkId);
    if (!link) return;
    link.value = value;
    (link.members || []).forEach((m) => {
      const s = (state.models.slots || []).find((x) => x.id === m.slotId);
      if (s) { s.inputs = s.inputs || {}; s.inputs[m.input] = value; }
    });
    notify();
    try { state.models = await API.saveModels(state.project?.id, state.models); notify(); }
    catch (e) { console.error("saveModels failed", e); }
  }

  // ── media bin + libraries ─────────────────────────────────────────────────────
  async function loadMedia() { try { state.mediaBin = (await API.listMedia()).media || []; } catch (_) { state.mediaBin = []; } notify(); }
  async function uploadMedia(files) {
    for (const f of files) { try { await API.uploadMedia(f); } catch (e) { console.error("upload failed", e); } }
    await loadMedia();
  }
  async function deleteMedia(id) { try { await API.deleteMedia(id); } catch (_) {} await loadMedia(); }

  async function loadTransitions() { try { state.transitions = (await API.transitions()).transitions || []; } catch (_) {} notify(); }
  async function loadShortcuts() { try { state.shortcuts = (await API.shortcuts()).shortcuts || []; } catch (_) { state.shortcuts = []; } notify(); }
  async function saveShortcut(item) { try { state.shortcuts = (await API.saveShortcut(item)).shortcuts || state.shortcuts; notify(); } catch (e) { alert("Save failed: " + e.message); } }
  async function deleteShortcut(name) { try { state.shortcuts = (await API.deleteShortcut(name)).shortcuts || []; notify(); } catch (e) { console.error(e); } }
  async function saveTransition(item) { try { state.transitions = (await API.saveTransition(item)).transitions || state.transitions; notify(); } catch (e) { alert("Save failed: " + e.message); } }
  async function deleteTransition(name) { try { state.transitions = (await API.deleteTransition(name)).transitions || []; notify(); } catch (e) { console.error(e); } }
  async function importShortcuts(file) {
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      const r = await API.importShortcuts(data);
      state.shortcuts = r.shortcuts || state.shortcuts; notify();
      return r.imported;
    } catch (e) { alert("Import failed: " + e.message); return 0; }
  }
  async function importTransitions(file) {
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      const r = await API.importTransitions(data);
      state.transitions = r.transitions || state.transitions; notify();
      return r.imported;
    } catch (e) { alert("Import failed: " + e.message); return 0; }
  }

  // Apply a library item to the selected scene.
  function applyTransitionToSelection(trigger) {
    const s = scene(state.selectedSceneId); if (!s) return false;
    patchScene(s.id, { transition_to_next: trigger }); return true;
  }
  function insertShortcutIntoSelection(trigger) {
    const s = scene(state.selectedSceneId); if (!s) return false;
    const t = (s.text || "").trim();
    patchScene(s.id, { text: t ? `${t} ${trigger}` : trigger }); return true;
  }
  function assignMediaToScene(sceneId, mediaId) {
    const s = scene(sceneId); if (!s) return;
    patchScene(sceneId, { source: { ...(s.source || {}), type: "image", media_ref: mediaId } });
  }

  // ── boot ─────────────────────────────────────────────────────────────────────
  async function init() {
    try { state.health = await API.health(); } catch (_) { state.health = { ok: false }; }
    try { const t = await API.transitions(); state.transitions = t.transitions || []; } catch (_) { state.transitions = []; }
    await loadShortcuts();
    await loadMedia();
    try { state.ratingLabels = (await API.ratingLabels()).labels || []; } catch (_) { state.ratingLabels = []; }
    await loadModels();
    window.addEventListener("funpack-models-changed", loadModels);
    await refreshProjectList();
    if (state.projects[0]) await loadProject(state.projects[0].id);
    else await newProject("My first montage");
    notify();
  }

  window.Store = {
    get, set, subscribe, init,
    refreshProjectList, loadProject, newProject, deleteProject, downloadProject, importProject,
    patchProject, patchProjectQuiet, patchScene, patchSceneQuiet, flushSave, selectScene, addScene, removeScene, moveScene, moveSceneTo, scene,
    addAudioTrack, updateAudioTrack, removeAudioTrack,
    resizeScene, splitScene, snapFrames,
    refreshPreview, syncFromPreview, applyGlobalPrompt, generate, generateMontage, renderFinal, exportSelected, interrupt, loadModels, loadImageTargets, setModelInput, setModelLink,
    setConditioningSlot, setSamplerSlot, setSamplerInput, setSamplerInputNow, setStudioInput, setStudioInputNow,
    loadMedia, uploadMedia, deleteMedia, assignMediaToScene,
    loadShortcuts, saveShortcut, deleteShortcut, importShortcuts, loadTransitions, saveTransition, deleteTransition, importTransitions,
    applyTransitionToSelection, insertShortcutIntoSelection,
    setSceneRating: (id, v) => patchScene(id, { rating: v }),
    resetStudioSession,
  };
})();
