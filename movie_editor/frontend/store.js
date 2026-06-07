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
    // Rendered video segments: [{sceneIds, media, startSec (computed), durationSec (computed)}]
    // Not persisted to disk — cleared on project switch. Player uses these to map
    // timeline positions to video files and compute seek offsets.
    renderedSegments: [],
    saving: false,
    models: { slots: [] },   // pluggable node config (shared with Models modal)
    mediaBin: [],            // uploaded assets [{id,name,kind,...}]
    shortcuts: [],           // prompt shortcut library
    imageTargets: [],        // where an image asset can be wired [{value,label}]
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
    state.renderedSegments = [];  // segments are per-session; not persisted
    notify();
    refreshPreview();
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
  function scheduleSave() {
    state.saving = true; notify();
    clearTimeout(saveTimer);
    saveTimer = setTimeout(commit, 900);
  }

  // For free-text / number fields: update state and queue a save WITHOUT re-rendering,
  // so typing isn't interrupted. The eventual commit() re-renders once, after a pause.
  function scheduleSaveSilent() { clearTimeout(saveTimer); saveTimer = setTimeout(commit, 1100); }

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
    const s = { text: "", transition_to_next: "", source: { type: "empty" }, excluded: false };
    state.project.scenes.push(s);
    notify(); scheduleSave(); // server assigns id; reselect after commit
  }

  function removeScene(id) {
    if (!state.project) return;
    state.project.scenes = state.project.scenes.filter((s) => s.id !== id);
    if (state.selectedSceneId === id) state.selectedSceneId = state.project.scenes[0]?.id || null;
    notify(); scheduleSave();
  }

  // LTX-valid frame counts are 8k+1 (so (frames-1) % 8 === 0); snap to the nearest.
  function snapFrames(n) { return Math.max(9, Math.round((Math.round(n) - 1) / 8) * 8 + 1); }

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
    delete second.id;
    second.frames = snapFrames(frames - cut);
    second.transition_to_next = s.transition_to_next || "";
    second.transition_frames = s.transition_frames || null;
    s.frames = cut; s.transition_to_next = ""; s.transition_frames = null;
    arr.splice(i + 1, 0, second);
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

  // ── sync scenes from preview (distribute parsed anchor/transitions back) ──────
  function syncFromPreview() {
    if (!state.project || !state.preview) return;
    const parsed = state.preview.parsed || {};
    const parsedScenes = parsed.scenes || [];
    const parsedTrans = parsed.transitions || [];
    if (!parsedScenes.length) return;

    // Sync anchor
    if (parsed.anchor != null) state.project.anchor = parsed.anchor;

    // Sync scene texts (only if count matches to avoid destructive mismatches)
    const activeScenes = state.project.scenes.filter((s) => !s.excluded);
    if (parsedScenes.length === activeScenes.length) {
      parsedScenes.forEach((ps, i) => { if (ps.text) activeScenes[i].text = ps.text; });
    }

    // Sync transitions: parsed.transitions[i] → scene[i].transition_to_next
    // We derive the trigger from the library by matching visual_effect.
    parsedTrans.forEach((t) => {
      const idx = t.after_scene;
      const sceneIdx = idx >= 0 ? idx : -1;
      if (sceneIdx >= 0 && sceneIdx < activeScenes.length) {
        const s = activeScenes[sceneIdx];
        if (!s.transition_to_next && t.visual_effect) {
          const lib = (state.transitions || []).find((tr) =>
            (tr.visual_effect || "none") === t.visual_effect);
          if (lib) s.transition_to_next = lib.trigger || lib.name || "";
        }
      }
    });

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
    let parsed;
    try { parsed = (await API.parsePrompt(state.project.id, text)).parsed; }
    catch (e) { alert("Could not parse the global prompt: " + e.message); return; }
    if (!parsed || !(parsed.scenes || []).length) { alert("Nothing parsed — no scenes detected."); return; }

    if (parsed.anchor != null) state.project.anchor = parsed.anchor;
    const old = state.project.scenes || [];
    const next = (parsed.scenes || []).map((ps, i) => {
      const base = old[i] ? JSON.parse(JSON.stringify(old[i])) : { source: { type: "empty" }, excluded: false };
      base.text = ps.text || "";
      base.transition_to_next = "";
      return base;
    });
    (parsed.transitions || []).forEach((t) => {
      const idx = t.after_scene;
      if (idx >= 0 && idx < next.length && t.visual_effect) {
        const lib = (state.transitions || []).find((tr) => (tr.visual_effect || "none") === t.visual_effect);
        if (lib) next[idx].transition_to_next = lib.trigger || lib.name || "";
      }
    });
    state.project.scenes = next;
    state.selectedSceneId = null;  // show project view so the result is visible
    notify(); scheduleSave();
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
    [/no source available/i,     "A required pipeline input has no source — open Models and check the Pipeline requirements checklist."],
    [/not installed/i,           "A configured node class is not installed in ComfyUI — check Models for details."],
    [/Node registry unavailable/i, "ComfyUI is offline or still starting up — wait a moment and try again."],
  ];

  function _friendlyGenError(raw) {
    for (const [pat, msg] of GEN_ERROR_MAP) if (pat.test(raw)) return msg;
    return raw;
  }

  let pollTimer = null;
  let pollStart = 0;
  async function generate(onlyScene) {
    if (!state.project) return;
    // Capture which scene ids this generation covers so the player can map
    // the output video to the correct timeline positions.
    const activeScenes = state.project.scenes.filter((s) => !s.excluded);
    const targetSceneIds = onlyScene ? [onlyScene] : activeScenes.map((s) => s.id);
    set({ gen: { state: "queuing", promptId: null, media: [], msg: onlyScene ? "Queuing scene…" : "Queuing montage…" } });
    try {
      const r = await API.generate(state.project.id, onlyScene);
      if (!r.prompt_id) { set({ gen: { ...state.gen, state: "error", msg: "No prompt id returned." } }); return; }
      pollStart = Date.now();
      set({ gen: { state: "running", promptId: r.prompt_id, media: [], msg: "Generating…" } });
      poll(r.prompt_id, targetSceneIds);
    } catch (e) {
      set({ gen: { state: "error", promptId: null, media: [], msg: _friendlyGenError(e.message) } });
    }
  }

  function _elapsed() {
    const s = Math.floor((Date.now() - pollStart) / 1000);
    return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
  }

  function poll(promptId, targetSceneIds) {
    clearInterval(pollTimer);
    let pendingStreak = 0;
    pollTimer = setInterval(async () => {
      try {
        const s = await API.status(state.project.id, promptId);
        if (s.state === "error") {
          clearInterval(pollTimer);
          const msg = s.error ? `ComfyUI error: ${s.error}` : "Generation failed inside ComfyUI — check the ComfyUI terminal for details.";
          set({ gen: { state: "error", promptId, media: [], msg } });
        } else if (s.state === "completed") {
          clearInterval(pollTimer);
          // Record the segment so the player can map it to timeline positions.
          if (s.media.length && targetSceneIds?.length) {
            const keep = (state.renderedSegments || []).filter(
              (seg) => !(seg.sceneIds || []).some((id) => targetSceneIds.includes(id))
            );
            // Primary media (first video/gif) is the playable output.
            const primary = s.media.find((m) => m.kind === "videos" || m.kind === "gifs") || s.media[0];
            state.renderedSegments = [...keep, { sceneIds: targetSceneIds, media: primary }];
          }
          set({ gen: { state: "done", promptId, media: s.media, msg: s.media.length ? "" : "Completed but no output media found — check ComfyUI terminal." } });
        } else {
          // "pending" after being "running" means the job left the queue without a
          // history entry — it likely crashed or was interrupted by ComfyUI.
          if (s.state === "pending") pendingStreak++;
          else pendingStreak = 0;
          if (pendingStreak >= 3) {
            clearInterval(pollTimer);
            set({ gen: { state: "error", promptId, media: [], msg: "Job disappeared from ComfyUI queue — it may have crashed or been interrupted. Check the ComfyUI terminal." } });
            return;
          }
          set({ gen: { ...state.gen, state: s.state, msg: `Generating… ${_elapsed()}` } });
        }
      } catch (e) {
        clearInterval(pollTimer);
        set({ gen: { ...state.gen, state: "error", msg: e.message } });
      }
    }, 2000);
  }

  // ── pluggable models / exposed controls ──────────────────────────────────────
  async function loadModels() {
    try { state.models = await API.getModels(); } catch (_) { state.models = { slots: [] }; }
    notify();
    loadImageTargets();
  }
  async function loadImageTargets() {
    try { state.imageTargets = (await API.imageTargets()).targets || []; } catch (_) { state.imageTargets = []; }
    notify();
  }

  // Edit a configured node input from the main editor (an "exposed" control) and persist.
  async function setModelInput(slotId, name, value) {
    const slot = (state.models.slots || []).find((s) => s.id === slotId);
    if (!slot) return;
    slot.inputs = slot.inputs || {}; slot.inputs[name] = value;
    notify();
    try { state.models = await API.saveModels(state.models); notify(); }
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
    try { state.models = await API.saveModels(state.models); notify(); }
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
    patchProject, patchProjectQuiet, patchScene, patchSceneQuiet, selectScene, addScene, removeScene, moveScene, scene,
    resizeScene, splitScene, snapFrames,
    refreshPreview, syncFromPreview, applyGlobalPrompt, generate, loadModels, loadImageTargets, setModelInput, setModelLink,
    setConditioningSlot, setSamplerSlot, setSamplerInput, setSamplerInputNow, setStudioInput, setStudioInputNow,
    loadMedia, uploadMedia, deleteMedia, assignMediaToScene,
    loadShortcuts, saveShortcut, deleteShortcut, importShortcuts, loadTransitions, saveTransition, deleteTransition, importTransitions,
    applyTransitionToSelection, insertShortcutIntoSelection,
  };
})();
