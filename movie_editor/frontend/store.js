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
    saving: false,
    models: { slots: [] },   // pluggable node config (shared with Models modal)
    mediaBin: [],            // uploaded assets [{id,name,kind,...}]
    shortcuts: [],           // prompt shortcut library
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
      // keep selection valid
      if (!state.project.scenes.some((s) => s.id === state.selectedSceneId))
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
  function resizeScene(id, durationSec) {
    if (!state.project) return;
    const s = scene(id); if (!s) return;
    const fps = s.fps != null ? s.fps : state.project.frame_rate;
    s.frames = snapFrames(Math.max(1, durationSec) * fps);
    notify(); scheduleSave();
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
  let pollTimer = null;
  async function generate(onlyScene) {
    if (!state.project) return;
    set({ gen: { state: "queuing", promptId: null, media: [], msg: onlyScene ? "Queuing scene…" : "Queuing montage…" } });
    try {
      const r = await API.generate(state.project.id, onlyScene);
      if (!r.prompt_id) { set({ gen: { ...state.gen, state: "error", msg: "No prompt id returned." } }); return; }
      set({ gen: { state: "running", promptId: r.prompt_id, media: [], msg: "Generating…" } });
      poll(r.prompt_id);
    } catch (e) {
      set({ gen: { state: "error", promptId: null, media: [], msg: e.message } });
    }
  }

  function poll(promptId) {
    clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      try {
        const s = await API.status(state.project.id, promptId);
        if (s.state === "completed") {
          clearInterval(pollTimer);
          set({ gen: { state: "done", promptId, media: s.media, msg: s.media.length ? "" : "Completed (no media found)." } });
        } else {
          set({ gen: { ...state.gen, state: s.state, msg: `Generating… (${s.state})` } });
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
    refreshProjectList, loadProject, newProject, deleteProject,
    patchProject, patchProjectQuiet, patchScene, patchSceneQuiet, selectScene, addScene, removeScene, moveScene, scene,
    resizeScene, splitScene, snapFrames,
    refreshPreview, generate, loadModels, setModelInput, setModelLink,
    loadMedia, uploadMedia, deleteMedia, assignMediaToScene,
    loadShortcuts, saveShortcut, deleteShortcut, loadTransitions, saveTransition, deleteTransition,
    applyTransitionToSelection, insertShortcutIntoSelection,
  };
})();
