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
    saveTimer = setTimeout(commit, 550);
  }

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

  function scene(id) { return state.project?.scenes.find((s) => s.id === id) || null; }

  function patchScene(id, patch) {
    const s = scene(id); if (!s) return;
    Object.assign(s, patch); notify(); scheduleSave();
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

  // ── boot ─────────────────────────────────────────────────────────────────────
  async function init() {
    try { state.health = await API.health(); } catch (_) { state.health = { ok: false }; }
    try { const t = await API.transitions(); state.transitions = t.transitions || []; } catch (_) { state.transitions = []; }
    await refreshProjectList();
    if (state.projects[0]) await loadProject(state.projects[0].id);
    else await newProject("My first montage");
    notify();
  }

  window.Store = {
    get, set, subscribe, init,
    refreshProjectList, loadProject, newProject, deleteProject,
    patchProject, patchScene, selectScene, addScene, removeScene, moveScene, scene,
    refreshPreview, generate,
  };
})();
