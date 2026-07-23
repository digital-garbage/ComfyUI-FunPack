// Minimal state container for Easy Gen — NOT the Editor's store.js (that's a
// 4000+ line timeline/composer/overlay engine tightly coupled to the full NLE
// DOM). Easy Gen only ever holds one project with exactly one scene, so this
// is a small pub-sub wrapper around the same /funpack/movie/api/* endpoints.
//
// Exposed as window.Store (same global name the Editor uses) purely so
// reused sections — models.js, and the built-in "Refinement & Taste" /
// "Updates & ComfyUI" sections baked into settings_window.js — work
// unmodified: they all read window.Store.get().project/.health and call
// window.Store.subscribe(fn) without needing to know which app they're in.
(function () {
  const API = window.MovieEditorAPI;

  const state = {
    project: null,
    health: null,
    resetSessionArmed: false,
  };
  const listeners = new Set();

  function get() { return state; }

  function notify() { listeners.forEach((fn) => { try { fn(state); } catch (_) {} }); }

  function subscribe(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  }

  function set(patch) {
    Object.assign(state, patch);
    notify();
  }

  async function refreshHealth() {
    try { set({ health: await API.health() }); } catch (_) { set({ health: { ok: false } }); }
  }

  function emptyScene() {
    return { text: "", source: { type: "empty" } };
  }

  async function listProjects() {
    return (await API.listProjects()).projects || [];
  }

  async function createProject(name) {
    const p = await API.createProject(name || "Untitled");
    p.scenes = [emptyScene()];
    const saved = await API.saveProject(p.id, p);
    set({ project: saved });
    return saved;
  }

  async function loadProject(id) {
    const p = await API.getProject(id);
    if (!p.scenes || !p.scenes.length) p.scenes = [emptyScene()];
    set({ project: p });
    return p;
  }

  async function deleteProject(id) {
    await API.deleteProject(id);
    if (state.project && state.project.id === id) set({ project: null });
  }

  // Local-only edit — call save() to persist. Keeps the prompt box from
  // round-tripping to the server on every keystroke.
  function setPromptText(text) {
    if (!state.project) return;
    state.project.scenes[0].text = text;
    notify();
  }

  // kind: "image" | "video" | null (null clears the upload → pure t2v).
  function setSceneMedia(mediaId, kind) {
    if (!state.project) return;
    state.project.scenes[0].source = mediaId
      ? { type: kind === "video" ? "v2v" : "image", media_ref: mediaId }
      : { type: "empty" };
    notify();
  }

  async function save() {
    if (!state.project) return;
    const saved = await API.saveProject(state.project.id, state.project);
    set({ project: saved });
    return saved;
  }

  function resetStudioSession() {
    set({ resetSessionArmed: !state.resetSessionArmed });
  }

  // Consumed by the Generate flow: reset_session applies to the NEXT run only.
  function takeResetSessionFlag() {
    const armed = state.resetSessionArmed;
    if (armed) set({ resetSessionArmed: false });
    return armed;
  }

  window.Store = {
    get, subscribe, set,
    refreshHealth,
    listProjects, createProject, loadProject, deleteProject,
    setPromptText, setSceneMedia, save,
    resetStudioSession, takeResetSessionFlag,
  };
})();
