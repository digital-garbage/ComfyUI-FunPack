// Minimal state container for Easy Gen — NOT the Editor's store.js (that's a
// 4000+ line timeline/composer/overlay engine tightly coupled to the full NLE
// DOM). This is a small pub-sub wrapper around the same /funpack/movie/api/*
// endpoints, but it DOES support the same one-prompt-splits-into-N-scenes
// flow as the Editor's Composer — the user only ever sees one textarea, the
// resulting scenes are never surfaced in any per-scene UI, and Generate always
// runs the whole project as one chain (no scene_ids filter), so N scenes come
// back as one stitched video with no manual cut/trim ability, by design.
//
// Exposed as window.Store (same global name the Editor uses) purely so
// reused sections — models.js, autocomplete.js, and the built-in "Refinement
// & Taste" / "Updates & ComfyUI" sections baked into settings_window.js —
// work unmodified: they all read window.Store.get().project/.shortcuts/etc.
// and call window.Store.subscribe(fn) without needing to know which app they're in.
(function () {
  const API = window.MovieEditorAPI;

  const state = {
    project: null,
    health: null,
    resetSessionArmed: false,
    shortcuts: [],     // prompt shortcut library (for autocomplete)
    transitions: [],   // transition trigger library (for split-marker inference)
    mediaBin: [],      // uploaded assets [{id,name,kind,...}] — feeds engine_settings.js's MediaPicker
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

  // ── per-browser editor settings ──────────────────────────────────────────
  // Same localStorage key/shape as movie_editor/frontend/store.js so toggling
  // "Prompt autocomplete" or "Use anchor" is shared between the Editor and
  // Easy Gen on the same browser (matches the Editor's own cross-entry-point
  // behavior — these are per-browser prompt-editing prefs, not project data).
  const EDITOR_SETTINGS_KEY = "funpack_editor_settings";
  const EDITOR_SETTINGS_DEFAULTS = { autocomplete: true, anchorEnabled: true };
  function _loadEditorSettings() {
    try {
      const raw = JSON.parse(localStorage.getItem(EDITOR_SETTINGS_KEY) || "{}");
      return { ...EDITOR_SETTINGS_DEFAULTS, ...(raw && typeof raw === "object" ? raw : {}) };
    } catch (_) { return { ...EDITOR_SETTINGS_DEFAULTS }; }
  }
  let _editorSettings = _loadEditorSettings();
  function getEditorSetting(key) { return _editorSettings[key]; }
  function setEditorSetting(key, val) {
    if (_editorSettings[key] === val) return;
    _editorSettings = { ..._editorSettings, [key]: val };
    try { localStorage.setItem(EDITOR_SETTINGS_KEY, JSON.stringify(_editorSettings)); } catch (_) {}
    notify();
    if (key === "anchorEnabled" && state.project && (state.project.global_prompt || "").trim()) {
      applyGlobalPrompt(state.project.global_prompt);
    }
  }

  async function loadLibraries() {
    try { const r = await API.shortcuts(); set({ shortcuts: r.shortcuts || [] }); } catch (_) {}
    try { const r = await API.transitions(); set({ transitions: r.transitions || r || [] }); } catch (_) {}
  }

  async function loadMedia() {
    try { set({ mediaBin: (await API.listMedia()).media || [] }); } catch (_) { set({ mediaBin: [] }); }
  }

  // Import/export the same shortcut/transition JSON libraries the ComfyUI node graph
  // and the Movie Editor read and write (one shared FunPack library, no Easy Gen copy).
  // mode: "merge" (overlay onto the existing library) | "replace" (wipe first).
  async function importShortcuts(file, mode) {
    const data = JSON.parse(await file.text());
    const r = await API.importShortcuts(data, mode);
    set({ shortcuts: r.shortcuts || state.shortcuts });
    return r.imported;
  }
  async function importTransitions(file, mode) {
    const data = JSON.parse(await file.text());
    const r = await API.importTransitions(data, mode);
    set({ transitions: r.transitions || state.transitions });
    return r.imported;
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

  // Fallback for projects saved before the global-prompt split existed, or
  // edited scene-by-scene in the Editor without ever using its Composer:
  // reconstruct a verbatim prompt from anchor + scene texts + transition
  // markers so the textarea shows something sensible instead of going blank.
  function buildPromptFallback(p) {
    const parts = [];
    const anchor = (p.anchor || "").trim();
    if (anchor) parts.push(anchor);
    (p.scenes || []).forEach((s, i) => {
      if (i > 0) parts.push((p.scenes[i - 1].transition_to_next || "cut").trim());
      const t = (s.text || "").trim();
      if (t) parts.push(t);
    });
    return parts.filter(Boolean).join(" ");
  }

  async function loadProject(id) {
    const p = await API.getProject(id);
    if (!p.scenes || !p.scenes.length) p.scenes = [emptyScene()];
    if (!(p.global_prompt || "").trim()) p.global_prompt = buildPromptFallback(p);
    set({ project: p });
    return p;
  }

  async function deleteProject(id) {
    await API.deleteProject(id);
    if (state.project && state.project.id === id) set({ project: null });
  }

  // ── global prompt → N scenes (mirrors the Editor's Composer Apply, minus
  // the timeline/ghost/history machinery Easy Gen has no use for) ──────────

  // Ported from movie_editor/frontend/store.js's _normalizeGlobalPromptParse:
  // folds the parser's "anchor" back into Scene 1 when the anchor setting is off.
  function _normalizeGlobalPromptParse(trimmed, v) {
    v = v || {};
    const text = String(trimmed || "").trim();
    const anchor = String(v.anchor || "").trim();
    const scenes = Array.isArray(v.scenes) ? v.scenes : [];
    const hasSceneText = scenes.some((s) => String(s.text || "").trim());

    if (!hasSceneText && text) return { anchor: "", scenes: [{ text }] };
    if (!hasSceneText && anchor) return { anchor: "", scenes: [{ text: anchor }] };
    if (!_editorSettings.anchorEnabled && anchor) {
      return { anchor: "", scenes: [{ text: anchor }, ...scenes] };
    }
    return { anchor: v.anchor || "", scenes };
  }

  // Ported from movie_editor/frontend/store.js's _inferSplitMarkersFromPrompt:
  // finds which transition trigger (if any) sits in the raw gap text between
  // two consecutive verbatim scene chunks, so the chain sampler gets the same
  // seam behavior ("cut", etc.) the Editor would have produced.
  function _inferSplitMarkers(scenes, fullPrompt) {
    if (!fullPrompt || !scenes.length) return;
    const markers = (state.transitions || []).filter((m) => m.enabled !== false && (m.trigger || m.name));
    let pos = 0;
    for (let i = 0; i < scenes.length; i++) {
      const t = scenes[i].text || "";
      if (!t) continue;
      const idx = fullPrompt.indexOf(t, pos);
      if (idx < 0) continue;
      const endOfScene = idx + t.length;
      if (i < scenes.length - 1) {
        const nextT = scenes[i + 1].text || "";
        const nextIdx = nextT ? fullPrompt.indexOf(nextT, endOfScene) : endOfScene;
        const gapEnd = nextIdx >= 0 ? nextIdx : fullPrompt.length;
        const gap = fullPrompt.slice(endOfScene, gapEnd);
        for (const m of markers) {
          const trig = m.trigger || m.name;
          if (trig && gap.includes(trig)) { scenes[i].transition_to_next = trig; break; }
        }
      }
      pos = endOfScene;
    }
  }

  // Sequence guard: if the user keeps typing while a parse is in flight, an
  // older (now-stale) response must not overwrite scenes built from newer text.
  let _applySeq = 0;

  // Splits `text` into project.anchor + project.scenes via the same backend
  // parser the Editor's Composer uses (POST /projects/{id}/parse, lossless —
  // no shortcut expansion, no scene UI). Scene 0 keeps whatever upload/guide
  // it already had; new scenes 1..N-1 default to "carry" (continue the chain),
  // matching how the Editor treats anchorless continuation scenes.
  async function applyGlobalPrompt(text) {
    if (!state.project) return false;
    const trimmed = String(text || "").trim();
    state.project.global_prompt = trimmed;
    if (!trimmed) {
      const prevSource = state.project.scenes[0]?.source || { type: "empty" };
      state.project.scenes = [{ text: "", source: prevSource }];
      state.project.anchor = "";
      notify();
      return true;
    }
    const seq = ++_applySeq;
    let res;
    try { res = await API.parsePrompt(state.project.id, trimmed); }
    catch (e) { console.warn("Prompt parse failed:", e); return false; }
    if (seq !== _applySeq) return false; // superseded by a newer keystroke
    if (!state.project || state.project.global_prompt !== trimmed) return false; // project changed underneath us
    const v = _normalizeGlobalPromptParse(trimmed, res.parsed_verbatim || res.parsed_raw || res.parsed || {});
    if (!(v.scenes || []).length) return false;
    const prevSource = state.project.scenes[0]?.source || { type: "empty" };
    state.project.anchor = v.anchor || "";
    state.project.scenes = v.scenes.map((s, i) => ({
      text: String(s.text || "").trim(),
      transition_to_next: "",
      source: i === 0 ? prevSource : { type: "carry" },
    }));
    _inferSplitMarkers(state.project.scenes, trimmed);
    notify();
    return true;
  }

  let _applyTimer = null;
  function scheduleGlobalPromptApply(text) {
    if (_applyTimer) clearTimeout(_applyTimer);
    _applyTimer = setTimeout(() => { applyGlobalPrompt(text).then(() => save().catch(() => {})); }, 500);
  }

  // kind: "image" | "video" | null (null clears the upload → pure t2v).
  function setSceneMedia(mediaId, kind) {
    if (!state.project) return;
    state.project.scenes[0].source = mediaId
      ? { type: kind === "video" ? "v2v" : "image", media_ref: mediaId }
      : { type: "empty" };
    notify();
  }

  // Two overlapping save() calls (e.g. a debounced auto-save from an earlier
  // edit still in flight when a later action calls save() directly) can land
  // on the server out of order — whichever PUT is written to disk last wins,
  // even if it was the older/now-stale one, silently reverting a newer edit.
  // Serializing through one chain means a save only starts once the previous
  // one has FULLY completed, and reads state.project live at that point, so
  // it always picks up whatever changed while it waited.
  let saveQueue = Promise.resolve();
  function save() {
    const run = saveQueue.then(async () => {
      if (!state.project) return null;
      const saved = await API.saveProject(state.project.id, state.project);
      set({ project: saved });
      return saved;
    });
    saveQueue = run.catch(() => {}); // keep the queue alive even if this save failed
    return run;
  }

  // ── Engine settings (Studio / Chain Sampler / continuity) ──────────────────
  // Minimal re-implementation of movie_editor/frontend/store.js's project-patch
  // helpers, just enough for the shared engine_settings.js (loaded via the same
  // frontend-fallback mechanism as autocomplete.js/suggestions.js) to run
  // unmodified against Easy Gen's project. No undo/history, no global-prompt
  // resync — Easy Gen's scenes come from the split parser, not manual editing.
  let _engineSaveTimer = null;
  function _scheduleEngineSave(ms) {
    if (_engineSaveTimer) clearTimeout(_engineSaveTimer);
    _engineSaveTimer = setTimeout(() => { _engineSaveTimer = null; save().catch(() => {}); }, ms);
  }

  function patchProject(patch) {
    if (!state.project) return;
    Object.assign(state.project, patch);
    notify();
    save().catch(() => {});
  }
  function patchProjectQuiet(patch) {
    if (!state.project) return;
    Object.assign(state.project, patch);
    notify();
    _scheduleEngineSave(600);
  }

  function setConditioningSlot(slotId) { patchProject({ conditioning_slot: slotId || "funpack" }); }
  function setSamplerSlot(slotId) { patchProject({ sampler_slot: slotId || "funpack" }); }
  function setSamplerInput(name, value) {
    if (!state.project) return;
    patchProjectQuiet({ sampler_inputs: { ...(state.project.sampler_inputs || {}), [name]: value } });
  }
  function setSamplerInputNow(name, value) {
    if (!state.project) return;
    patchProject({ sampler_inputs: { ...(state.project.sampler_inputs || {}), [name]: value } });
  }
  function unsetSamplerInput(name) {
    if (!state.project) return;
    const prev = { ...(state.project.sampler_inputs || {}) };
    delete prev[name];
    patchProjectQuiet({ sampler_inputs: prev });
  }
  function setStudioInput(name, value) {
    if (!state.project) return;
    patchProjectQuiet({ studio_inputs: { ...(state.project.studio_inputs || {}), [name]: value } });
  }
  function setStudioInputNow(name, value) {
    if (!state.project) return;
    patchProject({ studio_inputs: { ...(state.project.studio_inputs || {}), [name]: value } });
  }

  const ENGINE_PRESETS = {
    draft: { label: "Fast draft", studio_inputs: { "refiner.split_by_transitions": false }, sampler_inputs: { steps: 20 } },
    quality: { label: "Quality", studio_inputs: { "refiner.split_by_transitions": true }, sampler_inputs: { steps: 40 } },
    continuity: {
      label: "Continuity-heavy",
      guide_settings: { stack_enabled: true, auto_guides: true, identity_pins: true },
      continuity_settings: { enabled: true },
      sampler_inputs: { steps: 36 },
    },
  };
  function applyEnginePreset(key) {
    const preset = ENGINE_PRESETS[key];
    if (!preset || !state.project) return;
    const patch = {};
    if (preset.studio_inputs) patch.studio_inputs = { ...(state.project.studio_inputs || {}), ...preset.studio_inputs };
    if (preset.sampler_inputs) patch.sampler_inputs = { ...(state.project.sampler_inputs || {}), ...preset.sampler_inputs };
    if (preset.guide_settings) patch.guide_settings = { ...(state.project.guide_settings || {}), ...preset.guide_settings };
    if (preset.continuity_settings) patch.continuity_settings = { ...(state.project.continuity_settings || {}), ...preset.continuity_settings };
    patchProject(patch);
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
    getEditorSetting, setEditorSetting,
    loadLibraries, importShortcuts, importTransitions,
    loadMedia,
    listProjects, createProject, loadProject, deleteProject,
    applyGlobalPrompt, scheduleGlobalPromptApply,
    setSceneMedia, save,
    patchProject, patchProjectQuiet,
    setConditioningSlot, setSamplerSlot,
    setSamplerInput, setSamplerInputNow, unsetSamplerInput,
    setStudioInput, setStudioInputNow,
    ENGINE_PRESETS, applyEnginePreset,
    resetStudioSession, takeResetSessionFlag,
  };
})();
