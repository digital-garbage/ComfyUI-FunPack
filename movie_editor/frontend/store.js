// Central state + data actions. Views subscribe and re-render their zone.
(function () {
  const API = window.MovieEditorAPI;

  const state = {
    projects: [],          // [{id,name,scene_count,updated_at}]
    project: null,         // full Project
    selectedSceneId: null,   // focus clip (inspector / toolbar)
    selectedSceneIds: [],    // multi-select for generate / timeline
    transitions: [],       // library [{trigger/name, visual_effect}]
    health: null,          // {ok, comfy_url, template_exists}
    preview: null,         // {combined_prompt, parsed, warning, parse_error}
    gen: { state: "idle", promptId: null, media: [], msg: "" },
    // Per-scene render mapping: { sceneId: { media, inSec } } — `inSec` is the IN-point
    // within that source video this clip starts at (so split halves / deleted clips play
    // the correct portion). Not persisted to disk; cleared on project switch.
    sceneRenders: {},
    // Removed scenes that still had a render: shown as timeline/preview ghosts until the
    // neighboring run is regenerated (preview-only; not included in generation/export).
    sceneGhosts: [],
    saving: false,
    models: { slots: [] },   // pluggable node config (shared with Models modal)
    mediaBin: [],            // uploaded assets [{id,name,kind,...}]
    mediaPreviewId: null,    // transient image preview in the player (not scene assignment)
    shortcuts: [],           // prompt shortcut library
    characters: [],          // global character library
    imageTargets: [],        // where an image asset can be wired [{value,label}]
    ratingLabels: [],        // FunPack Studio V2 rating options
  };

  const listeners = new Set();
  let saveTimer = null;
  let _selectionAnchorId = null;  // shift-click range anchor

  function notify() { listeners.forEach((fn) => { try { fn(state); } catch (e) { console.error(e); } }); }
  function subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }
  function set(patch) { Object.assign(state, patch); notify(); }

  // Update generation progress WITHOUT a full notify(): a store notify rebuilds the whole
  // editor (inspector/timeline/player), which during a long generation kept interrupting the
  // user (couldn't save frames to the bin or add effects). Progress ticks instead mutate
  // state.gen quietly and fire a DOM event the player listens to, updating only the readout.
  function updateGenProgress(patch) {
    Object.assign(state.gen, patch);
    try { window.dispatchEvent(new CustomEvent("funpack-gen-progress", { detail: state.gen })); } catch (_) {}
  }
  function get() { return state; }

  // ── project lifecycle ──────────────────────────────────────────────────────
  async function refreshProjectList() {
    const { projects } = await API.listProjects();
    state.projects = projects; notify();
  }

  async function loadProject(id) {
    state.project = await API.getProject(id);
    const first = state.project.scenes[0]?.id || null;
    state.selectedSceneId = first;
    state.selectedSceneIds = first ? [first] : [];
    _selectionAnchorId = first;
    state.gen = { state: "idle", promptId: null, media: [], msg: "" };
    state.sceneRenders = {};  // per-session; not persisted
    state.sceneGhosts = [];
    _genInFlightIds.clear();
    _removedDuringGen.clear();
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
    state.project = null; state.selectedSceneId = null; state.selectedSceneIds = [];
    _selectionAnchorId = null;
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
  function unsetSamplerInput(name) {
    if (!state.project) return;
    const prev = { ...(state.project.sampler_inputs || {}) };
    delete prev[name];
    patchProjectQuiet({ sampler_inputs: prev });
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
      _pruneSelection();
      notify();
      refreshProjectList();
      refreshPreview(true);
    } catch (e) {
      state.saving = false; notify();
      console.error("save failed", e);
    }
  }

  function patchProject(patch) { if (!state.project) return; Object.assign(state.project, patch); notify(); scheduleSave(); }
  function patchProjectQuiet(patch) { if (!state.project) return; Object.assign(state.project, patch); scheduleSaveSilent(); }

  function scene(id) { return state.project?.scenes.find((s) => s.id === id) || null; }

  function genUnitId(sc) { return (sc && sc.gen_unit_id) || (sc && sc.id) || ""; }
  function isGenSubclip(sc) { return !!((sc && sc.cut_offset_frames) > 0); }
  function genUnitRoot(unitId) {
    const group = (state.project?.scenes || []).filter((s) => genUnitId(s) === unitId);
    return group.find((s) => !isGenSubclip(s)) || group[0] || null;
  }
  function genUnitSceneIds(unitId) {
    return (state.project?.scenes || []).filter((s) => genUnitId(s) === unitId && !s.excluded).map((s) => s.id);
  }

  function _patchSceneTarget(id, patch) {
    const s = scene(id); if (!s) return null;
    const root = isGenSubclip(s) ? genUnitRoot(genUnitId(s)) : s;
    const targetId = (root && isGenSubclip(s) && (patch.text != null || patch.rating != null || patch.source != null || patch.character_ids != null))
      ? root.id : id;
    return scene(targetId);
  }

  function _anchorPatchChanged(before, patch) {
    if (!patch.source) return false;
    const b = before?.source || {};
    const next = { ...b, ...patch.source };
    return next.media_ref !== b.media_ref
      || (patch.source.type != null && next.type !== (b.type || "carry"));
  }

  // Dropping or picking a new i2v anchor must drop the cached render and rating —
  // otherwise preview keeps the old video and Studio still refines from the prior rating.
  function _invalidateSceneAfterAnchorChange(target) {
    if (!target || !state.project) return;
    const uid = genUnitId(target);
    (state.project.scenes || []).filter((sc) => genUnitId(sc) === uid).forEach((sc) => {
      delete state.sceneRenders[sc.id];
    });
    const root = genUnitRoot(uid);
    if (root?.rating) root.rating = "";
  }

  function _applyScenePatch(id, patch, quiet) {
    const t = _patchSceneTarget(id, patch); if (!t) return;
    if (_anchorPatchChanged(t, patch)) _invalidateSceneAfterAnchorChange(t);
    const merged = { ...patch };
    if (merged.source) merged.source = { ...(t.source || {}), ...merged.source };
    Object.assign(t, merged);
    if (quiet) scheduleSaveSilent(); else { notify(); scheduleSave(); }
  }

  function patchScene(id, patch) { _applyScenePatch(id, patch, false); }
  function patchSceneQuiet(id, patch) {
    const s = scene(id); if (!s) return;
    if (_anchorPatchChanged(s, patch)) _invalidateSceneAfterAnchorChange(s);
    const merged = { ...patch };
    if (merged.source) merged.source = { ...(s.source || {}), ...merged.source };
    Object.assign(s, merged); scheduleSaveSilent();
  }

  function _sceneOrder() { return (state.project?.scenes || []).map((s) => s.id); }

  function _pruneSelection() {
    if (!state.project) {
      state.selectedSceneIds = [];
      state.selectedSceneId = null;
      _selectionAnchorId = null;
      return;
    }
    const valid = new Set(_sceneOrder());
    state.selectedSceneIds = (state.selectedSceneIds || []).filter((sid) => valid.has(sid));
    // null selectedSceneId = project settings view — keep it unless focus id vanished.
    if (state.selectedSceneId !== null && !valid.has(state.selectedSceneId))
      state.selectedSceneId = state.selectedSceneIds[0] || state.project.scenes[0]?.id || null;
    if (state.selectedSceneId && !state.selectedSceneIds.includes(state.selectedSceneId))
      state.selectedSceneIds = [...state.selectedSceneIds, state.selectedSceneId];
    if (!state.selectedSceneIds.length && state.selectedSceneId)
      state.selectedSceneIds = [state.selectedSceneId];
    if (_selectionAnchorId && !valid.has(_selectionAnchorId))
      _selectionAnchorId = state.selectedSceneId;
  }

  // Plain click = single; ⌘/Ctrl = toggle; Shift = range from anchor.
  function selectScene(id, opts) {
    opts = opts || {};
    const additive = !!opts.additive;
    const range = !!opts.range;
    const order = _sceneOrder();

    if (id == null) {
      state.selectedSceneId = null;
      state.selectedSceneIds = [];
      _selectionAnchorId = null;
      notify();
      return;
    }
    if (!scene(id)) return;

    if (range && _selectionAnchorId && order.includes(_selectionAnchorId)) {
      const a = order.indexOf(_selectionAnchorId);
      const b = order.indexOf(id);
      const lo = Math.min(a, b);
      const hi = Math.max(a, b);
      const rangeIds = order.slice(lo, hi + 1);
      if (additive) {
        const set = new Set([...(state.selectedSceneIds || []), ...rangeIds]);
        state.selectedSceneIds = order.filter((sid) => set.has(sid));
      } else state.selectedSceneIds = rangeIds;
      state.selectedSceneId = id;
      notify();
      return;
    }

    if (additive) {
      const set = new Set(state.selectedSceneIds || []);
      if (set.has(id)) {
        set.delete(id);
        if (state.selectedSceneId === id)
          state.selectedSceneId = [...set].pop() || null;
      } else {
        set.add(id);
        state.selectedSceneId = id;
      }
      state.selectedSceneIds = order.filter((sid) => set.has(sid));
      _selectionAnchorId = id;
      notify();
      return;
    }

    state.selectedSceneId = id;
    state.selectedSceneIds = [id];
    _selectionAnchorId = id;
    notify();
  }

  function selectedSceneCount() {
    return (state.selectedSceneIds || []).filter((sid) => {
      const s = scene(sid);
      return s && !s.excluded;
    }).length;
  }

  function addScene() {
    if (!state.project) return;
    // Default to "carry": a new scene continues the previous one (overlap) unless the
    // user picks an anchor/empty. Scene 1 carrying just starts a fresh run.
    const s = { text: "", transition_to_next: "", source: { type: "carry" }, excluded: false };
    state.project.scenes.push(s);
    window.Timeline?.requestAutoFit?.();
    notify(); scheduleSave(); // server assigns id; reselect after commit
  }

  function removeScene(id) {
    if (!state.project) return;
    const arr = state.project.scenes;
    const idx = arr.findIndex((s) => s.id === id);
    if (idx < 0) return;
    const sc = arr[idx];
    const unitId = genUnitId(sc);
    const unitMates = arr.filter((s) => genUnitId(s) === unitId);
    if (!isGenSubclip(sc) && unitMates.length > 1) {
      const next = unitMates.find((s) => s.id !== id);
      if (next) {
        next.cut_offset_frames = 0;
        next.text = sc.text || next.text;
        next.rating = sc.rating || next.rating;
        next.source = JSON.parse(JSON.stringify(sc.source || next.source || {}));
        const removedOffset = sc.frames || 0;
        unitMates.filter((s) => s.id !== id).forEach((s) => {
          if ((s.cut_offset_frames || 0) > (sc.cut_offset_frames || 0))
            s.cut_offset_frames = Math.max(0, (s.cut_offset_frames || 0) - removedOffset);
        });
      }
    }
    const inFlight = _genInFlightIds.has(id);
    if (inFlight) _removedDuringGen.add(id);
    const newAnchor = idx > 0 ? arr[idx - 1].id : null;
    state.sceneGhosts = (state.sceneGhosts || []).map((g) =>
      g.afterSceneId === id ? { ...g, afterSceneId: newAnchor } : g
    );
    const r = state.sceneRenders[id];
    if (sc && (r?.media || inFlight)) {
      state.sceneGhosts = (state.sceneGhosts || []).filter((g) => g.id !== id);
      state.sceneGhosts.push({
        id,
        afterSceneId: newAnchor,
        gen_unit_id: genUnitId(sc),
        text: sc.text || "",
        frames: sc.frames,
        frames_mode: sc.frames_mode,
        fps: sc.fps,
        fps_mode: sc.fps_mode,
        effects: sc.effects || {},
        audio_volume: sc.audio_volume,
        media: r?.media || null,
        inSec: r?.inSec || 0,
        pendingGen: inFlight && !r?.media,
      });
    } else {
      state.sceneGhosts = (state.sceneGhosts || []).filter((g) => g.id !== id);
    }
    state.project.scenes = arr.filter((s) => s.id !== id);
    delete state.sceneRenders[id];
    state.selectedSceneIds = (state.selectedSceneIds || []).filter((sid) => sid !== id);
    if (state.selectedSceneId === id)
      state.selectedSceneId = state.selectedSceneIds[0] || state.project.scenes[0]?.id || null;
    if (_selectionAnchorId === id) _selectionAnchorId = state.selectedSceneId;
    window.Timeline?.requestAutoFit?.();
    notify(); scheduleSave();
  }

  function dismissGhost(id) {
    state.sceneGhosts = (state.sceneGhosts || []).filter((g) => g.id !== id);
    _removedDuringGen.delete(id);
    notify();
  }

  // Duration of a live scene or a removed-scene ghost snapshot.
  function segmentDurationSec(seg) {
    const p = state.project;
    if (!p || !seg) return 0;
    if (seg.kind === "scene") return sceneDurationSec(seg.scene);
    const g = seg.ghost;
    const fps = (g.fps_mode !== "project" && g.fps != null) ? g.fps : p.frame_rate;
    const frames = (g.frames_mode !== "project" && g.frames != null) ? g.frames : p.num_frames_per_scene;
    return (frames || 1) / (fps || 25);
  }

  // Preview/timeline layout: live scenes interleaved with removed-scene ghosts.
  function buildPreviewSegments() {
    const p = state.project;
    if (!p) return [];
    const ghosts = state.sceneGhosts || [];
    const byAnchor = new Map();
    for (const g of ghosts) {
      const key = g.afterSceneId || "__start__";
      if (!byAnchor.has(key)) byAnchor.set(key, []);
      byAnchor.get(key).push(g);
    }
    const placed = new Set();
    const ordered = [];
    const pushGhost = (g) => {
      if (placed.has(g.id)) return;
      placed.add(g.id);
      ordered.push({ kind: "ghost", ghost: g, id: `ghost:${g.id}` });
    };
    for (const g of (byAnchor.get("__start__") || [])) pushGhost(g);
    for (const sc of (p.scenes || [])) {
      ordered.push({ kind: "scene", scene: sc, id: sc.id });
      for (const g of (byAnchor.get(sc.id) || [])) pushGhost(g);
    }
    // Orphans (stale afterSceneId) still show — appended after live clips.
    for (const g of ghosts) { if (!placed.has(g.id)) pushGhost(g); }
    let acc = 0;
    return ordered.map((seg) => {
      const durationSec = segmentDurationSec(seg);
      const out = { ...seg, offsetSec: acc, durationSec };
      acc += durationSec;
      return out;
    });
  }

  function previewTotalSec() {
    const segs = buildPreviewSegments();
    return segs.length ? segs[segs.length - 1].offsetSec + segs[segs.length - 1].durationSec : 0;
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

  // Split a clip in two at `atFrames` (defaults to the midpoint). Both halves stay one
  // generative unit — Generate collapses them back into a single uncut scene.
  function splitScene(id, atFrames) {
    if (!state.project) return;
    const arr = state.project.scenes;
    const i = arr.findIndex((s) => s.id === id); if (i < 0) return;
    const s = arr[i];
    const fps = s.fps != null ? s.fps : state.project.frame_rate;
    const frames = s.frames != null ? s.frames : state.project.num_frames_per_scene;
    const cut = snapFrames(atFrames != null ? atFrames : frames / 2);
    if (cut <= 9 || cut >= frames) return;
    const unitId = genUnitId(s);
    const baseOffset = s.cut_offset_frames || 0;
    const second = JSON.parse(JSON.stringify(s));
    // Assign a client id up-front (server honors it) so the rendered video can be mapped
    // to BOTH halves immediately — like an NLE, a split yields two clips of one source.
    second.id = "c" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
    second.gen_unit_id = unitId;
    second.cut_offset_frames = baseOffset + cut;
    second.frames = snapFrames(frames - cut);
    second.text = "";
    second.rating = "";
    second.transition_to_next = s.transition_to_next || "";
    second.transition_frames = s.transition_frames || null;
    s.gen_unit_id = unitId;
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
    if (!state.project) return false;
    state.project.global_prompt = text;
    let res;
    try { res = await API.parsePrompt(state.project.id, text); }
    catch (e) { alert("Could not parse the global prompt: " + e.message); return false; }
    // The verbatim split is authoritative: correct (shortcut-aware) boundaries, scene
    // text kept exactly as typed, anchor + scenes reproduce the global prompt.
    const v = res.parsed_verbatim || res.parsed_raw || res.parsed || {};
    if (!(v.scenes || []).length) { alert("Nothing parsed — no scenes detected."); return false; }

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
    state.sceneGhosts = [];  // scene ids changed — old ghosts no longer anchor correctly
    const firstId = next[0]?.id || null;
    state.selectedSceneId = firstId;
    state.selectedSceneIds = firstId ? [firstId] : [];
    _selectionAnchorId = firstId;
    // Refresh the inspector immediately — don't wait for save debounce + preview round-trip.
    state.preview = {
      ...(state.preview || {}),
      combined_prompt: text,
      display_prompt: text,
      parsed: res.parsed,
      parsed_raw: res.parsed_raw,
      parsed_verbatim: res.parsed_verbatim,
      for_generation: false,
    };
    notify();
    clearTimeout(saveTimer);
    saveTimer = null;
    state.saving = true;
    notify();
    try {
      await commit();
    } catch (e) {
      console.error("save after apply failed", e);
      return false;
    }
    return true;
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
  function refreshPreview(immediate = false) {
    clearTimeout(previewTimer);
    previewTimer = null;
    const run = async () => {
      if (!state.project) return;
      try { state.preview = await API.preview(state.project.id, false, true); }
      catch (e) { state.preview = { parse_error: e.message }; }
      notify();
    };
    if (immediate) return run();
    previewTimer = setTimeout(run, 250);
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
  // Scene ids currently inside an in-flight generate/poll (montage runs included).
  const _genInFlightIds = new Set();
  // Removed from the timeline while their run was still generating — completion updates
  // the ghost only, never sceneRenders, and ghosts are not pruned as "regenerated".
  const _removedDuringGen = new Set();

  function _markGenInFlight(ids) { (ids || []).forEach((id) => _genInFlightIds.add(id)); }
  function _clearGenInFlight(ids) { (ids || []).forEach((id) => _genInFlightIds.delete(id)); }

  function _clearGenTimers() { clearInterval(pollTimer); clearInterval(progressTimer); }

  // Ask ComfyUI to stop the current generation. The running poll then resolves as
  // "Interrupted" and the run loop stops.
  async function interrupt() {
    _interrupted = true;
    updateGenProgress({ msg: "Interrupting…" });
    try { await API.interrupt(); } catch (_) {}
  }

  function _elapsed() {
    const s = Math.floor((Date.now() - pollStart) / 1000);
    return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
  }

  // Split the active scenes into generation runs. Each anchored scene (empty / image /
  // generated_frame / mixed) is its own queue request. Only "carry" continues the
  // previous run (continuous chain overlap). Mixed = solo i2v + prior-scene guides.
  // True if a scene's image anchor still exists in the media bin (deleted image → fall
  // back to carry / i2v guides, not a broken anchor).
  function _anchorAvailable(s) {
    const t = s.source && s.source.type;
    if (t !== "image" && t !== "generated_frame" && t !== "mixed") return false;
    const ref = s.source.media_ref;
    return !!(ref && (state.mediaBin || []).some((m) => m.id === ref));
  }
  function _runs() {
    const active = state.project.scenes.filter((s) => !s.excluded);
    const runs = [];
    for (const s of active) {
      const t = (s.source && s.source.type) || "empty";
      // Editorial subclips always continue the current run; carry / broken anchors too.
      const isCarry = isGenSubclip(s)
        || t === "carry"
        || ((t === "image" || t === "generated_frame" || t === "mixed") && !_anchorAvailable(s));
      if (isCarry && runs.length) runs[runs.length - 1].push(s.id);
      else runs.push([s.id]);
    }
    return runs;
  }

  // Expand selection to full generative units, then slice each chain run from the
  // first to last selected scene in that run (keeps carry / overlap context).
  function _expandSelection(sceneIds) {
    const wanted = new Set();
    for (const id of sceneIds || []) {
      const sc = scene(id);
      if (!sc || sc.excluded) continue;
      genUnitSceneIds(genUnitId(sc)).forEach((sid) => wanted.add(sid));
    }
    return wanted;
  }

  function _runsForSceneIds(sceneIds) {
    const wanted = _expandSelection(sceneIds);
    if (!wanted.size) return [];
    const out = [];
    for (const run of _runs()) {
      const hit = run.filter((id) => wanted.has(id));
      if (!hit.length) continue;
      const idxs = hit.map((id) => run.indexOf(id));
      out.push(run.slice(Math.min(...idxs), Math.max(...idxs) + 1));
    }
    return out;
  }

  // Record a completed run's output: map each of the run's scenes to the one source
  // video at its cumulative in-point, so splits/deletes later play the right portions.
  function _pruneGhostsAfterRegen(recordedSceneIds) {
    const ids = new Set(recordedSceneIds || []);
    state.sceneGhosts = (state.sceneGhosts || []).filter((g) => !ids.has(g.afterSceneId) && !ids.has(g.id));
  }

  function _ghostDurationSec(g) {
    const p = state.project;
    const fps = (g.fps_mode !== "project" && g.fps != null) ? g.fps : p.frame_rate;
    const frames = (g.frames_mode !== "project" && g.frames != null) ? g.frames : p.num_frames_per_scene;
    return (frames || 1) / (fps || 25);
  }

  function _chainInSec(lastChain, sourceEnd, curr) {
    if (!lastChain) return 0;
    const sameUnit = genUnitId(lastChain) === genUnitId(curr);
    return sameUnit ? sourceEnd : Math.max(0, sourceEnd - _overlapBetweenScenes(lastChain, curr));
  }

  // Pixel overlap between consecutive scenes inside one carry chain run.
  function _overlapBetweenScenes(prev, curr) {
    if (!prev || !curr) return 0;
    const frames = +(state.project?.sampler_inputs?.frame_overlap ?? 16);
    const fps = state.project?.frame_rate || 25;
    return Math.max(0, frames / fps);
  }

  function _recordSegment(mediaList, targetSceneIds) {
    if (!mediaList || !mediaList.length || !targetSceneIds || !targetSceneIds.length) return;
    const primary = mediaList.find((m) => m.kind === "videos" || m.kind === "gifs") || mediaList[0];
    let sourceEnd = 0;
    let lastChain = null;
    let clearedRating = false;
    const recordedSceneIds = [];
    for (let i = 0; i < targetSceneIds.length; i++) {
      const id = targetSceneIds[i];
      if (_removedDuringGen.has(id)) {
        const ghosts = state.sceneGhosts || [];
        const gi = ghosts.findIndex((g) => g.id === id);
        if (gi >= 0) {
          const ghost = ghosts[gi];
          const inSec = _chainInSec(lastChain, sourceEnd, ghost);
          ghosts[gi] = { ...ghost, media: primary, inSec, pendingGen: false };
          state.sceneGhosts = ghosts;
          sourceEnd = inSec + _ghostDurationSec(ghost);
          lastChain = ghost;
        }
        _removedDuringGen.delete(id);
        continue;
      }
      const sc = scene(id); if (!sc) continue;
      const inSec = _chainInSec(lastChain, sourceEnd, sc);
      state.sceneRenders[id] = { media: primary, inSec };
      recordedSceneIds.push(id);
      const root = genUnitRoot(genUnitId(sc));
      if (root && root.rating) { root.rating = ""; clearedRating = true; }
      sourceEnd = inSec + sceneDurationSec(sc);
      lastChain = sc;
    }
    _pruneGhostsAfterRegen(recordedSceneIds);
    if (clearedRating) scheduleSaveSilent();
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
          if (pr && pr.max > 0) {
            const head = (state.gen.msg || prefix).replace(/\s*·\s*sampling \d+\/\d+$/, "");
            updateGenProgress({
              step: pr.value,
              maxStep: pr.max,
              msg: `${head}  ·  sampling ${pr.value}/${pr.max}`,
            });
          }
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
            const step = (state.gen.maxStep > 0) ? `  ·  sampling ${state.gen.step}/${state.gen.maxStep}` : "";
            updateGenProgress({ state: s.state, msg: `${prefix} ${_elapsed()}${step}` });
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
    _markGenInFlight(sceneIds);
    set({ gen: { state: "queuing", promptId: null, media: [], msg: `${prefix}: queuing…`, step: 0, maxStep: 0 } });
    try {
      const r = await API.generate(state.project.id, onlyScene || null, onlyScene ? null : sceneIds, !!resetSession);
      if (!r.prompt_id) { set({ gen: { ...state.gen, state: "error", msg: "No prompt id returned." } }); return false; }
      if (r.validation && state.project) {
        state.project.generation_meta = {
          ...(state.project.generation_meta || {}),
          prompt_hash: r.validation.prompt_hash,
          run_hash: r.validation.run_hash,
        };
      }
      pollStart = Date.now();
      let runMsg = `${prefix}: generating…`;
      if (r.prompt_repairs_cleared) {
        const anchorOnly = r.validation?.anchors_changed_since_last_queue && !r.validation?.text_changed_since_last_queue;
        runMsg += anchorOnly
          ? " · guides refreshed, stale repairs cleared"
          : " · stale repairs cleared (training kept)";
      }
      else if (r.reset_session) runMsg += " · Studio session reset";
      set({ gen: { state: "running", promptId: r.prompt_id, media: [], msg: runMsg } });
      return await _pollPromise(r.prompt_id, sceneIds, prefix);
    } catch (e) {
      set({ gen: { state: "error", promptId: null, media: [], msg: _friendlyGenError(e.message) } });
      return false;
    } finally {
      _clearGenInFlight(sceneIds);
    }
  }

  async function generate(onlyScene) {
    if (!state.project) return;
    await flushSave();  // ensure the server has the latest edits before generating
    if (!onlyScene) return generateMontage();
    const reset = _resetSessionPending; _resetSessionPending = false;
    if (reset) state.resetSessionArmed = false;
    const sc = scene(onlyScene);
    const ids = sc ? genUnitSceneIds(genUnitId(sc)) : [onlyScene];
    await _generateRun(ids, onlyScene, "Generating scene", reset);
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

  async function generateSelected() {
    if (!state.project) return;
    const ids = (state.selectedSceneIds || []).filter((sid) => {
      const s = scene(sid);
      return s && !s.excluded;
    });
    if (!ids.length) {
      set({ gen: { state: "error", promptId: null, media: [], msg: "No scenes selected." } });
      return;
    }
    await flushSave();
    const runs = _runsForSceneIds(ids);
    if (!runs.length) {
      set({ gen: { state: "error", promptId: null, media: [], msg: "No generatable runs in the selection." } });
      return;
    }
    const reset = _resetSessionPending; _resetSessionPending = false;
    if (reset) state.resetSessionArmed = false;
    for (let i = 0; i < runs.length; i++) {
      const prefix = runs.length > 1 ? `Selected run ${i + 1}/${runs.length}` : "Generating selection";
      const ok = await _generateRun(runs[i], null, prefix, reset && i === 0);
      if (!ok) return;
    }
    const n = ids.length;
    set({
      gen: {
        state: "done", promptId: null, media: state.gen.media,
        msg: `${runs.length} run(s) generated for ${n} selected scene${n > 1 ? "s" : ""}.`,
      },
    });
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
      state.sceneGhosts = [];
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
  async function deleteMedia(id) {
    try { await API.deleteMedia(id); } catch (_) {}
    if (state.mediaPreviewId === id) state.mediaPreviewId = null;
    await loadMedia();
  }

  async function loadTransitions() { try { state.transitions = (await API.transitions()).transitions || []; } catch (_) {} notify(); }
  async function loadShortcuts() { try { state.shortcuts = (await API.shortcuts()).shortcuts || []; } catch (_) { state.shortcuts = []; } notify(); }
  async function loadCharacters() { try { state.characters = (await API.characters()).characters || []; } catch (_) { state.characters = []; } notify(); }
  async function saveCharacter(item) {
    try { state.characters = (await API.saveCharacter(item)).characters || state.characters; notify(); }
    catch (e) { alert("Save failed: " + e.message); }
  }
  async function deleteCharacter(id) {
    try { state.characters = (await API.deleteCharacter(id)).characters || []; notify(); }
    catch (e) { console.error(e); }
  }

  function sceneCharacterIds(sceneId) {
    const s = scene(sceneId); if (!s) return [];
    const root = isGenSubclip(s) ? genUnitRoot(genUnitId(s)) : s;
    return [...(root?.character_ids || s.character_ids || [])];
  }

  function toggleSceneCharacter(sceneId, charId) {
    const s = scene(sceneId); if (!s || !charId) return;
    const ids = sceneCharacterIds(sceneId);
    const next = ids.includes(charId) ? ids.filter((x) => x !== charId) : [...ids, charId];
    patchScene(sceneId, { character_ids: next });
  }
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
  // True when this scene continues the current chain run (not a new run anchor).
  function _continuesChainRun(s) {
    const active = state.project.scenes.filter((sc) => !sc.excluded);
    const idx = active.findIndex((sc) => sc.id === s.id);
    if (idx <= 0) return false;
    const t = (s.source && s.source.type) || "empty";
    if (t === "empty") return false;
    if (isGenSubclip(s)) return true;
    if (t === "carry" || t === "mixed") return true;
    if ((t === "image" || t === "generated_frame") && _anchorAvailable(s)) return false;
    return true;
  }

  function previewMedia(mediaId) {
    const m = (state.mediaBin || []).find((x) => x.id === mediaId);
    if (!m || m.kind !== "image") return;
    if (state.mediaPreviewId === mediaId) {
      clearMediaPreview();
      return;
    }
    try { window.Player?.pause?.(); } catch (_) {}
    set({ mediaPreviewId: mediaId });
  }

  function clearMediaPreview() {
    if (!state.mediaPreviewId) return;
    set({ mediaPreviewId: null });
  }

  function assignMediaToScene(sceneId, mediaId) {
    const s = scene(sceneId); if (!s) return;
    // Drag-drop is an explicit new anchor — use image i2v (not mixed/carry guides).
    const patch = { source: { ...(s.source || {}), type: "image", media_ref: mediaId } };
    if ((s.source?.media_ref || null) !== mediaId) patch.guides = [];
    patchScene(sceneId, patch);
  }

  // ── boot ─────────────────────────────────────────────────────────────────────
  async function init() {
    try { state.health = await API.health(); } catch (_) { state.health = { ok: false }; }
    try { const t = await API.transitions(); state.transitions = t.transitions || []; } catch (_) { state.transitions = []; }
    await loadShortcuts();
    await loadCharacters();
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
    patchProject, patchProjectQuiet, patchScene, patchSceneQuiet, flushSave, selectScene, addScene, removeScene, dismissGhost, moveScene, moveSceneTo, scene,
    sceneCharacterIds, toggleSceneCharacter,
    genUnitId, isGenSubclip, genUnitRoot, genUnitSceneIds,
    buildPreviewSegments, previewTotalSec, segmentDurationSec,
    addAudioTrack, updateAudioTrack, removeAudioTrack,
    resizeScene, splitScene, snapFrames,
    refreshPreview, syncFromPreview, applyGlobalPrompt, generate, generateMontage, generateSelected, selectedSceneCount, renderFinal, exportSelected, interrupt, loadModels, loadImageTargets, setModelInput, setModelLink,
    setConditioningSlot, setSamplerSlot, setSamplerInput, setSamplerInputNow, unsetSamplerInput, setStudioInput, setStudioInputNow,
    loadMedia, uploadMedia, deleteMedia, previewMedia, clearMediaPreview, assignMediaToScene,
    loadShortcuts, saveShortcut, deleteShortcut, importShortcuts,
    loadCharacters, saveCharacter, deleteCharacter,
    loadTransitions, saveTransition, deleteTransition, importTransitions,
    applyTransitionToSelection, insertShortcutIntoSelection,
    setSceneRating: (id, v) => patchScene(id, { rating: v }),
    resetStudioSession,
  };
})();
