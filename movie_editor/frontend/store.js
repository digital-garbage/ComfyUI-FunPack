// Central state + data actions. Views subscribe and re-render their zone.
(function () {
  const API = window.MovieEditorAPI;

  const state = {
    projects: [],          // [{id,name,scene_count,updated_at}]
    project: null,         // full Project
    selectedSceneId: null,   // focus clip (inspector / toolbar)
    selectedSceneIds: [],    // multi-select for generate / timeline
    transitions: [],       // prompt split-marker library (FunPack transition DB; generation only)
    nleEffects: [],        // post-render clip effects (zoom, blur, fades)
    nleVideoTransitions: [], // post-render seam blends (crossfade, wipe, …)
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
    notice: "",
    saving: false,
    unsaved: false,
    models: { slots: [] },   // pluggable node config (shared with Models modal)
    mediaBin: [],            // uploaded assets [{id,name,kind,...}]
    mediaPreviewId: null,    // transient image preview in the player (not scene assignment)
    shortcuts: [],           // prompt shortcut library
    characters: [],          // global character library
    imageTargets: [],        // where an image asset can be wired [{value,label}]
    ratingLabels: [],        // FunPack Studio V2 rating options
  };

  const listeners = new Set();
  const SAVE_DEBOUNCE_MS = 5000;        // discrete edits (dropdowns, toggles)
  const SAVE_SILENT_DEBOUNCE_MS = 20000; // typing in fields — flushed on blur / generate
  let saveTimer = null;
  let _localDirty = false;       // local edits newer than the in-flight save snapshot
  let _commitPromise = null;     // avoid overlapping saves stomping newer anchor edits
  let _commitQueued = false;
  let _selectionAnchorId = null;  // shift-click range anchor

  function notify() { listeners.forEach((fn) => { try { fn(state); } catch (e) { console.error(e); } }); }
  function subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }
  function set(patch) { Object.assign(state, patch); notify(); }

  function _syncEditorStateToProject() {
    if (!state.project) return;
    state.project.scene_renders = JSON.parse(JSON.stringify(state.sceneRenders || {}));
    state.project.scene_ghosts = JSON.parse(JSON.stringify(state.sceneGhosts || []));
  }

  function _historyRecord() {
    if (window.EditorHistory?.isCoalescing()) return;
    if (window.EditorHistory && !window.EditorHistory.isApplying()) window.EditorHistory.record(window.Store);
  }

  function notifyHistoryState() {
    try { window.dispatchEvent(new CustomEvent("funpack-history-state")); } catch (_) {}
  }

  function scheduleSaveFromHistory() {
    _localDirty = true;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => { saveTimer = null; commit(); }, SAVE_DEBOUNCE_MS);
    _notifySaveChip();
  }

  // Update generation progress WITHOUT a full notify(): a store notify rebuilds the whole
  // editor (inspector/timeline/player), which during a long generation kept interrupting the
  // user (couldn't save frames to the bin or add effects). Progress ticks instead mutate
  // state.gen quietly and fire a DOM event the player listens to, updating only the readout.
  function updateGenProgress(patch) {
    Object.assign(state.gen, patch);
    try { window.dispatchEvent(new CustomEvent("funpack-gen-progress", { detail: state.gen })); } catch (_) {}
  }
  function get() { return state; }

  function _assignedCharacterIds(project) {
    const ids = new Set();
    (project?.scenes || []).forEach((s) => (s.character_ids || []).forEach((id) => ids.add(id)));
    return ids;
  }

  function _characterPreviewKey(project) {
    const assigned = _assignedCharacterIds(project);
    return (state.characters || [])
      .filter((c) => assigned.has(c.id))
      .map((c) => [
        c.id, c.name, c.appearance, c.body, c.wardrobe, c.always_include, c.never_include,
        c.face_ref, c.body_ref, c.detail_ref,
      ].join("\x1f"))
      .sort()
      .join("|");
  }

  function _promptPreviewKey(project) {
    if (!project) return "";
    return JSON.stringify({
      anchor: project.anchor,
      global_prompt: project.global_prompt,
      intro_transition: project.intro_transition,
      character_bible: project.character_bible,
      characters: _characterPreviewKey(project),
      scenes: (project.scenes || []).map((s) => ({
        text: s.text,
        excluded: s.excluded,
        character_ids: s.character_ids,
        transition_to_next: s.transition_to_next,
        source_type: s.source?.type,
        media_ref: s.source?.media_ref,
      })),
    });
  }

  let _lastPreviewKey = "";

  function _invalidateGlobalPromptDraft() {
    try { window.dispatchEvent(new CustomEvent("funpack-invalidate-global-prompt")); } catch (_) {}
  }

  function _dispatchGlobalPromptUpdated(text) {
    try {
      window.dispatchEvent(new CustomEvent("funpack-global-prompt-updated", { detail: { text: text || "" } }));
    } catch (_) {}
  }

  function _groupGenerativeUnits(scenes) {
    const units = [];
    for (const sc of scenes || []) {
      const uid = genUnitId(sc);
      if (units.length && units[units.length - 1][0] === uid) units[units.length - 1][1].push(sc);
      else units.push([uid, [sc]]);
    }
    return units;
  }

  function _rootFromGroup(group) {
    return (group || []).find((s) => !isGenSubclip(s)) || group[0] || null;
  }

  // Verbatim montage text rebuilt from timeline fields (display mode — transitions between scenes).
  function buildGlobalPromptFromTimeline(project) {
    if (!project) return "";
    const parts = [];
    const anchor = (project.anchor || "").trim();
    if (anchor) parts.push(anchor);
    const active = (project.scenes || []).filter((s) => !s.excluded && isGenerativeScene(s));
    const units = _groupGenerativeUnits(active);
    units.forEach(([, group], i) => {
      const root = _rootFromGroup(group);
      if (!root) return;
      if (i === 0) {
        const intro = (project.intro_transition || "").trim();
        if (intro && anchor) parts.push(intro);
      } else {
        const prevRoot = _rootFromGroup(units[i - 1][1]);
        const tr = (prevRoot?.transition_to_next || "").trim();
        if (tr) parts.push(tr);
      }
      const text = (root.text || "").trim();
      if (text) parts.push(text);
    });
    return parts.join(" ").trim();
  }

  function _patchAffectsCombinedPrompt(patch) {
    if (!patch || typeof patch !== "object") return false;
    return patch.text != null
      || patch.transition_to_next != null
      || patch.excluded != null
      || patch.anchor != null
      || patch.intro_transition != null
      || patch.scenes != null;
  }

  let _timelinePromptTimer = null;
  let _pinnedGlobalPrompt = null;

  function _restorePinnedGlobalPrompt() {
    const pin = (_pinnedGlobalPrompt || "").trim();
    if (!pin || !state.project) return;
    state.project.global_prompt = pin;
    state.preview = {
      ...(state.preview || {}),
      display_prompt: pin,
      combined_prompt: pin,
    };
    _dispatchGlobalPromptUpdated(pin);
  }

  function syncGlobalPromptFromTimeline() {
    if (!state.project) return;
    _pinnedGlobalPrompt = null;
    const built = buildGlobalPromptFromTimeline(state.project);
    const prev = (state.project.global_prompt || "").trim();
    if (built === prev && (state.preview?.display_prompt || "") === built) return;
    state.project.global_prompt = built;
    state.preview = {
      ...(state.preview || {}),
      display_prompt: built,
      combined_prompt: built,
    };
    _lastPreviewKey = _promptPreviewKey(state.project);
    _invalidateGlobalPromptDraft();
    _dispatchGlobalPromptUpdated(built);
    notify();
    clearTimeout(_timelinePromptTimer);
    _timelinePromptTimer = setTimeout(() => refreshPreview(true), 300);
  }

  let _globalApplyTimer = null;
  function scheduleGlobalPromptApply(text) {
    clearTimeout(_globalApplyTimer);
    _globalApplyTimer = setTimeout(() => {
      applyGlobalPromptQuiet(text).catch((e) => console.warn("Global prompt apply failed:", e));
    }, 500);
  }

  function _normalizeSceneText(t) {
    return String(t || "").trim().replace(/\s+/g, " ");
  }

  function _newSceneId() {
    return "s" + Math.random().toString(36).slice(2, 11);
  }

  function _newSceneFromParsed(ps) {
    const id = _newSceneId();
    return {
      id,
      gen_unit_id: id,
      text: ps.text || "",
      transition_to_next: "",
      source: { type: "carry" },
      excluded: false,
    };
  }

  // Restore a ghost as a fresh timeline clip (new id — never reuse the ghost's id).
  function _materializeSceneFromGhost(ghost, text) {
    const id = _newSceneId();
    const sc = {
      id,
      gen_unit_id: id,
      text: text || ghost.text || "",
      transition_to_next: "",
      source: { type: "carry" },
      excluded: false,
      frames: ghost.frames,
      frames_mode: ghost.frames_mode,
      fps: ghost.fps,
      fps_mode: ghost.fps_mode,
      effects: JSON.parse(JSON.stringify(ghost.effects || {})),
      audio_volume: ghost.audio_volume,
      audio_separated: false,
    };
    if (ghost.media) {
      state.sceneRenders[id] = {
        media: ghost.media,
        inSec: ghost.inSec || 0,
      };
    }
    return sc;
  }

  function _afterAnchorForRemoved(removedSc, oldActive, keptSceneIds) {
    const idx = oldActive.findIndex((s) => s.id === removedSc.id);
    for (let j = idx - 1; j >= 0; j--) {
      if (keptSceneIds.has(oldActive[j].id)) return oldActive[j].id;
    }
    return null;
  }

  // Ghost a removed scene when it had a render; otherwise drop it entirely.
  function _ghostOrDropScene(sc, afterSceneId, ghosts) {
    const r = state.sceneRenders[sc.id];
    const inFlight = _genInFlightIds.has(sc.id);
    state.project.audio_tracks = (state.project.audio_tracks || []).filter(
      (t) => !(t.kind === "separated" && t.scene_id === sc.id)
    );
    delete state.sceneRenders[sc.id];
    let out = (ghosts || []).filter((g) => g.id !== sc.id);
    if (!r?.media && !inFlight) return out;
    out.push({
      id: sc.id,
      afterSceneId: afterSceneId || null,
      gen_unit_id: genUnitId(sc),
      text: sc.text || "",
      frames: sc.frames,
      frames_mode: sc.frames_mode,
      fps: sc.fps,
      fps_mode: sc.fps_mode,
      effects: JSON.parse(JSON.stringify(sc.effects || {})),
      audio_volume: sc.audio_volume,
      media: r?.media || null,
      inSec: r?.inSec || 0,
      pendingGen: inFlight && !r?.media,
    });
    return out;
  }

  function _mergeGenerativeWithVideoClips(oldScenes, generativeNext) {
    const oldOrder = (oldScenes || []).filter((s) => !s.excluded);
    const videoIds = new Set(oldOrder.filter(isVideoClip).map((s) => s.id));
    if (!videoIds.size) return generativeNext;
    const byId = new Map((oldScenes || []).map((s) => [s.id, s]));
    const out = [];
    let g = 0;
    for (const sc of oldOrder) {
      if (videoIds.has(sc.id)) {
        out.push(JSON.parse(JSON.stringify(byId.get(sc.id) || sc)));
      } else if (isGenerativeScene(sc)) {
        if (g < generativeNext.length) out.push(generativeNext[g++]);
      }
    }
    while (g < generativeNext.length) out.push(generativeNext[g++]);
    return out;
  }

  function _generativeRoots(oldScenes) {
    const active = (oldScenes || []).filter((s) => !s.excluded && isGenerativeScene(s));
    return _groupGenerativeUnits(active).map(([, group]) => _rootFromGroup(group)).filter(Boolean);
  }

  function _markGenUnitUsed(uid, usedOld, oldAll) {
    for (const s of oldAll) {
      if (genUnitId(s) === uid) usedOld.add(s.id);
    }
  }

  function _reuseGenerativeRoot(root, text) {
    const sc = JSON.parse(JSON.stringify(root));
    sc.text = text || "";
    sc.cut_offset_frames = 0;
    sc.gen_unit_id = sc.id;
    return sc;
  }

  // Global prompt is authoritative. Parsed scene slots map positionally onto timeline
  // roots first (scene 1 always reuses the first generative clip), then fall back to
  // text matching for extras. Unmatched old roots are ghosted or dropped.
  function _alignScenesFromParsed(oldScenes, parsedScenes, ghosts) {
    const oldAll = (oldScenes || []).filter((s) => !s.excluded && isGenerativeScene(s));
    const oldRoots = _generativeRoots(oldScenes);
    const usedOld = new Set();
    const usedGhost = new Set();
    const next = [];
    let ghostsOut = [...(ghosts || [])];

    for (let i = 0; i < (parsedScenes || []).length; i++) {
      const ps = parsedScenes[i];
      const pt = _normalizeSceneText(ps.text);
      let match = null;

      if (i < oldRoots.length && !usedOld.has(oldRoots[i].id)) {
        match = oldRoots[i];
      }

      if (!match && pt) {
        match = oldAll.find((s) => !usedOld.has(s.id) && _normalizeSceneText(s.text) === pt);
      }

      if (match) {
        const root = isGenSubclip(match) ? (genUnitRoot(genUnitId(match)) || match) : match;
        _markGenUnitUsed(genUnitId(root), usedOld, oldAll);
        next.push(_reuseGenerativeRoot(root, ps.text));
        continue;
      }

      const ghost = pt
        ? ghostsOut.find((g) => !usedGhost.has(g.id) && _normalizeSceneText(g.text) === pt)
        : null;
      if (ghost) {
        usedGhost.add(ghost.id);
        next.push(_materializeSceneFromGhost(ghost, ps.text));
        continue;
      }

      next.push(_newSceneFromParsed(ps));
    }

    const keptSceneIds = new Set(next.map((s) => s.id).filter(Boolean));
    for (const root of oldRoots) {
      if (usedOld.has(root.id)) continue;
      _markGenUnitUsed(genUnitId(root), usedOld, oldAll);
      const afterId = _afterAnchorForRemoved(root, oldRoots, keptSceneIds);
      ghostsOut = _ghostOrDropScene(root, afterId, ghostsOut);
    }
    ghostsOut = ghostsOut.filter((g) => !usedGhost.has(g.id));

    const merged = _mergeGenerativeWithVideoClips(oldScenes, next);
    return { next: merged, ghosts: ghostsOut, usedGhost };
  }

  function _normalizeGlobalPromptParse(trimmed, v) {
    v = v || {};
    const text = String(trimmed || "").trim();
    const anchor = String(v.anchor || "").trim();
    const scenes = Array.isArray(v.scenes) ? v.scenes : [];
    const transitions = v.transitions || [];
    const hasSceneText = scenes.some((s) => String(s.text || "").trim());

    // No split markers / shortcuts detected: the whole global prompt is one scene.
    if (!hasSceneText && text) {
      return { anchor: "", scenes: [{ text }], transitions: [] };
    }
    // Parser kept everything in anchor with no scene chunks — fold into one scene.
    if (!hasSceneText && anchor) {
      return { anchor: "", scenes: [{ text: anchor }], transitions: [] };
    }
    return { anchor: v.anchor || "", scenes, transitions };
  }

  function _afterTimelineStructureChange() {
    _pinnedGlobalPrompt = null;
    _syncSeparatedAudioTracks();
    syncGlobalPromptFromTimeline();
  }

  async function _distributeGlobalPrompt(text, res) {
    const trimmed = String(text || "").trim();
    const v = _normalizeGlobalPromptParse(trimmed, res.parsed_verbatim || res.parsed_raw || res.parsed || {});
    if (!(v.scenes || []).length) return false;
    _pinnedGlobalPrompt = trimmed;
    state.project.global_prompt = trimmed;
    state.project.anchor = v.anchor || "";
    const old = state.project.scenes || [];
    const ghosts = state.sceneGhosts || [];
    const { next, ghosts: ghostsOut } = _alignScenesFromParsed(old, v.scenes, ghosts);
    for (const sc of next) {
      sc.transition_to_next = "";
    }
    _applyDetectedTransitions(next, v.transitions, text);
    state.project.scenes = next;
    state.sceneGhosts = ghostsOut;
    const prevSel = state.selectedSceneId;
    const stillSel = prevSel && next.some((s) => s.id === prevSel);
    const firstId = stillSel ? prevSel : (next[0]?.id || null);
    state.selectedSceneId = firstId;
    state.selectedSceneIds = firstId ? [firstId] : [];
    _selectionAnchorId = firstId;
    state.preview = {
      ...(state.preview || {}),
      combined_prompt: trimmed,
      display_prompt: trimmed,
      parsed: res.parsed,
      parsed_raw: res.parsed_raw,
      parsed_verbatim: res.parsed_verbatim,
      for_generation: false,
    };
    _lastPreviewKey = _promptPreviewKey(state.project);
    _invalidateGlobalPromptDraft();
    _dispatchGlobalPromptUpdated(trimmed);
    notify();
    return true;
  }

  async function applyGlobalPromptQuiet(text) {
    if (!state.project) return false;
    const trimmed = String(text || "").trim();
    if (!trimmed) return false;
    if (trimmed === buildGlobalPromptFromTimeline(state.project) && !_pinnedGlobalPrompt) return true;
    _historyRecord();
    let res;
    try { res = await API.parsePrompt(state.project.id, trimmed); }
    catch (e) {
      console.warn("Global prompt parse failed:", e);
      return false;
    }
    if (!await _distributeGlobalPrompt(trimmed, res)) return false;
    clearTimeout(saveTimer);
    saveTimer = null;
    _localDirty = true;
    scheduleSaveSilent();
    refreshPreview(true);
    return true;
  }

  function _syncPromptPreview() {
    return (async () => {
      try {
        await flushSave();
        await refreshPreview(true);
      } catch (e) { console.error("prompt preview sync failed", e); }
    })();
  }

  function _notifySaveChip() {
    state.unsaved = _localDirty || !!saveTimer;
    try {
      window.dispatchEvent(new CustomEvent("funpack-save-status", {
        detail: { saving: !!state.saving, unsaved: !!state.unsaved },
      }));
    } catch (_) {}
  }

  // ── project lifecycle ──────────────────────────────────────────────────────
  async function refreshProjectList(quiet) {
    const { projects } = await API.listProjects();
    state.projects = projects;
    if (!quiet) notify();
  }

  function _genUnitHasRender(unitId) {
    return (state.project?.scenes || []).some(
      (s) => genUnitId(s) === unitId && !!(state.sceneRenders[s.id]?.media?.filename)
    );
  }

  // Ratings apply to a specific render — drop stale ratings when nothing is playable.
  function _clearOrphanRatings() {
    if (!state.project) return false;
    let cleared = false;
    const seen = new Set();
    for (const sc of state.project.scenes || []) {
      if (isVideoClip(sc)) {
        if (sc.rating) { sc.rating = ""; cleared = true; }
        continue;
      }
      const uid = genUnitId(sc);
      if (seen.has(uid)) continue;
      seen.add(uid);
      const root = genUnitRoot(uid) || sc;
      if (root.rating && !_genUnitHasRender(uid)) {
        root.rating = "";
        cleared = true;
      }
      for (const s of state.project.scenes) {
        if (genUnitId(s) === uid && s.id !== root.id && s.rating) {
          s.rating = "";
          cleared = true;
        }
      }
    }
    return cleared;
  }

  let _validateRendersToken = 0;

  async function _validateSceneRenders() {
    if (!state.project) return;
    const token = ++_validateRendersToken;
    const snapshot = JSON.parse(JSON.stringify(state.sceneRenders || {}));
    const ids = Object.keys(snapshot);
    if (!ids.length) {
      if (_clearOrphanRatings()) {
        _syncEditorStateToProject();
        scheduleSave();
        notify();
      }
      return;
    }
    let missing = 0;
    await Promise.all(ids.map(async (id) => {
      if (token !== _validateRendersToken) return;
      const r = snapshot[id];
      if (!r?.media?.filename) {
        if (token !== _validateRendersToken) return;
        if (state.sceneRenders[id]) delete state.sceneRenders[id];
        missing++;
        return;
      }
      try {
        const res = await fetch(API.resultUrl(state.project.id, r.media), { method: "HEAD" });
        if (!res.ok) {
          if (token !== _validateRendersToken) return;
          const cur = state.sceneRenders[id];
          if (cur && cur.media?.filename === r.media.filename) delete state.sceneRenders[id];
          missing++;
        }
      } catch (_) {
        if (token !== _validateRendersToken) return;
        const cur = state.sceneRenders[id];
        if (cur && cur.media?.filename === r.media.filename) delete state.sceneRenders[id];
        missing++;
      }
    }));
    if (token !== _validateRendersToken) return;
    const clearedRatings = _clearOrphanRatings();
    if (missing || clearedRatings) {
      if (missing) {
        state.notice = `${missing} saved render${missing > 1 ? "s" : ""} no longer on disk — regenerate those clips.`;
      }
      _syncEditorStateToProject();
      scheduleSave();
      notify();
    }
  }

  function clearNotice() {
    if (!state.notice) return;
    state.notice = "";
    notify();
  }

  async function loadProject(id) {
    state.project = await API.getProject(id);
    state.notice = "";
    const first = state.project.scenes[0]?.id || null;
    state.selectedSceneId = first;
    state.selectedSceneIds = first ? [first] : [];
    _selectionAnchorId = first;
    state.gen = { state: "idle", promptId: null, media: [], msg: "" };
    state.sceneRenders = JSON.parse(JSON.stringify(state.project.scene_renders || {}));
    state.sceneGhosts = JSON.parse(JSON.stringify(state.project.scene_ghosts || []));
    if (_clearOrphanRatings()) {
      _syncEditorStateToProject();
      scheduleSaveSilent();
    }
    _genInFlightIds.clear();
    _removedDuringGen.clear();
    if (window.EditorHistory) window.EditorHistory.clear();
    notify();
    syncGlobalPromptFromTimeline();
    _syncSeparatedAudioTracks();
    refreshPreview();
    loadModels();
    _validateSceneRenders();
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
  // Autosave persists in the background — scheduling a save must NOT rebuild the UI
  // (that was closing dropdowns and stealing clicks). Full notify only when the user
  // edits something visible, or when prompt preview actually needs refreshing.
  function scheduleSave() {
    _localDirty = true;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => { saveTimer = null; commit(); }, SAVE_DEBOUNCE_MS);
    _notifySaveChip();
  }

  // For free-text / number fields: queue a save without re-rendering; blur/generate flush.
  function scheduleSaveSilent() {
    _localDirty = true;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => { saveTimer = null; commit(); }, SAVE_SILENT_DEBOUNCE_MS);
    _notifySaveChip();
  }

  // Commit any pending save immediately (field blur / before generate). Returns the
  // commit promise when it flushed, else null.
  function flushSave() {
    clearTimeout(saveTimer); saveTimer = null;
    if (!_localDirty && !_commitPromise) return null;
    return commit();
  }

  async function commit(opts = {}) {
    if (!state.project) return;
    if (_commitPromise) { _commitQueued = true; return _commitPromise; }
    const skipPreviewRefresh = !!opts.skipPreviewRefresh;
    _commitPromise = (async () => {
      const snapshot = JSON.parse(JSON.stringify(state.project));
      _syncEditorStateToProject();
      snapshot.scene_renders = state.project.scene_renders;
      snapshot.scene_ghosts = state.project.scene_ghosts;
      const previewKeyBefore = _promptPreviewKey(snapshot);
      const selBefore = JSON.stringify(state.selectedSceneIds || []);
      const metaBefore = { name: snapshot.name, scene_count: (snapshot.scenes || []).length };
      _localDirty = false;
      state.saving = true;
      _notifySaveChip();
      try {
        const saved = await API.saveProject(snapshot.id, snapshot);
        // If the user edited again while this save was in flight (e.g. dropped a new
        // anchor), do NOT replace local state with the older snapshot we just wrote.
        if (_localDirty) scheduleSave();
        else state.project = saved;
        state.saving = false;
        _pruneSelection();
        const selChanged = JSON.stringify(state.selectedSceneIds || []) !== selBefore;
        const previewKeyNow = _promptPreviewKey(state.project);
        const previewStale = previewKeyNow !== previewKeyBefore || previewKeyNow !== _lastPreviewKey;
        const metaChanged = saved.name !== metaBefore.name
          || (saved.scenes || []).length !== metaBefore.scene_count;
        if (metaChanged) refreshProjectList(true);
        if (previewStale && !skipPreviewRefresh) refreshPreview(true);
        else if (selChanged) notify();
        _notifySaveChip();
      } catch (e) {
        _localDirty = true;
        state.saving = false;
        _notifySaveChip();
        console.error("save failed", e);
      } finally {
        _commitPromise = null;
        if (_commitQueued) { _commitQueued = false; commit(); }
      }
    })();
    return _commitPromise;
  }

  function patchProject(patch) {
    if (!state.project) return;
    _historyRecord();
    Object.assign(state.project, patch);
    if (_patchAffectsCombinedPrompt(patch)) syncGlobalPromptFromTimeline();
    notify();
    scheduleSave();
  }
  function patchProjectQuiet(patch) {
    if (!state.project) return;
    Object.assign(state.project, patch);
    if (_patchAffectsCombinedPrompt(patch)) syncGlobalPromptFromTimeline();
    scheduleSaveSilent();
  }

  function scene(id) { return state.project?.scenes.find((s) => s.id === id) || null; }

  function genUnitId(sc) { return (sc && sc.gen_unit_id) || (sc && sc.id) || ""; }
  function isVideoClip(sc) { return (sc && sc.source && sc.source.type) === "video"; }
  function isGenerativeScene(sc) { return !!sc && !isVideoClip(sc); }
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

  // Dropping or picking a new i2v anchor keeps the cached render for preview/rating but
  // stamps the pre-change source on the snapshot so UI can warn (like prompt mismatch).
  function _stampRenderSourceBeforeAnchorChange(target) {
    if (!target || !state.project) return;
    const uid = genUnitId(target);
    const root = genUnitRoot(uid) || target;
    const src = root?.source || {};
    const sourceType = src.type || "carry";
    const mediaRef = src.media_ref || null;
    const text = (root?.text || "").trim();
    const anchor = (state.project.anchor || "").trim();
    (state.project.scenes || []).filter((sc) => genUnitId(sc) === uid).forEach((sc) => {
      const r = state.sceneRenders[sc.id];
      if (!r?.media) return;
      const rp = r.renderPrompt || {};
      r.renderPrompt = {
        text: rp.text ?? text,
        anchor: rp.anchor ?? anchor,
        sourceType: rp.sourceType ?? sourceType,
        mediaRef: rp.mediaRef !== undefined ? rp.mediaRef : mediaRef,
        sourceSnapshotted: true,
      };
    });
  }

  // Source lives on the gen-unit root; editorial subclips must mirror it for UI + saves.
  function _syncGenUnitSource(root) {
    if (!root || !state.project) return;
    const src = JSON.parse(JSON.stringify(root.source || {}));
    const uid = genUnitId(root);
    (state.project.scenes || []).forEach((sc) => {
      if (genUnitId(sc) === uid) sc.source = JSON.parse(JSON.stringify(src));
    });
  }

  function _applyScenePatch(id, patch, quiet) {
    const t = _patchSceneTarget(id, patch); if (!t) return;
    const charsChanged = patch.character_ids != null;
    const anchorChanged = _anchorPatchChanged(t, patch);
    if (anchorChanged) _stampRenderSourceBeforeAnchorChange(t);
    const merged = { ...patch };
    if (merged.source) merged.source = { ...(t.source || {}), ...merged.source };
    if (!quiet && !window.EditorHistory?.isApplying()) _historyRecord();
    Object.assign(t, merged);
    if (merged.source) _syncGenUnitSource(genUnitRoot(genUnitId(t)) || t);
    if (charsChanged) _invalidateGlobalPromptDraft();
    if (anchorChanged) {
      clearTimeout(saveTimer); saveTimer = null;
      _localDirty = true; notify(); commit();
    } else if (charsChanged) {
      notify();
      _syncPromptPreview();
    } else if (_patchAffectsCombinedPrompt(merged)) {
      syncGlobalPromptFromTimeline();
      if (quiet) scheduleSaveSilent();
      else { notify(); scheduleSave(); }
    } else if (quiet) scheduleSaveSilent();
    else { notify(); scheduleSave(); }
  }

  function patchScene(id, patch) { _applyScenePatch(id, patch, false); }
  function patchSceneQuiet(id, patch) {
    const t = _patchSceneTarget(id, patch) || scene(id); if (!t) return;
    const anchorChanged = _anchorPatchChanged(t, patch);
    if (anchorChanged) _stampRenderSourceBeforeAnchorChange(t);
    const merged = { ...patch };
    if (merged.source) merged.source = { ...(t.source || {}), ...merged.source };
    Object.assign(t, merged);
    if (merged.source) _syncGenUnitSource(genUnitRoot(genUnitId(t)) || t);
    if (anchorChanged) {
      clearTimeout(saveTimer); saveTimer = null;
      _localDirty = true; notify(); commit();
    } else {
      if (_patchAffectsCombinedPrompt(merged)) syncGlobalPromptFromTimeline();
      scheduleSaveSilent();
    }
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
    _historyRecord();
    // Default to "carry": a new scene continues the previous one (overlap) unless the
    // user picks an anchor/empty. Scene 1 carrying just starts a fresh run.
    const s = { text: "", transition_to_next: "", source: { type: "carry" }, excluded: false };
    state.project.scenes.push(s);
    window.Timeline?.requestAutoFit?.();
    notify(); scheduleSave(); // server assigns id; reselect after commit
  }

  function _snapshotSceneForArchive(sc) {
    const r = state.sceneRenders[sc.id];
    return {
      text: sc.text || "",
      source: JSON.parse(JSON.stringify(sc.source || {})),
      guides: JSON.parse(JSON.stringify(sc.guides || [])),
      character_ids: [...(sc.character_ids || [])],
      effects: JSON.parse(JSON.stringify(sc.effects || {})),
      frames: sc.frames,
      frames_mode: sc.frames_mode,
      fps: sc.fps,
      fps_mode: sc.fps_mode,
      transition_to_next: sc.transition_to_next || "",
      video_transition: sc.video_transition || "",
      transition_frames: sc.transition_frames,
      rating: sc.rating || "",
      audio_volume: sc.audio_volume,
      audio_separated: sc.audio_separated,
      gen_unit_id: sc.gen_unit_id,
      cut_offset_frames: sc.cut_offset_frames,
      source_in: sc.source_in,
      source_dur: sc.source_dur,
      width: sc.width,
      height: sc.height,
      render: r ? JSON.parse(JSON.stringify(r)) : null,
    };
  }

  function _applySceneArchive(sc, archive) {
    if (!archive) return;
    sc.text = archive.text || "";
    sc.source = JSON.parse(JSON.stringify(archive.source || { type: "carry" }));
    sc.guides = JSON.parse(JSON.stringify(archive.guides || []));
    sc.character_ids = [...(archive.character_ids || [])];
    sc.effects = JSON.parse(JSON.stringify(archive.effects || {}));
    sc.frames = archive.frames;
    sc.frames_mode = archive.frames_mode;
    sc.fps = archive.fps;
    sc.fps_mode = archive.fps_mode;
    sc.transition_to_next = archive.transition_to_next || "";
    sc.video_transition = archive.video_transition || "";
    sc.transition_frames = archive.transition_frames;
    sc.rating = archive.rating || "";
    sc.audio_volume = archive.audio_volume;
    sc.audio_separated = !!archive.audio_separated;
    sc.gen_unit_id = archive.gen_unit_id;
    sc.cut_offset_frames = archive.cut_offset_frames || 0;
    sc.source_in = archive.source_in || 0;
    sc.source_dur = archive.source_dur;
    sc.width = archive.width;
    sc.height = archive.height;
    sc.scene_archive = null;
    if (archive.render) state.sceneRenders[sc.id] = JSON.parse(JSON.stringify(archive.render));
  }

  function convertToVideo(sceneId) {
    if (!state.project) return;
    const sc = scene(sceneId);
    if (!sc || isVideoClip(sc)) return;
    _historyRecord();
    const uid = genUnitId(sc);
    const root = genUnitRoot(uid) || sc;
    const archive = _snapshotSceneForArchive(root);
    const mediaRef = (root.source && root.source.media_ref) || null;
    state.project.scenes = state.project.scenes.filter(
      (s) => genUnitId(s) !== uid || s.id === root.id
    );
    Object.assign(root, {
      scene_archive: archive,
      source: { type: "video", media_ref: mediaRef },
      gen_unit_id: root.id,
      cut_offset_frames: 0,
      rating: "",
    });
    if (state.selectedSceneId && !scene(state.selectedSceneId)) {
      state.selectedSceneId = root.id;
      state.selectedSceneIds = [root.id];
      _selectionAnchorId = root.id;
    }
    _afterTimelineStructureChange();
    notify(); scheduleSave();
  }

  function convertToScene(sceneId) {
    if (!state.project) return;
    const sc = scene(sceneId);
    if (!sc || !isVideoClip(sc)) return;
    _historyRecord();
    if (sc.scene_archive) {
      _applySceneArchive(sc, sc.scene_archive);
    } else {
      const mediaRef = sc.source?.media_ref || null;
      sc.scene_archive = null;
      sc.source = { type: "v2v", media_ref: mediaRef };
      if (!sc.text) sc.text = "";
      sc.gen_unit_id = sc.gen_unit_id || sc.id;
    }
    _afterTimelineStructureChange();
    notify(); scheduleSave();
  }

  function addVideoClip(mediaId) {
    if (!state.project || !mediaId) return;
    const asset = (state.mediaBin || []).find((m) => m.id === mediaId);
    if (!asset || asset.kind !== "video") return;
    _historyRecord();
    const id = _newSceneId();
    state.project.scenes.push({
      id,
      gen_unit_id: id,
      text: "",
      transition_to_next: "",
      source: { type: "video", media_ref: mediaId },
      excluded: false,
    });
    state.selectedSceneId = id;
    state.selectedSceneIds = [id];
    _selectionAnchorId = id;
    window.Timeline?.requestAutoFit?.();
    notify(); scheduleSave();
  }

  function _removeSceneNoHistory(id) {
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
    state.sceneGhosts = _ghostOrDropScene(sc, newAnchor, state.sceneGhosts);
    state.project.scenes = arr.filter((s) => s.id !== id);
    state.selectedSceneIds = (state.selectedSceneIds || []).filter((sid) => sid !== id);
    if (state.selectedSceneId === id)
      state.selectedSceneId = state.selectedSceneIds[0] || state.project.scenes[0]?.id || null;
    if (_selectionAnchorId === id) _selectionAnchorId = state.selectedSceneId;
  }

  function removeScene(id) {
    if (!state.project) return;
    _historyRecord();
    _removeSceneNoHistory(id);
    _afterTimelineStructureChange();
    window.Timeline?.requestAutoFit?.();
    notify(); scheduleSave();
  }

  function removeSelectedScenes() {
    if (!state.project) return;
    let ids = [...new Set(state.selectedSceneIds || [])].filter((sid) => scene(sid));
    if (!ids.length && state.selectedSceneId) ids = [state.selectedSceneId];
    if (!ids.length) return;
    if (ids.length > 1 && !confirm(`Remove ${ids.length} selected clips?`)) return;
    if (ids.length === 1) { removeScene(ids[0]); return; }
    _historyRecord();
    ids.map((id) => ({ id, idx: state.project.scenes.findIndex((s) => s.id === id) }))
      .filter((x) => x.idx >= 0)
      .sort((a, b) => b.idx - a.idx)
      .forEach(({ id }) => _removeSceneNoHistory(id));
    _afterTimelineStructureChange();
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

  function trimSceneLeft(id, trimSec) {
    if (!state.project) return;
    const s = scene(id); if (!s || s.frames_mode === "custom") return;
    const fps = (s.fps_mode !== "project" && s.fps != null) ? s.fps : state.project.frame_rate;
    const curFrames = s.frames != null ? s.frames : state.project.num_frames_per_scene;
    const trimFrames = snapFrames(Math.max(0.05, trimSec) * fps);
    if (trimFrames >= curFrames - 8) return;
    _historyRecord();
    s.frames = snapFrames(curFrames - trimFrames);
    s.source_in = (s.source_in || 0) + trimFrames / fps;
    if (s.frames_mode == null || s.frames_mode === "project") s.frames_mode = "timeline";
    notify();
    scheduleSave();
  }

  function slipScene(id, deltaSec) {
    if (!state.project) return;
    const s = scene(id); if (!s) return;
    _historyRecord();
    s.source_in = Math.max(0, (s.source_in || 0) + deltaSec);
    notify();
    scheduleSave();
  }

  function undo() { window.EditorHistory?.undo(window.Store); }
  function redo() { window.EditorHistory?.redo(window.Store); }


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
    if (s.frames_mode === "custom") return;
    _historyRecord();
    const fps = (s.fps_mode !== "project" && s.fps != null) ? s.fps : state.project.frame_rate;
    s.frames = snapFrames(Math.max(1, durationSec) * fps);
    // Dragging the trim handle opts the scene into timeline-driven length.
    if (s.frames_mode == null || s.frames_mode === "project") s.frames_mode = "timeline";
    _syncSeparatedAudioTracks();
    scheduleSaveSilent();
  }

  // Split a clip in two at `atFrames` (defaults to the midpoint). Both halves stay one
  // generative unit — Generate collapses them back into a single uncut scene.
  function splitScene(id, atFrames) {
    if (!state.project) return;
    _historyRecord();
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
      state.sceneRenders[second.id] = {
        media: r.media,
        inSec: (r.inSec || 0) + cut / (fps || 25),
        renderPrompt: r.renderPrompt ? { ...r.renderPrompt } : _snapshotRenderPrompt(id),
      };
    }
    notify(); scheduleSave();
    _syncSeparatedAudioTracks();
  }

  function moveScene(id, delta) {
    if (!state.project) return;
    _historyRecord();
    const arr = state.project.scenes;
    const i = arr.findIndex((s) => s.id === id);
    const j = i + delta;
    if (i < 0 || j < 0 || j >= arr.length) return;
    [arr[i], arr[j]] = [arr[j], arr[i]];
    notify(); scheduleSave();
    _syncSeparatedAudioTracks();
  }

  // Move a scene to an absolute index (post-removal index). Used by timeline drag-reorder.
  function moveSceneTo(id, toIndex) {
    if (!state.project) return;
    _historyRecord();
    const arr = state.project.scenes;
    const from = arr.findIndex((s) => s.id === id);
    if (from < 0) return;
    toIndex = Math.max(0, Math.min(arr.length - 1, toIndex));
    if (toIndex === from) return;
    const [it] = arr.splice(from, 1);
    arr.splice(toIndex, 0, it);
    notify(); scheduleSave();
    _syncSeparatedAudioTracks();
  }

  // ── audio editing ─────────────────────────────────────────────────────────────
  function _uid() { return "a" + Math.random().toString(36).slice(2, 9) + Date.now().toString(36); }

  function isOverlayAudioTrack(t) {
    return !!(t && t.kind === "overlay" && (t.media_ref || t.render_media?.filename));
  }

  function isSeparatedAudioTrack(t) {
    return !!(t && t.kind === "separated" && t.scene_id);
  }

  function _syncSeparatedAudioTracks() {
    if (!state.project) return;
    let moved = false;
    for (const t of state.project.audio_tracks || []) {
      if (!isSeparatedAudioTrack(t)) continue;
      const sc = scene(t.scene_id);
      if (!sc || sc.excluded) continue;
      const off = sceneTimelineOffsetSec(t.scene_id);
      if (Math.abs((t.start_sec || 0) - off) > 0.001) {
        t.start_sec = off;
        moved = true;
      }
    }
    if (moved) scheduleSaveSilent();
  }

  function addAudioTrack(mediaId, startSec) {
    if (!state.project || !mediaId) return;
    _historyRecord();
    const asset = (state.mediaBin || []).find((m) => m.id === mediaId);
    state.project.audio_tracks = state.project.audio_tracks || [];
    state.project.audio_tracks.push({
      id: _uid(),
      kind: "overlay",
      media_ref: mediaId,
      start_sec: +(startSec || 0),
      source_in_sec: 0,
      source_dur: asset?.duration_sec || null,
      volume: 1.0,
      label: (asset && asset.name) || "Overlay",
    });
    notify(); scheduleSave();
  }

  function sceneTimelineOffsetSec(sceneId) {
    if (!state.project) return 0;
    for (const seg of buildPreviewSegments()) {
      if (seg.kind === "scene" && seg.scene.id === sceneId) return seg.offsetSec;
    }
    return 0;
  }

  function separatedTrackForScene(sceneId) {
    return (state.project?.audio_tracks || []).find((t) => t.kind === "separated" && t.scene_id === sceneId);
  }

  function separatedTrackMedia(track) {
    if (!track) return null;
    if (track.pinned_media?.filename) return track.pinned_media;
    const r = state.sceneRenders[track.scene_id];
    return r?.media || null;
  }

  function separatedTrackInSec(track) {
    if (!track) return 0;
    if (track.pinned_in_sec != null) return track.pinned_in_sec;
    if (track.source_in_sec != null) return track.source_in_sec;
    const r = state.sceneRenders[track.scene_id];
    const sc = scene(track.scene_id);
    return (r?.inSec || 0) + (sc?.source_in || 0);
  }

  function separatedTrackDurSec(track) {
    if (!track) return 1;
    if (track.pinned_dur != null) return track.pinned_dur;
    if (track.source_dur != null) return track.source_dur;
    const sc = scene(track.scene_id);
    return sc ? sceneDurationSec(sc) : 1;
  }

  function separateSceneAudio(sceneId) {
    if (!state.project) return;
    const sc = scene(sceneId);
    if (!sc || sc.audio_separated) return;
    const r = state.sceneRenders[sceneId];
    if (!r?.media) { alert("Generate this clip first."); return; }
    if (separatedTrackForScene(sceneId)) return;
    _historyRecord();
    const idx = state.project.scenes.indexOf(sc);
    const inSec = (r.inSec || 0) + (sc.source_in || 0);
    const dur = sc.source_dur != null ? sc.source_dur : sceneDurationSec(sc);
    const savedVol = sc.audio_volume != null ? sc.audio_volume : 1;
    state.project.audio_tracks = state.project.audio_tracks || [];
    state.project.audio_tracks.push({
      id: _uid(),
      kind: "separated",
      scene_id: sceneId,
      start_sec: sceneTimelineOffsetSec(sceneId),
      source_in_sec: inSec,
      source_dur: dur,
      pinned_media: JSON.parse(JSON.stringify(r.media)),
      pinned_in_sec: inSec,
      pinned_dur: dur,
      volume: savedVol,
      label: `S${idx + 1} audio`,
    });
    sc.audio_separated = true;
    sc.audio_volume = 0;
    notify(); scheduleSave();
  }

  function updateAudioTrack(id, patch, quiet) {
    const t = (state.project?.audio_tracks || []).find((x) => x.id === id);
    if (!t) return;
    if (!quiet) _historyRecord();
    Object.assign(t, patch);
    notify(); quiet ? scheduleSaveSilent() : scheduleSave();
  }
  function _syncSeparatedTracksAfterRegen(recordedIds) {
    for (const id of recordedIds || []) {
      const t = separatedTrackForScene(id);
      if (!t) continue;
      const sc = scene(id);
      if (!sc || sc.excluded) continue;
      t.start_sec = sceneTimelineOffsetSec(id);
    }
    _syncSeparatedAudioTracks();
  }

  function removeAudioTrack(id, opts) {
    if (!state.project) return;
    const t = (state.project?.audio_tracks || []).find((x) => x.id === id);
    if (!t) return;
    if (!opts?.skipConfirm) {
      const msg = isSeparatedAudioTrack(t)
        ? "Remove separated audio? The clip will use embedded audio from its current video render again."
        : "Remove this overlay audio track?";
      if (!confirm(msg)) return;
    }
    _historyRecord();
    if (isSeparatedAudioTrack(t) && t.scene_id) {
      const sc = scene(t.scene_id);
      if (sc) {
        sc.audio_separated = false;
        if (t.volume != null) sc.audio_volume = t.volume;
      }
    }
    state.project.audio_tracks = (state.project.audio_tracks || []).filter((x) => x.id !== id);
    notify(); scheduleSave();
  }

  function setSourceTrim(id, patch) {
    if (!state.project) return;
    const p = {};
    if (patch.source_in != null) p.source_in = Math.max(0, +patch.source_in);
    if (patch.source_dur !== undefined) p.source_dur = patch.source_dur == null ? null : Math.max(0.1, +patch.source_dur);
    if (Object.keys(p).length) patchScene(id, p);
  }

  const ENGINE_PRESETS = {
    draft: {
      label: "Fast draft",
      studio_inputs: { "refiner.split_by_transitions": false },
      sampler_inputs: { steps: 20 },
    },
    quality: {
      label: "Quality",
      studio_inputs: { "refiner.split_by_transitions": true },
      sampler_inputs: { steps: 40 },
    },
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
    _historyRecord();
    const patch = {};
    if (preset.studio_inputs) patch.studio_inputs = { ...(state.project.studio_inputs || {}), ...preset.studio_inputs };
    if (preset.sampler_inputs) patch.sampler_inputs = { ...(state.project.sampler_inputs || {}), ...preset.sampler_inputs };
    if (preset.guide_settings) patch.guide_settings = { ...(state.project.guide_settings || {}), ...preset.guide_settings };
    if (preset.continuity_settings) patch.continuity_settings = { ...(state.project.continuity_settings || {}), ...preset.continuity_settings };
    Object.assign(state.project, patch);
    notify();
    scheduleSave();
  }

  // ── sync scenes from preview (distribute parsed anchor/transitions back) ──────
  function syncFromPreview() {
    if (!state.project || !state.preview) return;
    _historyRecord();
    // Authoritative lossless split (verbatim text, shortcut-aware boundaries).
    const v0 = state.preview.parsed_verbatim || state.preview.parsed_raw || state.preview.parsed || {};
    const fullPrompt = state.preview.combined_prompt || state.project.global_prompt || "";
    const v = _normalizeGlobalPromptParse(fullPrompt, v0);
    const parsedScenes = v.scenes || [];
    if (!parsedScenes.length) return;

    state.project.anchor = v.anchor || "";
    const { next, ghosts: ghostsOut } = _alignScenesFromParsed(
      state.project.scenes, parsedScenes, state.sceneGhosts
    );
    for (const sc of next) sc.transition_to_next = "";
    _applyDetectedTransitions(next, null, fullPrompt);
    state.project.scenes = next;
    state.sceneGhosts = ghostsOut;
    _pinnedGlobalPrompt = null;
    syncGlobalPromptFromTimeline();

    notify();
    scheduleSave();
  }

  // Infer prompt split markers from text between verbatim scene chunks (not visual_effect).
  function _inferSplitMarkersFromPrompt(scenes, fullPrompt) {
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
          if (trig && gap.includes(trig)) {
            scenes[i].transition_to_next = trig;
            break;
          }
        }
      }
      pos = endOfScene;
    }
  }

  function _applyDetectedTransitions(scenes, _transitions, fullPrompt) {
    if (fullPrompt) _inferSplitMarkersFromPrompt(scenes, fullPrompt);
  }

  // ── preview ──────────────────────────────────────────────────────────────────
  let previewTimer = null;
  function refreshPreview(immediate = false) {
    clearTimeout(previewTimer);
    previewTimer = null;
    const run = async () => {
      if (!state.project) return;
      try {
        state.preview = await API.preview(state.project.id, false, true);
        _lastPreviewKey = _promptPreviewKey(state.project);
        _restorePinnedGlobalPrompt();
      } catch (e) { state.preview = { parse_error: e.message }; }
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
  // Prompt text captured when a run was queued — applied to sceneRenders on completion.
  const _queuedRenderPrompts = {};

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
    if (t !== "image" && t !== "generated_frame" && t !== "mixed" && t !== "v2v") return false;
    const ref = s.source.media_ref;
    return !!(ref && (state.mediaBin || []).some((m) => m.id === ref));
  }
  function _runs() {
    const active = state.project.scenes.filter((s) => !s.excluded && isGenerativeScene(s));
    const runs = [];
    for (const s of active) {
      const t = (s.source && s.source.type) || "empty";
      // Editorial subclips always continue the current run; carry / broken anchors too.
      const isCarry = isGenSubclip(s)
        || t === "carry"
        || ((t === "image" || t === "generated_frame" || t === "mixed" || t === "v2v") && !_anchorAvailable(s));
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

  // Frames per scene inside one chain-sampler run (mirrors server generate effective_frames).
  function _effectiveRunFrames(scenes) {
    const p = state.project;
    if (!p) return 65;
    const trimmed = (scenes || [])
      .filter((s) => s && s.frames_mode !== "project" && s.frames != null)
      .map((s) => s.frames);
    if (trimmed.length && trimmed.every((f) => f === trimmed[0])) return trimmed[0];
    return p.num_frames_per_scene || 65;
  }

  // Source layout for mapping one chain output video onto timeline clips.
  function _chainRunLayout(sceneIds) {
    const p = state.project;
    const scenes = (sceneIds || []).map((id) => scene(id)).filter(Boolean);
    const fps = p?.frame_rate || 25;
    const frames = _effectiveRunFrames(scenes);
    const overlapFrames = scenes.length > 1 ? +(p?.sampler_inputs?.frame_overlap ?? 16) : 0;
    const overlapSec = Math.max(0, overlapFrames / fps);
    const sourceSpanSec = frames / fps;
    return { frames, fps, overlapSec, sourceSpanSec, overlapFrames };
  }

  // Chain units in a run (subclips share one layout slot).
  function _chainUnitCount(sceneIds) {
    let n = 0;
    let lastUid = null;
    for (const id of sceneIds || []) {
      const sc = scene(id);
      if (!sc) continue;
      const uid = genUnitId(sc);
      if (uid !== lastUid) { n += 1; lastUid = uid; }
    }
    return n;
  }

  // Mirrors movie_editor/backend/chain_layout.py for client-side fallback.
  function _latentToPixelFrame(latentFrame, timeScale) {
    const lf = Math.max(0, Math.floor(latentFrame));
    const ts = Math.max(1, Math.floor(timeScale));
    if (ts > 1) return lf > 0 ? (lf - 1) * ts + 1 : 0;
    return lf;
  }
  function _expectedLatentFrames(pixelFrames, timeScale) {
    return Math.floor((Math.max(1, pixelFrames) - 1) / Math.max(1, timeScale)) + 1;
  }
  function _latentOverlapFrames(frameOverlap, timeScale) {
    if (frameOverlap <= 0) return 0;
    return _expectedLatentFrames(frameOverlap + 1, timeScale) - 1;
  }
  function _fallbackChainPlaybackLayout(sceneIds) {
    const unitCount = _chainUnitCount(sceneIds);
    if (unitCount <= 1) return null;
    const run = _chainRunLayout(sceneIds);
    const timeScale = 8; // LTX VAE default; sampler boundaries JSON carries the authoritative value
    const latentFrames = _expectedLatentFrames(run.frames, timeScale);
    const latentOverlap = _latentOverlapFrames(run.overlapFrames, timeScale);
    const starts = [0];
    for (let i = 1; i < unitCount; i++) {
      const cum = latentFrames + (i - 1) * Math.max(1, latentFrames - latentOverlap);
      starts.push(_latentToPixelFrame(cum, timeScale));
    }
    return starts.map((start, i) => ({
      scene_index: i,
      start_frame: start,
      in_sec: Math.round(start / run.fps * 1e6) / 1e6,
    }));
  }
  function _resolveSceneLayout(sceneIds, sceneLayout) {
    const need = _chainUnitCount(sceneIds);
    if (need <= 1) return sceneLayout;
    const fallback = _fallbackChainPlaybackLayout(sceneIds);
    if (!sceneLayout || !sceneLayout.length) return fallback;
    if (sceneLayout.length >= need) return sceneLayout;
    if (!fallback) return sceneLayout;
    const out = sceneLayout.slice();
    for (let i = out.length; i < need; i++) {
      if (fallback[i]) out.push(fallback[i]);
    }
    return out;
  }
  function _layoutSlot(sceneLayout, chainSceneIdx) {
    if (!sceneLayout || !sceneLayout.length) return null;
    const byIndex = sceneLayout.find((s) => (s.scene_index ?? s.sceneIndex) === chainSceneIdx);
    if (byIndex) return byIndex;
    return sceneLayout[chainSceneIdx] || null;
  }

  function _recordInSec(lastChain, sourceEnd, curr, sceneLayout, chainSceneIdx) {
    const slot = _layoutSlot(sceneLayout, chainSceneIdx);
    if (slot) return slot.inSec != null ? slot.inSec : (slot.in_sec != null ? slot.in_sec : 0);
    if (!lastChain) return 0;
    if (genUnitId(lastChain) === genUnitId(curr)) return sourceEnd;
    // Decoded chain output advances by full scene spans; overlap is blended in latent space only.
    return Math.max(0, sourceEnd);
  }

  function _recordSourceStepSec(lastChain, curr, layout, sceneLayout, chainSceneIdx, inSec) {
    if (lastChain && genUnitId(lastChain) === genUnitId(curr)) return sceneDurationSec(curr);
    const nextSlot = _layoutSlot(sceneLayout, chainSceneIdx + 1);
    if (nextSlot) {
      const nextIn = nextSlot.inSec != null ? nextSlot.inSec : (nextSlot.in_sec != null ? nextSlot.in_sec : 0);
      return Math.max(0, nextIn - inSec);
    }
    return layout.sourceSpanSec;
  }

  function _snapshotRenderPrompt(sceneId) {
    const sc = scene(sceneId);
    if (!sc) return null;
    const root = genUnitRoot(genUnitId(sc));
    const src = root?.source || sc.source || {};
    return {
      text: (root?.text || "").trim(),
      anchor: (state.project?.anchor || "").trim(),
      sourceType: src.type || "carry",
      mediaRef: src.media_ref || null,
      sourceSnapshotted: true,
    };
  }

  function _snapshotRenderSource(sceneId) {
    const sc = scene(sceneId);
    if (!sc) return null;
    const root = genUnitRoot(genUnitId(sc));
    const src = root?.source || sc.source || {};
    return { sourceType: src.type || "carry", mediaRef: src.media_ref || null };
  }

  function renderMediaLabel(mediaRef) {
    if (!mediaRef) return "(none)";
    const m = (state.mediaBin || []).find((x) => x.id === mediaRef);
    return m?.name || mediaRef;
  }

  function renderPromptForScene(sceneId) {
    return state.sceneRenders[sceneId]?.renderPrompt || null;
  }

  function renderPromptMismatch(sceneId) {
    const snap = renderPromptForScene(sceneId);
    if (!snap) return null;
    const current = (_snapshotRenderPrompt(sceneId)?.text || "");
    const rendered = (snap.text || "");
    if (current === rendered) return null;
    return { current, rendered, anchor: snap.anchor || "" };
  }

  function renderAnchorMismatch(sceneId) {
    const snap = renderPromptForScene(sceneId);
    if (!snap?.sourceSnapshotted) return null;
    const current = _snapshotRenderSource(sceneId);
    if (!current) return null;
    const renderedType = snap.sourceType || "carry";
    const renderedRef = snap.mediaRef || null;
    if (renderedType === current.sourceType && renderedRef === current.mediaRef) return null;
    return {
      renderedType,
      renderedRef,
      renderedLabel: renderedRef ? renderMediaLabel(renderedRef) : null,
      currentType: current.sourceType,
      currentRef: current.mediaRef,
      currentLabel: current.mediaRef ? renderMediaLabel(current.mediaRef) : null,
    };
  }

  function renderIsStale(sceneId) {
    return !!(renderPromptMismatch(sceneId) || renderAnchorMismatch(sceneId));
  }

  function _queueRenderPrompts(sceneIds) {
    for (const id of sceneIds || []) {
      const snap = _snapshotRenderPrompt(id);
      if (snap) _queuedRenderPrompts[id] = snap;
    }
  }

  function _recordSegment(mediaList, targetSceneIds, opts) {
    if (!mediaList || !mediaList.length || !targetSceneIds || !targetSceneIds.length) return;
    const completedScenes = opts?.completedScenes ?? targetSceneIds.length;
    const ids = targetSceneIds.slice(0, Math.max(0, completedScenes));
    if (!ids.length) return;
    const primary = mediaList.find((m) => m.kind === "videos" || m.kind === "gifs") || mediaList[0];
    const sceneLayout = _resolveSceneLayout(ids, opts?.sceneLayout || null);
    const layout = _chainRunLayout(ids);
    let sourceEnd = 0;
    let lastChain = null;
    let chainSceneIdx = 0;
    let clearedRating = false;
    const recordedSceneIds = [];
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      if (_removedDuringGen.has(id)) {
        const ghosts = state.sceneGhosts || [];
        const gi = ghosts.findIndex((g) => g.id === id);
        if (gi >= 0) {
          const ghost = ghosts[gi];
          const sameUnit = lastChain && genUnitId(lastChain) === genUnitId(ghost);
          const inSec = sameUnit ? sourceEnd : _recordInSec(lastChain, sourceEnd, ghost, sceneLayout, chainSceneIdx);
          ghosts[gi] = { ...ghost, media: primary, inSec, pendingGen: false };
          state.sceneGhosts = ghosts;
          sourceEnd = inSec + _ghostDurationSec(ghost);
          lastChain = ghost;
          if (!sameUnit) chainSceneIdx += 1;
        }
        _removedDuringGen.delete(id);
        continue;
      }
      const sc = scene(id); if (!sc) continue;
      const sameUnit = lastChain && genUnitId(lastChain) === genUnitId(sc);
      const inSec = sameUnit ? sourceEnd : _recordInSec(lastChain, sourceEnd, sc, sceneLayout, chainSceneIdx);
      const renderPrompt = _queuedRenderPrompts[id] || _snapshotRenderPrompt(id);
      delete _queuedRenderPrompts[id];
      state.sceneRenders[id] = { media: primary, inSec, renderPrompt };
      recordedSceneIds.push(id);
      const root = genUnitRoot(genUnitId(sc));
      if (root && root.rating) { root.rating = ""; clearedRating = true; }
      sourceEnd = inSec + _recordSourceStepSec(lastChain, sc, layout, sceneLayout, chainSceneIdx, inSec);
      lastChain = sc;
      if (!sameUnit) chainSceneIdx += 1;
    }
    _pruneGhostsAfterRegen(recordedSceneIds);
    _syncSeparatedTracksAfterRegen(recordedSceneIds);
    if (clearedRating) scheduleSaveSilent();
    if (recordedSceneIds.length) {
      _validateRendersToken++;
      state.notice = "";
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
            _recordSegment(s.media, targetSceneIds, { sceneLayout: s.scene_layout });
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
      _queueRenderPrompts(sceneIds);
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
    await flushSave();
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
    await flushSave();
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
      return s && !s.excluded && isGenerativeScene(s);
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
  async function _saveBlobAs(url, name, opts) {
    opts = opts || {};
    const pickTypes = opts.types || [{ description: "MP4 video", accept: { "video/mp4": [".mp4"] } }];
    let blob;
    try { const r = await fetch(url); if (!r.ok) throw new Error(r.statusText); blob = await r.blob(); }
    catch (e) { alert("Could not fetch the file: " + e.message); return; }
    if (window.showSaveFilePicker) {
      try {
        const h = await window.showSaveFilePicker({ suggestedName: name, types: pickTypes });
        const w = await h.createWritable(); await w.write(blob); await w.close();
        return;
      } catch (e) { if (e && e.name === "AbortError") return; }
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }

  function _exportMediaFilename(m) {
    let name = (m.name || m.filename || m.id || "export").trim();
    const fileExt = (m.filename || "").match(/(\.[^.]+)$/)?.[1] || "";
    if (fileExt && !/\.\w+$/.test(name)) name += fileExt;
    return name;
  }

  async function exportMediaAsset(mediaId) {
    const m = (state.mediaBin || []).find((x) => x.id === mediaId);
    if (!m || (m.kind !== "image" && m.kind !== "video")) {
      alert("Only images and videos can be exported.");
      return;
    }
    const name = _exportMediaFilename(m);
    const types = m.kind === "video"
      ? [{ description: "Video", accept: { "video/mp4": [".mp4"], "video/webm": [".webm"], "video/quicktime": [".mov"] } }]
      : [{ description: "Image", accept: { "image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"], "image/webp": [".webp"] } }];
    await _saveBlobAs(API.mediaUrl(mediaId), name, { types });
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
      const fps = (sc.fps_mode !== "project" && sc.fps != null ? sc.fps : p.frame_rate) || 25;
      const tFrames = sc.transition_frames || 0;
      const base = {
        inSec: (r?.inSec || 0) + (sc.source_in || 0),
        durationSec: sc.source_dur != null ? sc.source_dur : sceneDurationSec(sc),
        fx: sc.effects || {}, fps,
        w: (sc.width != null ? sc.width : p.width) || null,
        h: (sc.height != null ? sc.height : p.height) || null,
        transition: sc.video_transition || "",
        tdur: tFrames > 0 ? tFrames / fps : 0,
        sceneId: sc.id,
        volume: sc.audio_separated ? 0 : (sc.audio_volume != null ? sc.audio_volume : 1),
      };
      if (r && r.media) {
        out.push({ media: r.media, ...base });
      } else if (isVideoClip(sc) && sc.source?.media_ref) {
        out.push({ bin_media_ref: sc.source.media_ref, ...base });
      }
    }
    return out;
  }

  function _selectedClipFocusId() {
    const ids = state.selectedSceneIds?.length
      ? state.selectedSceneIds
      : (state.selectedSceneId ? [state.selectedSceneId] : []);
    if (!ids.length) return null;
    if (state.selectedSceneId && ids.includes(state.selectedSceneId)) return state.selectedSceneId;
    return ids[0];
  }

  function _hasTimelineClipSelection() {
    return !!_selectedClipFocusId();
  }

  // Build export/import clip spec for the selected scene (render trim or media-bin video).
  function _selectedClipExportSpec() {
    if (!state.project) return null;
    const id = _selectedClipFocusId();
    if (!id) return null;
    const sc = scene(id);
    if (!sc) return null;
    const r = state.sceneRenders[id];
    const idx = state.project.scenes.findIndex((s) => s.id === id);
    const proj = (state.project.name || "montage").replace(/[^\w.-]+/g, "_");
    if (isVideoClip(sc) && sc.source?.media_ref && !r?.media) {
      return {
        clip: {
          bin_media_ref: sc.source.media_ref,
          in: sc.source_in || 0,
          dur: sc.source_dur != null ? sc.source_dur : sceneDurationSec(sc),
        },
        name: `${proj}_video_${idx >= 0 ? idx + 1 : "x"}_${_stamp()}.mp4`,
      };
    }
    if (!r?.media) return null;
    return {
      clip: {
        filename: r.media.filename,
        subfolder: r.media.subfolder || "",
        type: r.media.type || "output",
        in: (r.inSec || 0) + (sc.source_in || 0),
        dur: sc.source_dur != null ? sc.source_dur : sceneDurationSec(sc),
      },
      name: `${proj}_scene${idx >= 0 ? idx + 1 : "x"}_${_stamp()}.mp4`,
    };
  }

  function clipSaveableToMediaBin(sceneId) {
    if (!sceneId || !state.project) return false;
    const sc = scene(sceneId);
    if (!sc) return false;
    if (isVideoClip(sc) && sc.source?.media_ref) return true;
    return !!(state.sceneRenders[sceneId]?.media);
  }

  // Export the render covering the selected clip via a Save dialog.
  async function exportSelected() {
    if (!state.project) return;
    if (!_hasTimelineClipSelection()) { alert("Select a clip first."); return; }
    const spec = _selectedClipExportSpec();
    if (!spec) { alert("Generate the clip first, or select a video clip with media."); return; }
    const focusId = _selectedClipFocusId();
    const sc = scene(focusId);
    if (isVideoClip(sc) && sc.source?.media_ref && !state.sceneRenders[focusId]?.media) {
      const asset = (state.mediaBin || []).find((m) => m.id === sc.source.media_ref);
      const name = asset?.name || spec.name;
      await _saveBlobAs(API.mediaUrl(sc.source.media_ref), name.replace(/[^\w.-]+/g, "_"));
      return;
    }
    try {
      const out = await API.exportClip(state.project.id, spec.clip);
      await _saveBlobAs(API.resultUrl(state.project.id, out.media), spec.name);
    } catch (e) {
      alert("Export failed: " + (e.message || e));
    }
  }

  async function saveSelectedToMediaBin() {
    if (!state.project) return;
    if (!_hasTimelineClipSelection()) { alert("Select a clip first."); return; }
    const spec = _selectedClipExportSpec();
    if (!spec) {
      alert("Generate the clip first, or select a video clip with media.");
      return;
    }
    try {
      const res = await API.importClipToMediaBin(spec.clip, spec.name);
      await loadMedia();
      const entry = res?.media || res;
      state.notice = `Saved to Media bin: ${entry?.name || spec.name}`;
      notify();
    } catch (e) {
      alert("Save to media bin failed: " + (e.message || e));
    }
  }

  // Stitch the kept clips (in/out per clip, hard cut, video + audio) into one final file.
  async function renderFinal() {
    if (!state.project) return;
    const rc = _renderClips();
    if (!rc.length) { alert("No generated clips to stitch yet — generate first."); return; }
    if (!confirm(
      `Stitch ${rc.length} clip(s) into one final video?\n\nThis replaces per-scene renders with a single stitched file on the timeline.`
    )) return;
    const clips = rc.map((c) => ({
      filename: c.media?.filename, subfolder: c.media?.subfolder || "", type: c.media?.type || "output",
      bin_media_ref: c.bin_media_ref || "",
      in: +c.inSec.toFixed(3), dur: +c.durationSec.toFixed(3),
      fx: c.fx, fps: c.fps, w: c.w, h: c.h, transition: c.transition, tdur: +(c.tdur || 0).toFixed(3),
      volume: c.volume, scene_id: c.sceneId,
    }));
    await flushSave();  // audio tracks / keep-original live on the project — render reads it from disk
    set({ gen: { state: "running", promptId: null, media: [], msg: `Stitching ${clips.length} clip(s)…` } });
    try {
      const r = await API.renderFinal(state.project.id, clips);
      // Play the stitched file across the whole timeline (single source, in-point 0).
      const order = state.project.scenes.filter((s) => !s.excluded);
      state.sceneRenders = {};
      state.sceneGhosts = [];
      _syncEditorStateToProject();
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
  async function deleteMediaMany(ids) {
    const list = [...new Set(ids || [])].filter(Boolean);
    if (!list.length) return;
    for (const id of list) {
      try { await API.deleteMedia(id); } catch (_) {}
      if (state.mediaPreviewId === id) state.mediaPreviewId = null;
    }
    await loadMedia();
  }

  async function loadTransitions() { try { state.transitions = (await API.transitions()).transitions || []; } catch (_) {} notify(); }
  async function loadNleLibrary() {
    try {
      const lib = await API.nleLibrary();
      state.nleEffects = lib.effects || [];
      state.nleVideoTransitions = lib.video_transitions || [];
    } catch (_) {
      state.nleEffects = [];
      state.nleVideoTransitions = [];
    }
    notify();
  }
  async function loadShortcuts() { try { state.shortcuts = (await API.shortcuts()).shortcuts || []; } catch (_) { state.shortcuts = []; } notify(); }
  async function loadCharacters() { try { state.characters = (await API.characters()).characters || []; } catch (_) { state.characters = []; } notify(); }
  function _characterAssignedToProject(charId) {
    if (!charId || !state.project) return false;
    return (state.project.scenes || []).some((s) => (s.character_ids || []).includes(charId));
  }

  async function saveCharacter(item) {
    try {
      state.characters = (await API.saveCharacter(item)).characters || state.characters;
      const cid = item.id || item.original_id;
      if (_characterAssignedToProject(cid)) {
        _invalidateGlobalPromptDraft();
        await refreshPreview(true);
      } else notify();
    } catch (e) { alert("Save failed: " + e.message); }
  }
  async function deleteCharacter(id) {
    try {
      const wasAssigned = _characterAssignedToProject(id);
      state.characters = (await API.deleteCharacter(id)).characters || [];
      if (wasAssigned) {
        _invalidateGlobalPromptDraft();
        await refreshPreview(true);
      } else notify();
    } catch (e) { console.error(e); }
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
  async function saveTransition(item) {
    try {
      state.transitions = (await API.saveTransition(item)).transitions || state.transitions;
      notify();
    } catch (e) { alert("Save failed: " + e.message); }
  }
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

  // Apply a split-marker library item to the selected scene seam (generation prompt).
  function applySplitMarkerToSelection(trigger) {
    const s = scene(state.selectedSceneId); if (!s) return false;
    patchScene(s.id, { transition_to_next: trigger }); return true;
  }
  function applyTransitionToSelection(trigger) { return applySplitMarkerToSelection(trigger); }

  function applyNleEffect(effectId, paramValue) {
    const sc = scene(state.selectedSceneId); if (!sc) return false;
    const patches = {
      zoom_in: () => ({ effects: { ...(sc.effects || {}), zoom: "in" } }),
      zoom_out: () => ({ effects: { ...(sc.effects || {}), zoom: "out" } }),
      blur: (v) => ({ effects: { ...(sc.effects || {}), blur: v } }),
      fade_in: (v) => ({ effects: { ...(sc.effects || {}), fade_in: v } }),
      fade_out: (v) => ({ effects: { ...(sc.effects || {}), fade_out: v } }),
    };
    const fn = patches[effectId]; if (!fn) return false;
    patchScene(sc.id, fn(paramValue)); return true;
  }

  function applyNleVideoTransition(transitionId, paramValue) {
    const sc = scene(state.selectedSceneId); if (!sc) return false;
    const item = (state.nleVideoTransitions || []).find((x) => x.id === transitionId);
    if (!item) return false;
    const frames = paramValue != null ? Math.round(paramValue)
      : Math.round(item.param?.default || 16);
    patchScene(sc.id, { video_transition: item.type || transitionId, transition_frames: frames });
    return true;
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
    if (!m || (m.kind !== "image" && m.kind !== "audio" && m.kind !== "video")) return;
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
    const asset = (state.mediaBin || []).find((m) => m.id === mediaId);
    if (asset?.kind === "video") {
      if (isVideoClip(s)) {
        patchScene(sceneId, { source: { ...(s.source || {}), type: "video", media_ref: mediaId } });
      } else if (confirm("Replace this scene with the imported video clip? Use Convert to video in the inspector to keep a backup of scene settings.")) {
        patchScene(sceneId, { source: { type: "video", media_ref: mediaId } });
      }
      return;
    }
    // Drag-drop is an explicit new anchor — use image i2v (not mixed/carry guides).
    const patch = { source: { ...(s.source || {}), type: "image", media_ref: mediaId } };
    if ((s.source?.media_ref || null) !== mediaId) patch.guides = [];
    patchScene(sceneId, patch);
  }

  // ── boot ─────────────────────────────────────────────────────────────────────
  async function init() {
    try { state.health = await API.health(); } catch (_) { state.health = { ok: false }; }
    if (state.gen?.state === "error" && /Setup incomplete|Missing workflow template/.test(state.gen.msg || "")) {
      state.gen = { state: "idle", promptId: null, media: [], msg: "" };
    }
    try { const t = await API.transitions(); state.transitions = t.transitions || []; } catch (_) { state.transitions = []; }
    await loadNleLibrary();
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
    get, set, subscribe, notify, init,
    scheduleSaveFromHistory, notifyHistoryState,
    refreshProjectList, loadProject, newProject, deleteProject, downloadProject, importProject,
    patchProject, patchProjectQuiet, patchScene, patchSceneQuiet, flushSave, selectScene, addScene, removeScene, removeSelectedScenes, dismissGhost, moveScene, moveSceneTo, scene,
    addVideoClip, convertToVideo, convertToScene, isVideoClip, isGenerativeScene,
    sceneCharacterIds, toggleSceneCharacter,
    genUnitId, isGenSubclip, genUnitRoot, genUnitSceneIds,
    renderPromptForScene, renderPromptMismatch, renderAnchorMismatch, renderMediaLabel, renderIsStale,
    buildPreviewSegments, previewTotalSec, segmentDurationSec,
    addAudioTrack, updateAudioTrack, removeAudioTrack, separateSceneAudio, separatedTrackForScene,
    separatedTrackMedia, separatedTrackInSec, separatedTrackDurSec,
    isOverlayAudioTrack, isSeparatedAudioTrack,
    resizeScene, splitScene, snapFrames, setSourceTrim, trimSceneLeft, slipScene,
    applyEnginePreset, ENGINE_PRESETS, undo, redo,
    refreshPreview, syncFromPreview, applyGlobalPromptQuiet, scheduleGlobalPromptApply, buildGlobalPromptFromTimeline, syncGlobalPromptFromTimeline, generate, generateMontage, generateSelected, selectedSceneCount, renderFinal, exportSelected, saveSelectedToMediaBin, clipSaveableToMediaBin, interrupt, loadModels, loadImageTargets, setModelInput, setModelLink, clearNotice,
    setConditioningSlot, setSamplerSlot, setSamplerInput, setSamplerInputNow, unsetSamplerInput, setStudioInput, setStudioInputNow,
    loadMedia, uploadMedia, deleteMedia, deleteMediaMany, previewMedia, clearMediaPreview, assignMediaToScene, exportMediaAsset,
    loadShortcuts, saveShortcut, deleteShortcut, importShortcuts,
    loadCharacters, saveCharacter, deleteCharacter,
    loadTransitions, saveTransition, deleteTransition, importTransitions,
    loadNleLibrary, applyNleEffect, applyNleVideoTransition,
    applySplitMarkerToSelection, applyTransitionToSelection, insertShortcutIntoSelection,
    setSceneRating: (id, v) => {
      patchScene(id, { rating: v || "" });
      if (window.Timeline?.syncClipRatings) window.Timeline.syncClipRatings(get());
    },
    resetStudioSession,
  };
})();
