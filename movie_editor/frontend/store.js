// Central state + data actions. Views subscribe and re-render their zone.
(function () {
  const API = window.MovieEditorAPI;

  const state = {
    projects: [],          // [{id,name,scene_count,updated_at}]
    project: null,         // full Project
    selectedSceneId: null,   // focus clip (inspector / toolbar)
    selectedSceneIds: [],    // multi-select for generate / timeline
    selectedOverlayId: null, // graphics overlay (text/image) on overlay lane
    selectedAudioTrackId: null, // inserted audio lane (overlay / separated)
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
    shortcutCategories: [],  // managed grouping list: [{name, sub_categories:[]}]
    imageTargets: [],        // where an image asset can be wired [{value,label}]
    ratingLabels: [],        // FunPack Studio V2 rating options
  };

  // ── editor settings ─────────────────────────────────────────────────────────
  // Editor preferences. Govern how the prompt is parsed/edited, not what is generated
  // from already-distributed scenes. They live in this browser's localStorage AND, once
  // a project is open, are mirrored into `project.editor_settings`: on a fresh machine
  // (a newly rented GPU box) localStorage is empty and the server-side revolver sidecar
  // does not exist, so without the mirror every one of these is back at its default and
  // has to be set by hand again. Opening a project restores what it remembers.
  const EDITOR_SETTINGS_KEY = "funpack_editor_settings";
  const EDITOR_SETTINGS_DEFAULTS = {
    autocomplete: true, suggestions: true, anchorEnabled: true,
    // "Anchor as guide" i2v bypass: declare a custom i2v node so it can be forced
    // to a state (value-only) on anchor_guide runs. anchorGuideBypass = {node, input, value}.
    anchorGuideHasI2v: false, anchorGuideBypass: null,
    // Snapshot of the shortcut revolver, which is NOT a browser setting — it lives
    // server-side next to the shortcut DB because previews and generation share one
    // cycle. Carried here purely so the project can put it back on a new instance;
    // the server copy stays the single source of truth while the app is running.
    revolver: null,   // {enabled, random} | null = never recorded
  };
  function _loadEditorSettings() {
    try {
      const raw = JSON.parse(localStorage.getItem(EDITOR_SETTINGS_KEY) || "{}");
      return { ...EDITOR_SETTINGS_DEFAULTS, ...(raw && typeof raw === "object" ? raw : {}) };
    } catch (_) { return { ...EDITOR_SETTINGS_DEFAULTS }; }
  }
  let _editorSettings = _loadEditorSettings();
  function _persistEditorSettings() {
    try { localStorage.setItem(EDITOR_SETTINGS_KEY, JSON.stringify(_editorSettings)); } catch (_) {}
  }
  function getEditorSettings() { return { ..._editorSettings }; }
  function getEditorSetting(key) { return _editorSettings[key]; }
  function setEditorSetting(key, val) {
    // Mirrored into the open project so it travels with the file. Deliberately BEFORE the
    // unchanged-value return: a browser that was already configured before the project
    // learned to remember would otherwise never record anything, because re-picking the
    // value you already have is a no-op — and the setting you never touch again is exactly
    // the one you lose on the next machine. Silent, since changing a preference is not a
    // timeline edit and should not mark the project dirty in the UI.
    if (state.project) {
      const stored = state.project.editor_settings || {};
      if (JSON.stringify(stored[key]) !== JSON.stringify(val)) {
        state.project.editor_settings = { ...stored, [key]: val };
        scheduleSaveSilent();
      }
    }
    if (JSON.stringify(_editorSettings[key]) === JSON.stringify(val)) return;
    _editorSettings = { ..._editorSettings, [key]: val };
    _persistEditorSettings();
    notify();
    // Anchor toggle changes how the global prompt maps onto the timeline — re-split now.
    if (key === "anchorEnabled" && state.project) {
      const cur = (state.project.global_prompt
        || state.preview?.display_prompt || state.preview?.combined_prompt || "").trim();
      if (cur) { _pinnedGlobalPrompt = cur; applyGlobalPromptQuiet(cur).catch(() => {}); }
    }
  }

  /** Put a freshly-opened project's remembered preferences back in force.
   *
   * Key-by-key, and only for keys the project actually carries: a project saved before
   * this existed has no `editor_settings` at all, and that must leave the current browser
   * alone rather than resetting it to defaults. Applied wholesale (not via
   * setEditorSetting) so the anchor toggle's re-split side effect doesn't fire — the
   * caller re-syncs the prompt from the timeline straight after this anyway.
   *
   * The revolver is the one that isn't ours to set: it is a server setting shared by every
   * browser and by generation. We POST it only when it actually differs, so opening a
   * project that agrees with the server costs nothing and never restarts a live cycle.
   */
  function _applyProjectEditorSettings(project) {
    if (!project) return;
    const saved = project.editor_settings || {};
    const keys = Object.keys(saved).filter((k) => k !== "revolver" && k in EDITOR_SETTINGS_DEFAULTS);
    if (keys.length) {
      const next = { ..._editorSettings };
      keys.forEach((k) => { next[k] = saved[k]; });
      _editorSettings = next;
      _persistEditorSettings();
    } else {
      // A project from before this existed, opened on a machine that IS configured: adopt
      // this browser's settings rather than leaving the project blank, so the very next
      // move carries them. Nothing changes here and now — these already are the settings
      // in force — it only stops the project from travelling empty.
      project.editor_settings = { ...(project.editor_settings || {}), ...getEditorSettings() };
      delete project.editor_settings.revolver;   // the server's copy is authoritative below
      scheduleSaveSilent();
    }
    const rev = saved.revolver;
    if (rev && typeof rev === "object") {
      _editorSettings = { ..._editorSettings, revolver: { enabled: !!rev.enabled, random: !!rev.random } };
      _persistEditorSettings();
      API.revolverSettings().then((cur) => {
        if (!!cur.enabled === !!rev.enabled && !!cur.random === !!rev.random) return;
        return API.setRevolverSettings({ enabled: !!rev.enabled, random: !!rev.random })
          .then(() => notify());
      }).catch(() => {});
      return;
    }
    // Nothing recorded yet: take whatever the server has now, so the mode you already set
    // by hand travels with the project without you having to toggle it to register it.
    API.revolverSettings().then((cur) => {
      const snap = { enabled: !!cur.enabled, random: !!cur.random };
      _editorSettings = { ..._editorSettings, revolver: snap };
      _persistEditorSettings();
      if (state.project === project) {
        project.editor_settings = { ...(project.editor_settings || {}), revolver: snap };
        scheduleSaveSilent();
      }
    }).catch(() => {});
  }

  const listeners = new Set();
  const SAVE_DEBOUNCE_MS = 5000;        // discrete edits (dropdowns, toggles)
  const SAVE_SILENT_DEBOUNCE_MS = 20000; // typing in fields — flushed on blur / generate
  let saveTimer = null;
  let _localDirty = false;       // local edits newer than the in-flight save snapshot
  let _commitPromise = null;     // avoid overlapping saves stomping newer anchor edits
  let _commitQueued = false;
  // Force a re-render from the authoritative saved project after the next commit. Set by
  // source/anchor changes: those don't change the prompt-preview key or selection, so commit()
  // would replace state.project = saved WITHOUT re-rendering, leaving the timeline showing the
  // stale optimistic DOM (scene prompt text could read blank until reload/regenerate).
  let _renderAfterSave = false;
  let _selectionAnchorId = null;  // shift-click range anchor
  let _modelsSaveTimer = null;    // debounce exposed-control (node input) saves
  let _modelsSaveDirty = false;
  // Autosave suspension, held while a settings surface is open (see suspendSave). A
  // debounce firing mid-session saves a half-edited value and echoes it back over
  // state.project — "I set 0.5 and it went back to 1.0". Counted: surfaces nest.
  let _saveSuspended = 0;
  // In-flight reload of state.models (funpack-models-changed). commit() copies it into
  // project.models, so committing mid-reload would write the pre-edit node list.
  let _modelsLoad = null;

  function notify() { listeners.forEach((fn) => { try { fn(state); } catch (e) { console.error(e); } }); }
  function subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }
  function set(patch) { Object.assign(state, patch); notify(); }

  function _syncEditorStateToProject() {
    if (!state.project) return;
    state.project.scene_renders = JSON.parse(JSON.stringify(state.sceneRenders || {}));
    state.project.scene_ghosts = JSON.parse(JSON.stringify(state.sceneGhosts || []));
    // Keep project.models aligned with the Models modal / saveModels API so timeline
    // autosave (commit before generate) cannot resurrect deleted pipeline nodes.
    state.project.models = JSON.parse(JSON.stringify(state.models || { slots: [] }));
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
    if (_saveSuspended) { _notifySaveChip(); return; }
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

  function _promptPreviewKey(project) {
    if (!project) return "";
    return JSON.stringify({
      anchor: project.anchor,
      postfix: project.postfix,
      postfix_enabled: project.postfix_enabled,
      global_prompt: project.global_prompt,
      intro_transition: project.intro_transition,
      scenes: (project.scenes || []).map((s) => ({
        text: s.text,
        excluded: s.excluded,
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
    // The pin is the last COMPLETED apply — while newer typed text is still debouncing or
    // mid-parse, broadcasting it would rewind the Compose textarea to older text.
    if (!globalPromptApplyPending()) _dispatchGlobalPromptUpdated(pin);
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
  let _pendingGlobalPromptText = null;   // latest typed global prompt awaiting distribution
  let _applySeq = 0;                     // ordering token — an older apply must never clobber a newer one
  let _applyInFlight = 0;

  // True while typed global-prompt text is still debouncing or mid-parse. During that window
  // any funpack-global-prompt-updated dispatch would carry STALE text (the last completed
  // apply), so consumers (Composer textarea) must not overwrite the user's draft with it.
  function globalPromptApplyPending() {
    return _pendingGlobalPromptText != null || !!_globalApplyTimer || _applyInFlight > 0;
  }

  function scheduleGlobalPromptApply(text) {
    _pendingGlobalPromptText = text;
    clearTimeout(_globalApplyTimer);
    _globalApplyTimer = setTimeout(() => {
      _globalApplyTimer = null;
      const t = _pendingGlobalPromptText; _pendingGlobalPromptText = null;
      applyGlobalPromptQuiet(t).catch((e) => console.warn("Global prompt apply failed:", e));
    }, 500);
  }

  // Materialize a still-debounced global-prompt edit into scene.text NOW (screen = truth). Called
  // before generate/save so a prompt typed faster than the 500ms debounce isn't left undistributed
  // — otherwise we'd save (and generate from) stale scene text while the screen shows the new prompt.
  async function flushGlobalPromptApply() {
    if (_globalApplyTimer) { clearTimeout(_globalApplyTimer); _globalApplyTimer = null; }
    if (_pendingGlobalPromptText == null) return;
    const t = _pendingGlobalPromptText; _pendingGlobalPromptText = null;
    await applyGlobalPromptQuiet(t);
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
      source: { type: window.PipelineCaps?.defaultSceneSourceType(state) || "carry" },
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
      source: { type: window.PipelineCaps?.defaultSceneSourceType(state) || "carry" },
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
        durationSec: ghost.durationSec != null ? ghost.durationSec : undefined,
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

  // Keep separated audio on the timeline when its source clip is removed.
  function _detachSeparatedAudioForScene(sc) {
    if (!state.project?.audio_tracks) return;
    for (const t of state.project.audio_tracks) {
      if (t.kind !== "separated" || t.scene_id !== sc.id) continue;
      t.start_sec = sceneTimelineOffsetSec(sc.id);
      delete t.scene_id;
    }
  }

  // Ghost a removed scene when it had a render; otherwise drop it entirely.
  function _ghostOrDropScene(sc, afterSceneId, ghosts) {
    const r = state.sceneRenders[sc.id];
    const inFlight = _genInFlightIds.has(sc.id);
    _detachSeparatedAudioForScene(sc);
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

  // Global prompt is authoritative. A scene's id carries its source/anchor AND its render
  // (state.sceneRenders[id]), so a parsed slot must map onto the old root that owns its
  // CONTENT — never just the root that happens to sit at the same array index. Matching by
  // POSITION first was the "stale scene" bug: a reorder/insert/delete in the global prompt
  // shifted indices, gluing one scene's text onto another scene's anchor+render (and that
  // wrong mapping then got saved, so it survived restart and key-clears). So:
  //   Pass 1 — exact normalized-text match (nearest original position breaks ties): pins
  //            unchanged scenes to their own root through any reorder/insert/delete.
  //   Pass 2 — leftover slots inherit the leftover roots IN ORDER: an in-place text edit
  //            keeps its render/anchor; a slot with no root left becomes a fresh scene.
  // Unmatched old roots are ghosted or dropped (unchanged). The only irreducible case is a
  // scene that is BOTH reordered AND fully rewritten with no unchanged neighbour to anchor
  // it — genuinely ambiguous from text alone, so Pass 2 falls back to a positional guess.
  function _alignScenesFromParsed(oldScenes, parsedScenes, ghosts) {
    const oldAll = (oldScenes || []).filter((s) => !s.excluded && isGenerativeScene(s));
    const oldRoots = _generativeRoots(oldScenes);
    const usedOld = new Set();
    const usedGhost = new Set();
    const next = [];
    let ghostsOut = [...(ghosts || [])];
    const parsed = parsedScenes || [];

    // Match each parsed slot to the old root that owns its CONTENT — exact text first
    // (nearest original position breaks ties), then leftover slots take leftover roots in
    // order. The pure index-based kernel lives in scene_align.js so the two passes can be
    // unit-tested in node (see scene_align.test.js). Both sides are normalized identically
    // here before matching, so the kernel can compare raw strings.
    const matchIdx = window.SceneAlign.matchParsedToRoots(
      parsed.map((p) => _normalizeSceneText(p.text)),
      oldRoots.map((r) => _normalizeSceneText(r.text)),
    );
    const matchByIdx = matchIdx.map((ri) => (ri >= 0 ? oldRoots[ri] : null));
    for (const root of matchByIdx) {
      if (root) _markGenUnitUsed(genUnitId(root), usedOld, oldAll);
    }

    // Materialize: matched root → reuse (keeps id/source/render); else ghost by text; else new.
    for (let i = 0; i < parsed.length; i++) {
      const ps = parsed[i];
      const pt = _normalizeSceneText(ps.text);
      const match = matchByIdx[i];

      if (match) {
        const root = isGenSubclip(match) ? (genUnitRoot(genUnitId(match)) || match) : match;
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
    // Anchor disabled (editor setting): the leading pre-first-split text is Scene 1,
    // not a shared anchor. Fold it in as the first scene; the split trigger that began
    // the old Scene 1 becomes the seam between new Scene 1 and Scene 2 (inferred from text).
    if (!_editorSettings.anchorEnabled && anchor) {
      return { anchor: "", scenes: [{ text: anchor }, ...scenes], transitions };
    }
    return { anchor: v.anchor || "", scenes, transitions };
  }

  function _afterTimelineStructureChange() {
    _pinnedGlobalPrompt = null;
    _ensureTimelineOrder();
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
    // The global prompt is authoritative on plan order — reconcile the cut order to it now
    // (no-op when the user has pinned a manual order; otherwise tracks the plan).
    _ensureTimelineOrder();
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
    const seq = ++_applySeq;
    _applyInFlight++;
    try {
      _historyRecord();
      let res;
      try { res = await API.parsePrompt(state.project.id, trimmed); }
      catch (e) {
        console.warn("Global prompt parse failed:", e);
        return false;
      }
      // The user typed more while this parse was in flight: a newer apply is scheduled (or
      // already running). Distributing this OLDER text now would re-split the timeline to a
      // stale prompt and dispatch a stale updated event — drop it and let the newer one win.
      if (seq !== _applySeq || _pendingGlobalPromptText != null) return false;
      if (!await _distributeGlobalPrompt(trimmed, res)) return false;
      clearTimeout(saveTimer);
      saveTimer = null;
      _localDirty = true;
      scheduleSaveSilent();
      refreshPreview(true);
      return true;
    } finally {
      _applyInFlight--;
    }
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
        // Drop a render ONLY on a definitive "file is gone" (404). A busy server, a dropped
        // tunnel, or any 5xx is transient — deleting here (and autosaving the deletion!)
        // permanently lost clips when the UI reloaded while another run was still saving.
        if (res.status === 404) {
          if (token !== _validateRendersToken) return;
          const cur = state.sceneRenders[id];
          if (cur && cur.media?.filename === r.media.filename) delete state.sceneRenders[id];
          missing++;
        } else if (res.ok && token === _validateRendersToken) {
          // File is on disk: reconcile a stale plan-estimate length with the real encoded
          // duration for standalone renders (retroactively fixes pre-existing clips on load).
          _probeRenderDuration(id);
        }
      } catch (_) {
        // Network-level failure — can't tell missing from unreachable; keep the render.
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
    let loaded;
    try {
      loaded = await API.getProject(id);
    } catch (e) {
      // Corrupt / incompatible project file: keep the current view, surface why instead of
      // leaving the app in a half-loaded state.
      state.notice = `Could not open this project: ${e && e.message ? e.message : e}`;
      notify();
      return;
    }
    state.project = loaded;
    state.notice = "";
    // Before anything reads a preference: anchorEnabled decides how the global prompt
    // splits into scenes, and syncGlobalPromptFromTimeline() below runs on that answer.
    _applyProjectEditorSettings(loaded);
    const first = state.project.scenes[0]?.id || null;
    state.selectedSceneId = first;
    state.selectedSceneIds = first ? [first] : [];
    state.selectedOverlayId = null;
    state.selectedAudioTrackId = null;
    _selectionAnchorId = first;
    state.gen = { state: "idle", promptId: null, media: [], msg: "" };
    state.sceneRenders = JSON.parse(JSON.stringify(state.project.scene_renders || {}));
    state.sceneGhosts = JSON.parse(JSON.stringify(state.project.scene_ghosts || []));
    _ensureTimelineOrder();
    normalizeOverlayLanes(state.project);
    state.models = JSON.parse(JSON.stringify(state.project.models || { slots: [] }));
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
    await loadModels();
    _validateSceneRenders();
    window.PipelineSetup?.maybePrompt?.();
  }

  async function newProject(name) {
    const p = await API.createProject(name || "Untitled montage");
    await refreshProjectList();
    await loadProject(p.id);
    // A project born on a configured machine inherits that configuration, so it can carry
    // it away later. Without this it would only start remembering once a preference was
    // toggled — and the settings you never have to touch are exactly the ones you'd lose.
    if (state.project && state.project.id === p.id) {
      state.project.editor_settings = { ...getEditorSettings() };
      scheduleSaveSilent();
    }
  }

  async function deleteProject(id) {
    await API.deleteProject(id);
    state.project = null; state.selectedSceneId = null; state.selectedSceneIds = [];
    _selectionAnchorId = null;
    await refreshProjectList();
    if (state.projects[0]) await loadProject(state.projects[0].id);
    else {
      notify();
      window.Onboarding?.open?.();
    }
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
    if (_saveSuspended) { _notifySaveChip(); return; }
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => { saveTimer = null; commit(); }, SAVE_DEBOUNCE_MS);
    _notifySaveChip();
  }

  // For free-text / number fields: queue a save without re-rendering; blur/generate flush.
  function scheduleSaveSilent() {
    _localDirty = true;
    if (_saveSuspended) { _notifySaveChip(); return; }
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => { saveTimer = null; commit(); }, SAVE_SILENT_DEBOUNCE_MS);
    _notifySaveChip();
  }

  // Hold scheduled saves until the matching resumeSave(). Edits still land locally and
  // still mark the project unsaved; only the network write waits. An explicit flushSave()
  // (Generate, restart, tab hidden) ignores the hold.
  function suspendSave() {
    _saveSuspended++;
    clearTimeout(saveTimer);
    saveTimer = null;
  }

  // Balanced with suspendSave(). Returns the commit promise if this released the last
  // hold and there was something to write, else null.
  function resumeSave() {
    if (_saveSuspended > 0) _saveSuspended--;
    if (_saveSuspended > 0) return null;
    return flushSave();
  }

  // Commit any pending save immediately (field blur / before generate). Returns the
  // commit promise when it flushed, else null.
  function flushSave() {
    clearTimeout(saveTimer); saveTimer = null;
    flushModelsSave();  // persist any debounced exposed-control (node input) edits too
    if (!_localDirty && !_commitPromise) return null;
    return commit();
  }

  async function commit(opts = {}) {
    if (!state.project) return;
    if (_commitPromise) { _commitQueued = true; return _commitPromise; }
    const skipPreviewRefresh = !!opts.skipPreviewRefresh;
    _commitPromise = (async () => {
      if (_modelsLoad) { try { await _modelsLoad; } catch (_) {} }
      // Sync BEFORE snapshotting: the snapshot is what gets uploaded, and a snapshot taken
      // first carries the models config (and renders/ghosts) from the PREVIOUS sync — every
      // autosave then silently reverted exposed-control edits (bypass/input/link changes
      // saved via /models) by overwriting p.models on the server with the stale copy.
      _syncEditorStateToProject();
      const snapshot = JSON.parse(JSON.stringify(state.project));
      const previewKeyBefore = _promptPreviewKey(snapshot);
      const selBefore = JSON.stringify(state.selectedSceneIds || []);
      const metaBefore = { name: snapshot.name, scene_count: (snapshot.scenes || []).length };
      _localDirty = false;
      state.saving = true;
      _notifySaveChip();
      try {
        const saved = await _retryOnTunnel(() => API.saveProject(snapshot.id, snapshot));
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
        if (previewStale && !skipPreviewRefresh) { refreshPreview(true); _renderAfterSave = false; }
        else if (selChanged || _renderAfterSave) { notify(); _renderAfterSave = false; }
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

  function _sceneHasManualSource(sc) {
    if (!sc?.source) return false;
    if (isVideoClip(sc)) return true;
    const t = sc.source.type;
    if (t === "image" || t === "v2v" || t === "video") return true;
    return t === "mixed" && !!sc.source.media_ref;
  }

  function _propagateProjectSettingsToScenes(before, patch) {
    const p = state.project;
    if (!p) return;
    const touches = ["num_frames_per_scene", "frame_rate", "width", "height"].some((k) => k in patch);
    if (!touches) return;
    for (const sc of p.scenes || []) {
      if (patch.num_frames_per_scene != null && (sc.frames_mode || "project") === "project" && !isVideoClip(sc)) {
        sc.frames = null;
      }
      if (patch.frame_rate != null && (sc.fps_mode || "project") === "project") {
        sc.fps = null;
      }
      if (_sceneHasManualSource(sc)) continue;
      if (patch.width != null && (sc.width == null || sc.width === before.width)) sc.width = null;
      if (patch.height != null && (sc.height == null || sc.height === before.height)) sc.height = null;
    }
  }

  function patchProject(patch) {
    if (!state.project) return;
    _historyRecord();
    const before = {
      num_frames_per_scene: state.project.num_frames_per_scene,
      frame_rate: state.project.frame_rate,
      width: state.project.width,
      height: state.project.height,
    };
    Object.assign(state.project, patch);
    _propagateProjectSettingsToScenes(before, patch);
    if (_patchAffectsCombinedPrompt(patch)) syncGlobalPromptFromTimeline();
    notify();
    scheduleSave();
  }
  function patchProjectQuiet(patch) {
    if (!state.project) return;
    const before = {
      num_frames_per_scene: state.project.num_frames_per_scene,
      frame_rate: state.project.frame_rate,
      width: state.project.width,
      height: state.project.height,
    };
    Object.assign(state.project, patch);
    _propagateProjectSettingsToScenes(before, patch);
    if (_patchAffectsCombinedPrompt(patch)) syncGlobalPromptFromTimeline();
    if (["num_frames_per_scene", "frame_rate", "width", "height"].some((k) => k in patch)) notify();
    scheduleSaveSilent();
  }

  // ── prompt variables ($name) + global-prompt templates ──────────────────────
  function projectVariables() { return (state.project && state.project.variables) || []; }
  function setProjectVariables(list) {
    if (!state.project) return;
    // Normalize: strip a leading $, drop fully-empty rows on persist (kept while editing in the UI).
    patchProjectQuiet({ variables: (Array.isArray(list) ? list : []).map((v) => ({
      name: String((v && v.name) || "").replace(/^\$+/, "").trim(),
      value: String((v && v.value != null) ? v.value : ""),
    })) });
  }
  function promptTemplates() { return (state.project && state.project.prompt_templates) || []; }
  function savePromptTemplate(name, promptText) {
    if (!state.project) return;
    const nm = String(name || "").trim();
    if (!nm) return;
    const snapshot = {
      name: nm,
      prompt: String(promptText != null ? promptText : (state.project.global_prompt || "")),
      variables: JSON.parse(JSON.stringify(state.project.variables || [])),
    };
    const tpls = (state.project.prompt_templates || []).filter((t) => t && t.name !== nm);
    tpls.push(snapshot);
    patchProject({ prompt_templates: tpls });
  }
  function deletePromptTemplate(name) {
    if (!state.project) return;
    patchProject({ prompt_templates: (state.project.prompt_templates || []).filter((t) => t && t.name !== name) });
  }
  async function applyPromptTemplate(name) {
    if (!state.project) return;
    const tpl = (state.project.prompt_templates || []).find((t) => t && t.name === name);
    if (!tpl) return;
    // Restore the saved variables first, then distribute the prompt (re-splits the timeline).
    patchProjectQuiet({ variables: JSON.parse(JSON.stringify(tpl.variables || [])) });
    await applyGlobalPromptQuiet(tpl.prompt || "");
  }

  function scene(id) { return state.project?.scenes.find((s) => s.id === id) || null; }

  function genUnitId(sc) { return (sc && sc.gen_unit_id) || (sc && sc.id) || ""; }
  function isVideoClip(sc) { return (sc && sc.source && sc.source.type) === "video"; }
  function isGenerativeScene(sc) { return !!sc && !isVideoClip(sc); }
  function isGenSubclip(sc) { return !!((sc && sc.cut_offset_frames) > 0); }
  /** Timeline clip whose preview is driven by source_in on a media-bin file (not render inSec alone). */
  function playsFromSourceMedia(sc) {
    if (!sc?.source) return false;
    if (isVideoClip(sc)) return !!sc.source.media_ref;
    return sc.source.type === "v2v" && !!sc.source.media_ref;
  }
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
    const targetId = (root && isGenSubclip(s) && (patch.text != null || patch.rating != null || patch.source != null))
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
    const anchorChanged = _anchorPatchChanged(t, patch);
    if (anchorChanged) _stampRenderSourceBeforeAnchorChange(t);
    const merged = { ...patch };
    if (merged.source) merged.source = { ...(t.source || {}), ...merged.source };
    if (!quiet && !window.EditorHistory?.isApplying()) _historyRecord();
    Object.assign(t, merged);
    if (merged.source) _syncGenUnitSource(genUnitRoot(genUnitId(t)) || t);
    if (anchorChanged) {
      clearTimeout(saveTimer); saveTimer = null;
      _localDirty = true; _renderAfterSave = true; notify(); commit();
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
      _localDirty = true; _renderAfterSave = true; notify(); commit();
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
    state.selectedOverlayId = null;
    state.selectedAudioTrackId = null;
    state.mediaPreviewId = null;  // tapping a clip exits the transient media-bin preview

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
    const srcType = window.PipelineCaps?.defaultSceneSourceType(state) || "carry";
    const s = { text: "", transition_to_next: "", source: { type: srcType }, excluded: false, frames_mode: "project", fps_mode: "project" };
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
    const mediaRef = (sc.source && sc.source.media_ref) || null;
    const asset = mediaRef ? (state.mediaBin || []).find((m) => m.id === mediaRef) : null;
    if (asset?.kind !== "video") {
      alert("Assign a video file to this clip first (Media Browser or drag onto the clip), then Convert to video.");
      return;
    }
    _lockSceneAsVideo(sceneId, mediaRef, { archive: true });
  }

  function _clearUnitSceneRenders(unitId) {
    const live = new Set((state.project?.scenes || []).filter((s) => genUnitId(s) === unitId).map((s) => s.id));
    for (const id of Object.keys(state.sceneRenders || {})) {
      const s = scene(id);
      if (!s) continue;
      if (genUnitId(s) === unitId && !live.has(id)) delete state.sceneRenders[id];
    }
    live.forEach((id) => { delete state.sceneRenders[id]; });
  }

  function _lockSceneAsVideo(sceneId, mediaId, { archive = true } = {}) {
    if (!state.project || !mediaId) return;
    const asset = (state.mediaBin || []).find((m) => m.id === mediaId);
    if (!asset || asset.kind !== "video") return;
    _historyRecord();
    const sc = scene(sceneId);
    if (!sc) return;
    const uid = genUnitId(sc);
    const root = genUnitRoot(uid) || sc;
    const archiveSnap = archive && !isVideoClip(root) ? _snapshotSceneForArchive(root) : null;
    state.project.scenes = state.project.scenes.filter(
      (s) => genUnitId(s) !== uid || s.id === root.id
    );
    _clearUnitSceneRenders(uid);
    const fps = state.project.frame_rate || 25;
    const dur = asset.duration_sec || root.source_dur || null;
    Object.assign(root, {
      scene_archive: archiveSnap || root.scene_archive || null,
      source: { type: "video", media_ref: mediaId },
      gen_unit_id: root.id,
      cut_offset_frames: 0,
      source_dur: dur,
      rating: root.rating || "",
    });
    if (dur != null) {
      root.frames_mode = "timeline";
      root.frames = snapFrames(dur * fps);
    }
    if (state.selectedSceneId && !scene(state.selectedSceneId)) {
      state.selectedSceneId = root.id;
      state.selectedSceneIds = [root.id];
      _selectionAnchorId = root.id;
    }
    if (!asset.duration_sec) _probeVideoClipDuration(root.id, mediaId);
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
    const fps = state.project.frame_rate || 25;
    const dur = asset.duration_sec || null;
    state.project.scenes.push({
      id,
      gen_unit_id: id,
      text: "",
      transition_to_next: "",
      source: { type: "video", media_ref: mediaId },
      source_dur: dur,
      frames_mode: dur != null ? "timeline" : "project",
      frames: dur != null ? snapFrames(dur * fps) : null,
      excluded: false,
    });
    state.selectedSceneId = id;
    state.selectedSceneIds = [id];
    _selectionAnchorId = id;
    window.Timeline?.requestAutoFit?.();
    notify(); scheduleSave();
    if (!dur) _probeVideoClipDuration(id, mediaId);
  }

  // Drop an image onto the timeline → a new GENERATIVE scene at the project's
  // default length, anchored to that image (i2v). Empty prompt; the user fills it
  // in. Mirrors addScene (server assigns the id) but pre-sets the image anchor.
  function addImageClip(mediaId) {
    if (!state.project || !mediaId) return;
    const asset = (state.mediaBin || []).find((m) => m.id === mediaId);
    if (!asset || asset.kind !== "image") return;
    _historyRecord();
    state.project.scenes.push({
      text: "",
      transition_to_next: "",
      source: { type: "image", media_ref: mediaId },
      excluded: false,
      frames_mode: "project",
      fps_mode: "project",
    });
    window.Timeline?.requestAutoFit?.();
    notify(); scheduleSave(); // server assigns id; reselect after commit
  }

  function _applyVideoClipDuration(sceneId, durSec) {
    const sc = scene(sceneId);
    if (!sc || !isVideoClip(sc) || !durSec || !isFinite(durSec)) return;
    const fps = (sc.fps_mode !== "project" && sc.fps != null) ? sc.fps : (state.project?.frame_rate || 25);
    patchSceneQuiet(sceneId, {
      source_dur: durSec,
      frames: snapFrames(durSec * fps),
      frames_mode: sc.frames_mode === "custom" ? "custom" : "timeline",
    });
  }

  function _probeVideoClipDuration(sceneId, mediaId) {
    const v = document.createElement("video");
    v.preload = "metadata";
    v.src = API.mediaUrl(mediaId);
    // Always release the probe's media connection — detached <video>s hold their socket
    // until GC and count against Chrome's per-origin connection cap.
    const release = () => { try { v.removeAttribute("src"); v.load(); } catch (_) {} };
    v.onerror = release;
    v.onloadedmetadata = () => {
      const dur = v.duration;
      release();
      if (!dur || !isFinite(dur)) return;
      const asset = (state.mediaBin || []).find((m) => m.id === mediaId);
      if (asset && !asset.duration_sec) asset.duration_sec = dur;
      _applyVideoClipDuration(sceneId, dur);
    };
  }

  // True when a render's media file belongs to this scene ALONE — i.e. the scene spans the
  // whole encoded file from its start. Multi-scene chains share one file and partition it via
  // inSec/durationSec slices, so for those the plan-based slice length is the source of truth
  // and must NOT be replaced by the whole-file duration.
  function _renderIsStandalone(sceneId) {
    const r = state.sceneRenders[sceneId];
    if (!r?.media?.filename) return false;
    if ((r.inSec || 0) > 0.001) return false;
    const fname = r.media.filename;
    let n = 0;
    for (const k of Object.keys(state.sceneRenders)) {
      if (state.sceneRenders[k]?.media?.filename === fname) n++;
    }
    return n === 1;
  }

  // A generated render's frozen durationSec is the PLAN estimate (snapped frames/fps). The
  // actual encoded file can be slightly longer (LTXAV muxes audio; container fps rounding), so
  // the clip drew shorter than the real media. For a standalone render (whole file = this scene)
  // probe the real duration and adopt it — sceneDurationSec already routes render.durationSec for
  // project-mode scenes, so the timeline/playback then match the file. Idempotent: once adopted,
  // the values match and re-probing is a no-op.
  function _probeRenderDuration(sceneId) {
    if (!state.project) return;
    const r = state.sceneRenders[sceneId];
    if (!r?.media || !(r.media.kind === "videos" || r.media.kind === "gifs")) return;
    if (!_renderIsStandalone(sceneId)) return;
    const fname = r.media.filename;
    const tok = r._editTok || 0;                                  // snapshot before the async probe
    const v = document.createElement("video");
    v.preload = "metadata";
    v.muted = true;
    const release = () => { try { v.removeAttribute("src"); v.load(); } catch (_) {} };
    v.onloadedmetadata = () => {
      const dur = v.duration;
      release(); // metadata read — free the connection before touching state
      if (!dur || !isFinite(dur)) return;
      const cur = state.sceneRenders[sceneId];
      if (!cur || cur.media?.filename !== fname) return;        // scene moved on; don't clobber
      if ((cur._editTok || 0) !== tok) return;                  // split/re-length happened mid-probe; stale
      const old = cur.durationSec;
      if (old != null && Math.abs(old - dur) < 0.05) return;
      cur.durationSec = dur;                                     // adopt the real encoded length
      // A separated-audio track was pinned to the OLD (plan) length at separation time. If it
      // still matches that length it was auto-pinned, so carry it along; if it differs the user
      // trimmed it by hand — leave that alone.
      if (old != null) {
        const t = separatedTrackForScene(sceneId);
        if (t && t.pinned_dur != null && Math.abs(t.pinned_dur - old) < 0.02) {
          t.pinned_dur = dur;
          if (t.source_dur != null) t.source_dur = dur;
        }
      }
      notify();
      scheduleSaveSilent();
    };
    v.onerror = release;
    v.src = API.resultUrl(state.project.id, r.media);
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
        if (playsFromSourceMedia(next)) {
          const fps = (sc.fps_mode !== "project" && sc.fps != null) ? sc.fps : (state.project.frame_rate || 25);
          const removedDur = (isVideoClip(sc) && sc.source_dur != null)
            ? sc.source_dur
            : ((sc.frames != null ? sc.frames : state.project.num_frames_per_scene) / fps);
          // Legacy splits never bumped source_in — advance the kept half past the deleted head.
          if ((next.source_in || 0) <= (sc.source_in || 0) + 0.001) {
            next.source_in = (sc.source_in || 0) + removedDur;
          }
        }
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

  // Remove a scene from the PLAN. If its gen-unit has a generated clip, KEEP that clip on the
  // timeline: mark the unit `excluded` (so generation + prompt skip it everywhere) and
  // `removed_from_plan` (hide from the plan strip; export keeps it because it has a render).
  // If nothing was generated, it's a plain delete.
  function removeFromPlan(id) {
    if (!state.project) return;
    const s = scene(id); if (!s) return;
    const uid = genUnitId(s);
    const unit = (state.project.scenes || []).filter((x) => genUnitId(x) === uid);
    const rendered = isVideoClip(s) || unit.some((x) => state.sceneRenders[x.id] && state.sceneRenders[x.id].media);
    if (!rendered) { removeScene(id); return; }
    _historyRecord();
    unit.forEach((x) => { x.excluded = true; x.removed_from_plan = true; });
    syncGlobalPromptFromTimeline();
    notify(); scheduleSave();
  }

  // Bring a removed-from-plan scene back into the plan (re-enable generation).
  function restoreToPlan(id) {
    if (!state.project) return;
    const s = scene(id); if (!s) return;
    const uid = genUnitId(s);
    _historyRecord();
    (state.project.scenes || []).filter((x) => genUnitId(x) === uid)
      .forEach((x) => { x.excluded = false; x.removed_from_plan = false; });
    syncGlobalPromptFromTimeline();
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
    if (seg.kind === "gap") return Math.max(0, +(seg.gapSec || 0));
    if (seg.kind === "scene") {
      const sc = seg.scene;
      if (isVideoClip(sc) && sc.source_dur != null) return sc.source_dur;
      return sceneDurationSec(sc);
    }
    return _ghostDurationSec(seg.ghost);
  }

  // ── timeline (cut) order — independent of plan (scenes) order ─────────────────
  // The timeline is the RESULT; its order is project.timeline_order (scene ids), which the
  // plan can never move. Self-healing: keep existing order, drop ids whose scene is gone,
  // append new scenes in plan order. Empty falls back to plan order (back-compat / fresh).
  function _ensureTimelineOrder() {
    const p = state.project; if (!p) return;
    const ids = (p.scenes || []).map((s) => s.id);
    // Until the user explicitly pins a custom cut order (◀ ▶), the timeline follows the plan.
    // This keeps an edit to the global prompt — which rebuilds the plan in its own order — from
    // leaving a stale id sequence behind that scrambles the clips against their plan badges.
    if (!p.timeline_manually_ordered) { p.timeline_order = ids; return; }
    const idSet = new Set(ids);
    const cur = (Array.isArray(p.timeline_order) ? p.timeline_order : []).filter((id) => idSet.has(id));
    const inCur = new Set(cur);
    for (const id of ids) if (!inCur.has(id)) cur.push(id);
    p.timeline_order = cur;
  }

  function orderedTimelineScenes() {
    const p = state.project; if (!p) return [];
    const byId = new Map((p.scenes || []).map((s) => [s.id, s]));
    const order = (Array.isArray(p.timeline_order) && p.timeline_order.length)
      ? p.timeline_order : (p.scenes || []).map((s) => s.id);
    const out = []; const seen = new Set();
    for (const id of order) { const s = byId.get(id); if (s && !seen.has(id)) { out.push(s); seen.add(id); } }
    for (const s of (p.scenes || [])) if (!seen.has(s.id)) { out.push(s); seen.add(s.id); }
    return out;
  }

  // Reorder the CUT (timeline_order) only — never the plan. Selected clip ± 1 slot.
  function moveTimelineClip(id, delta) {
    const p = state.project; if (!p) return;
    _ensureTimelineOrder();
    const order = p.timeline_order;
    const i = order.indexOf(id); if (i < 0) return;
    const j = Math.max(0, Math.min(order.length - 1, i + delta));
    if (i === j) return;
    _historyRecord();
    order.splice(i, 1); order.splice(j, 0, id);
    // The user has now pinned a custom cut order — a global-prompt edit must preserve it
    // rather than re-deriving the order from the plan.
    p.timeline_manually_ordered = true;
    notify(); scheduleSave();
  }

  // Preview/timeline layout: result clips (cut order) interleaved with removed-scene ghosts.
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
    for (const sc of orderedTimelineScenes()) {
      ordered.push({ kind: "scene", scene: sc, id: sc.id });
      for (const g of (byAnchor.get(sc.id) || [])) pushGhost(g);
      const gapSec = Math.max(0, +(sc.gap_after_sec || 0));
      if (gapSec > 0.001) {
        ordered.push({ kind: "gap", id: `gap:${sc.id}`, afterSceneId: sc.id, gapSec });
      }
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

  function audioTrackEndSec(t) {
    if (!t) return 0;
    const start = t.start_sec || 0;
    if (t.source_dur != null) return start + t.source_dur;
    if (t.pinned_dur != null) return start + t.pinned_dur;
    if (isSeparatedAudioTrack(t)) return start + separatedTrackDurSec(t);
    const asset = (state.mediaBin || []).find((m) => m.id === t.media_ref);
    if (asset?.duration_sec) return start + asset.duration_sec;
    return start + 1;
  }

  function previewTotalSec() {
    const p = state.project;
    if (!p) return 0;
    const segs = buildPreviewSegments();
    let total = segs.length ? segs[segs.length - 1].offsetSec + segs[segs.length - 1].durationSec : 0;
    for (const ov of p.overlay_tracks || []) {
      const end = (ov.start_sec || 0) + (ov.duration_sec || 0);
      if (end > total) total = end;
    }
    for (const t of p.audio_tracks || []) {
      const end = audioTrackEndSec(t);
      if (end > total) total = end;
    }
    return total;
  }

  function hasOverlayOrAudioContent() {
    const p = state.project;
    if (!p) return false;
    if ((p.audio_tracks || []).length) return true;
    return (p.overlay_tracks || []).some((ov) => (ov.duration_sec || 0) > 0);
  }

  // Valid frame counts are per model family — LTX 8k+1, MiniMax H3 17k+5 — so the grid comes
  // from PipelineCaps rather than being hardcoded here. Off-grid lengths do not fail softly:
  // the H3 latent node snaps its own length up while the sampler expects the number the
  // project asked for, and the run dies on the mismatch.
  function _snap(n, mode) {
    if (window.PipelineCaps?.snapFramesTo) return window.PipelineCaps.snapFramesTo(n, state, mode);
    n = Math.round(Number(n) || 0);
    const k = (n - 1) / 8;
    const r = mode === "floor" ? Math.floor(k) : mode === "ceil" ? Math.ceil(k) : Math.round(k);
    return Math.max(9, r * 8 + 1);
  }
  function snapFrames(n) { return _snap(n, "round"); }
  function snapFramesFloor(n) { return _snap(n, "floor"); }
  function snapFramesCeil(n) { return _snap(n, "ceil"); }

  function _framesFromDuration(durationSec, fps, curFrames) {
    const raw = Math.max(1, durationSec) * fps;
    const cur = curFrames != null ? curFrames : snapFrames(raw);
    let frames;
    if (raw < cur - 0.01) frames = snapFramesFloor(raw);
    else if (raw > cur + 0.01) frames = snapFramesCeil(raw);
    else frames = cur;
    if (frames === cur && Math.abs(durationSec - cur / fps) > 0.02) {
      // one whole grid step in the direction the drag went (8 frames on LTX, 17 on H3)
      const step = window.PipelineCaps?.frameGrid ? window.PipelineCaps.frameGrid(state).step : 8;
      frames = raw < cur ? snapFramesFloor(cur - step) : cur + step;
    }
    return frames;
  }

  function sceneEffFrames(sc, project) {
    const p = project || state.project;
    if (!sc || !p) return p?.num_frames_per_scene || 97;
    if ((sc.frames_mode || "project") === "project") return p.num_frames_per_scene || 97;
    return sc.frames != null ? sc.frames : p.num_frames_per_scene;
  }

  // Generative scenes whose own length overrides the project's Frames field. Any timeline
  // resize, trim or split flips a scene out of "project" mode, and from then on editing the
  // project number does nothing to it — silently, which reads as the field being ignored.
  function scenesOverridingProjectFrames() {
    if (!state.project) return [];
    return (state.project.scenes || []).filter(
      (s) => s && !isVideoClip(s) && (s.frames_mode || "project") !== "project");
  }

  // Put those scenes back on the project's length. Video clips keep theirs: their length is
  // the source file's, not a preference.
  function useProjectFramesEverywhere() {
    if (!state.project) return;
    const targets = scenesOverridingProjectFrames();
    if (!targets.length) return;
    _historyRecord();
    targets.forEach((s) => { s.frames_mode = "project"; s.frames = null; });
    _afterTimelineStructureChange();
    notify();
    scheduleSave();
  }

  function sceneEffFps(sc, project) {
    const p = project || state.project;
    if (!sc || !p) return p?.frame_rate || 25;
    if ((sc.fps_mode || "project") === "project") return p.frame_rate || 25;
    return sc.fps != null ? sc.fps : p.frame_rate;
  }

  function trimSceneLeft(id, trimSec) {
    if (!state.project) return;
    const s = scene(id); if (!s || s.frames_mode === "custom") return;
    const trim = Math.max(0.05, trimSec);
    _historyRecord();
    if (isVideoClip(s)) {
      const dur = s.source_dur != null ? s.source_dur : sceneDurationSec(s);
      if (trim >= dur - 0.05) return;
      s.source_in = (s.source_in || 0) + trim;
      s.source_dur = Math.max(0.1, dur - trim);
      if ((s.frames_mode || "project") === "project") s.frames_mode = "timeline";
      _syncSeparatedAudioTracks();
      notify();
      scheduleSave();
      return;
    }
    const fps = sceneEffFps(s);
    const curFrames = Math.max(1, Math.round(sceneDurationSec(s) * fps)); // trim from what's shown
    const trimFrames = Math.max(1, Math.round(trim * fps));
    const rawNext = curFrames - trimFrames;
    if (rawNext < 9) return;
    let nextFrames = snapFramesFloor(rawNext);
    if (nextFrames >= curFrames) nextFrames = Math.max(9, curFrames - 8);
    if (nextFrames >= curFrames) return;
    const actualTrimFrames = curFrames - nextFrames;
    s.frames = nextFrames;
    s.source_in = (s.source_in || 0) + actualTrimFrames / fps;
    if ((s.frames_mode || "project") === "project") s.frames_mode = "timeline";
    _syncSeparatedAudioTracks();
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


  // Length (s) this scene was ACTUALLY rendered at, frozen at generation time. Null until
  // generated (or for old renders saved before this was captured — they fall back to plan).
  function renderDurationSec(sceneId) {
    const r = state.sceneRenders[sceneId];
    return r && r.durationSec != null ? r.durationSec : null;
  }

  // The PLANNED length (s) from the per-scene frames/fps modes — what the next Generate will
  // produce. Ignores any existing render. Use this for "what will generate", not playback.
  function _planDurationSec(sc) {
    const p = state.project; if (!p || !sc) return 0;
    if (isVideoClip(sc) && sc.source_dur != null) return sc.source_dur;
    const fps = sceneEffFps(sc, p) || 25;
    const frames = sceneEffFrames(sc, p) || 1;
    return frames / fps;
  }

  // A scene's duration in seconds for layout / playback / trimming. A generated render is
  // IMMUTABLE: once made it keeps the length it was rendered at, so changing the project or
  // scene default can never truncate it or make its tail unreachable — regenerate or trim to
  // change it. An explicit per-scene length (timeline/custom mode), a slip-trim duration, or
  // a split (which clears the frozen length) all take precedence and fall back to the plan.
  function sceneDurationSec(sc) {
    if (!sc) return 0;
    const rd = renderDurationSec(sc.id);
    if (rd != null && !isVideoClip(sc) && sc.source_dur == null
        && (sc.frames_mode || "project") === "project") {
      return rd;
    }
    return _planDurationSec(sc);
  }

  // Insert or adjust a black/silent pause after this clip (seconds).
  function setSceneGapAfter(id, sec) {
    if (!state.project) return;
    const s = scene(id);
    if (!s) return;
    const gap = Math.max(0, +sec || 0);
    if (Math.abs((s.gap_after_sec || 0) - gap) < 0.001) return;
    patchScene(id, { gap_after_sec: gap });
  }

  // Resize a clip by setting its frame count from a target duration (seconds) × fps.
  function resizeScene(id, durationSec) {
    if (!state.project) return;
    const s = scene(id); if (!s) return;
    if (s.frames_mode === "custom") return;
    const fps = sceneEffFps(s);
    const curDur = sceneDurationSec(s);
    const target = Math.max(0.1, durationSec);
    if (Math.abs(target - curDur) <= 0.02) return;
    _historyRecord();
    if (isVideoClip(s)) {
      s.source_dur = target;
      s.frames = snapFrames(target * fps);
    } else {
      const curFrames = Math.max(1, Math.round(sceneDurationSec(s) * fps)); // resize from what's shown
      const nextFrames = _framesFromDuration(target, fps, curFrames);
      if (nextFrames === curFrames) return;
      s.frames = nextFrames;
    }
    if ((s.frames_mode || "project") === "project") s.frames_mode = "timeline";
    _syncSeparatedAudioTracks();
    notify();
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
    const effFps = sceneEffFps(s);
    let frames = sceneEffFrames(s);
    if (isVideoClip(s) && s.source_dur != null) frames = snapFrames(s.source_dur * effFps);
    const cut = snapFrames(atFrames != null ? atFrames : frames / 2);
    if (cut <= 9 || cut >= frames) return;
    const cutSec = cut / effFps;
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
    // The split seam between the two halves is an internal HARD CUT: the original clip's
    // outgoing edge now belongs to the second half. Clear ALL transition fields on the first
    // half — the prompt transition (transition_to_next), the NLE video blend (video_transition),
    // and its frame count — so the seam doesn't show/render a phantom (the deep clone already
    // carried these onto the second half, which correctly keeps them).
    s.frames = cut; s.transition_to_next = ""; s.transition_frames = null; s.video_transition = "";
    // "project" frames_mode IGNORES the per-scene `frames` value (sceneEffFrames / backend
    // eff_frames both fall back to the project default), so a split of a default-mode scene
    // would leave the first half at full project length and render the second half as a whole
    // extra scene. Pin both halves to their cut frame counts — same flip resizeScene does.
    if ((s.frames_mode || "project") === "project") s.frames_mode = "timeline";
    if ((second.frames_mode || "project") === "project") second.frames_mode = "timeline";
    if (playsFromSourceMedia(s)) {
      const srcIn = s.source_in || 0;
      const totalDur = s.source_dur != null ? s.source_dur : (frames / effFps);
      const secondDur = second.frames / effFps;
      s.source_dur = cutSec;
      second.source_in = srcIn + cutSec;
      second.source_dur = Math.max(0.1, Math.min(secondDur, totalDur - cutSec));
    }
    arr.splice(i + 1, 0, second);
    // With a pinned custom cut order, _ensureTimelineOrder appends unknown ids at the END of
    // the timeline — keep the second half glued to its first half in the cut as well.
    if (state.project.timeline_manually_ordered && Array.isArray(state.project.timeline_order)) {
      const oi = state.project.timeline_order.indexOf(s.id);
      if (oi >= 0) state.project.timeline_order.splice(oi + 1, 0, second.id);
    }
    // Keep the render across the cut: the second half plays the SAME source video starting
    // at the first half's out-point (its in-point + first-half duration).
    const r = state.sceneRenders[id];
    if (r && r.media) {
      delete r.durationSec; // a split is an explicit re-length: both halves revert to plan layout
      r._editTok = (r._editTok || 0) + 1; // invalidate any in-flight _probeRenderDuration for this scene
      state.sceneRenders[second.id] = {
        media: r.media,
        inSec: (r.inSec || 0) + cutSec,
        renderPrompt: r.renderPrompt ? { ...r.renderPrompt } : _snapshotRenderPrompt(id),
      };
    }
    notify(); scheduleSave();
    _syncSeparatedAudioTracks();
  }

  // Any scene with playable content, regardless of type: an explicit "video" clip plays
  // straight off media via source_in/source_dur; a generative scene plays off its OWN
  // render record (state.sceneRenders[id].media + inSec, with duration driven by `frames`,
  // same convention used by removeFromPlan/renderMediaLabel elsewhere in this file). Returns
  // null when neither applies, so callers can treat any scene uniformly via this one shape.
  function _renderRef(s) {
    if (!s) return null;
    const fps = sceneEffFps(s);
    if (isVideoClip(s)) {
      return { isVideo: true, media: null, renderPrompt: null,
               baseInSec: s.source_in || 0, fps, totalSec: s.source_dur != null ? s.source_dur : sceneDurationSec(s) };
    }
    const r = state.sceneRenders[s.id];
    if (!r || !r.media) return null;
    return { isVideo: false, media: r.media, renderPrompt: r.renderPrompt || null,
             baseInSec: r.inSec || 0, fps, totalSec: sceneDurationSec(s) };
  }

  function hasPlayableRender(s) { return _renderRef(s) !== null; }

  // Auto Montage: chop an already-rendered "lead" scene into shrinking-length segments
  // (trailer pacing — cuts get faster toward the end) and interleave a randomly-drawn
  // segment from a pool of other rendered scenes after each one. Pure timeline construction,
  // no new generation involved — every built segment reuses an existing scene's already-
  // rendered media, via source_in/source_dur for an explicit video clip or via a cloned
  // sceneRenders entry (media + inSec) for a generative scene's own render.
  function autoMontage({ leadId, poolIds, segmentFrames, decay } = {}) {
    if (!state.project) return;
    const lead = scene(leadId);
    const leadRef = _renderRef(lead);
    if (!leadRef) return;
    const pool = (poolIds || []).map((id) => scene(id)).filter((s) => s && hasPlayableRender(s));
    if (!pool.length) return;

    const base = Math.max(9, Math.round(segmentFrames || 100));
    const dk = decay != null ? Math.min(1, Math.max(0.3, decay)) : 1;

    const leadTotal = snapFrames(leadRef.totalSec * leadRef.fps);

    // Lead segments stay in chronological order (one continuous shot being interrupted),
    // shrinking each cycle so cutaways come faster as the take progresses.
    const leadSegs = [];
    let consumed = 0, i = 0;
    while (consumed < leadTotal - 8) {
      const len = Math.min(leadTotal - consumed, Math.max(9, Math.round(base * Math.pow(dk, i))));
      leadSegs.push({ start: consumed, frames: snapFrames(len) });
      consumed += len;
      i++;
    }
    if (!leadSegs.length) return;

    // Pool: chop every selected scene into its own ORDERED queue of base-length
    // chunks (chronological — never reordered within a scene). Randomness only picks
    // WHICH scene's queue to draw from next; each draw advances that scene's own cursor,
    // so a scene's later content can never surface before its own earlier content.
    const queues = pool.map((s) => {
      const ref = _renderRef(s);
      const total = snapFrames(ref.totalSec * ref.fps);
      const chunks = [];
      let c = 0;
      while (c < total - 8) {
        const len = Math.min(total - c, base);
        if (len >= 9) chunks.push({ scene: s, ref, start: c, frames: snapFrames(len) });
        c += len;
      }
      return { chunks, cursor: 0 };
    }).filter((q) => q.chunks.length);
    if (!queues.length) return;

    function nextPoolChunk() {
      let available = queues.filter((q) => q.cursor < q.chunks.length);
      if (!available.length) {
        queues.forEach((q) => { q.cursor = 0; }); // every scene exhausted — start each over from its own beginning
        available = queues;
      }
      const q = available[Math.floor(Math.random() * available.length)];
      return q.chunks[q.cursor++];
    }

    function buildSegment(src, ref, startFrames, lenFrames) {
      const out = JSON.parse(JSON.stringify(src));
      const newId = "c" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
      out.id = newId;
      out.gen_unit_id = src.id;
      out.frames = lenFrames;
      out.rating = "";
      out.transition_to_next = "cut";
      out.transition_frames = null;
      const offsetSec = startFrames / ref.fps;
      if (ref.isVideo) {
        out.source_in = ref.baseInSec + offsetSec;
        out.source_dur = lenFrames / ref.fps;
        out.frames_mode = "timeline";
      } else {
        out.cut_offset_frames = (src.cut_offset_frames || 0) + startFrames;
        state.sceneRenders[newId] = {
          media: ref.media,
          inSec: ref.baseInSec + offsetSec,
          renderPrompt: ref.renderPrompt ? { ...ref.renderPrompt } : undefined,
        };
      }
      return out;
    }

    _historyRecord();
    const built = [];
    leadSegs.forEach((seg) => {
      built.push(buildSegment(lead, leadRef, seg.start, seg.frames));
      const chunk = nextPoolChunk();
      built.push(buildSegment(chunk.scene, chunk.ref, chunk.start, chunk.frames));
    });
    built[built.length - 1].transition_to_next = "";

    state.project.scenes.push(...built);
    window.Timeline?.requestAutoFit?.();
    notify(); scheduleSave();
    return built.length;
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
    return !!(t && t.kind === "separated");
  }

  function sceneHasEmbeddedAudio(sc) {
    if (!sc || sc.excluded || sc.audio_separated) return false;
    if (isVideoClip(sc)) {
      const ref = sc.source?.media_ref;
      if (!ref) return false;
      const asset = (state.mediaBin || []).find((m) => m.id === ref);
      return asset?.kind === "video";
    }
    if (sc.source?.type === "v2v" && sc.source?.media_ref) {
      const asset = (state.mediaBin || []).find((m) => m.id === sc.source.media_ref);
      if (asset?.kind === "video") return true;
    }
    return !!(state.sceneRenders[sc.id]?.media);
  }

  function separatedTrackAudioUrl(st, track) {
    if (!track) return null;
    if (track.pinned_bin_ref) {
      const asset = (st?.mediaBin || state.mediaBin || []).find((m) => m.id === track.pinned_bin_ref);
      return asset?.kind === "video" ? window.MovieEditorAPI.mediaUrl(track.pinned_bin_ref) : null;
    }
    const media = separatedTrackMedia(track);
    if (media && st?.project?.id) return window.MovieEditorAPI.resultUrl(st.project.id, media);
    if (media && state.project?.id) return window.MovieEditorAPI.resultUrl(state.project.id, media);
    return null;
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
    if (!sc || sc.audio_separated || !sceneHasEmbeddedAudio(sc)) return;
    if (separatedTrackForScene(sceneId)) return;
    _historyRecord();
    const idx = state.project.scenes.indexOf(sc);
    const dur = sc.source_dur != null ? sc.source_dur : sceneDurationSec(sc);
    const savedVol = sc.audio_volume != null ? sc.audio_volume : 1;
    let inSec = sc.source_in || 0;
    let pinned_media = null;
    let pinned_bin_ref = null;

    if (isVideoClip(sc) || (sc.source?.type === "v2v" && sc.source?.media_ref && !state.sceneRenders[sceneId]?.media)) {
      pinned_bin_ref = sc.source.media_ref;
    } else {
      const r = state.sceneRenders[sceneId];
      if (!r?.media) { alert("Generate this clip first."); return; }
      inSec = clipSourceInSec(sc, r);
      pinned_media = JSON.parse(JSON.stringify(r.media));
    }

    state.project.audio_tracks = state.project.audio_tracks || [];
    state.project.audio_tracks.push({
      id: _uid(),
      kind: "separated",
      scene_id: sceneId,
      start_sec: sceneTimelineOffsetSec(sceneId),
      source_in_sec: inSec,
      source_dur: dur,
      pinned_media,
      pinned_bin_ref,
      pinned_in_sec: inSec,
      pinned_dur: dur,
      volume: savedVol,
      label: (isVideoClip(sc) ? `V${idx + 1}` : `S${idx + 1}`) + " audio",
    });
    sc.audio_separated = true;
    sc.audio_volume = 0;
    notify(); scheduleSave();
  }

  function audioTrackInSec(t) {
    if (!t) return 0;
    if (t.pinned_in_sec != null) return t.pinned_in_sec;
    if (t.source_in_sec != null) return t.source_in_sec;
    return 0;
  }

  function audioTrackDurSec(t) {
    if (!t) return 0.1;
    if (t.pinned_dur != null) return t.pinned_dur;
    if (t.source_dur != null) return t.source_dur;
    const asset = t.media_ref ? (state.mediaBin || []).find((m) => m.id === t.media_ref) : null;
    if (asset?.duration_sec) return asset.duration_sec;
    return 1;
  }

  function audioTrackSourceDurSec(t) {
    if (!t) return null;
    if (t.pinned_bin_ref) {
      const asset = (state.mediaBin || []).find((m) => m.id === t.pinned_bin_ref);
      return asset?.duration_sec ?? null;
    }
    if (t.media_ref) {
      const asset = (state.mediaBin || []).find((m) => m.id === t.media_ref);
      return asset?.duration_sec ?? null;
    }
    return null;
  }

  function audioTrackMaxDurSec(t) {
    const inSec = audioTrackInSec(t);
    const srcDur = audioTrackSourceDurSec(t);
    if (srcDur != null) return Math.max(0.1, srcDur - inSec);
    return Math.max(audioTrackDurSec(t), 86400);
  }

  function _applyAudioTrimFields(t, patch) {
    if (patch.start_sec != null) t.start_sec = Math.max(0, +patch.start_sec);
    if (patch.source_in_sec != null) t.source_in_sec = Math.max(0, +patch.source_in_sec);
    if (patch.source_dur != null) t.source_dur = Math.max(0.1, +patch.source_dur);
    if (isSeparatedAudioTrack(t)) {
      if (patch.source_in_sec != null) t.pinned_in_sec = t.source_in_sec;
      if (patch.source_dur != null) t.pinned_dur = t.source_dur;
    }
  }

  function trimAudioTrackLeft(id, trimSec, opts) {
    if (!state.project) return;
    const t = (state.project.audio_tracks || []).find((x) => x.id === id);
    if (!t) return;
    const trim = Math.max(0, +trimSec || 0);
    if (trim < 0.02) return;
    const inSec = audioTrackInSec(t);
    const dur = audioTrackDurSec(t);
    const start = t.start_sec || 0;
    const linked = isSeparatedAudioTrack(t) && t.scene_id && scene(t.scene_id);
    if (opts?.slip) {
      slipAudioTrack(id, trim, opts);
      return;
    }
    if (trim >= dur - 0.1) return;
    if (!opts?.quiet) _historyRecord();
    if (linked) {
      _applyAudioTrimFields(t, { source_in_sec: inSec + trim, source_dur: dur - trim });
    } else {
      _applyAudioTrimFields(t, {
        start_sec: start + trim,
        source_in_sec: inSec + trim,
        source_dur: dur - trim,
      });
    }
    notify(); opts?.quiet ? scheduleSaveSilent() : scheduleSave();
  }

  function slipAudioTrack(id, deltaSec, opts) {
    if (!state.project) return;
    const t = (state.project.audio_tracks || []).find((x) => x.id === id);
    if (!t) return;
    const delta = +deltaSec || 0;
    if (Math.abs(delta) < 0.02) return;
    const inSec = audioTrackInSec(t);
    const dur = audioTrackDurSec(t);
    const srcDur = audioTrackSourceDurSec(t);
    let newIn = Math.max(0, inSec + delta);
    if (srcDur != null) newIn = Math.min(newIn, Math.max(0, srcDur - dur));
    if (Math.abs(newIn - inSec) < 0.02) return;
    if (!opts?.quiet) _historyRecord();
    _applyAudioTrimFields(t, { source_in_sec: newIn, source_dur: dur });
    notify(); opts?.quiet ? scheduleSaveSilent() : scheduleSave();
  }

  function resizeAudioTrack(id, durationSec, opts) {
    if (!state.project) return;
    const t = (state.project.audio_tracks || []).find((x) => x.id === id);
    if (!t) return;
    let dur = Math.max(0.1, +durationSec || 0);
    dur = Math.min(dur, audioTrackMaxDurSec(t));
    if (Math.abs(dur - audioTrackDurSec(t)) < 0.02) return;
    if (!opts?.quiet) _historyRecord();
    _applyAudioTrimFields(t, { source_dur: dur });
    notify(); opts?.quiet ? scheduleSaveSilent() : scheduleSave();
  }

  function splitAudioTrack(id, timelineSec) {
    if (!state.project) return;
    const t = (state.project.audio_tracks || []).find((x) => x.id === id);
    if (!t) return;
    const start = t.start_sec || 0;
    const inSec = audioTrackInSec(t);
    const dur = audioTrackDurSec(t);
    const rel = (+timelineSec || 0) - start;
    if (rel <= 0.05 || rel >= dur - 0.05) return;
    _historyRecord();
    _applyAudioTrimFields(t, { start_sec: start, source_in_sec: inSec, source_dur: rel });
    const second = {
      id: _uid(),
      kind: t.kind || "overlay",
      start_sec: start + rel,
      source_in_sec: inSec + rel,
      source_dur: dur - rel,
      volume: t.volume != null ? t.volume : 1,
      label: (t.label || "audio") + " (2)",
    };
    if (t.media_ref) second.media_ref = t.media_ref;
    if (t.render_media) second.render_media = JSON.parse(JSON.stringify(t.render_media));
    if (t.pinned_bin_ref) second.pinned_bin_ref = t.pinned_bin_ref;
    if (t.pinned_media) second.pinned_media = JSON.parse(JSON.stringify(t.pinned_media));
    if (isSeparatedAudioTrack(t)) {
      second.pinned_in_sec = second.source_in_sec;
      second.pinned_dur = second.source_dur;
    }
    state.project.audio_tracks.push(second);
    state.selectedAudioTrackId = second.id;
    state.selectedSceneId = null;
    state.selectedSceneIds = [];
    state.selectedOverlayId = null;
    notify(); scheduleSave();
  }

  function selectAudioTrack(id) {
    state.selectedAudioTrackId = id || null;
    if (id) {
      state.selectedSceneId = null;
      state.selectedSceneIds = [];
      state.selectedOverlayId = null;
      _selectionAnchorId = null;
    }
    notify();
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
        ? (t.scene_id && scene(t.scene_id)
          ? "Remove separated audio? The clip will use embedded audio from its current video render again."
          : "Remove this detached audio track?")
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
    if (state.selectedAudioTrackId === id) state.selectedAudioTrackId = null;
    notify(); scheduleSave();
  }

  function _playheadSec() {
    try { return window.Player?.getPlayhead?.() ?? 0; } catch (_) { return 0; }
  }

  function overlayTrack(id) {
    const t = (state.project?.overlay_tracks || []).find((x) => x.id === id) || null;
    return t ? normalizeOverlayTrack(t) : null;
  }

  function projectCanvasSize() {
    return { w: state.project?.width || 768, h: state.project?.height || 512 };
  }

  function normalizeOverlayTrack(ov) {
    if (!ov) return ov;
    const cw = state.project?.width || 768;
    const ch = state.project?.height || 512;
    if (ov.kind === "image") {
      if (ov.width_px == null && ov.scale != null) ov.width_px = Math.max(8, Math.round(+ov.scale * cw));
      if (ov.width_px == null) ov.width_px = Math.max(8, Math.round(cw * 0.35));
      if (ov.keep_aspect === undefined) ov.keep_aspect = true;
      if (!ov.keep_aspect && ov.height_px == null) ov.height_px = ov.width_px;
    }
    ov.flip_h = !!ov.flip_h;
    ov.flip_v = !!ov.flip_v;
    return ov;
  }

  function normalizeOverlayLanes(project) {
    if (!project) return;
    project.overlay_tracks = project.overlay_tracks || [];
    let lanes = project.overlay_lanes = project.overlay_lanes || [];
    if (!lanes.length) {
      const id = _uid();
      lanes = project.overlay_lanes = [{ id, label: "Overlay 1" }];
    }
    const ids = new Set(lanes.map((l) => l.id));
    const fallback = lanes[0].id;
    project.overlay_tracks.forEach((t) => {
      if (!t.lane_id || !ids.has(t.lane_id)) t.lane_id = fallback;
      normalizeOverlayTrack(t);
    });
  }

  function ensureOverlayLanes() {
    if (!state.project) return [];
    normalizeOverlayLanes(state.project);
    return state.project.overlay_lanes;
  }

  function overlayLaneById(laneId) {
    return ensureOverlayLanes().find((l) => l.id === laneId) || null;
  }

  function overlayLaneIndex(laneId) {
    return ensureOverlayLanes().findIndex((l) => l.id === laneId);
  }

  function sortedOverlayTracks() {
    if (!state.project) return [];
    ensureOverlayLanes();
    const tracks = [...(state.project.overlay_tracks || [])];
    tracks.sort((a, b) => {
      const ai = overlayLaneIndex(a.lane_id);
      const bi = overlayLaneIndex(b.lane_id);
      if (ai !== bi) return ai - bi;
      return (a.start_sec || 0) - (b.start_sec || 0);
    });
    return tracks;
  }

  function defaultOverlayLaneId() {
    const lanes = ensureOverlayLanes();
    return lanes[lanes.length - 1]?.id || lanes[0]?.id;
  }

  function addOverlayLane(opts) {
    if (!state.project) return null;
    _historyRecord();
    ensureOverlayLanes();
    const id = _uid();
    const n = state.project.overlay_lanes.length + 1;
    const lane = { id, label: `Overlay ${n}` };
    if (opts?.belowId) {
      const idx = state.project.overlay_lanes.findIndex((l) => l.id === opts.belowId);
      state.project.overlay_lanes.splice(Math.max(0, idx), 0, lane);
    } else {
      state.project.overlay_lanes.push(lane);
    }
    notify(); scheduleSave();
    return id;
  }

  function removeOverlayLane(laneId, opts) {
    if (!state.project) return;
    ensureOverlayLanes();
    if (state.project.overlay_lanes.length <= 1) return;
    const hasClips = state.project.overlay_tracks.some((t) => t.lane_id === laneId);
    if (hasClips && !opts?.skipConfirm && !confirm("Remove this overlay track and all clips on it?")) return;
    _historyRecord();
    state.project.overlay_tracks = state.project.overlay_tracks.filter((t) => t.lane_id !== laneId);
    state.project.overlay_lanes = state.project.overlay_lanes.filter((l) => l.id !== laneId);
    notify(); scheduleSave();
  }

  function _setOverlayLane(overlayId, laneId) {
    const t = overlayTrack(overlayId);
    if (!t || !overlayLaneById(laneId)) return;
    t.lane_id = laneId;
    notify(); scheduleSave();
  }

  function bringOverlayToFront(overlayId) {
    const lanes = ensureOverlayLanes();
    if (!lanes.length) return;
    _historyRecord();
    _setOverlayLane(overlayId, lanes[lanes.length - 1].id);
  }

  function sendOverlayToBack(overlayId) {
    const lanes = ensureOverlayLanes();
    if (!lanes.length) return;
    _historyRecord();
    _setOverlayLane(overlayId, lanes[0].id);
  }

  function bringOverlayForward(overlayId) {
    const t = overlayTrack(overlayId);
    if (!t) return;
    const lanes = ensureOverlayLanes();
    const idx = overlayLaneIndex(t.lane_id);
    if (idx < 0 || idx >= lanes.length - 1) return;
    _historyRecord();
    _setOverlayLane(overlayId, lanes[idx + 1].id);
  }

  function sendOverlayBackward(overlayId) {
    const t = overlayTrack(overlayId);
    if (!t) return;
    const lanes = ensureOverlayLanes();
    const idx = overlayLaneIndex(t.lane_id);
    if (idx <= 0) return;
    _historyRecord();
    _setOverlayLane(overlayId, lanes[idx - 1].id);
  }

  function assignMediaToOverlayLane(laneId, mediaId) {
    if (!state.project || !laneId || !mediaId) return;
    ensureOverlayLanes();
    if (!overlayLaneById(laneId)) return;
    const asset = (state.mediaBin || []).find((m) => m.id === mediaId);
    if (asset?.kind === "image") addImageOverlay(mediaId, undefined, laneId);
  }

  function selectOverlay(id) {
    state.selectedOverlayId = id || null;
    if (id) {
      state.selectedSceneId = null;
      state.selectedSceneIds = [];
      state.selectedAudioTrackId = null;
      _selectionAnchorId = null;
    }
    notify();
  }

  function addImageOverlay(mediaId, startSec, laneId) {
    if (!state.project || !mediaId) return;
    _historyRecord();
    ensureOverlayLanes();
    const asset = (state.mediaBin || []).find((m) => m.id === mediaId);
    const lane = laneId && overlayLaneById(laneId) ? laneId : defaultOverlayLaneId();
    state.project.overlay_tracks = state.project.overlay_tracks || [];
    const id = _uid();
    const { w: cw, h: ch } = projectCanvasSize();
    state.project.overlay_tracks.push({
      id,
      lane_id: lane,
      kind: "image",
      media_ref: mediaId,
      start_sec: Math.max(0, +(startSec ?? _playheadSec())),
      duration_sec: 3,
      x: 0.5,
      y: 0.5,
      width_px: Math.max(8, Math.round(cw * 0.35)),
      keep_aspect: true,
      flip_h: false,
      flip_v: false,
      opacity: 1,
      label: (asset && asset.name) || "Image",
    });
    selectOverlay(id);
    scheduleSave();
  }

  function addTextOverlay(text, startSec, laneId) {
    if (!state.project) return;
    _historyRecord();
    ensureOverlayLanes();
    const lane = laneId && overlayLaneById(laneId) ? laneId : defaultOverlayLaneId();
    state.project.overlay_tracks = state.project.overlay_tracks || [];
    const id = _uid();
    state.project.overlay_tracks.push({
      id,
      lane_id: lane,
      kind: "text",
      text: String(text || "Title").trim() || "Title",
      font_size: 42,
      font_family: "system-ui",
      color: "#ffffff",
      start_sec: Math.max(0, +(startSec ?? _playheadSec())),
      duration_sec: 3,
      x: 0.5,
      y: 0.5,
      flip_h: false,
      flip_v: false,
      opacity: 1,
      label: "Text",
    });
    selectOverlay(id);
    scheduleSave();
  }

  function updateOverlayTrack(id, patch, quiet) {
    const t = overlayTrack(id);
    if (!t) return;
    if (!quiet) _historyRecord();
    Object.assign(t, patch);
    notify();
    quiet ? scheduleSaveSilent() : scheduleSave();
  }

  function removeOverlayTrack(id, opts) {
    if (!state.project) return;
    const t = overlayTrack(id);
    if (!t) return;
    if (!opts?.skipConfirm && !confirm("Remove this overlay?")) return;
    _historyRecord();
    state.project.overlay_tracks = (state.project.overlay_tracks || []).filter((x) => x.id !== id);
    if (state.selectedOverlayId === id) state.selectedOverlayId = null;
    notify(); scheduleSave();
  }

  function removeSelectedOverlay() {
    const id = state.selectedOverlayId;
    if (!id) return;
    removeOverlayTrack(id, { skipConfirm: true });
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
    [/prompt_no_outputs|Prompt has no outputs/i,
      "The graph reached ComfyUI without any output node. Usually the built-in pipeline is disabled for this project (Models → Enable built-in pipeline), or your custom pipeline needs a final IMAGE wired to the 🌐 Global video output."],
    [/HTTP 53[0-9]|HTTP 52[24]|HTTP 502|Failed to fetch|NetworkError|Load failed/i,
      "Cloudflare tunnel lost contact with the vast.ai instance. GPU work may still be running — wait, refresh, and check ComfyUI output. If it keeps failing, use vast.ai direct port access or restart the tunnel."],
  ];

  function _isTransientTunnelError(err) {
    const m = String(err?.message || err || "");
    return /HTTP 53[0-9]|HTTP 52[24]|HTTP 502|Failed to fetch|NetworkError|Load failed|fetch/i.test(m);
  }

  async function _retryOnTunnel(fn, attempts) {
    attempts = attempts || 8;
    let last;
    for (let i = 0; i < attempts; i++) {
      try {
        return await fn();
      } catch (e) {
        last = e;
        if (!_isTransientTunnelError(e) || i >= attempts - 1) throw e;
        await _sleep(1500 + i * 500);
      }
    }
    throw last;
  }

  async function _flushSaveForGenerate() {
    await flushGlobalPromptApply();   // distribute any just-typed global prompt into scenes first
    await flushModelsSave();          // persist exposed-control edits before building the graph
    await flushSave();
    if (_localDirty) {
      throw new Error(
        "Could not save your latest edits before generate (tunnel may have dropped). "
        + "Wait for the save indicator to clear, then try again."
      );
    }
  }

  function _sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

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
  // Wall clock for the WHOLE operation the user started, as distinct from pollStart, which
  // restarts on every run of a multi-run montage. The transport buttons count from the press
  // to the finished video, so this has to span every run that one press kicked off.
  let genWallStart = 0;
  let genWallSource = null;
  let genWallApprox = false;
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

  // Re-attach to a generation that was already running when the UI (re)loaded. The job
  // lives in ComfyUI's still-running queue; bridge.active_generation hands back its
  // prompt id + the project/scenes it targets. We restore the "running" gen state so
  // Generate stays blocked and Interrupt is offered — exactly like a live run — and,
  // when the matching project is loaded, record the result onto its scenes on completion.
  async function resumeRunningGeneration(act) {
    if (!act || !act.running || !act.prompt_id) return false;
    _interrupted = false;
    // A reloaded montage can have further runs still queued behind the running one. Re-attach
    // to each in turn so Generate stays blocked (and outputs get recorded) until the queue
    // drains — not just for the one prompt that happened to be running at reload.
    let cur = act;
    let lastPromptId = null;
    // Re-attaching means we never saw the press, so the button's clock can only count from
    // here. Flagged approximate so it reads "1m 4s+" rather than claiming a total it can't know.
    _genClockStart("all", true);
    try {
      while (cur && cur.running && cur.prompt_id && cur.prompt_id !== lastPromptId && !_interrupted) {
        lastPromptId = cur.prompt_id;
        pollStart = Date.now();
        const sceneIds = (cur.scene_ids && cur.scene_ids.length)
          ? cur.scene_ids
          : (cur.only_scene ? [cur.only_scene] : []);
        const more = cur.pending > 0 ? ` · ${cur.pending} run(s) queued` : "";
        set({ gen: { state: "running", promptId: cur.prompt_id, media: [], msg: `Generation running (reconnected after reload)…${more}` } });
        if (state.project && cur.pid && state.project.id === cur.pid && sceneIds.length) {
          // Full re-attach: poll history and place the output on the run's scenes when done.
          _markGenInFlight(sceneIds);
          try { await _pollPromise(cur.prompt_id, sceneIds, "Generation running (reconnected)"); }
          finally { _clearGenInFlight(sceneIds); }
        } else {
          // No matching project loaded (or single-scene run on another project): we can't map
          // the output, but still surface Interrupt and clear busy when it stops.
          await _monitorActive(cur.prompt_id);
        }
        if (_interrupted) break;
        // Another queued run may now be executing — re-attach to it.
        try { cur = await API.active(); } catch (_) { cur = null; }
      }
    } finally { _genClockStop(); }
    return true;
  }

  // Lightweight queue watcher used when we can't fully re-attach (see above). Polls
  // /active until the prompt leaves the queue or the user interrupts.
  function _monitorActive(promptId) {
    return new Promise((resolve) => {
      _clearGenTimers();
      pollTimer = setInterval(async () => {
        if (_interrupted) {
          _clearGenTimers();
          set({ gen: { state: "idle", promptId, media: [], msg: "Interrupted." } });
          resolve();
          return;
        }
        try {
          const a = await API.active();
          if (!a || !a.running || a.prompt_id !== promptId) {
            _clearGenTimers();
            set({ gen: { state: "done", promptId, media: [], msg: "Generation finished. Open Settings ▸ Temp files to view or save the output, then regenerate to place it." } });
            resolve();
          } else {
            updateGenProgress({ msg: `Generation running (reconnected) ${_elapsed()}` });
          }
        } catch (_) { /* transient — keep watching */ }
      }, 2000);
    });
  }

  function _elapsed() {
    const s = Math.floor((Date.now() - pollStart) / 1000);
    return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
  }

  // A nested call (generate -> generateMontage) must keep the ORIGINAL press time, so the
  // first start wins and later ones are no-ops until the clock is stopped.
  function _genClockStart(source, approx) {
    if (genWallStart) return;
    genWallStart = Date.now();
    genWallSource = source;
    genWallApprox = !!approx;
  }
  function _genClockStop() { genWallStart = 0; genWallSource = null; genWallApprox = false; }

  // Read by the transport buttons on their own 1s tick. Deliberately NOT part of `state`:
  // a value that changes every second would churn every view fingerprint that hashed it,
  // rebuilding whole zones once a second for a number that one button owns.
  function genElapsed() {
    if (!genWallStart) return null;
    return { ms: Date.now() - genWallStart, source: genWallSource, approx: genWallApprox };
  }

  // Split the active scenes into generation runs. Each anchored scene (empty / image /
  // generated_frame / mixed) is its own queue request. Only "carry" continues the
  // previous run (continuous chain overlap). Mixed = solo i2v + prior-scene guides.
  // True if a scene's image anchor still exists in the media bin (deleted image → fall
  // back to carry / i2v guides, not a broken anchor).
  function _anchorAvailable(s) {
    const t = s.source && s.source.type;
    if (t !== "image" && t !== "generated_frame" && t !== "mixed" && t !== "v2v" && t !== "anchor_guide") return false;
    const ref = s.source.media_ref;
    return !!(ref && (state.mediaBin || []).some((m) => m.id === ref));
  }
  function _runs() {
    const active = state.project.scenes.filter((s) => !s.excluded && isGenerativeScene(s));
    if (window.PipelineCaps && !window.PipelineCaps.usesChainSampler(state)) {
      return active.length ? [active.map((s) => s.id)] : [];
    }
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
  function _pruneGhostsAfterRegen(recordedSceneIds, protectedGhostIds) {
    const ids = new Set(recordedSceneIds || []);
    const keep = new Set(protectedGhostIds || []);
    // Drop ghosts superseded by this run's fresh renders (anchored after, or sharing id with, a
    // recorded scene) — but never a ghost that just received its own finished media this pass.
    state.sceneGhosts = (state.sceneGhosts || []).filter(
      (g) => keep.has(g.id) || (!ids.has(g.afterSceneId) && !ids.has(g.id))
    );
  }

  function _ghostDurationSec(g) {
    if (!g) return 0;
    if (g.durationSec != null) return g.durationSec; // frozen real length once rendered
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

  // Map one chain output file onto timeline clips: inSec from scene_layout timestamps only.
  // Overlap is internal to generation — preview/export split at boundaries, not (span - overlap).
  function _chainRunLayout(sceneIds) {
    const p = state.project;
    const scenes = (sceneIds || []).map((id) => scene(id)).filter(Boolean);
    const fps = p?.frame_rate || 25;
    const frames = _effectiveRunFrames(scenes);
    const sourceSpanSec = frames / fps;
    return { frames, fps, sourceSpanSec };
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

  function _fallbackChainPlaybackLayout(sceneIds) {
    const unitCount = _chainUnitCount(sceneIds);
    if (unitCount <= 1) return null;
    const run = _chainRunLayout(sceneIds);
    return Array.from({ length: unitCount }, (_, i) => ({
      scene_index: i,
      start_frame: Math.round(i * run.frames),
      in_sec: Math.round(i * run.sourceSpanSec * 1e6) / 1e6,
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

  function _layoutInSec(sceneLayout, chainSceneIdx, sourceSpanSec) {
    const slot = _layoutSlot(sceneLayout, chainSceneIdx);
    if (slot) return slot.inSec != null ? slot.inSec : (slot.in_sec != null ? slot.in_sec : 0);
    return chainSceneIdx * sourceSpanSec;
  }

  function _recordInSec(lastChain, sourceEnd, curr, sceneLayout, chainSceneIdx, sourceSpanSec) {
    if (lastChain && genUnitId(lastChain) === genUnitId(curr)) return sourceEnd;
    return _layoutInSec(sceneLayout, chainSceneIdx, sourceSpanSec);
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
    const sourceSpanSec = layout.sourceSpanSec;
    let sourceEnd = 0;
    let lastChain = null;
    let chainSceneIdx = 0;
    let clearedRating = false;
    const recordedSceneIds = [];
    // Ghosts of scenes removed mid-generation that just received their finished media in this
    // pass. They must survive _pruneGhostsAfterRegen — they ARE the fresh result, not the stale
    // pre-regen render the prune is meant to drop.
    const completedGhostIds = [];
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      if (_removedDuringGen.has(id)) {
        const ghosts = state.sceneGhosts || [];
        const gi = ghosts.findIndex((g) => g.id === id);
        if (gi >= 0) {
          const ghost = ghosts[gi];
          const sameUnit = lastChain && genUnitId(lastChain) === genUnitId(ghost);
          const inSec = sameUnit ? sourceEnd : _recordInSec(lastChain, sourceEnd, ghost, sceneLayout, chainSceneIdx, sourceSpanSec);
          const ghostDur = _ghostDurationSec(ghost); // freeze before any project change can shrink it
          ghosts[gi] = { ...ghost, media: primary, inSec, durationSec: ghostDur, pendingGen: false };
          state.sceneGhosts = ghosts;
          completedGhostIds.push(id);
          sourceEnd = inSec + ghostDur;
          lastChain = ghost;
          if (!sameUnit) chainSceneIdx += 1;
        }
        _removedDuringGen.delete(id);
        continue;
      }
      const sc = scene(id); if (!sc) continue;
      const sameUnit = lastChain && genUnitId(lastChain) === genUnitId(sc);
      const inSec = sameUnit ? sourceEnd : _recordInSec(lastChain, sourceEnd, sc, sceneLayout, chainSceneIdx, sourceSpanSec);
      const renderPrompt = _queuedRenderPrompts[id] || _snapshotRenderPrompt(id);
      delete _queuedRenderPrompts[id];
      const realDur = _planDurationSec(sc); // freeze the length this render was actually made at
      state.sceneRenders[id] = { media: primary, inSec, renderPrompt, durationSec: realDur };
      recordedSceneIds.push(id);
      const root = genUnitRoot(genUnitId(sc));
      if (root && root.rating) { root.rating = ""; clearedRating = true; }
      sourceEnd = inSec + realDur;
      lastChain = sc;
      if (!sameUnit) chainSceneIdx += 1;
    }
    _pruneGhostsAfterRegen(recordedSceneIds, completedGhostIds);
    _syncSeparatedTracksAfterRegen(recordedSceneIds);
    // Adopt the real encoded length for standalone renders so the clip matches the file (the
    // frozen length above is the plan estimate). Chains are skipped by the standalone guard.
    for (const id of recordedSceneIds) _probeRenderDuration(id);
    if (clearedRating) scheduleSaveSilent();
    if (recordedSceneIds.length) {
      _validateRendersToken++;
      state.notice = "";
      _syncEditorStateToProject();
      scheduleSaveSilent();
    }
  }

  // Poll a single queued prompt to completion. Resolves true on success, false on error.
  function _pollPromise(promptId, targetSceneIds, prefix) {
    prefix = prefix || "Generating…";
    return new Promise((resolve) => {
      _clearGenTimers();
      let pendingStreak = 0;
      let transientStreak = 0;
      // Faster step-progress poll (sampler current/total steps, plus what it says it's
      // doing — "scene 2/3 · pass 2 of 2"). The message is rebuilt from prefix + elapsed
      // the same way the slower poll below builds it, rather than regex-stripping the last
      // one off the end: a suffix the strip pattern doesn't know about (the phase label)
      // would otherwise be re-appended every tick and grow without limit.
      progressTimer = setInterval(async () => {
        if (_interrupted) return;
        try {
          const pr = await API.progress();
          if (pr && pr.max > 0) {
            const phase = pr.label ? `  ·  ${pr.label}` : "";
            updateGenProgress({
              step: pr.value,
              maxStep: pr.max,
              phase: pr.label || "",
              msg: `${prefix} ${_elapsed()}  ·  sampling ${pr.value}/${pr.max}${phase}`,
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
          transientStreak = 0;
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
            const phase = state.gen.phase ? `  ·  ${state.gen.phase}` : "";
            updateGenProgress({ state: s.state, msg: `${prefix} ${_elapsed()}${step}${phase}` });
          }
        } catch (e) {
          if (_isTransientTunnelError(e)) {
            transientStreak++;
            if (transientStreak < 120) {
              updateGenProgress({
                msg: `${prefix} ${_elapsed()} · tunnel reconnect (${transientStreak})…`,
              });
              return;
            }
          }
          _clearGenTimers();
          set({ gen: { ...state.gen, state: "error", msg: _friendlyGenError(e.message) } });
          resolve(false);
        }
      }, 2000);
    });
  }

  async function _pollFfmpegJob(statusFn, jobId, clipCount, busyLabel) {
    let transientStreak = 0;
    for (;;) {
      if (_interrupted) throw new Error("Interrupted.");
      try {
        const st = await statusFn(jobId);
        transientStreak = 0;
        if (st.state === "done") return st;
        if (st.state === "error") throw new Error(st.detail || `${busyLabel} failed`);
        updateGenProgress({ msg: `${busyLabel} ${clipCount} clip(s)…` });
      } catch (e) {
        if (_isTransientTunnelError(e)) {
          transientStreak++;
          if (transientStreak < 120) {
            updateGenProgress({
              msg: `${busyLabel} ${clipCount} clip(s) · tunnel reconnect (${transientStreak})…`,
            });
            await _sleep(2000);
            continue;
          }
        }
        throw e;
      }
      await _sleep(2000);
    }
  }

  async function _pollRenderJob(jobId, clipCount) {
    return _pollFfmpegJob(
      (id) => API.renderFinalStatus(state.project.id, id),
      jobId,
      clipCount,
      "Stitching",
    );
  }

  async function _pollExportClipsJob(jobId, clipCount) {
    return _pollFfmpegJob(
      (id) => API.exportClipsStatus(state.project.id, id),
      jobId,
      clipCount,
      "Combining",
    );
  }

  // Every refinement key a reset will wipe: the project/default key plus every non-default key
  // whose shortcut fires anywhere in the prompt. Uses the scene-count-independent
  // refinement_key_pool, which mirrors the backend _v2_reset_prompt_keys EXACTLY — the per-scene
  // scene_refinement_keys can drop a key via its divergence fallback, which would under-report and
  // let the reset silently wipe a key the user wasn't warned about. Falls back to the per-scene
  // list only for an older preview payload without the pool.
  function sessionResetKeys() {
    const pv = state.preview || {};
    const defKey = (state.project?.refinement_key || "default").trim() || "default";
    const out = new Set([defKey]);
    if (Array.isArray(pv.refinement_key_pool)) {
      pv.refinement_key_pool.forEach((k) => { if (k) out.add(k); });
    } else {
      (pv.scene_refinement_keys || []).forEach((rk) => {
        if (rk && !rk.uses_default) (rk.keys || []).forEach((k) => { if (k) out.add(k); });
      });
    }
    return [...out];
  }

  // Toggle a pending Studio session reset — applied to the FIRST run of the next
  // generation. Clicking again disarms it (in case of a mis-click). Arming asks for
  // confirmation, listing every key it will wipe so non-default keys aren't lost silently.
  let _resetSessionPending = false;
  async function resetStudioSession() {
    if (!_resetSessionPending) {
      // Read the key list off a FRESH preview: the cached one can lag the prompt (debounced
      // refresh, or _distributeGlobalPrompt carrying old scene_refinement_keys forward), so a
      // just-added key-bound shortcut would otherwise be missing from the wipe list.
      try { await refreshPreview(true); } catch (_) {}
      const keys = sessionResetKeys();
      const msg = `This action will reset Studio learning for keys: ${keys.join(", ")}.\n\n`
        + "It is applied on the next generation. To avoid resetting a non-default key, "
        + "remove the shortcut that activates it from the prompt.\n\nProceed?";
      if (!confirm(msg)) return;
    }
    _resetSessionPending = !_resetSessionPending;
    set({ resetSessionArmed: _resetSessionPending });
  }

  // Generate one run (single scene, or an explicit list of scene ids). Returns success.
  // i2v-node bypass overrides for a run that contains an "Anchor as guide" scene, when the
  // user has declared a custom i2v node in Editor settings (built-in pipeline needs none).
  function _anchorGuideNodeOverrides(sceneIds) {
    if (!getEditorSetting("anchorGuideHasI2v")) return null;
    const cfg = getEditorSetting("anchorGuideBypass");
    if (!cfg || !cfg.node || !cfg.input) return null;
    const hasAG = (sceneIds || []).some((id) => {
      const s = scene(id);
      return s && s.source && s.source.type === "anchor_guide";
    });
    if (!hasAG) return null;
    return [{ node: cfg.node, input: cfg.input, value: cfg.value }];
  }

  async function _generateRun(sceneIds, onlyScene, prefix, resetSession) {
    _interrupted = false;
    _markGenInFlight(sceneIds);
    set({ gen: { state: "queuing", promptId: null, media: [], msg: `${prefix}: queuing…`, step: 0, maxStep: 0 } });
    try {
      const overrides = _anchorGuideNodeOverrides(sceneIds);
      const r = await _retryOnTunnel(
        () => API.generate(state.project.id, onlyScene || null, onlyScene ? null : sceneIds, !!resetSession, overrides),
        10,
      );
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
    try {
      await _flushSaveForGenerate();
    } catch (e) {
      set({ gen: { state: "error", promptId: null, media: [], msg: _friendlyGenError(e.message) } });
      return;
    }
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
    try {
      await _flushSaveForGenerate();
    } catch (e) {
      set({ gen: { state: "error", promptId: null, media: [], msg: _friendlyGenError(e.message) } });
      return;
    }
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
    try {
      await _flushSaveForGenerate();
    } catch (e) {
      set({ gen: { state: "error", promptId: null, media: [], msg: _friendlyGenError(e.message) } });
      return;
    }
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

  // The clock is wrapped around the PUBLIC entry points rather than set inside them: every
  // one of these has several early-return error paths, and a `finally` here cannot miss one.
  // `await` (not `return fn()`) is what makes the finally wait for the whole montage.
  async function _clockedGenerate(onlyScene) {
    _genClockStart(onlyScene ? "scene" : "all");
    try { return await generate(onlyScene); } finally { _genClockStop(); }
  }
  async function _clockedMontage() {
    _genClockStart("all");
    try { return await generateMontage(); } finally { _genClockStop(); }
  }
  async function _clockedSelected() {
    _genClockStart("selected");
    try { return await generateSelected(); } finally { _genClockStop(); }
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

  // Source in-point for preview/export: chain segment offset + per-scene trim.
  function clipSourceInSec(sc, render) {
    if (!sc) return 0;
    const trim = sc.source_in || 0;
    if (render?.media) return trim + (render.inSec || 0);
    return trim;
  }

  // Ordered list of rendered clips in timeline order: {media, inSec, durationSec} for
  // each non-excluded scene that has a render. This is what plays/exports — so splits
  // and deletes are honoured (deleted scenes simply aren't here).
  function _renderClips() {
    const p = state.project;
    const out = [];
    for (const sc of orderedTimelineScenes()) {
      if (sc.excluded && !sc.removed_from_plan) continue; // removed-from-plan clips stay in the cut
      const r = state.sceneRenders[sc.id];
      const fps = (sc.fps_mode !== "project" && sc.fps != null ? sc.fps : p.frame_rate) || 25;
      const tFrames = sc.transition_frames || 0;
      const base = {
        inSec: clipSourceInSec(sc, r),
        durationSec: sc.source_dur != null ? sc.source_dur : sceneDurationSec(sc),
        fx: sc.effects || {}, fps,
        w: (sc.width != null ? sc.width : p.width) || null,
        h: (sc.height != null ? sc.height : p.height) || null,
        transition: sc.video_transition || "",
        tdur: tFrames > 0 ? tFrames / fps : 0,
        sceneId: sc.id,
        volume: sc.audio_separated ? 0 : (sc.audio_volume != null ? sc.audio_volume : 1),
        gap_after: Math.max(0, +(sc.gap_after_sec || 0)),
      };
      if (r && r.media) {
        out.push({ media: r.media, ...base });
      } else if (isVideoClip(sc) && sc.source?.media_ref) {
        out.push({ bin_media_ref: sc.source.media_ref, ...base });
      }
    }
    return out;
  }

  function _saveableSelectedSceneIds() {
    return _orderedSelectedSceneIds().filter((id) => clipSaveableToMediaBin(id));
  }

  function _renderClipsForSceneIds(sceneIds) {
    const set = new Set(sceneIds);
    return _renderClips().filter((c) => set.has(c.sceneId));
  }

  function _plainExportClipsFromRenderClips(rc) {
    return rc.map((c) => ({
      filename: c.media?.filename, subfolder: c.media?.subfolder || "", type: c.media?.type || "output",
      bin_media_ref: c.bin_media_ref || "",
      in: +c.inSec.toFixed(3), dur: +c.durationSec.toFixed(3),
    }));
  }

  function _combinedSelectionFilename(clipCount) {
    const proj = (state.project.name || "montage").replace(/[^\w.-]+/g, "_");
    return `${proj}_${clipCount}clips_${_stamp()}.mp4`;
  }

  // Trim one clip, or hard-cut concat multiple selected clips (no effects/transitions).
  async function _stitchSelectedClips(saveableIds) {
    const rc = _renderClipsForSceneIds(saveableIds);
    if (!rc.length) return null;
    if (rc.length === 1) {
      const spec = _clipExportSpecForScene(rc[0].sceneId);
      if (!spec) return null;
      const out = await API.exportClip(state.project.id, spec.clip);
      return { media: out.media, clips: 1, exportName: spec.name };
    }
    const queued = await API.exportClipsCombined(state.project.id, _plainExportClipsFromRenderClips(rc));
    const r = queued?.job_id
      ? await _pollExportClipsJob(queued.job_id, rc.length)
      : queued;
    return { ...r, exportName: _combinedSelectionFilename(rc.length) };
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

  function _orderedSelectedSceneIds() {
    const raw = state.selectedSceneIds?.length
      ? state.selectedSceneIds
      : (state.selectedSceneId ? [state.selectedSceneId] : []);
    if (!raw.length) return [];
    const order = _sceneOrder();
    const set = new Set(raw);
    return order.filter((id) => set.has(id));
  }

  // Build export/import clip spec for one scene (render trim or media-bin video).
  function _clipExportSpecForScene(sceneId) {
    if (!state.project) return null;
    const sc = scene(sceneId);
    if (!sc) return null;
    const r = state.sceneRenders[sceneId];
    const idx = state.project.scenes.findIndex((s) => s.id === sceneId);
    const proj = (state.project.name || "montage").replace(/[^\w.-]+/g, "_");
    if (isVideoClip(sc) && sc.source?.media_ref && !r?.media) {
      return {
        sceneId,
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
      sceneId,
      clip: {
        filename: r.media.filename,
        subfolder: r.media.subfolder || "",
        type: r.media.type || "output",
        in: clipSourceInSec(sc, r),
        dur: sc.source_dur != null ? sc.source_dur : sceneDurationSec(sc),
      },
      name: `${proj}_scene${idx >= 0 ? idx + 1 : "x"}_${_stamp()}.mp4`,
    };
  }

  function _selectedClipExportSpecs() {
    return _orderedSelectedSceneIds().map((id) => _clipExportSpecForScene(id)).filter(Boolean);
  }

  // Build export/import clip spec for the focused selected scene (render trim or media-bin video).
  function _selectedClipExportSpec() {
    const id = _selectedClipFocusId();
    return id ? _clipExportSpecForScene(id) : null;
  }

  function clipSaveableToMediaBin(sceneId) {
    if (!sceneId || !state.project) return false;
    const sc = scene(sceneId);
    if (!sc) return false;
    if (isVideoClip(sc) && sc.source?.media_ref) return true;
    return !!(state.sceneRenders[sceneId]?.media);
  }

  // Export selected clip(s): one file; multi-select is hard-cut concat with trim only.
  async function exportSelected() {
    if (!state.project) return;
    if (!_hasTimelineClipSelection()) { alert("Select a clip first."); return; }
    const allIds = _orderedSelectedSceneIds();
    const saveableIds = _saveableSelectedSceneIds();
    if (!saveableIds.length) {
      alert("Generate the clip first, or select a video clip with media.");
      return;
    }
    const skipped = allIds.length - saveableIds.length;
    if (skipped > 0 && !confirm(
      `${skipped} selected clip(s) have no render yet and will be skipped. Export ${saveableIds.length} clip(s)?`
    )) return;
    if (saveableIds.length > 1 && !confirm(
      `Combine ${saveableIds.length} selected clips into one file and export?\n\nTimeline trim is applied per clip. No effects, transitions, or extra audio — use Render Final for that.`
    )) return;
    const showProgress = saveableIds.length > 1;
    if (showProgress) {
      set({ gen: { state: "running", promptId: null, media: [], msg: `Exporting ${saveableIds.length} clips…` } });
    }
    try {
      const r = await _stitchSelectedClips(saveableIds);
      if (!r?.media) throw new Error("Export produced no media.");
      await _saveBlobAs(API.resultUrl(state.project.id, r.media), r.exportName);
      const notice = saveableIds.length === 1
        ? "Clip exported"
        : `Exported ${saveableIds.length} clips as one file`;
      set({ gen: { state: "idle", promptId: null, media: [], msg: "" }, notice });
    } catch (e) {
      set({ gen: { state: "error", promptId: null, media: [], msg: "Export failed: " + _friendlyGenError(e.message) } });
    }
  }

  async function saveSelectedToMediaBin() {
    if (!state.project) return;
    if (!_hasTimelineClipSelection()) { alert("Select a clip first."); return; }
    const allIds = _orderedSelectedSceneIds();
    const saveableIds = _saveableSelectedSceneIds();
    if (!saveableIds.length) {
      alert("Generate the clip first, or select a video clip with media.");
      return;
    }
    const skipped = allIds.length - saveableIds.length;
    if (skipped > 0 && !confirm(
      `${skipped} selected clip(s) have no render yet and will be skipped. Save ${saveableIds.length} clip(s) to Media bin?`
    )) return;
    if (saveableIds.length > 1 && !confirm(
      `Combine ${saveableIds.length} selected clips into one file and save to Media bin?\n\nTimeline trim is applied per clip. No effects, transitions, or extra audio — use Render Final for that.`
    )) return;
    const showProgress = saveableIds.length > 1;
    if (showProgress) {
      set({ gen: { state: "running", promptId: null, media: [], msg: `Saving ${saveableIds.length} clips to Media bin…` } });
    }
    try {
      let clip;
      let name;
      if (saveableIds.length === 1) {
        const spec = _clipExportSpecForScene(saveableIds[0]);
        if (!spec) throw new Error("Nothing to save.");
        clip = spec.clip;
        name = spec.name;
      } else {
        const r = await _stitchSelectedClips(saveableIds);
        if (!r?.media) throw new Error("Stitch produced no media.");
        clip = {
          filename: r.media.filename,
          subfolder: r.media.subfolder || "",
          type: r.media.type || "temp",
        };
        name = r.exportName;
      }
      const res = await API.importClipToMediaBin(clip, name);
      await loadMedia();
      const entry = res?.media || res;
      const notice = saveableIds.length === 1
        ? `Saved to Media bin: ${entry?.name || name}`
        : `Saved ${saveableIds.length} clips to Media bin as one file: ${entry?.name || name}`;
      set({ gen: { state: "idle", promptId: null, media: [], msg: "" }, notice });
    } catch (e) {
      set({ gen: { state: "error", promptId: null, media: [], msg: "Save to media bin failed: " + _friendlyGenError(e.message) } });
    }
  }

  // Stitch the kept clips (in/out per clip, hard cut, video + audio) into one final file.
  async function renderFinal() {
    if (!state.project) return;
    const rc = _renderClips();
    const totalSec = previewTotalSec();
    const graphicsOnly = !rc.length && hasOverlayOrAudioContent() && totalSec > 0;
    if (!rc.length && !graphicsOnly) {
      alert("Nothing to render yet — generate clips or add overlay/audio tracks first.");
      return;
    }
    const confirmMsg = graphicsOnly
      ? `Render the overlay/audio timeline (${totalSec.toFixed(1)}s) to a final video?\n\nUses a blank canvas at project resolution (${state.project.width || 768}×${state.project.height || 512}).`
      : `Stitch ${rc.length} clip(s) into one final video?\n\nThis replaces per-scene renders with a single stitched file on the timeline.`;
    if (!confirm(confirmMsg)) return;
    const clips = rc.map((c) => ({
      filename: c.media?.filename, subfolder: c.media?.subfolder || "", type: c.media?.type || "output",
      bin_media_ref: c.bin_media_ref || "",
      in: +c.inSec.toFixed(3), dur: +c.durationSec.toFixed(3),
      fx: c.fx, fps: c.fps, w: c.w, h: c.h, transition: c.transition, tdur: +(c.tdur || 0).toFixed(3),
      volume: c.volume, scene_id: c.sceneId,
      gap_after: +(c.gap_after || 0).toFixed(3),
    }));
    await flushSave();  // audio tracks / keep-original live on the project — render reads it from disk
    set({
      gen: {
        state: "running",
        promptId: null,
        media: [],
        msg: graphicsOnly ? `Rendering ${totalSec.toFixed(1)}s overlay/audio…` : `Stitching ${clips.length} clip(s)…`,
      },
    });
    try {
      const queued = await API.renderFinal(state.project.id, clips);
      const r = queued?.job_id
        ? await _pollRenderJob(queued.job_id, clips.length || 1)
        : queued;
      if (!graphicsOnly) {
        // Play the stitched file across the whole timeline (single source, in-point 0).
        const order = state.project.scenes.filter((s) => !s.excluded);
        state.sceneRenders = {};
        state.sceneGhosts = [];
        _syncEditorStateToProject();
        let acc = 0;
        for (const sc of order) {
          state.sceneRenders[sc.id] = { media: r.media, inSec: acc };
          const dur = sc.source_dur != null ? sc.source_dur : sceneDurationSec(sc);
          acc += dur + Math.max(0, +(sc.gap_after_sec || 0));
        }
      }
      set({
        gen: {
          state: "done",
          promptId: null,
          media: [r.media],
          msg: graphicsOnly
            ? "Final overlay/audio video rendered — saving…"
            : `Final video rendered from ${r.clips} clip(s) — saving…`,
        },
      });
      const name = (state.project.name || "montage").replace(/[^\w.-]+/g, "_") + `_final_${_stamp()}.mp4`;
      await _saveBlobAs(API.resultUrl(state.project.id, r.media), name);
    } catch (e) {
      set({ gen: { state: "error", promptId: null, media: [], msg: "Render failed: " + _friendlyGenError(e.message) } });
    }
  }

  // ── pluggable models / exposed controls ──────────────────────────────────────
  async function loadModels() {
    const p = (async () => {
      try { state.models = await API.getModels(state.project?.id); } catch (_) { state.models = { slots: [] }; }
    })();
    _modelsLoad = p;
    await p;
    if (_modelsLoad === p) _modelsLoad = null;
    if (state.project) {
      state.project.models = JSON.parse(JSON.stringify(state.models || { slots: [] }));
    }
    if (window.PipelineCaps && !window.PipelineCaps.usesFunpackStudio(state)) {
      _resetSessionPending = false;
      state.resetSessionArmed = false;
    }
    notify();
    loadImageTargets();
  }
  async function loadImageTargets() {
    try { state.imageTargets = (await API.imageTargets(state.project?.id)).targets || []; } catch (_) { state.imageTargets = []; }
    notify();
  }

  // Persisting a node input hits the server; typing a weight used to fire one save PER
  // KEYSTROKE and notify twice (before + after the await), which rebuilt the inspector and
  // yanked the field. Now: update in place + notify once (cheap, and the field is data-k
  // protected from rebuild), and DEBOUNCE the network save. flushModelsSave() runs on field
  // blur and before generate (via flushSave / _flushSaveForGenerate).
  function _scheduleModelsSave() {
    _modelsSaveDirty = true;
    clearTimeout(_modelsSaveTimer);
    _modelsSaveTimer = setTimeout(() => { _modelsSaveTimer = null; flushModelsSave(); }, 500);
  }
  async function flushModelsSave() {
    if (!_modelsSaveTimer && !_modelsSaveDirty) return;
    clearTimeout(_modelsSaveTimer); _modelsSaveTimer = null; _modelsSaveDirty = false;
    try {
      const saved = await API.saveModels(state.project?.id, state.models);
      // Same guard commit() has: an edit made during this write is newer than the echo,
      // so adopting the echo would roll it back. Keep local; the queued save wins.
      if (!_modelsSaveDirty) state.models = saved;
      notify();
    } catch (e) { console.error("saveModels failed", e); }
  }

  // Edit a configured node input from the main editor (an "exposed" control) and persist.
  function setModelInput(slotId, name, value) {
    const slot = (state.models.slots || []).find((s) => s.id === slotId);
    if (!slot) return;
    slot.inputs = slot.inputs || {}; slot.inputs[name] = value;
    notify();
    _scheduleModelsSave();
  }

  // Edit a slot's bypass flag from the main editor (an "exposed" control) and persist.
  // Top-level on the slot, not slot.inputs, so it never gets sent to ComfyUI as a fake widget.
  function setModelBypass(slotId, value) {
    const slot = (state.models.slots || []).find((s) => s.id === slotId);
    if (!slot) return;
    slot.bypassed = !!value;
    notify();
    _scheduleModelsSave();
  }

  // Set a linked control's shared value (writes through to all member inputs) and persist.
  function setModelLink(linkId, value) {
    const link = (state.models.links || []).find((l) => l.id === linkId);
    if (!link) return;
    link.value = value;
    (link.members || []).forEach((m) => {
      const s = (state.models.slots || []).find((x) => x.id === m.slotId);
      if (s) { s.inputs = s.inputs || {}; s.inputs[m.input] = value; }
    });
    notify();
    _scheduleModelsSave();
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
  async function renameMedia(id, name) {
    const label = String(name || "").trim();
    if (!label) return;
    try {
      const res = await API.renameMedia(id, label);
      const entry = res?.media;
      if (entry) {
        state.mediaBin = (state.mediaBin || []).map((m) => (m.id === id ? { ...m, ...entry } : m));
        notify();
      } else {
        await loadMedia();
      }
    } catch (e) {
      alert("Rename failed: " + (e.message || e));
    }
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
  async function loadShortcuts() { try { const r = await API.shortcuts(); state.shortcuts = r.shortcuts || []; state.shortcutCategories = r.categories || []; } catch (_) { state.shortcuts = []; state.shortcutCategories = []; } notify(); }
  async function saveShortcut(item) { try { const r = await API.saveShortcut(item); state.shortcuts = r.shortcuts || state.shortcuts; if (r.categories) state.shortcutCategories = r.categories; notify(); } catch (e) { alert("Save failed: " + e.message); } }
  async function deleteShortcut(name) { try { const r = await API.deleteShortcut(name); state.shortcuts = r.shortcuts || []; if (r.categories) state.shortcutCategories = r.categories; notify(); } catch (e) { console.error(e); } }
  async function addCategory(category, subCategory) {
    try {
      const r = await API.saveCategory({ category, sub_category: subCategory || "" });
      if (r.categories) state.shortcutCategories = r.categories;
      notify();
      return true;
    } catch (e) { alert("Add category failed: " + e.message); return false; }
  }
  async function saveTransition(item) {
    try {
      state.transitions = (await API.saveTransition(item)).transitions || state.transitions;
      notify();
    } catch (e) { alert("Save failed: " + e.message); }
  }
  async function deleteTransition(name) { try { state.transitions = (await API.deleteTransition(name)).transitions || []; notify(); } catch (e) { console.error(e); } }
  async function importShortcuts(file, mode) {
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      const r = await API.importShortcuts(data, mode);
      state.shortcuts = r.shortcuts || state.shortcuts;
      if (r.categories) state.shortcutCategories = r.categories;
      notify();
      return r.imported;
    } catch (e) { alert("Import failed: " + e.message); return null; }
  }
  async function importTransitions(file, mode) {
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      const r = await API.importTransitions(data, mode);
      state.transitions = r.transitions || state.transitions; notify();
      return r.imported;
    } catch (e) { alert("Import failed: " + e.message); return null; }
  }
  async function clearShortcuts() {
    try {
      const r = await API.clearShortcuts();
      state.shortcuts = r.shortcuts || [];
      state.shortcutCategories = r.categories || [];
      notify();
      return true;
    } catch (e) { alert("Delete-all failed: " + e.message); return false; }
  }
  async function clearTransitions() {
    try {
      const r = await API.clearTransitions();
      state.transitions = r.transitions || [];
      notify();
      return true;
    } catch (e) { alert("Delete-all failed: " + e.message); return false; }
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
      zoom_in: () => ({
        effects: {
          ...(sc.effects || {}),
          zoom: "in",
          zoom_ratio: sc.effects?.zoom_ratio ?? 0.15,
          zoom_frames: sc.effects?.zoom_frames ?? 25,
          zoom_start_frame: sc.effects?.zoom_start_frame ?? 0,
        },
      }),
      zoom_out: () => ({
        effects: {
          ...(sc.effects || {}),
          zoom: "out",
          zoom_ratio: sc.effects?.zoom_ratio ?? 0.15,
          zoom_frames: sc.effects?.zoom_frames ?? 25,
          zoom_start_frame: sc.effects?.zoom_start_frame ?? 0,
        },
      }),
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
    if (window.PipelineCaps && !window.PipelineCaps.usesChainSampler(state)) return false;
    const active = state.project.scenes.filter((sc) => !sc.excluded);
    const idx = active.findIndex((sc) => sc.id === s.id);
    if (idx <= 0) return false;
    const t = (s.source && s.source.type) || "empty";
    if (t === "empty") return false;
    if (isGenSubclip(s)) return true;
    if (t === "carry" || t === "mixed") return true;
    if ((t === "image" || t === "generated_frame" || t === "anchor_guide") && _anchorAvailable(s)) return false;
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
        patchScene(sceneId, {
          source: { ...(s.source || {}), type: "video", media_ref: mediaId },
          source_dur: asset.duration_sec || s.source_dur || null,
        });
        delete state.sceneRenders[sceneId];
        notify(); scheduleSave();
      } else if (confirm("Replace this scene with the imported video clip? Use Convert to video in the inspector to keep a backup of scene settings.")) {
        _lockSceneAsVideo(sceneId, mediaId, { archive: true });
      }
      if (!asset.duration_sec) _probeVideoClipDuration(sceneId, mediaId);
      return;
    }
    // Drag-drop is an explicit new anchor — use image i2v (not mixed/carry guides), but
    // keep an existing anchor_guide scene in guide mode (its image is still the anchor).
    const keepType = s.source?.type === "anchor_guide" ? "anchor_guide" : "image";
    const patch = { source: { ...(s.source || {}), type: keepType, media_ref: mediaId } };
    if ((s.source?.media_ref || null) !== mediaId) patch.guides = [];
    patchScene(sceneId, patch);
  }

  // ── boot ─────────────────────────────────────────────────────────────────────
  async function init() {
    if (window.__FUNPACK_TOUR__ && window.TourSandbox) {
      window.TourSandbox.patchApi?.();
      window.TourSandbox.patchStore?.(window.Store);
      await window.TourSandbox.bootstrap(state, notify);
      notify();
      document.body.classList.add("tour-mode");
      setTimeout(() => window.TourGuide?.start?.(), 180);
      return;
    }
    try { state.health = await API.health(); } catch (_) { state.health = { ok: false }; }
    if (state.gen?.state === "error" && /Setup incomplete|Missing workflow template/.test(state.gen.msg || "")) {
      state.gen = { state: "idle", promptId: null, media: [], msg: "" };
    }
    try { const t = await API.transitions(); state.transitions = t.transitions || []; } catch (_) { state.transitions = []; }
    await loadNleLibrary();
    await loadShortcuts();
    await loadMedia();
    try { state.ratingLabels = (await API.ratingLabels()).labels || []; } catch (_) { state.ratingLabels = []; }
    await loadModels();
    window.addEventListener("funpack-models-changed", loadModels);
    await refreshProjectList();
    // Re-attach to a generation already running in ComfyUI (UI was reloaded mid-run).
    // Load its project first so Generate is blocked, Interrupt is shown, and the result
    // lands on the right scenes — then skip the welcome page, there's work in flight.
    let resumed = false;
    try {
      const act = await API.active();
      if (act && act.running && act.prompt_id) {
        if (act.pid && (state.projects || []).some((p) => p.id === act.pid)
            && (!state.project || state.project.id !== act.pid)) {
          await loadProject(act.pid);
        }
        notify();
        resumeRunningGeneration(act);  // fire-and-forget poll loop
        resumed = true;
      }
    } catch (_) { /* queue unreachable — boot normally */ }
    notify();
    // One front door: the full-screen welcome, unless a running generation was resumed.
    if (resumed) return;
    window.Onboarding?.maybeOpen();
  }

  // Backstop for a suspended save the user walks away from. Unconditional is safe:
  // nobody is mid-edit in a tab that just went to the background.
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") { try { flushSave(); } catch (_) {} }
  });

  window.Store = {
    get, set, subscribe, notify, init,
    scheduleSaveFromHistory, notifyHistoryState,
    refreshProjectList, loadProject, newProject, deleteProject, downloadProject, importProject,
    patchProject, patchProjectQuiet, patchScene, patchSceneQuiet, flushSave, suspendSave, resumeSave, selectScene, addScene, removeScene, removeSelectedScenes, removeFromPlan, restoreToPlan, dismissGhost, moveScene, moveSceneTo, moveTimelineClip, scene,
    addVideoClip, addImageClip, convertToVideo, convertToScene, isVideoClip, isGenerativeScene,
    genUnitId, isGenSubclip, genUnitRoot, genUnitSceneIds,
    renderPromptForScene, renderPromptMismatch, renderAnchorMismatch, renderMediaLabel, renderIsStale,
    buildPreviewSegments, previewTotalSec, audioTrackEndSec, hasOverlayOrAudioContent,
    segmentDurationSec, sceneDurationSec, clipSourceInSec,
    addAudioTrack, updateAudioTrack, removeAudioTrack, separateSceneAudio, separatedTrackForScene,
    selectAudioTrack, trimAudioTrackLeft, slipAudioTrack, resizeAudioTrack, splitAudioTrack,
    sceneHasEmbeddedAudio, separatedTrackAudioUrl,
    separatedTrackMedia, separatedTrackInSec, separatedTrackDurSec, audioTrackInSec, audioTrackDurSec,
    audioTrackMaxDurSec,
    overlayTrack, selectOverlay, ensureOverlayLanes, sortedOverlayTracks, overlayLaneById, overlayLaneIndex,
    projectCanvasSize, normalizeOverlayTrack,
    addOverlayLane, removeOverlayLane, assignMediaToOverlayLane,
    bringOverlayToFront, sendOverlayToBack, bringOverlayForward, sendOverlayBackward,
    addImageOverlay, addTextOverlay, updateOverlayTrack, removeOverlayTrack, removeSelectedOverlay,
    isOverlayAudioTrack, isSeparatedAudioTrack,
    resizeScene, setSceneGapAfter, splitScene, autoMontage, hasPlayableRender, snapFrames, snapFramesFloor, snapFramesCeil, sceneEffFrames, sceneEffFps, scenesOverridingProjectFrames, useProjectFramesEverywhere, setSourceTrim, trimSceneLeft, slipScene,
    applyEnginePreset, ENGINE_PRESETS, undo, redo,
    refreshPreview, syncFromPreview, applyGlobalPromptQuiet, scheduleGlobalPromptApply, globalPromptApplyPending, buildGlobalPromptFromTimeline, syncGlobalPromptFromTimeline, generate: _clockedGenerate, generateMontage: _clockedMontage, generateSelected: _clockedSelected, genElapsed, selectedSceneCount, renderFinal, exportSelected, saveSelectedToMediaBin, clipSaveableToMediaBin, interrupt, loadModels, loadImageTargets, setModelInput, setModelBypass, setModelLink, clearNotice,
    projectVariables, setProjectVariables, promptTemplates, savePromptTemplate, deletePromptTemplate, applyPromptTemplate,
    setConditioningSlot, setSamplerSlot, setSamplerInput, setSamplerInputNow, unsetSamplerInput, setStudioInput, setStudioInputNow,
    loadMedia, uploadMedia, deleteMedia, deleteMediaMany, renameMedia, previewMedia, clearMediaPreview, assignMediaToScene, exportMediaAsset,
    loadShortcuts, saveShortcut, deleteShortcut, importShortcuts, clearShortcuts, addCategory,
    getEditorSettings, getEditorSetting, setEditorSetting,
    loadTransitions, saveTransition, deleteTransition, importTransitions, clearTransitions,
    loadNleLibrary, applyNleEffect, applyNleVideoTransition,
    applySplitMarkerToSelection, applyTransitionToSelection, insertShortcutIntoSelection,
    setSceneRating: (id, v) => {
      patchScene(id, { rating: v || "" });
      if (window.Timeline?.syncClipRatings) window.Timeline.syncClipRatings(get());
    },
    resetStudioSession,
  };
})();
