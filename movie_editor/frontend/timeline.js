// Bottom zone: a real NLE timeline. Clips are laid out on a time axis (width =
// duration × zoom), with a HH:MM:SS ruler, a playhead, drag-to-trim edges (which
// recompute frame counts from duration × fps), split, and per-seam crossfades.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const body = document.getElementById("timeline-body");
  const meta = document.getElementById("timeline-meta");

  const SRC_ICON = { empty: "▦", image: "◐", generated_frame: "⛶", carry: "⇥", mixed: "◑", video: "▶", v2v: "⟳", anchor_guide: "◓" };

  function genUnitRootScene(scene, p) {
    const uid = scene.gen_unit_id || scene.id;
    return (p.scenes || []).find((s) => (s.gen_unit_id || s.id) === uid && !(s.cut_offset_frames > 0))
      || scene;
  }

  function sceneSourceForClip(scene, p) {
    const root = genUnitRootScene(scene, p);
    return root.source || scene.source || {};
  }

  function anchorMediaRef(scene, p) {
    const src = sceneSourceForClip(scene, p);
    const t = src.type;
    if (t === "image" || t === "mixed" || t === "generated_frame" || t === "video" || t === "v2v" || t === "anchor_guide") return src.media_ref || null;
    return null;
  }

  function appendSrcBadge(head, srcType) {
    if (srcType === "mixed") {
      const stack = el("span", "clip-src-stack");
      stack.title = "Img2Video anchor + prior i2v guides active";
      stack.append(el("span", "clip-anchor-mark", "◐"));
      stack.append(el("span", "clip-guide-mark", "⇥"));
      head.append(stack);
      head.append(el("span", "clip-src-label", "mixed"));
      return;
    }
    if (srcType === "carry") {
      const g = el("span", "clip-src carry-only");
      g.title = "Continues from prior scene — i2v guides only, no new anchor";
      g.textContent = "⇥";
      head.append(g);
      return;
    }
    if (srcType === "anchor_guide") {
      const stack = el("span", "clip-src-stack");
      stack.title = "Anchor as guide — image steers via a frame-0 guide; latent empty (i2v bypassed)";
      stack.append(el("span", "clip-anchor-mark", "◓"));
      stack.append(el("span", "clip-guide-mark", "⇥"));
      head.append(stack);
      head.append(el("span", "clip-src-label", "guide"));
      return;
    }
    head.append(el("span", "clip-src", SRC_ICON[srcType] || "▦"));
  }

  // `reorder` (optional): { canLeft, canRight, onLeft, onRight } adds inline move
  // controls so clips can be reordered directly on the timeline (not just the Edit menu).
  function appendClipHeadBar(head, durSec, fps, onRemove, removeTitle, reorder) {
    const bar = el("div", "clip-head-bar");
    bar.append(el("span", "clip-dur", timecode(durSec, fps)));
    if (reorder) {
      const mk = (glyph, can, fn, title) => {
        const b = el("button", "clip-move btn ghost tiny" + (can ? "" : " disabled"), glyph);
        b.type = "button"; b.title = title; b.disabled = !can;
        if (can) b.onclick = (e) => { e.stopPropagation(); fn(); };
        return b;
      };
      bar.append(mk("◀", reorder.canLeft, reorder.onLeft, "Move clip left"));
      bar.append(mk("▶", reorder.canRight, reorder.onRight, "Move clip right"));
    }
    const rm = el("button", "clip-rm btn ghost tiny danger", "Remove");
    rm.type = "button";
    rm.title = removeTitle || "Remove clip (Delete / Backspace)";
    rm.onclick = (e) => { e.stopPropagation(); onRemove(); };
    bar.append(rm);
    head.append(bar);
  }

  const ZOOM_KEY = "funpack_tl_zoom";
  const GUTTER_W = 56;  // sticky lane labels (Video / Audio) — shared time origin in tl-content
  const VIDEO_LANE_H = 96;  // matches .tl-track2 padding + clip band
  const AUDIO_LANE_H = 44;
  const OVERLAY_LANE_H = 40;

  function isAudioAsset(m) {
    return !!m && (m.kind === "audio" || /\.(mp3|wav|m4a|aac|ogg|flac|opus|weba)$/i.test(m.name || ""));
  }

  function _audioTrackStartSec() {
    const ph = window.Player?.getPlayhead?.();
    return ph != null && ph >= 0 ? ph : 0;
  }

  function _attachWaveform(canvas, key, url, opts) {
    const WF = window.FunPackWaveform;
    if (!WF || !canvas) return;
    if (url) WF.attach(canvas, key, url, opts || {});
    else WF.paintPlaceholder(canvas, opts || {});
  }
  let pxPerSec = parseFloat(localStorage.getItem(ZOOM_KEY)) || 80;  // zoom
  let scrollLeft = 0;
  let _scrollFromCode = false;      // ignore scroll events we triggered (playback follow)
  let _playbackScrollFollow = true; // auto-pan during playback until user scrolls manually
  let _userScrollTimer = null;
  let _renderDeferred = false;

  function _markManualTimelineScroll() {
    _playbackScrollFollow = false;
    clearTimeout(_userScrollTimer);
    _userScrollTimer = setTimeout(() => {
      _userScrollTimer = null;
      if (_renderDeferred) render(S.get());
    }, 250);
  }

  function _isUserScrollingTimeline() {
    return _userScrollTimer != null;
  }

  function _setTimelineScrollLeft(px) {
    if (!tlScrollEl) return;
    _scrollFromCode = true;
    tlScrollEl.scrollLeft = px;
    scrollLeft = px;
    requestAnimationFrame(() => { _scrollFromCode = false; });
  }

  function _bindTimelineScroll(scroll) {
    scroll.addEventListener("scroll", () => {
      scrollLeft = scroll.scrollLeft;
      if (_scrollFromCode) return;
      _markManualTimelineScroll();
    });
    scroll.addEventListener("wheel", () => { _markManualTimelineScroll(); }, { passive: true });
    scroll.addEventListener("mousedown", (e) => {
      const r = scroll.getBoundingClientRect();
      const barH = Math.max(0, scroll.offsetHeight - scroll.clientHeight);
      const barW = Math.max(0, scroll.offsetWidth - scroll.clientWidth);
      const onHBar = barH > 0 && e.clientY >= r.bottom - barH;
      const onVBar = barW > 0 && e.clientX >= r.right - barW;
      if (onHBar || onVBar) _markManualTimelineScroll();
    });
  }

  // Refs updated on each render so the Player playhead listener can update
  // just the needle and timecode without a full re-render.
  let tlPhEl = null;
  let tlTcEl = null;
  let tlScrollEl = null;
  let tlTotalSec = 0;
  let tlFps = 25;

  // ── helpers ──────────────────────────────────────────────────────────────────
  const sFps = (sc, p) => (S.sceneEffFps ? S.sceneEffFps(sc, p) : p.frame_rate) || 25;
  const sFrames = (sc, p) => (S.sceneEffFrames ? S.sceneEffFrames(sc, p) : p.num_frames_per_scene) || 1;
  const sDur = (sc, p) => (S.sceneDurationSec ? S.sceneDurationSec(sc) : (sFrames(sc, p) / sFps(sc, p)));

  function snapPx(x, anchors, threshold) {
    threshold = threshold || 10;
    let best = x;
    for (const a of anchors) {
      if (Math.abs(x - a) < threshold) best = a;
    }
    return best;
  }

  function seamAnchorsPx(lay) {
    return lay.map(({ o }) => o * pxPerSec);
  }

  function timelineSnapAnchorsPx(st, p) {
    const segs = S.buildPreviewSegments ? S.buildPreviewSegments() : [];
    const anchors = segs.map((s) => s.offsetSec * pxPerSec);
    const endSec = segs.length ? segs[segs.length - 1].offsetSec + segs[segs.length - 1].durationSec : 0;
    anchors.push(endSec * pxPerSec);
    anchors.push((window.Player?.getPlayhead() ?? 0) * pxPerSec);
    for (let sec = 0; sec <= Math.ceil(endSec); sec++) anchors.push(sec * pxPerSec);
    return anchors;
  }

  function _syncSceneAudioClipVisual(sceneId, leftPx, widthPx) {
    try {
      const aud = document.querySelector(`.tl-aud-clip.scene-aud[data-scene-id="${sceneId}"]`);
      if (!aud) return;
      if (leftPx != null) aud.style.left = leftPx + "px";
      if (widthPx != null) {
        const w = Math.max(widthPx, 8);
        aud.style.width = w + "px";
        aud.style.maxWidth = w + "px";
      }
    } catch (_) {}
  }

  function coalescedDrag(startEvt, onMove, onUp) {
    window.EditorHistory?.beginCoalesce(S);
    onDrag(startEvt, onMove, (dx, upEvt) => {
      try { if (onUp) onUp(dx, upEvt); } finally { window.EditorHistory?.endCoalesce(S); }
    });
  }

  // Filmstrip probes run through a small shared queue: painting a big (montage) timeline
  // used to start one preload=auto <video> per clip AT ONCE — each of them a server-side
  // ffmpeg segment encode — saturating both the browser's ~6 per-origin connections and
  // the server, and starving the player pool (clips stuck "loading" on tunneled
  // instances). Two at a time keeps thumbnails flowing without wedging playback.
  const _fsQueue = [];
  let _fsActive = 0;
  const FS_CONCURRENCY = 2;
  function _fsPump() {
    while (_fsActive < FS_CONCURRENCY && _fsQueue.length) {
      const job = _fsQueue.shift();
      _fsActive++;
      job().catch(() => {}).then(() => { _fsActive--; _fsPump(); });
    }
  }
  function _fsEnqueue(job) { _fsQueue.push(job); _fsPump(); }

  function appendFilmstrip(clip, st, scene, widthPx) {
    if (S.isVideoClip && S.isVideoClip(scene)) return;
    if (!hasRender(st, scene.id)) return;
    if (clip.querySelector(".clip-filmstrip")) return;
    const r = st.sceneRenders[scene.id];
    if (!r?.media || widthPx < 40) return;
    const strip = el("div", "clip-filmstrip");
    clip.append(strip);
    const n = Math.min(10, Math.max(3, Math.floor(widthPx / 28)));
    const API = window.MovieEditorAPI;
    const durSec = sDur(scene, st.project);
    // Route through the same trimmed, faststart preview segment the player uses instead of
    // the raw chain output: that file is moov-at-end (deep seeks stall/black in the browser)
    // and shared across every scene in the run, so a filmstrip per clip meant N full-file
    // downloads with preload=auto fighting the player pool over Chrome's 6-per-origin cap.
    const segUrl = API.previewSegmentUrl && st.project.id
      ? API.previewSegmentUrl(st.project.id, scene.id, { media: r.media, renderIn: r.inSec || 0, dur: durSec })
      : null;
    const url = segUrl || API.resultUrl(st.project.id, r.media);
    _fsEnqueue(() => new Promise((finish) => {
      if (!strip.isConnected) { finish(); return; }  // re-rendered before its turn came up
      const vid = document.createElement("video");
      vid.muted = true; vid.preload = "auto"; vid.src = url;
      // Free the media connection once the strip is captured (or abandoned) — a detached
      // <video> keeps its socket until GC, and those add up against Chrome's 6-per-origin
      // cap. Also settles the queue job; the guard timer keeps a wedged load (server busy,
      // dead tunnel) from clogging a queue slot forever.
      let settled = false;
      const release = () => {
        if (settled) return;
        settled = true;
        clearTimeout(guard);
        try { vid.removeAttribute("src"); vid.load(); } catch (_) {}
        finish();
      };
      const guard = setTimeout(release, 20000);
      const captureAt = (time) => new Promise((resolve) => {
        if (!strip.isConnected) { resolve(); return; }
        const cell = el("div", "clip-fs-cell");
        const c = document.createElement("canvas");
        c.width = 48; c.height = 27;
        const ctx = c.getContext("2d");
        const thumb = el("img", "clip-fs-thumb");
        thumb.alt = "";
        const done = () => {
          if (!strip.isConnected) { resolve(); return; }
          try { ctx.drawImage(vid, 0, 0, 48, 27); thumb.src = c.toDataURL("image/jpeg", 0.55); } catch (_) {}
          cell.append(thumb);
          strip.append(cell);
          resolve();
        };
        vid.onseeked = done;
        const dur = vid.duration || durSec;
        vid.currentTime = Math.max(0, Math.min(time, dur - 0.01));
      });
      vid.onerror = release;
      vid.onloadeddata = async () => {
        if (!strip.isConnected) { release(); return; }
        const dur = vid.duration || durSec;
        // A segment starts at 0 (it's already trimmed to this scene); the raw-file fallback
        // still needs the absolute in-point.
        const inSec = segUrl ? 0 : (r.inSec || 0) + (scene.source_in || 0);
        try {
          for (let i = 0; i < n; i++) {
            if (!strip.isConnected) return;
            const t = inSec + (dur * (i + 0.5) / n);
            await captureAt(t);
          }
        } finally {
          release();
        }
      };
    }));
  }

  const hasRender = (st, sceneId) => !!(st.sceneRenders && st.sceneRenders[sceneId] && st.sceneRenders[sceneId].media);
  const REORDER_PX = 12;
  const IS_MAC = /Mac|iPhone|iPad|iPod/.test(navigator.platform || "") || /Mac OS/.test(navigator.userAgent || "");
  function isAdditiveMod(e) { return IS_MAC ? !!e.metaKey : !!(e.ctrlKey || e.metaKey); }
  function isClipSelectBlocked(e) {
    return !!e.target.closest(".clip-head-bar, .clip-rm, .clip-trim, .clip-vt-tail, .seam-cut, button, .tl-aud-controls");
  }

  function normId(v) { return v == null ? "" : String(v); }
  function selectedIds(st) {
    return st.selectedSceneIds?.length ? st.selectedSceneIds : (st.selectedSceneId ? [st.selectedSceneId] : []);
  }
  function isSceneSelected(st, sceneId) {
    const id = normId(sceneId);
    return selectedIds(st).some((sid) => normId(sid) === id);
  }
  function isSceneFocus(st, sceneId) {
    return normId(sceneId) === normId(st.selectedSceneId);
  }
  /** Selection state shared by toolbar build + live sync. */
  function clipToolbarState(st) {
    const ids = selectedIds(st);
    const hasSel = ids.length > 0;
    const focusId = hasSel
      ? (st.selectedSceneId && ids.some((sid) => normId(sid) === normId(st.selectedSceneId))
        ? st.selectedSceneId
        : ids[0])
      : null;
    const saveableIds = ids.filter((id) => S.clipSaveableToMediaBin?.(id));
    const saveable = saveableIds.length > 0;
    return { ids, hasSel, focusId, saveable, saveableN: saveableIds.length };
  }
  function clipSelClass(st, sceneId) {
    let cls = "";
    if (isSceneSelected(st, sceneId)) cls += " selected";
    if (isSceneFocus(st, sceneId)) cls += " focus";
    return cls;
  }
  function onClipSelect(e, sceneId) {
    _tlEditing = false;
    const ae = document.activeElement;
    if (ae && /^(SELECT|INPUT|TEXTAREA)$/.test(ae.tagName)) ae.blur();
    S.selectScene(sceneId, { additive: isAdditiveMod(e), range: e.shiftKey });
  }

  function attachClipSelect(clip, sceneId) {
    clip.addEventListener("mousedown", (e) => {
      if (e.button !== 0 || isClipSelectBlocked(e)) return;
      // Modifier clicks select immediately — do not start spacing drag (it eats the gesture).
      if (e.metaKey || e.shiftKey || isAdditiveMod(e)) {
        e.preventDefault();
        clip.dataset.modSelect = "1";
        onClipSelect(e, sceneId);
      }
    });
    clip.addEventListener("click", (e) => {
      if (isClipSelectBlocked(e)) return;
      e.stopPropagation();
      if (clip.dataset.modSelect) {
        delete clip.dataset.modSelect;
        return;
      }
      onClipSelect(e, sceneId);
    });
  }

  function syncClipSelection(st) {
    const ids = new Set(selectedIds(st).map(normId));
    const focus = normId(st.selectedSceneId);
    body.querySelectorAll(".tl-track2 > .clip, .tl-aud-clip.scene-aud").forEach((node) => {
      const id = normId(node.dataset.sceneId);
      if (!id) return;
      node.classList.toggle("selected", ids.has(id));
      node.classList.toggle("focus", id === focus);
    });
  }

  function genUnitHasRender(st, scene) {
    const uid = S.genUnitId(scene);
    return (st.project?.scenes || []).some(
      (s) => S.genUnitId(s) === uid && hasRender(st, s.id)
    );
  }

  function sceneRatingRaw(st, scene) {
    if (S.isVideoClip && S.isVideoClip(scene)) return "";
    if (!genUnitHasRender(st, scene)) return "";
    const root = S.genUnitRoot(S.genUnitId(scene)) || scene;
    return ((root && root.rating) || scene.rating || "").trim();
  }

  function sceneRatingDisplay(st, scene) {
    const raw = sceneRatingRaw(st, scene);
    if (!raw) return "";
    return window.MovieRatingPicker?.formatLabel(raw) || raw;
  }

  function syncClipRatings(st) {
    if (!st?.project) return;
    body.querySelectorAll(".clip[data-scene-id]").forEach((clipEl) => {
      const sc = S.scene(clipEl.dataset.sceneId);
      if (!sc) return;
      const label = sceneRatingDisplay(st, sc);
      let badge = clipEl.querySelector(".clip-rated");
      if (!label) {
        badge?.remove();
        return;
      }
      if (!badge) {
        badge = el("div", "clip-rated");
        const anchor = clipEl.querySelector(".clip-text");
        if (anchor && anchor.nextSibling) clipEl.insertBefore(badge, anchor.nextSibling);
        else clipEl.append(badge);
      }
      badge.title = "Rated for FunPack Studio on next generation";
      badge.textContent = "★ " + label;
    });
  }

  function exportSaveTitles(hasSel, saveable, saveableN, selN) {
    const multi = selN > 1;
    const exportTitle = !hasSel
      ? "Select a clip first"
      : saveable
        ? (multi
          ? `Combine ${saveableN} selected clips into one file (trim only, hard cuts)`
          : "Save the selected clip's rendered video to disk (renders are temporary)")
        : "Generate the clip first, or select a video clip with media";
    const binTitle = !hasSel
      ? "Select a clip first"
      : saveable
        ? (multi
          ? `Combine ${saveableN} selected clips into one file and save to Media bin (trim only)`
          : "Copy the selected clip into the Media bin - add it back as a plain video clip")
        : "Generate the clip first, or select a video clip with media";
    return { exportTitle, binTitle };
  }

  function syncToolbarSelection(st) {
    const bar = body.querySelector(".tl-toolbar");
    if (!bar || !st.project) return;
    const { ids, hasSel, focusId, saveable, saveableN } = clipToolbarState(st);
    const audioSel = st.selectedAudioTrackId;
    const canSplit = hasSel || audioSel;
    const canRemove = hasSel || audioSel;
    const selN = ids.length;
    const { exportTitle, binTitle } = exportSaveTitles(hasSel, saveable, saveableN, selN);
    bar.querySelectorAll("[data-needs-sel]").forEach((b) => {
      if (b.dataset.separateAudio != null || b.dataset.removeSepAudio != null) return;
      b.disabled = b.classList.contains("danger") ? !canRemove : !canSplit;
    });
    const splitBtn = [...bar.querySelectorAll("[data-needs-sel]")].find((b) => b.textContent === "Split");
    if (splitBtn) {
      splitBtn.disabled = !canSplit;
      splitBtn.title = audioSel
        ? "Split selected audio at the playhead (S)"
        : (hasSel ? "Split the selected clip at the playhead (S)" : "Select a clip or audio track first");
    }
    const delBtn = bar.querySelector(".btn.danger[data-needs-sel]");
    if (delBtn && delBtn.textContent === "Remove") {
      delBtn.disabled = !canRemove;
      delBtn.title = audioSel
        ? "Remove selected audio track (Delete / Backspace)"
        : (hasSel ? "Remove selected clip(s) (Delete / Backspace)" : "Select a clip or audio track first");
    }
    const badge = bar.querySelector("[data-tl-sel-badge]");
    if (badge) {
      if (selN > 0) {
        badge.hidden = false;
        badge.textContent = selN === 1 ? "1 clip selected" : `${selN} clips selected`;
      } else badge.hidden = true;
    }
    const exp = bar.querySelector("[data-export-scene]");
    if (exp) {
      exp.disabled = !(hasSel && saveable);
      exp.title = exportTitle;
    }
    const saveBin = bar.querySelector("[data-save-mediabin]");
    if (saveBin) {
      saveBin.disabled = !(hasSel && saveable);
      saveBin.title = binTitle;
    }
    const sep = bar.querySelector("[data-separate-audio]");
    if (sep) {
      const sc = focusId ? S.scene(focusId) : null;
      sep.disabled = !(sc && S.sceneHasEmbeddedAudio?.(sc));
    }
    const rmSep = bar.querySelector("[data-remove-sep-audio]");
    if (rmSep) {
      const sc = focusId ? S.scene(focusId) : null;
      const sepTrack = sc && S.separatedTrackForScene ? S.separatedTrackForScene(sc.id) : null;
      rmSep.disabled = !sepTrack;
    }
    const oldRating = bar.querySelector(".tl-rating-block");
    const freshRating = toolbarRatingBlock(st, st.project);
    if (oldRating) oldRating.replaceWith(freshRating);
    else {
      const spacer = bar.querySelector(".tl-spacer");
      if (spacer) bar.insertBefore(freshRating, spacer);
    }
  }

  function syncMetaSelection(st) {
    if (!meta || !st.project) return;
    const p = st.project;
    const scenes = p.scenes || [];
    const active = scenes.filter((s) => !s.excluded).length;
    const ghosts = (st.sceneGhosts || []).length;
    const selN = S.selectedSceneCount ? S.selectedSceneCount() : selectedIds(st).length;
    const totalSec = S.previewTotalSec ? S.previewTotalSec() : tlTotalSec;
    meta.textContent = `${scenes.length} clips · ${active} active`
      + (ghosts ? ` · ${ghosts} ghost${ghosts > 1 ? "s" : ""}` : "")
      + (selN > 0 ? ` · ${selN} selected` : "")
      + ` · ${timecode(totalSec, p.frame_rate)}`;
  }

  const p2 = (n) => String(n).padStart(2, "0");
  function timecode(sec, fps) {
    sec = Math.max(0, sec);
    const f = Math.round(sec * fps), ff = f % fps, t = Math.floor(f / fps);
    const s = t % 60, m = Math.floor(t / 60) % 60, h = Math.floor(t / 3600);
    return (h ? p2(h) + ":" : "") + p2(m) + ":" + p2(s) + ":" + p2(ff);
  }
  function rulerLabel(sec, interval) {
    if (interval < 1) return sec.toFixed(1) + "s";
    const t = Math.round(sec), s = t % 60, m = Math.floor(t / 60) % 60, h = Math.floor(t / 3600);
    return (h ? p2(h) + ":" : "") + p2(m) + ":" + p2(s);
  }
  function tickInterval() {
    const target = 92;  // ~px between labels
    for (const c of [0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600])
      if (c * pxPerSec >= target) return c;
    return 600;
  }

  const VT_SHORT = { crossfade: "dissolve", fadeblack: "fade ∅", wipeleft: "wipe ←", wiperight: "wipe →" };

  function videoTransitionState(scene, p) {
    const type = (scene.video_transition || "").trim();
    const fps = sFps(scene, p);
    if (!type) return { type: "", frames: 0, fps, sec: 0, active: false };
    const frames = scene.transition_frames > 0 ? scene.transition_frames : 16;
    return { type, frames, fps, sec: frames / fps, active: true };
  }

  function applyVideoTransition(sceneId, type, frames) {
    if (!type || !frames || frames <= 0) {
      S.patchScene(sceneId, { video_transition: "", transition_frames: null });
    } else {
      S.patchScene(sceneId, { video_transition: type, transition_frames: Math.round(frames) });
    }
  }

  function attachVtTailDrag(tail, clip, scene, vt, fps) {
    tail.addEventListener("mousedown", (e) => {
      e.stopPropagation();
      _seamDragging = true;
      const baseFrames = vt.frames || 16;
      const basePx = (baseFrames / fps) * pxPerSec;
      const tip = el("div", "trim-tip"); clip.append(tip);
      let frames = baseFrames;
      let type = vt.type || "crossfade";
      coalescedDrag(e, (dx) => {
        const sec = Math.max(0, (basePx + dx) / pxPerSec);
        frames = Math.min(120, Math.max(0, Math.round(sec * fps)));
        const w = frames > 0 ? Math.max(10, (frames / fps) * pxPerSec) : 0;
        tail.style.width = w ? w + "px" : "0";
        tail.hidden = frames <= 0;
        tip.textContent = frames > 0 ? `${(frames / fps).toFixed(2)}s · ${frames}f` : "release to clear";
      }, () => {
        tip.remove();
        _seamDragging = false;
        applyVideoTransition(scene.id, frames > 0 ? type : "", frames);
      });
    });
  }

  // ── drag plumbing ─────────────────────────────────────────────────────────────
  function onDrag(startEvt, onMove, onUp) {
    startEvt.preventDefault(); startEvt.stopPropagation();
    const x0 = startEvt.clientX;
    const move = (e) => onMove(e.clientX - x0, e);
    const up = (e) => { document.removeEventListener("mousemove", move); document.removeEventListener("mouseup", up); if (onUp) onUp(e.clientX - x0, e); };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  }

  // ── clip ───────────────────────────────────────────────────────────────────────
  function ghostClipEl(st, p, ghost, leftPx, widthPx) {
    const clip = el("div", "clip ghost");
    clip.style.left = leftPx + "px";
    clip.style.width = Math.max(widthPx, 8) + "px";
    clip.title = "Removed from timeline — shown in preview only; won't generate on the next run";
    const head = el("div", "clip-head");
    head.append(el("span", "clip-no", "∅"));
    head.append(el("span", "clip-src", "👻"));
    const fps = (ghost.fps_mode !== "project" && ghost.fps != null) ? ghost.fps : p.frame_rate;
    const frames = (ghost.frames_mode !== "project" && ghost.frames != null) ? ghost.frames : p.num_frames_per_scene;
    appendClipHeadBar(head, (frames || 1) / (fps || 25), fps || 25,
      () => S.dismissGhost(ghost.id), "Remove ghost from preview");
    clip.append(head);
    const label = ghost.pendingGen ? "removed · generating…" : (ghost.text || "removed scene");
    clip.append(el("div", "clip-text ghost-label", label));
    return clip;
  }

  // ── clip ───────────────────────────────────────────────────────────────────────
  function clipEl(st, p, scene, index, total, leftPx, widthPx) {
    const unitCuts = (p.scenes || []).filter((s) => (s.gen_unit_id || s.id) === (scene.gen_unit_id || scene.id)).length;
    const subclip = (scene.cut_offset_frames || 0) > 0;
    const src = sceneSourceForClip(scene, p);
    const srcType = src.type || "empty";
    const clip = el("div", "clip" + clipSelClass(st, scene.id)
      + (S.isVideoClip(scene) ? " clip-video" : "")
      + (scene.excluded && !scene.removed_from_plan ? " excluded" : "")
      + (scene.removed_from_plan ? " off-plan" : "")
      + (hasRender(st, scene.id) ? " rendered" : (!scene.excluded && S.isGenerativeScene(scene) ? " pending" : ""))
      + (hasRender(st, scene.id) && S.renderIsStale?.(scene.id) ? " stale-render" : "")
      + (unitCuts > 1 ? " gen-cut" : "") + (subclip ? " subclip" : "")
      + (srcType === "mixed" ? " src-mixed" : srcType === "carry" ? " src-carry" : srcType === "image" ? " src-image" : srcType === "video" ? " src-video" : srcType === "v2v" ? " src-v2v" : ""));
    clip.dataset.sceneId = scene.id;
    clip.style.left = leftPx + "px";
    clip.style.width = Math.max(widthPx, 8) + "px";

    // drag clip left/right to add or remove pause before it (flush with previous = no gap).
    attachClipSelect(clip, scene.id);
    if (index > 0) {
      clip.title = (clip.title ? clip.title + " · " : "") + "Drag left/right to adjust spacing from the previous clip";
      clip.addEventListener("mousedown", (e) => {
        if (e.button !== 0 || isClipSelectBlocked(e)) return;
        if (e.metaKey || e.shiftKey || isAdditiveMod(e)) return;
        const prevScene = (p.scenes || [])[index - 1];
        if (!prevScene) return;
        const gapBefore = +(prevScene.gap_after_sec || 0);
        const buttLeftPx = leftPx - gapBefore * pxPerSec;
        const startX = e.clientX;
        let dragging = false;
        const track = clip.parentNode;
        const onMove = (ev) => {
          const dx = ev.clientX - startX;
          if (!dragging) {
            if (Math.abs(dx) < REORDER_PX) return;
            dragging = true;
            clip.classList.add("spacing");
            document.body.classList.add("tl-spacing");
            _reordering = true;
            window.EditorHistory?.beginCoalesce(S);
          }
          const stNow = S.get();
          const pNow = stNow.project;
          const anchors = pNow ? timelineSnapAnchorsPx(stNow, pNow) : [];
          const clampedLeft = Math.max(buttLeftPx, snapPx(buttLeftPx + gapBefore * pxPerSec + dx, anchors, 10));
          clip.style.transform = `translateX(${clampedLeft - leftPx}px)`;
        };
        const onUp = (ev) => {
          document.removeEventListener("mousemove", onMove);
          document.removeEventListener("mouseup", onUp);
          document.body.classList.remove("tl-spacing");
          if (!dragging) return;
          clip.style.transform = "";
          clip.classList.remove("spacing");
          _reordering = false;
          ev.preventDefault();
          const dx = ev.clientX - startX;
          const stNow = S.get();
          const pNow = stNow.project;
          const anchors = pNow ? timelineSnapAnchorsPx(stNow, pNow) : [];
          const clampedLeft = Math.max(buttLeftPx, snapPx(buttLeftPx + gapBefore * pxPerSec + dx, anchors, 10));
          const finalGap = (clampedLeft - buttLeftPx) / pxPerSec;
          S.setSceneGapAfter(prevScene.id, finalGap < 0.02 ? 0 : finalGap);
          window.EditorHistory?.endCoalesce(S);
          track.addEventListener("click", (ce) => { ce.stopPropagation(); ce.preventDefault(); }, { capture: true, once: true });
        };
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
      });
    }

    // Drop an image (or video) from the Media bin onto a generative clip to set its i2v
    // anchor. This used to live on the Plan card; the single-timeline layout moved it here.
    if (!subclip && S.isGenerativeScene(scene)) {
      clip.addEventListener("dragover", (e) => {
        if (!e.dataTransfer.types.includes("application/funpack-media")) return;
        e.preventDefault(); e.stopPropagation();
        e.dataTransfer.dropEffect = "copy";
        clip.classList.add("drop-target");
      });
      clip.addEventListener("dragleave", (e) => {
        if (!clip.contains(e.relatedTarget)) clip.classList.remove("drop-target");
      });
      clip.addEventListener("drop", (e) => {
        clip.classList.remove("drop-target");
        const id = e.dataTransfer.getData("application/funpack-media");
        if (!id) return;
        e.preventDefault(); e.stopPropagation();
        S.assignMediaToScene(scene.id, id);
      });
    }

    // i2v anchor thumbnail (image + mixed + generated_frame)
    const mref = anchorMediaRef(scene, p);
    const asset = mref ? (st.mediaBin || []).find((m) => m.id === mref) : null;
    if (asset) {
      const th = el("div", "clip-thumb");
      if (asset.kind === "image") { const img = el("img"); img.src = window.MovieEditorAPI.mediaUrl(asset.id); img.loading = "lazy"; th.append(img); }
      else th.append(el("span", null, "▶"));
      if (srcType === "mixed") {
        const guide = el("div", "clip-thumb-guide");
        guide.title = "Prior-scene i2v guides carried with this anchor";
        guide.append(el("span", "clip-thumb-guide-icon", "⇥"));
        guide.append(el("span", "clip-thumb-guide-txt", "guides"));
        th.append(guide);
      }
      clip.append(th);
      clip.classList.add("has-media");
    } else if (srcType === "mixed") {
      clip.title = (clip.title ? clip.title + " · " : "") + "Mixed — assign an anchor image (guides stay active)";
    }

    const head = el("div", "clip-head");
    head.append(el("span", "clip-no", S.isVideoClip(scene) ? "V" : p2(index + 1)));
    appendSrcBadge(head, srcType);
    appendClipHeadBar(head, sDur(scene, p), sFps(scene, p), () => S.removeScene(scene.id), null, {
      canLeft: index > 0,
      canRight: total != null && index < total - 1,
      onLeft: () => S.moveTimelineClip(scene.id, -1),
      onRight: () => S.moveTimelineClip(scene.id, 1),
    });
    clip.append(head);

    const root = unitCuts > 1
      ? (p.scenes || []).find((s) => (s.gen_unit_id || s.id) === (scene.gen_unit_id || scene.id) && !(s.cut_offset_frames > 0))
      : null;
    // The timeline clip is the RESULT: show what it was actually generated with (the frozen
    // render prompt), not the live plan text — editing a scene in the plan must not mutate the
    // already-generated video here. Falls back to live plan text only when not yet generated.
    const frozen = !S.isVideoClip(scene) && S.renderPromptForScene ? S.renderPromptForScene(scene.id) : null;
    const label = S.isVideoClip(scene)
      ? ((mref && (st.mediaBin || []).find((m) => m.id === mref)?.name) || "Video clip")
      : ((frozen && frozen.text) || scene.text || (root && root.text) || (subclip ? "cut" : "empty scene"));
    clip.append(el("div", "clip-text" + (label && label !== "empty scene" && label !== "cut" && label !== "Video clip" ? "" : " empty"), label));

    const rating = sceneRatingDisplay(st, scene);
    if (rating) {
      const rated = el("div", "clip-rated");
      rated.title = "Rated for FunPack Studio conditioning on next generation";
      rated.textContent = "★ " + rating;
      clip.append(rated);
    }

    const fxTxt = effectSummary(scene.effects);
    if (fxTxt) {
      const fxChip = el("div", "clip-fx", fxTxt);
      fxChip.title = "Effects on this clip — click to remove them all";
      fxChip.onclick = (e) => {
        e.stopPropagation();
        if (confirm("Remove all effects from this clip?")) S.patchScene(scene.id, { effects: {} });
      };
      clip.append(fxChip);
    }

    const vt = videoTransitionState(scene, p);
    if (vt.active) {
      const tail = el("div", "clip-vt-tail vt-" + vt.type);
      tail.style.width = Math.max(10, vt.sec * pxPerSec) + "px";
      tail.title = `${VT_SHORT[vt.type] || vt.type} · drag to adjust length · type in inspector`;
      attachVtTailDrag(tail, clip, scene, vt, vt.fps);
      clip.append(tail);
    }

    if (hasRender(st, scene.id)) {
      const anchorMismatch = S.renderAnchorMismatch ? S.renderAnchorMismatch(scene.id) : null;
      if (anchorMismatch) {
        clip.classList.add("anchor-mismatch");
        const gen = el("div", "clip-gen-prompt clip-gen-anchor");
        gen.title = anchorMismatch.renderedLabel
          ? `Generated with i2v image: ${anchorMismatch.renderedLabel}`
          : "i2v anchor changed after generation - preview shows the previous render";
        gen.append(el("span", "clip-gen-prompt-label", "i2v image changed"));
        gen.append(el("span", "clip-gen-prompt-text", "Showing previous generation"));
        clip.append(gen);
      }
      if (S.renderPromptMismatch) {
        const mismatch = S.renderPromptMismatch(scene.id);
        if (mismatch) {
          clip.classList.add("prompt-mismatch");
          const gen = el("div", "clip-gen-prompt");
          gen.title = "Timeline prompt was edited after generation - rate against this text";
          gen.append(el("span", "clip-gen-prompt-label", "Generated with"));
          gen.append(el("span", "clip-gen-prompt-text", mismatch.rendered || "(empty)"));
          clip.append(gen);
        }
      }
    }

    // right-edge trim → new duration → frames recomputed (duration × fps).
    const locked = scene.frames_mode === "custom";
    const baseDur = sDur(scene, p);
    const baseLeftPx = leftPx;
    const baseWidthPx = widthPx;
    const leftHandle = el("div", "clip-trim clip-trim-left" + (locked ? " locked" : ""));
    leftHandle.title = locked ? "Length locked (custom mode)" : "Drag to trim start · Alt+drag to slip source when rendered";
    if (!locked) {
      leftHandle.addEventListener("mousedown", (e) => {
        e.stopPropagation();
        clip.classList.add("trimming");
        const tip = el("div", "trim-tip"); clip.append(tip);
        const anchors = timelineSnapAnchorsPx(st, p);
        let finalDelta = 0;
        coalescedDrag(e, (dx) => {
          const snappedLeft = snapPx(baseLeftPx + dx, anchors, 10);
          finalDelta = (snappedLeft - baseLeftPx) / pxPerSec;
          if (finalDelta > 0) {
            const w = Math.max(8, (baseDur - finalDelta) * pxPerSec);
            clip.style.left = snappedLeft + "px";
            clip.style.width = w + "px";
            _syncSceneAudioClipVisual(scene.id, snappedLeft, w);
          } else {
            clip.style.left = baseLeftPx + "px";
            clip.style.width = baseWidthPx + "px";
            _syncSceneAudioClipVisual(scene.id, baseLeftPx, baseWidthPx);
          }
          tip.textContent = e.altKey && hasRender(st, scene.id)
            ? `slip ${finalDelta >= 0 ? "+" : ""}${finalDelta.toFixed(2)}s`
            : `trim ${Math.max(0, finalDelta).toFixed(2)}s`;
        }, () => {
          clip.classList.remove("trimming"); tip.remove();
          clip.style.left = baseLeftPx + "px";
          clip.style.width = baseWidthPx + "px";
          if (e.altKey && hasRender(st, scene.id)) {
            if (Math.abs(finalDelta) > 0.02) S.slipScene(scene.id, finalDelta);
          } else if (finalDelta > 0.02) {
            S.trimSceneLeft(scene.id, finalDelta);
          }
        });
      });
    }
    clip.append(leftHandle);

    const handle = el("div", "clip-trim" + (locked ? " locked" : ""));
    handle.title = locked
      ? "Length is Custom — change it in the inspector (set Frames to Inherit project to trim here)"
      : "Drag to trim · length = duration × fps";
    if (locked) { clip.append(handle); return clip; }
    handle.addEventListener("mousedown", (e) => {
      e.stopPropagation();
      clip.classList.add("trimming");
      const tip = el("div", "trim-tip"); clip.append(tip);
      const anchors = timelineSnapAnchorsPx(st, p);
      let finalDur = baseDur;
      coalescedDrag(e, (dx) => {
        const rightPx = snapPx(baseLeftPx + baseDur * pxPerSec + dx, anchors, 10);
        finalDur = Math.max(0.1, (rightPx - baseLeftPx) / pxPerSec);
        const w = finalDur * pxPerSec;
        clip.style.width = Math.max(w, 8) + "px";
        _syncSceneAudioClipVisual(scene.id, baseLeftPx, Math.max(w, 8));
        const fps = sFps(scene, p);
        tip.textContent = `${timecode(finalDur, fps)} · ${S.snapFrames(finalDur * fps)}f`;
      }, () => {
        clip.classList.remove("trimming"); tip.remove();
        if (Math.abs(finalDur - baseDur) > 0.02) S.resizeScene(scene.id, finalDur);
      });
    });
    clip.append(handle);
    appendFilmstrip(clip, st, scene, widthPx);
    return clip;
  }

  // ── cut line at each clip boundary (NLE-style; transition lives on clip tail) ──
  function seamEl(st, p, scene, seamPx) {
    const vt = videoTransitionState(scene, p);
    const seam = el("div", "seam-cut" + (vt.active ? " has-vt vt-" + vt.type : ""));
    seam.style.left = seamPx + "px";
    if (vt.active) {
      seam.title = `${VT_SHORT[vt.type] || vt.type} · ${vt.frames}f — drag the clip edge or use inspector`;
    }
    return seam;
  }

  // ── + Add menu (effects, transitions, future: text/image/audio) ───────────────
  let _addModal = null;

  function closeAddModal() {
    if (_addModal) { _addModal.remove(); _addModal = null; }
  }

  // What's actually on this clip. Without it the effects are invisible — and several are
  // toggles, so "apply again to turn it off" is meaningless if you cannot see what is on.
  const FX_LABELS = [
    ["flip_h", () => "⇄ flip"],
    ["flip_v", () => "⇅ flip"],
    ["fit", (v) => (v === "fill" ? "fill" : null)],
    ["crop_inset", (v) => (v > 0 ? `crop ${Math.round(v * 100)}%` : null)],
    ["zoom", (v) => (v === "in" ? "zoom in" : v === "out" ? "zoom out" : null)],
    ["blur", (v) => (v > 0 ? `blur ${(+v).toFixed(2)}` : null)],
    ["fade_in", (v) => (v > 0 ? `fade in ${v}s` : null)],
    ["fade_out", (v) => (v > 0 ? `fade out ${v}s` : null)],
  ];
  function effectSummary(fx) {
    if (!fx) return "";
    const out = [];
    for (const [key, fmt] of FX_LABELS) {
      const raw = fx[key];
      if (raw === undefined || raw === null || raw === false || raw === "") continue;
      const txt = fmt(raw);
      if (txt) out.push(txt);
    }
    return out.join(" · ");
  }

  function openNleSettingsModal(kind, st) {
    closeAddModal();
    const isEffect = kind === "effect";
    const items = isEffect ? (st.nleEffects || []) : (st.nleVideoTransitions || []);
    if (!st.selectedSceneId) { alert("Select a clip first."); return; }
    if (!items.length) { alert("No presets loaded."); return; }

    _addModal = el("div", "modal-overlay");
    const box = el("div", "modal");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", isEffect ? "Add effect" : "Add transition"));
    const headRight = el("div", "modal-head-right");
    const closeBtn = el("button", "btn ghost tiny", "✕");
    closeBtn.onclick = closeAddModal;
    headRight.append(closeBtn);
    head.append(headRight);
    box.append(head);
    const content = el("div", "modal-content");
    const sel = el("select", "tl-dd-sel");
    items.forEach((d) => sel.append(new Option(d.name || d.id, d.id)));
    const valRow = el("label", "tl-dd-val");
    const valLbl = el("span", null, "");
    const valIn = el("input"); valIn.type = "number";
    valRow.append(valLbl, valIn);
    const sync = () => {
      const d = items.find((x) => x.id === sel.value);
      if (d && d.param) {
        valRow.hidden = false;
        valLbl.textContent = d.param.label;
        valIn.min = d.param.min; valIn.max = d.param.max; valIn.step = d.param.step;
        valIn.value = d.param.default;
      } else {
        valRow.hidden = true;
        valIn.value = "";
      }
    };
    sel.onchange = sync;
    sync();
    const actions = el("div", "lib-form-actions");
    const applyBtn = el("button", "btn primary tiny", "Apply");
    applyBtn.onclick = () => {
      const d = items.find((x) => x.id === sel.value); if (!d) return;
      const v = d.param ? parseFloat(valIn.value || d.param.default) : null;
      const ok = isEffect ? S.applyNleEffect(d.id, v) : S.applyNleVideoTransition(d.id, v);
      if (!ok) alert("Could not apply.");
      else closeAddModal();
    };
    const cancel = el("button", "btn ghost tiny", "Cancel");
    cancel.onclick = closeAddModal;
    actions.append(applyBtn, cancel);
    content.append(sel, valRow, actions);
    box.append(content);
    _addModal.append(box);
    _addModal.addEventListener("click", (e) => { if (e.target === _addModal) closeAddModal(); });
    document.body.append(_addModal);
  }

  function addMenuDropdown(st, p) {
    const wrap = el("div", "tl-dd");
    const btn = el("button", "btn ghost tiny", "＋ Add");
    btn.title = "Add clip, effect, transition, and more";
    const panel = el("div", "tl-dd-panel tl-add-panel");
    panel.hidden = true;

    const addRow = (label, title, onClick, disabled) => {
      const row = el("button", "tl-add-row" + (disabled ? " disabled" : ""), label);
      row.type = "button";
      row.title = title;
      row.disabled = !!disabled;
      if (onClick && !disabled) row.onclick = (e) => { e.stopPropagation(); panel.hidden = true; onClick(); };
      panel.append(row);
    };

    addRow("Clip", "Append a new generative scene clip to the timeline", () => S.addScene());
    addRow("Video", "Add an imported video clip from the Media Browser", () => openVideoClipModal(st));
    addRow("Effects", "Post-render clip effect (zoom, blur, fade…)", () => openNleSettingsModal("effect", st));
    addRow("Transitions", "Video blend on the outgoing edge of the clip", () => openNleSettingsModal("transition", st));
    addRow("Text", "Add a title or label on the top overlay track", () => openTextOverlayModal(st));
    addRow("Image", "Add a picture overlay from the Media Browser", () => openImageOverlayModal(st));
    addRow("Overlay track", "Add another compositing layer (higher tracks draw on top)", () => S.addOverlayLane());
    addRow("Audio", "Add an audio track from the Media Browser", () => openAudioTrackModal(st));

    btn.onclick = (e) => {
      e.stopPropagation();
      const show = panel.hidden;
      panel.hidden = !show;
      if (show) {
        const off = (ev) => { if (!wrap.contains(ev.target)) { panel.hidden = true; document.removeEventListener("mousedown", off, true); } };
        setTimeout(() => document.addEventListener("mousedown", off, true), 0);
      }
    };
    wrap.append(btn); wrap.append(panel);
    return wrap;
  }

  function openVideoClipModal(st) {
    closeAddModal();
    const videoAssets = (st.mediaBin || []).filter((m) => m.kind === "video");

    _addModal = el("div", "modal-overlay");
    const box = el("div", "modal");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", "Add video clip"));
    const headRight = el("div", "modal-head-right");
    const closeBtn = el("button", "btn ghost tiny", "✕");
    closeBtn.onclick = closeAddModal;
    headRight.append(closeBtn);
    head.append(headRight);
    box.append(head);

    const content = el("div", "modal-content");
    const pick = (mediaId) => {
      S.addVideoClip(mediaId);
      closeAddModal();
    };

    if (!videoAssets.length) {
      content.append(el("div", "pj-meta", "Upload video in the Media Browser (mp4, mov, webm…), then pick it here."));
      const uploadBtn = el("button", "btn primary tiny", "Upload video file");
      const fileIn = el("input");
      fileIn.type = "file";
      fileIn.accept = "video/*,.mp4,.mov,.webm,.mkv";
      fileIn.style.display = "none";
      uploadBtn.onclick = () => fileIn.click();
      fileIn.onchange = async () => {
        const files = [...(fileIn.files || [])];
        fileIn.value = "";
        if (!files.length) return;
        await S.uploadMedia(files);
        const fresh = S.get();
        const added = (fresh.mediaBin || []).filter((m) => m.kind === "video");
        if (added[0]) pick(added[0].id);
        else closeAddModal();
      };
      content.append(uploadBtn);
      content.append(fileIn);
    } else {
      content.append(el("div", "pj-meta", "Video clips play as-is and are skipped by Generate."));
      videoAssets.forEach((m) => {
        const row = el("button", "tl-add-row aud-pick-row", m.name || m.id);
        row.type = "button";
        row.onclick = () => pick(m.id);
        content.append(row);
      });
    }

    const cancel = el("button", "btn ghost tiny", "Cancel");
    cancel.onclick = closeAddModal;
    content.append(cancel);
    box.append(content);
    _addModal.append(box);
    _addModal.addEventListener("click", (e) => { if (e.target === _addModal) closeAddModal(); });
    document.body.append(_addModal);
  }

  // ── overlay modals ─────────────────────────────────────────────────────────────
  function openImageOverlayModal(st, existing) {
    closeAddModal();
    const OUI = window.OverlayUI;
    const API = window.MovieEditorAPI;
    const images = (st.mediaBin || []).filter((m) => m.kind === "image");
    const editing = existing && existing.kind === "image";
    const { body, foot } = OUI.openModal({
      title: editing ? "Image overlay" : "Add image",
      subtitle: editing ? "Pick a source and adjust size on the preview." : "Placed at the playhead · drag on preview to move.",
      widthClass: "ov-modal-md",
      onClose: () => { _addModal = null; },
    });
    _addModal = body.closest(".modal-overlay");

    const form = el("div", "ov-form");
    if (!images.length) {
      form.append(el("div", "ov-empty", "Upload images in the Media Browser first."));
      const uploadBtn = el("button", "btn primary", "Upload image");
      const fileIn = el("input");
      fileIn.type = "file";
      fileIn.accept = "image/*,.png,.jpg,.jpeg,.webp,.gif";
      fileIn.style.display = "none";
      uploadBtn.onclick = () => fileIn.click();
      fileIn.onchange = async () => {
        const files = [...(fileIn.files || [])];
        fileIn.value = "";
        if (!files.length) return;
        await S.uploadMedia(files);
        closeAddModal();
        openImageOverlayModal(S.get(), existing);
      };
      form.append(uploadBtn, fileIn);
    } else {
      const grid = el("div", "ov-image-grid");
      images.forEach((m) => {
        const card = el("button", "ov-image-card" + (editing && existing.media_ref === m.id ? " selected" : ""));
        card.type = "button";
        const img = el("img");
        img.src = API.mediaUrl(m.id);
        img.alt = m.name || "";
        img.loading = "lazy";
        card.append(img, el("span", "ov-image-name", m.name || m.id));
        card.onclick = () => {
          if (editing) S.updateOverlayTrack(existing.id, { media_ref: m.id, label: m.name || "Image" });
          else S.addImageOverlay(m.id);
          closeAddModal();
        };
        grid.append(card);
      });
      form.append(grid);
    }

    if (editing) {
      form.append(OUI.pixelSizeFields(existing, st.project?.width || 768, st.project?.height || 512, (patch) => {
        S.updateOverlayTrack(existing.id, patch, true);
      }));
      const op = el("input");
      op.type = "range"; op.min = "0"; op.max = "1"; op.step = "0.05";
      op.value = existing.opacity != null ? existing.opacity : 1;
      const opVal = el("span", "ov-range-val", Math.round(parseFloat(op.value) * 100) + "%");
      op.oninput = () => {
        opVal.textContent = Math.round(parseFloat(op.value) * 100) + "%";
        S.updateOverlayTrack(existing.id, { opacity: parseFloat(op.value) }, true);
      };
      form.append(OUI.rangeField("Opacity", op, opVal));
    }

    body.append(form);
    const cancel = el("button", "btn ghost", editing ? "Close" : "Cancel");
    cancel.onclick = closeAddModal;
    foot.append(cancel);
  }

  function openTextOverlayModal(st, existing) {
    closeAddModal();
    const OUI = window.OverlayUI;
    const editing = existing && existing.kind === "text";
    const state = {
      ...(OUI.TEXT_STYLE_DEFAULTS || {}),
      ...(editing ? existing : {}),
      text: editing ? (existing.text || "") : "Title",
      font_size: editing ? (existing.font_size != null ? existing.font_size : 42) : 42,
      font_family: editing ? (existing.font_family || "system-ui") : "system-ui",
      color: editing ? ((existing.color || "#ffffff").match(/^#/) ? existing.color : "#ffffff") : "#ffffff",
      opacity: editing ? (existing.opacity != null ? existing.opacity : 1) : 1,
    };

    const { body, foot } = OUI.openModal({
      title: editing ? "Text overlay" : "Add text",
      subtitle: "Drag on the program monitor to reposition.",
      widthClass: "ov-modal-sm",
      onClose: () => { _addModal = null; },
    });
    _addModal = body.closest(".modal-overlay");

    const preview = el("div", "ov-text-preview");
    const previewInner = el("div", "ov-text-preview-inner");
    preview.append(previewInner);

    const syncPreview = () => {
      previewInner.textContent = state.text || "Title";
      OUI.applyTextCss(previewInner, state, state.font_size);
      previewInner.style.opacity = state.opacity;
    };
    syncPreview();

    const form = el("div", "ov-form");
    const ta = el("textarea", "ov-input ov-textarea");
    ta.rows = 2;
    ta.value = state.text;
    ta.placeholder = "Enter text…";
    ta.oninput = () => { state.text = ta.value; syncPreview(); };
    form.append(OUI.field("Text", ta, { full: true }));

    const row = el("div", "ov-form-row");
    const size = el("input", "ov-input");
    size.type = "number"; size.min = "8"; size.max = "200"; size.step = "1";
    size.value = state.font_size;
    size.oninput = () => { state.font_size = parseInt(size.value || "42", 10); syncPreview(); };
    row.append(OUI.field("Font size (px)", size));

    const fontSel = OUI.fontSelect(state.font_family, (v) => { state.font_family = v; syncPreview(); });
    row.append(OUI.field("Font", fontSel));
    form.append(row);

    const row2 = el("div", "ov-form-row");
    const color = el("input", "ov-color");
    color.type = "color";
    color.value = state.color;
    color.oninput = () => { state.color = color.value; syncPreview(); };
    row2.append(OUI.field("Color", color));
    form.append(row2);

    form.append(OUI.textStyleControls(
      (k) => state[k],
      (patch) => { Object.assign(state, patch); syncPreview(); },
    ));

    const op = el("input");
    op.type = "range"; op.min = "0"; op.max = "1"; op.step = "0.05";
    op.value = state.opacity;
    const opVal = el("span", "ov-range-val", Math.round(state.opacity * 100) + "%");
    op.oninput = () => {
      state.opacity = parseFloat(op.value);
      opVal.textContent = Math.round(state.opacity * 100) + "%";
      syncPreview();
    };
    form.append(OUI.rangeField("Opacity", op, opVal));

    body.append(preview, form);

    const cancel = el("button", "btn ghost", "Cancel");
    cancel.onclick = closeAddModal;
    const ok = el("button", "btn primary", editing ? "Save" : "Add");
    ok.onclick = () => {
      const payload = {
        text: (state.text || "").trim() || "Title",
        font_size: state.font_size,
        font_family: state.font_family,
        color: state.color,
        opacity: state.opacity,
        bold: !!state.bold, italic: !!state.italic,
        text_align: state.text_align, line_spacing: state.line_spacing,
        stroke_width: state.stroke_width, stroke_color: state.stroke_color,
        shadow: !!state.shadow, shadow_color: state.shadow_color,
        bg_enabled: !!state.bg_enabled, bg_color: state.bg_color, bg_opacity: state.bg_opacity,
        label: "Text",
      };
      if (editing) S.updateOverlayTrack(existing.id, payload);
      else {
        S.addTextOverlay(payload.text);
        const ov = (S.get().project?.overlay_tracks || []).slice(-1)[0];
        if (ov) S.updateOverlayTrack(ov.id, payload, true);
      }
      closeAddModal();
    };
    foot.append(cancel, ok);
    ta.focus();
    if (!editing) ta.select();
  }

  function openOverlaySettingsModal(st, track) {
    if (!track) return;
    S.selectOverlay(track.id);
    if (track.kind === "text") openTextOverlayModal(st, track);
    else openImageOverlayModal(st, track);
  }

  // ── audio lanes (NLE-style, below video) ───────────────────────────────────────
  function openAudioTrackModal(st) {
    closeAddModal();
    const audioAssets = (st.mediaBin || []).filter(isAudioAsset);

    _addModal = el("div", "modal-overlay");
    const box = el("div", "modal");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", "Add audio track"));
    const headRight = el("div", "modal-head-right");
    const closeBtn = el("button", "btn ghost tiny", "✕");
    closeBtn.onclick = closeAddModal;
    headRight.append(closeBtn);
    head.append(headRight);
    box.append(head);

    const content = el("div", "modal-content");
    const pick = (mediaId) => {
      S.addAudioTrack(mediaId, _audioTrackStartSec());
      closeAddModal();
    };

    if (!audioAssets.length) {
      content.append(el("div", "pj-meta", "Upload audio in the Media Browser (wav, mp3, m4a, ogg, flac…), then pick it here."));
      const uploadBtn = el("button", "btn primary tiny", "Upload audio file");
      const fileIn = el("input");
      fileIn.type = "file";
      fileIn.accept = "audio/*,.mp3,.wav,.m4a,.aac,.ogg,.flac,.opus";
      fileIn.style.display = "none";
      uploadBtn.onclick = () => fileIn.click();
      fileIn.onchange = async () => {
        const files = [...(fileIn.files || [])];
        fileIn.value = "";
        if (!files.length) return;
        await S.uploadMedia(files);
        const fresh = S.get();
        const added = (fresh.mediaBin || []).filter(isAudioAsset);
        const last = added[0];
        if (last) pick(last.id);
        else closeAddModal();
      };
      content.append(uploadBtn);
      content.append(fileIn);
    } else {
      content.append(el("div", "pj-meta", "Inserts at the playhead. Upload more in the Media Browser."));
      audioAssets.forEach((m) => {
        const row = el("button", "tl-add-row aud-pick-row", m.name || m.id);
        row.type = "button";
        row.onclick = () => pick(m.id);
        content.append(row);
      });
    }

    const cancel = el("button", "btn ghost tiny", "Cancel");
    cancel.onclick = closeAddModal;
    content.append(cancel);
    box.append(content);
    _addModal.append(box);
    _addModal.addEventListener("click", (e) => { if (e.target === _addModal) closeAddModal(); });
    document.body.append(_addModal);
  }

  function sceneAudioClip(st, p, scene, index, leftPx, widthPx) {
    if (scene.audio_separated) return el("span");  // audio lives on its own lane
    const vol = scene.audio_volume != null ? scene.audio_volume : 1;
    const w = Math.max(widthPx, 8);
    const durSec = w / pxPerSec;
    const inSec = scene.source_in || 0;
    const clip = el("div", "tl-aud-clip scene-aud" + clipSelClass(st, scene.id));
    clip.dataset.sceneId = scene.id;
    clip.style.left = leftPx + "px";
    clip.style.width = w + "px";
    clip.style.maxWidth = w + "px";
    clip.onclick = (e) => {
      if (e.target.closest(".tl-aud-controls")) return;
      e.stopPropagation();
      onClipSelect(e, scene.id);
    };

    const wave = el("div", "tl-aud-wave");
    const canvas = el("canvas");
    wave.append(canvas);
    clip.append(wave);

    const r = (st.sceneRenders || {})[scene.id];
    const binRef = scene.source?.media_ref;
    const binAsset = binRef ? (st.mediaBin || []).find((m) => m.id === binRef) : null;
    if (r?.media && st.project?.id) {
      // Same reasoning as appendFilmstrip: the raw render is the shared, moov-at-end chain
      // output. A plain resultUrl fetch here downloaded and decoded the WHOLE multi-scene
      // file once per scene that shares it (fighting the player pool for connections); the
      // trimmed segment is this scene's span only, so no inSec slice is needed to paint it.
      const API = window.MovieEditorAPI;
      const segUrl = API.previewSegmentUrl
        ? API.previewSegmentUrl(st.project.id, scene.id, { media: r.media, renderIn: r.inSec || 0, dur: durSec })
        : null;
      _attachWaveform(canvas, `scene-aud:${scene.id}:${inSec.toFixed(3)}:${durSec.toFixed(3)}`, segUrl || API.resultUrl(st.project.id, r.media), segUrl ? {
        width: w,
        color: "rgba(45, 212, 191, 0.5)",
      } : {
        width: w,
        inSec,
        durationSec: durSec,
        color: "rgba(45, 212, 191, 0.5)",
      });
    } else if (binAsset?.kind === "video") {
      _attachWaveform(canvas, `scene-aud-bin:${scene.id}:${binRef}:${inSec.toFixed(3)}:${durSec.toFixed(3)}`, window.MovieEditorAPI.mediaUrl(binRef), {
        width: w,
        inSec,
        durationSec: durSec,
        color: "rgba(45, 212, 191, 0.5)",
      });
    } else {
      _attachWaveform(canvas, null, null, { color: "rgba(45, 138, 106, 0.28)" });
    }

    const controls = el("div", "tl-aud-controls");
    controls.append(el("span", "tl-aud-name", `S${index + 1}`));
    const slider = el("input", "tl-aud-vol"); slider.type = "range"; slider.min = "0"; slider.max = "2"; slider.step = "0.05";
    slider.value = vol; slider.title = `Clip ${index + 1} volume`;
    slider.oninput = (e) => { e.stopPropagation(); S.patchSceneQuiet(scene.id, { audio_volume: parseFloat(slider.value) }); };
    slider.onclick = (e) => e.stopPropagation();
    controls.append(slider);
    clip.append(controls);
    return clip;
  }

  function insertedAudioLane(st, p, track, laneH) {
    const lane = el("div", "tl-audio-lane" + (track.kind === "overlay" || (track.media_ref && track.kind !== "separated") ? " overlay-lane" : ""));
    lane.style.height = laneH + "px";
    const isSep = S.isSeparatedAudioTrack ? S.isSeparatedAudioTrack(track) : (track.kind === "separated" && track.scene_id);
    const isOverlay = S.isOverlayAudioTrack ? S.isOverlayAudioTrack(track) : (!isSep && (track.media_ref || track.render_media?.filename));
    const asset = isOverlay && track.media_ref ? (st.mediaBin || []).find((m) => m.id === track.media_ref) : null;
    const sepScene = isSep ? p.scenes.find((s) => s.id === track.scene_id) : null;
    const sepIdx = sepScene ? p.scenes.indexOf(sepScene) + 1 : 0;
    const linkedSep = isSep && !!track.scene_id && !!sepScene;
    const body = el("div", "tl-audio-lane-body");
    const startSec = track.start_sec || 0;
    const inSec = S.audioTrackInSec ? S.audioTrackInSec(track) : (track.source_in_sec || 0);
    let durSec = S.audioTrackDurSec ? S.audioTrackDurSec(track) : (track.source_dur || asset?.duration_sec || 0);
    const w = Math.max((durSec || 2) * pxPerSec, 48);
    const selected = st.selectedAudioTrackId === track.id;
    const block = el("div", "tl-aud-clip ins" + (isSep ? " sep" : " overlay") + (selected ? " selected" : ""));
    block.dataset.audioTrackId = track.id;
    block.style.left = (startSec * pxPerSec) + "px";
    block.style.width = w + "px";
    block.title = "Drag edges to trim · Alt+left drag to slip source · S splits at playhead";

    const wave = el("div", "tl-aud-wave");
    const canvas = el("canvas");
    wave.append(canvas);
    block.append(wave);
    if (isSep) {
      const pinned = track.pinned_media;
      const r = track.scene_id ? (st.sceneRenders || {})[track.scene_id] : null;
      const media = pinned || r?.media;
      const sepUrl = S.separatedTrackAudioUrl ? S.separatedTrackAudioUrl(st, track) : null;
      if (sepUrl) {
        const wfKey = track.pinned_bin_ref
          ? `sep-aud-bin:${track.id}:${track.pinned_bin_ref}:${inSec.toFixed(3)}:${durSec.toFixed(3)}`
          : `sep-aud:${track.id}:${media?.filename || ""}:${inSec.toFixed(3)}:${durSec.toFixed(3)}`;
        _attachWaveform(canvas, wfKey, sepUrl, {
          width: w,
          color: "rgba(45, 212, 191, 0.55)",
          onDuration: (sec) => {
            if (sec > 0 && !track.source_dur && !track.pinned_dur) {
              durSec = sec;
              block.style.width = Math.max(sec * pxPerSec, 48) + "px";
            }
          },
        });
      } else {
        _attachWaveform(canvas, null, null, { color: "rgba(45, 138, 106, 0.28)" });
      }
    } else if (asset) {
      _attachWaveform(canvas, `aud:${asset.id}:${inSec.toFixed(3)}:${durSec.toFixed(3)}`, window.MovieEditorAPI.mediaUrl(asset.id), {
        width: w,
        color: "rgba(96, 165, 250, 0.55)",
        onDuration: (sec) => {
          if (sec > 0 && !asset.duration_sec) {
            durSec = sec;
            block.style.width = Math.max(sec * pxPerSec, 48) + "px";
          }
        },
      });
    } else if (track.render_media?.filename && st.project?.id) {
      _attachWaveform(canvas, `aud-render:${track.id}:${inSec.toFixed(3)}`, window.MovieEditorAPI.resultUrl(st.project.id, track.render_media), {
        width: w,
        color: "rgba(96, 165, 250, 0.55)",
      });
    }

    const leftTrim = el("div", "tl-aud-trim left");
    leftTrim.title = linkedSep
      ? "Drag to trim start · Alt+drag to slip source in file"
      : "Drag to trim start · Alt+drag to slip source";
    leftTrim.addEventListener("mousedown", (e) => {
      e.stopPropagation();
      block.classList.add("trimming");
      const tip = el("div", "trim-tip"); block.append(tip);
      const baseLeft = startSec * pxPerSec;
      const anchors = timelineSnapAnchorsPx(st, p);
      let finalDelta = 0;
      coalescedDrag(e, (dx) => {
        const snappedLeft = snapPx(baseLeft + dx, anchors, 10);
        finalDelta = (snappedLeft - baseLeft) / pxPerSec;
        if (e.altKey) {
          tip.textContent = `slip ${finalDelta >= 0 ? "+" : ""}${finalDelta.toFixed(2)}s`;
          return;
        }
        const trim = Math.max(0, finalDelta);
        const newDur = Math.max(0.1, durSec - trim);
        if (linkedSep) {
          block.style.width = Math.max(newDur * pxPerSec, 48) + "px";
        } else {
          block.style.left = (baseLeft + trim * pxPerSec) + "px";
          block.style.width = Math.max(newDur * pxPerSec, 48) + "px";
        }
        tip.textContent = `trim ${trim.toFixed(2)}s · ${newDur.toFixed(2)}s`;
      }, () => {
        block.classList.remove("trimming"); tip.remove();
        if (e.altKey) {
          if (Math.abs(finalDelta) > 0.02) S.slipAudioTrack(track.id, finalDelta);
        } else if (finalDelta > 0.02) {
          S.trimAudioTrackLeft(track.id, finalDelta);
        }
      });
    });

    const rightTrim = el("div", "tl-aud-trim right");
    rightTrim.title = "Drag to shorten or extend";
    rightTrim.addEventListener("mousedown", (e) => {
      e.stopPropagation();
      block.classList.add("trimming");
      const tip = el("div", "trim-tip"); block.append(tip);
      const baseLeft = startSec * pxPerSec;
      const baseDur = durSec;
      const anchors = timelineSnapAnchorsPx(st, p);
      const maxDur = S.audioTrackMaxDurSec ? S.audioTrackMaxDurSec(track) : Infinity;
      let finalDur = durSec;
      coalescedDrag(e, (dx) => {
        const rightPx = snapPx(baseLeft + baseDur * pxPerSec + dx, anchors, 10);
        finalDur = Math.min(maxDur, Math.max(0.1, (rightPx - baseLeft) / pxPerSec));
        block.style.width = Math.max(finalDur * pxPerSec, 48) + "px";
        tip.textContent = finalDur >= maxDur ? `${finalDur.toFixed(2)}s (max — end of source)` : `${finalDur.toFixed(2)}s`;
      }, () => {
        block.classList.remove("trimming"); tip.remove();
        S.resizeAudioTrack(track.id, finalDur);
      });
    });
    block.append(leftTrim, rightTrim);

    const controls = el("div", "tl-aud-controls");
    const label = isSep ? (track.label || `S${sepIdx} audio`) : ((asset && asset.name) || "audio");
    controls.append(el("span", "tl-aud-name", label));
    const slider = el("input", "tl-aud-vol"); slider.type = "range"; slider.min = "0"; slider.max = "2"; slider.step = "0.05";
    slider.value = track.volume != null ? track.volume : 1;
    slider.oninput = () => S.updateAudioTrack(track.id, { volume: parseFloat(slider.value) }, true);
    slider.onclick = (e) => e.stopPropagation();
    controls.append(slider);
    const startIn = el("input"); startIn.type = "number"; startIn.min = "0"; startIn.step = "0.1";
    startIn.value = startSec; startIn.title = "Start (s)"; startIn.className = "tl-aud-start";
    startIn.oninput = (e) => {
      e.stopPropagation();
      const v = parseFloat(startIn.value || "0");
      if (linkedSep) return;
      S.updateAudioTrack(track.id, { start_sec: v }, true);
      block.style.left = (v * pxPerSec) + "px";
    };
    startIn.disabled = linkedSep;
    startIn.onclick = (e) => e.stopPropagation();
    controls.append(startIn);
    const rm = el("button", "ic-btn danger tl-aud-rm", "✕");
    rm.title = isSep
      ? (sepScene
        ? "Remove separated audio (restores embedded audio on the clip)"
        : "Remove detached audio track")
      : "Remove overlay audio track";
    rm.onclick = (e) => { e.stopPropagation(); S.removeAudioTrack(track.id); };
    controls.append(rm);
    block.append(controls);

    block.onclick = (e) => {
      if (e.target.closest(".tl-aud-trim, input, button, select")) return;
      e.stopPropagation();
      S.selectAudioTrack(track.id);
    };

    block.addEventListener("mousedown", (e) => {
      if (e.target.closest(".tl-aud-trim, input, button, select")) return;
      e.stopPropagation();
      S.selectAudioTrack(track.id);
      if (linkedSep) return;
      const baseLeft = startSec * pxPerSec;
      const anchors = timelineSnapAnchorsPx(st, p);
      coalescedDrag(e, (dx) => {
        const snapped = snapPx(baseLeft + dx, anchors, 10);
        const sec = Math.max(0, snapped / pxPerSec);
        block.style.left = (sec * pxPerSec) + "px";
        startIn.value = sec.toFixed(1);
      }, () => {
        S.updateAudioTrack(track.id, { start_sec: parseFloat(startIn.value || "0") });
      });
    });
    body.append(block);
    lane.append(body);
    return lane;
  }

  function overlayBlock(st, track, laneId) {
    const startSec = track.start_sec || 0;
    const durSec = Math.max(0.25, track.duration_sec || 3);
    const w = Math.max(durSec * pxPerSec, 40);
    const block = el("div", "tl-ov-clip" + (S.get().selectedOverlayId === track.id ? " selected" : "") + (track.kind === "text" ? " text" : " image"));
    block.style.left = (startSec * pxPerSec) + "px";
    block.style.width = w + "px";
    const label = track.kind === "text"
      ? (track.text || "Text")
      : ((st.mediaBin || []).find((m) => m.id === track.media_ref)?.name || track.label || "Image");
    const body = el("div", "tl-ov-body");
    body.append(el("span", "tl-ov-label", label.length > 18 ? label.slice(0, 17) + "…" : label));
    block.append(body);
    block.title = "Double-click to edit · drag edges to trim";
    block.onclick = (e) => { e.stopPropagation(); S.selectOverlay(track.id); };
    block.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      openOverlaySettingsModal(st, track);
    });

    const rm = el("button", "tl-ov-rm", "✕");
    rm.type = "button";
    rm.title = "Remove overlay (Delete)";
    rm.onclick = (e) => { e.stopPropagation(); S.removeOverlayTrack(track.id, { skipConfirm: true }); };
    rm.onmousedown = (e) => e.stopPropagation();
    block.append(rm);

    const leftTrim = el("div", "tl-ov-trim left");
    leftTrim.title = "Drag to trim start";
    leftTrim.addEventListener("mousedown", (e) => {
      e.stopPropagation();
      block.classList.add("trimming");
      const tip = el("div", "trim-tip"); block.append(tip);
      const baseLeft = startSec * pxPerSec;
      const baseDur = durSec;
      const anchors = timelineSnapAnchorsPx(st, st.project);
      let finalStart = startSec;
      let finalDur = durSec;
      coalescedDrag(e, (dx) => {
        const snappedLeft = snapPx(baseLeft + dx, anchors, 10);
        const deltaSec = (snappedLeft - baseLeft) / pxPerSec;
        finalStart = Math.max(0, startSec + deltaSec);
        finalDur = Math.max(0.25, durSec - (finalStart - startSec));
        block.style.left = (finalStart * pxPerSec) + "px";
        block.style.width = Math.max(finalDur * pxPerSec, 40) + "px";
        tip.textContent = `${finalStart.toFixed(1)}s · ${finalDur.toFixed(1)}s`;
      }, () => {
        block.classList.remove("trimming"); tip.remove();
        S.updateOverlayTrack(track.id, { start_sec: finalStart, duration_sec: finalDur });
      });
    });

    const rightTrim = el("div", "tl-ov-trim right");
    rightTrim.title = "Drag to extend or shorten";
    rightTrim.addEventListener("mousedown", (e) => {
      e.stopPropagation();
      block.classList.add("trimming");
      const tip = el("div", "trim-tip"); block.append(tip);
      const baseLeft = startSec * pxPerSec;
      const baseDur = durSec;
      const anchors = timelineSnapAnchorsPx(st, st.project);
      let finalDur = durSec;
      coalescedDrag(e, (dx) => {
        const rightPx = snapPx(baseLeft + baseDur * pxPerSec + dx, anchors, 10);
        finalDur = Math.max(0.25, (rightPx - baseLeft) / pxPerSec);
        block.style.width = Math.max(finalDur * pxPerSec, 40) + "px";
        tip.textContent = `${finalDur.toFixed(1)}s`;
      }, () => {
        block.classList.remove("trimming"); tip.remove();
        S.updateOverlayTrack(track.id, { duration_sec: finalDur });
      });
    });

    block.append(leftTrim, rightTrim);

    block.addEventListener("mousedown", (e) => {
      if (e.target.closest(".tl-ov-trim")) return;
      e.stopPropagation();
      S.selectOverlay(track.id);
      const baseLeft = startSec * pxPerSec;
      const anchors = timelineSnapAnchorsPx(st, st.project);
      document.querySelectorAll(".tl-overlay-lane").forEach((l) => l.classList.remove("drop-target"));
      coalescedDrag(e, (dx, moveEv) => {
        const snapped = snapPx(baseLeft + dx, anchors, 10);
        const sec = Math.max(0, snapped / pxPerSec);
        block.style.left = (sec * pxPerSec) + "px";
        if (moveEv) {
          document.querySelectorAll(".tl-overlay-lane").forEach((l) => l.classList.remove("drop-target"));
          const laneEl = document.elementFromPoint(moveEv.clientX, moveEv.clientY)?.closest?.(".tl-overlay-lane");
          if (laneEl) laneEl.classList.add("drop-target");
        }
      }, (dx, upEv) => {
        const snapped = snapPx(baseLeft + dx, anchors, 10);
        S.updateOverlayTrack(track.id, { start_sec: Math.max(0, snapped / pxPerSec) });
        if (upEv) {
          const hit = document.elementFromPoint(upEv.clientX, upEv.clientY);
          const laneEl = hit?.closest?.(".tl-overlay-lane");
          const newLaneId = laneEl?.dataset?.laneId;
          if (newLaneId && newLaneId !== laneId) {
            S.updateOverlayTrack(track.id, { lane_id: newLaneId }, true);
          }
        }
        document.querySelectorAll(".tl-overlay-lane").forEach((l) => l.classList.remove("drop-target"));
      });
    });

    return block;
  }

  function gutterLane(label, title, kind, heightPx) {
    const lane = el("div", "tl-gutter-lane " + kind, label);
    lane.title = title || label;
    if (heightPx) {
      lane.style.height = heightPx + "px";
      lane.style.minHeight = heightPx + "px";
      lane.style.flex = `0 0 ${heightPx}px`;
    }
    return lane;
  }

  function overlayGutterLane(st, lane, i, laneCount) {
    const top = i === laneCount - 1;
    const bottom = i === 0;
    const label = laneCount > 1 ? `O${i + 1}` : "Overlay";
    const hint = top ? "top layer" : bottom ? "bottom layer" : "middle layer";
    const row = el("div", "tl-gutter-lane overlay" + (top ? " top" : ""));
    row.title = `${lane.label || "Overlay"} · ${hint}`;
    row.style.height = OVERLAY_LANE_H + "px";
    row.style.minHeight = OVERLAY_LANE_H + "px";
    row.style.flex = `0 0 ${OVERLAY_LANE_H}px`;
    row.append(el("span", "tl-gutter-ov-label", label));
    if (laneCount > 1) {
      const rm = el("button", "tl-gutter-ov-rm", "✕");
      rm.type = "button";
      rm.title = "Remove this overlay track";
      rm.onclick = (e) => { e.stopPropagation(); S.removeOverlayLane(lane.id); };
      row.append(rm);
    }
    return row;
  }

  function timelineGutter(st, p) {
    const gutter = el("div", "tl-gutter");
    gutter.style.width = GUTTER_W + "px";
    gutter.append(el("div", "tl-gutter-ruler"));
    const gTracks = el("div", "tl-gutter-tracks");
    gTracks.append(gutterLane("Video", "Video", "video", VIDEO_LANE_H));
    const gAud = el("div", "tl-gutter-aud");
    gAud.append(gutterLane("Original", "Original audio from generated clips and video", "audio orig", AUDIO_LANE_H));
    (p.audio_tracks || []).forEach((t) => {
      let name, title;
      const isSep = S.isSeparatedAudioTrack ? S.isSeparatedAudioTrack(t) : (t.kind === "separated" && t.scene_id);
      const isOverlay = S.isOverlayAudioTrack ? S.isOverlayAudioTrack(t) : (!isSep && t.media_ref);
      if (isSep) {
        name = t.label || "Separated";
        title = `${name} (detached clip audio)`;
      } else if (isOverlay) {
        const asset = (st.mediaBin || []).find((m) => m.id === t.media_ref);
        name = t.label || (asset && asset.name) || "Overlay";
        title = `${name} — mixed on top of original audio`;
      } else {
        name = t.label || "Audio";
        title = name;
      }
      const short = name.length > 9 ? name.slice(0, 8) + "…" : name;
      gAud.append(gutterLane(short, title, "audio" + (isOverlay ? " overlay" : isSep ? " sep" : ""), AUDIO_LANE_H));
    });
    gTracks.append(gAud);
    S.ensureOverlayLanes();
    const ovLanes = p.overlay_lanes || [];
    const gOv = el("div", "tl-gutter-ov");
    ovLanes.forEach((lane, i) => gOv.append(overlayGutterLane(st, lane, i, ovLanes.length)));
    gTracks.append(gOv);
    gutter.append(gTracks);
    return gutter;
  }

  function audioLanes(st, p, lay) {
    const wrap = el("div", "tl-audio-lanes");
    const origLane = el("div", "tl-audio-lane orig-lane"); origLane.style.height = AUDIO_LANE_H + "px";
    const origBody = el("div", "tl-audio-lane-body");
    lay.forEach(({ sc, o, d }, i) => origBody.append(sceneAudioClip(st, p, sc, i, o * pxPerSec, d * pxPerSec)));
    origLane.append(origBody);
    wrap.append(origLane);
    (p.audio_tracks || []).forEach((t) => wrap.append(insertedAudioLane(st, p, t, AUDIO_LANE_H)));
    return wrap;
  }

  function attachOverlayLaneDrop(laneEl, body, laneId) {
    body.addEventListener("dragover", (e) => {
      if (e.dataTransfer.types.includes("application/funpack-media")) {
        e.preventDefault();
        e.dataTransfer.dropEffect = "copy";
        laneEl.classList.add("drop-target");
      }
    });
    body.addEventListener("dragleave", (e) => {
      if (!body.contains(e.relatedTarget)) laneEl.classList.remove("drop-target");
    });
    body.addEventListener("drop", (e) => {
      laneEl.classList.remove("drop-target");
      const id = e.dataTransfer.getData("application/funpack-media");
      if (!id) return;
      e.preventDefault();
      e.stopPropagation();
      S.assignMediaToOverlayLane(laneId, id);
    });
  }

  function overlayLaneEl(st, p, lane, laneIndex, laneCount) {
    const laneEl = el("div", "tl-overlay-lane" + (laneIndex === laneCount - 1 ? " top" : laneIndex === 0 ? " bottom" : ""));
    laneEl.style.height = OVERLAY_LANE_H + "px";
    laneEl.style.minHeight = OVERLAY_LANE_H + "px";
    laneEl.style.flex = `0 0 ${OVERLAY_LANE_H}px`;
    laneEl.dataset.laneId = lane.id;
    const body = el("div", "tl-overlay-lane-body");
    const tracks = (p.overlay_tracks || []).filter((t) => t.lane_id === lane.id);
    tracks.forEach((t) => body.append(overlayBlock(st, t, lane.id)));
    attachOverlayLaneDrop(laneEl, body, lane.id);
    laneEl.append(body);
    return laneEl;
  }

  function overlayLanes(st, p) {
    S.ensureOverlayLanes();
    const lanes = p.overlay_lanes || [];
    const wrap = el("div", "tl-overlay-lanes");
    lanes.forEach((lane, i) => wrap.append(overlayLaneEl(st, p, lane, i, lanes.length)));
    return wrap;
  }

  // ── toolbar ─────────────────────────────────────────────────────────────────────
  function toolbarRatingBlock(st, p) {
    const wrap = el("div", "tl-rating-block");
    const studioOn = window.PipelineCaps?.usesFunpackStudio(st);
    const hasSel = !!st.selectedSceneId;
    const Picker = window.MovieRatingPicker;
    if (!studioOn || !hasSel || !hasRender(st, st.selectedSceneId) || !Picker) return wrap;
    const sc = S.scene(st.selectedSceneId);
    if (!sc || !S.isGenerativeScene(sc)) return wrap;
    const sceneNo = p.scenes.indexOf(sc) + 1;
    const raw = sceneRatingRaw(st, sc);
    const rlabel = el("span", "tl-keys", `Scene ${sceneNo}`);
    const btn = el("button", "btn ghost tiny tl-rate-btn" + (raw ? " has-rating" : ""), Picker.buttonLabel(raw));
    btn.title = "Rate this scene's render — FunPack Studio refines from it on next generation";
    btn.onclick = (e) => {
      e.stopPropagation();
      Picker.open(e, raw, (label) => {
        const val = label === Picker.FORGET_LABEL ? "" : label;
        S.setSceneRating(sc.id, val);
      });
    };
    wrap.append(rlabel);
    wrap.append(btn);
    return wrap;
  }

  function toolbar(st, p, totalSec) {
    const bar = el("div", "tl-toolbar");
    const ph = window.Player?.getPlayhead() ?? 0;
    const tc = el("span", "tl-tc", timecode(Math.min(ph, totalSec), p.frame_rate) + " / " + timecode(totalSec, p.frame_rate));
    tlTcEl = tc;  // keep ref for Player's onPlayheadChanged updates
    bar.append(tc);
    bar.append(addMenuDropdown(st, p));

    // Clip actions on the selected clip (also bound to S / Delete).
    const { ids, hasSel, focusId, saveable, saveableN } = clipToolbarState(st);
    const audioSel = st.selectedAudioTrackId;
    const canSplit = hasSel || audioSel;
    const canRemove = hasSel || audioSel;
    const selN = ids.length;
    const { exportTitle, binTitle } = exportSaveTitles(hasSel, saveable, saveableN, selN);
    const selBadge = el("span", "tl-sel-badge");
    selBadge.dataset.tlSelBadge = "1";
    selBadge.hidden = selN === 0;
    if (selN > 0) selBadge.textContent = selN === 1 ? "1 clip selected" : `${selN} clips selected`;
    const split = el("button", "btn ghost tiny", "Split");
    split.dataset.needsSel = "1";
    split.title = audioSel
      ? "Split selected audio at the playhead (S)"
      : (hasSel ? "Split the selected clip at the playhead (S)" : "Select a clip or audio track first");
    split.disabled = !canSplit;
    split.onclick = () => splitSelectedAtPlayhead();
    const del = el("button", "btn ghost tiny danger", "Remove");
    del.dataset.needsSel = "1";
    del.title = audioSel
      ? "Remove selected audio track (Delete / Backspace)"
      : (hasSel ? "Remove selected clip(s) (Delete / Backspace)" : "Select a clip or audio track first");
    del.disabled = !canRemove;
    del.onclick = () => {
      if (audioSel) S.removeAudioTrack(audioSel);
      else S.removeSelectedScenes();
    };
    const exp = el("button", "btn ghost tiny", "⤓ Export");
    exp.dataset.tour = "export-scene";
    exp.dataset.exportScene = "1";
    exp.title = exportTitle;
    exp.disabled = !(hasSel && saveable);
    exp.onclick = () => S.exportSelected();
    const saveBin = el("button", "btn ghost tiny", "Save to media bin");
    saveBin.dataset.saveMediabin = "1";
    saveBin.title = binTitle;
    saveBin.disabled = !(hasSel && saveable);
    saveBin.onclick = () => S.saveSelectedToMediaBin();
    const sepAud = el("button", "btn ghost tiny", "⊟ Separate audio");
    sepAud.dataset.needsSel = "1";
    sepAud.dataset.separateAudio = "1";
    sepAud.title = "Detach this clip's audio onto its own track (video keeps picture only)";
    const selSc = focusId ? S.scene(focusId) : null;
    sepAud.disabled = !(selSc && S.sceneHasEmbeddedAudio?.(selSc));
    sepAud.onclick = () => { if (focusId) S.separateSceneAudio(focusId); };
    const sepTrack = selSc && S.separatedTrackForScene ? S.separatedTrackForScene(selSc.id) : null;
    const rmSepAud = el("button", "btn ghost tiny danger", "Remove audio");
    rmSepAud.dataset.removeSepAudio = "1";
    rmSepAud.title = "Remove the separated audio track for this clip";
    rmSepAud.disabled = !sepTrack;
    rmSepAud.onclick = () => { if (sepTrack) S.removeAudioTrack(sepTrack.id); };
    bar.append(split); bar.append(del); bar.append(selBadge); bar.append(exp); bar.append(saveBin); bar.append(sepAud); bar.append(rmSepAud);
    bar.append(toolbarRatingBlock(st, p));

    const spacer = el("div", "tl-spacer"); bar.append(spacer);
    const keys = el("span", "tl-keys", "J/K/L · S split · I/O in/out · +/- zoom");
    keys.title = IS_MAC
      ? "⌘-click toggles clip in selection · ⇧-click extends range · plain click replaces selection · S splits at playhead"
      : "Ctrl-click toggles clip in selection · Shift-click extends range · plain click replaces selection · S splits at playhead";
    bar.append(keys);
    const zlabel = el("span", "tl-zlabel", "zoom"); bar.append(zlabel);
    const zout = el("button", "btn ghost tiny", "−"); zout.onclick = () => setZoom(pxPerSec / 1.4, { manual: true });
    const zin = el("button", "btn ghost tiny", "＋"); zin.onclick = () => setZoom(pxPerSec * 1.4, { manual: true });
    const zfit = el("button", "btn ghost tiny", "fit"); zfit.onclick = () => fit(totalSec);
    bar.append(zout); bar.append(zin); bar.append(zfit);
    return bar;
  }
  let _autoFitEnabled = true;
  function setZoom(v, opts) {
    pxPerSec = Math.min(600, Math.max(8, v));
    localStorage.setItem(ZOOM_KEY, pxPerSec);
    if (opts?.manual) _autoFitEnabled = false;
    render(S.get());
  }
  function fit(totalSec) {
    _autoFitEnabled = true;
    const w = (body.querySelector(".tl-scroll")?.clientWidth || 800) - GUTTER_W - 8;
    if (totalSec > 0) setZoom(w / totalSec);
  }

  let _pendingAutoFit = false;
  window.Timeline = {
    requestAutoFit() { _pendingAutoFit = true; },
    fit: () => fit(S.previewTotalSec ? S.previewTotalSec() : tlTotalSec),
    syncClipRatings,
    openOverlaySettings: (track) => {
      if (!track) return;
      openOverlaySettingsModal(S.get(), track);
    },
  };

  // ── render ───────────────────────────────────────────────────────────────────────
  // Don't rebuild the timeline while the user is interacting with one of its controls
  // (e.g. the rating dropdown) — a store notify (autosave/progress) would close it.
  let _tlEditing = false;
  let _reordering = false;  // clip spacing drag in progress — never rebuild mid-drag
  let _seamDragging = false; // adjusting a video transition — never rebuild mid-drag

  function render(st) {
    if (_reordering || _seamDragging) return false;
    if (_isUserScrollingTimeline()) { _renderDeferred = true; return false; }
    _renderDeferred = false;
    if (_tlEditing) {
      // Only hold off if a control is genuinely still focused; otherwise the flag got
      // stuck (focused element removed without a focusout) — clear it and re-render.
      const a = document.activeElement;
      if (a && body.contains(a) && /^(SELECT|INPUT|TEXTAREA)$/.test(a.tagName)) return false;
      _tlEditing = false;
    }
    clear(body); clear(meta);
    if (!st.project) { body.append(el("div", "empty-stage", "Open a project to start cutting.")); return true; }
    const p = st.project;
    const scenes = p.scenes || [];
    const segs = S.buildPreviewSegments ? S.buildPreviewSegments() : [];
    const lay = segs.length
      ? segs.map((seg) => ({ seg, o: seg.offsetSec, d: seg.durationSec }))
      : scenes.map((sc) => { const d = sDur(sc, p); return { seg: { kind: "scene", scene: sc }, o: 0, d }; });
    const totalSec = S.previewTotalSec ? S.previewTotalSec() : lay.reduce((a, x) => a + x.d, 0);
    const sceneLay = lay.filter((x) => x.seg.kind === "scene").map((x) => ({ sc: x.seg.scene, o: x.o, d: x.d }));
    tlTotalSec = totalSec;
    tlFps = p.frame_rate;
    const contentW = Math.max(totalSec * pxPerSec + 40, 480);

    body.append(toolbar(st, p, totalSec));

    const scroll = el("div", "tl-scroll");
    _bindTimelineScroll(scroll);
    // Click empty timeline space (not a clip/seam) to clear the selection.
    scroll.addEventListener("click", (e) => {
      const cur = S.get();
      if ((cur.selectedSceneId || (cur.selectedSceneIds || []).length) && !e.target.closest(".clip") && !e.target.closest(".seam-cut") && !e.target.closest(".tl-ruler2") && !e.target.closest(".tl-aud-clip") && !e.target.closest(".tl-ov-clip"))
        S.selectScene(null);
      if (cur.selectedOverlayId && !e.target.closest(".tl-ov-clip")) S.selectOverlay(null);
      if (cur.selectedAudioTrackId && !e.target.closest(".tl-aud-clip.ins")) S.selectAudioTrack(null);
    });
    const stage = el("div", "tl-stage"); stage.style.width = (GUTTER_W + contentW) + "px";
    stage.append(timelineGutter(st, p));

    const content = el("div", "tl-content"); content.style.width = contentW + "px";

    // ruler
    const ruler = el("div", "tl-ruler2");
    const iv = tickInterval();
    for (let t = 0; t <= totalSec + iv; t += iv) {
      const tick = el("div", "tl-tick"); tick.style.left = (t * pxPerSec) + "px";
      tick.append(el("span", "tl-tick-label", rulerLabel(t, iv)));
      ruler.append(tick);
    }
    // Drag-to-scrub on ruler: smooth seek without full re-render (Player listener
    // updates just the needle + timecode element on every mousemove).
    ruler.addEventListener("mousedown", (e) => {
      e.preventDefault();
      window.Player?.beginScrub?.();
      const scrub = (ev) => {
        const r = ruler.getBoundingClientRect();
        const sec = Math.max(0, Math.min((ev.clientX - r.left) / pxPerSec, totalSec));
        window.Player?.seek(sec);
      };
      scrub(e);
      const move = (ev) => scrub(ev);
      const up = () => {
        document.removeEventListener("mousemove", move);
        document.removeEventListener("mouseup", up);
        window.Player?.endScrub?.();
      };
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", up);
    });
    content.append(ruler);

    const tracks = el("div", "tl-tracks");
    const track = el("div", "tl-track2");
    track.style.height = VIDEO_LANE_H + "px";
    track.style.minHeight = VIDEO_LANE_H + "px";
    track.style.flex = `0 0 ${VIDEO_LANE_H}px`;
    track.addEventListener("dragover", (e) => {
      if (e.dataTransfer?.types?.includes("application/funpack-media")) {
        e.preventDefault();
        track.classList.add("drop-target-track");
      }
    });
    track.addEventListener("dragleave", () => track.classList.remove("drop-target-track"));
    track.addEventListener("drop", (e) => {
      track.classList.remove("drop-target-track");
      const id = e.dataTransfer?.getData("application/funpack-media");
      if (!id) return;
      const asset = (st.mediaBin || []).find((m) => m.id === id);
      if (asset?.kind === "video") {
        e.preventDefault();
        S.addVideoClip(id); // a finished video is a result — it belongs on the timeline
      }
      // Images are i2v ANCHORS — drop them on a Plan card (or the Plan strip) instead, so
      // the timeline (the rendered result) is never changed by setting up a generation.
    });
    lay.forEach(({ seg, o, d }) => {
      if (seg.kind === "gap") return;
      if (seg.kind === "ghost") track.append(ghostClipEl(st, p, seg.ghost, o * pxPerSec, d * pxPerSec));
      else track.append(clipEl(st, p, seg.scene, scenes.indexOf(seg.scene), scenes.length, o * pxPerSec, d * pxPerSec));
    });
    for (let i = 0; i < scenes.length - 1; i++) {
      const prevSeg = segs.find((s) => s.kind === "scene" && s.scene.id === scenes[i].id);
      if (prevSeg) track.append(seamEl(st, p, scenes[i], (prevSeg.offsetSec + prevSeg.durationSec) * pxPerSec));
    }
    if (!lay.length) track.append(el("div", "tl-emptyhint", "No clips yet — add one from the toolbar."));
    tracks.append(track);
    tracks.append(audioLanes(st, p, sceneLay));
    tracks.append(overlayLanes(st, p));
    const phSec = Math.min(window.Player?.getPlayhead() ?? 0, totalSec);
    tlPhEl = el("div", "tl-playhead"); tlPhEl.style.left = (phSec * pxPerSec) + "px"; tracks.append(tlPhEl);
    content.append(tracks);

    stage.append(content);
    scroll.append(stage);
    body.append(scroll);
    scroll.scrollLeft = scrollLeft;
    tlScrollEl = scroll;

    const active = scenes.filter((s) => !s.excluded).length;
    const ghosts = (st.sceneGhosts || []).length;
    const selN = S.selectedSceneCount ? S.selectedSceneCount() : selectedIds(st).length;
    const metaTxt = `${scenes.length} clips · ${active} active` + (ghosts ? ` · ${ghosts} ghost${ghosts > 1 ? "s" : ""}` : "") + (selN > 0 ? ` · ${selN} selected` : "") + ` · ${timecode(totalSec, p.frame_rate)}`;
    meta.append(el("span", null, metaTxt));
    if (st.notice) {
      const note = el("span", "tl-notice", st.notice);
      const dismiss = el("button", "tl-notice-dismiss", "✕");
      dismiss.title = "Dismiss";
      dismiss.onclick = (e) => { e.stopPropagation(); S.clearNotice(); };
      note.append(dismiss);
      meta.append(note);
    }
    return true;
  }

  let _lastDataFp = null;
  let _lastSelFp = null;
  let _lastRatFp = null;

  function _ratingsFingerprint(st) {
    if (!st.project) return "";
    return JSON.stringify((st.project.scenes || []).map((s) => {
      const root = S.genUnitRoot(S.genUnitId(s)) || s;
      return [s.id, (root.rating || s.rating || "")];
    }));
  }

  function onStore(st) {
    const fpData = window.ViewBus?.fingerprints?.fpTimelineData?.(st)
      ?? JSON.stringify(st.project?.id);
    const fpSel = window.ViewBus?.fingerprints?.fpTimelineSel?.(st)
      ?? JSON.stringify({ sel: st.selectedSceneId, sels: st.selectedSceneIds });
    const fpRat = _ratingsFingerprint(st);
    if (fpData !== _lastDataFp) {
      const ok = render(st);
      if (!ok) return false; // render declined (focus/drag/scroll) — retry on next notify
      _lastDataFp = fpData;
      _lastSelFp = fpSel;
      _lastRatFp = fpRat;
      if (_pendingAutoFit && st.project) {
        _pendingAutoFit = false;
        if (_autoFitEnabled) {
          const total = S.previewTotalSec ? S.previewTotalSec() : tlTotalSec;
          requestAnimationFrame(() => { if (total > 0) fit(total); });
        }
      }
      return true;
    }
    if (fpRat !== _lastRatFp) {
      _lastRatFp = fpRat;
      syncClipRatings(st);
      syncToolbarSelection(st);
    }
    if (fpSel !== _lastSelFp) {
      _lastSelFp = fpSel;
      syncClipSelection(st);
      syncToolbarSelection(st);
      syncMetaSelection(st);
    }
  }

  if (window.ViewBus) window.ViewBus.subscribeTimeline(onStore);
  else S.subscribe(onStore);

  // Pause rebuilds while a timeline control (rating/transition dropdown, trim input) is
  // focused; re-sync shortly after it loses focus.
  body.addEventListener("focusin", (e) => {
    const t = e.target.tagName;
    if (t === "SELECT" || t === "INPUT" || t === "TEXTAREA") _tlEditing = true;
  });
  body.addEventListener("focusout", (e) => {
    const t = e.target.tagName;
    if (!(t === "SELECT" || t === "INPUT" || t === "TEXTAREA")) return;
    _tlEditing = false;
    setTimeout(() => { if (!_tlEditing) render(S.get()); }, 60);
  });

  // ── keyboard: S = split selected clip at playhead, Del/Backspace = remove it ──
  function splitSelectedAtPlayhead() {
    const st = S.get();
    if (!st.project) return;
    const ph = window.Player?.getPlayhead() ?? 0;
    if (st.selectedAudioTrackId) {
      const aud = (st.project.audio_tracks || []).find((t) => t.id === st.selectedAudioTrackId);
      if (!aud) return;
      const start = aud.start_sec || 0;
      const dur = S.audioTrackDurSec ? S.audioTrackDurSec(aud) : (aud.source_dur || 0);
      const splitAt = (ph > start + 0.05 && ph < start + dur - 0.05) ? ph : (start + dur / 2);
      S.splitAudioTrack(st.selectedAudioTrackId, splitAt);
      return;
    }
    if (!st.selectedSceneId) return;
    const p = st.project;
    const seg = (S.buildPreviewSegments ? S.buildPreviewSegments() : []).find((s) => s.kind === "scene" && s.scene.id === st.selectedSceneId);
    if (!seg) return;
    const target = seg.scene;
    const off = seg.offsetSec;
    const fps = sFps(target, p), dur = sDur(target, p);
    // Split at the playhead when it's inside the clip; otherwise at the midpoint.
    const at = (ph > off + 0.05 && ph < off + dur - 0.05) ? Math.round((ph - off) * fps) : null;
    S.splitScene(target.id, at);
  }

  window.addEventListener("keydown", (e) => {
    const a = document.activeElement;
    // Inline numeric fields on clips (e.g. audio track "Start (s)") can keep keyboard focus
    // after a stray click without the user noticing they're "in" a text field. Letters like
    // i/o/s/Delete are meaningless inside a number input anyway, so blur it and let the
    // clip-level shortcut through rather than silently swallowing the keystroke.
    if (a && a.tagName === "INPUT" && a.type === "number" && a.closest(".tl-aud-clip, .tl-clip")) {
      a.blur();
    } else if (a && (a.tagName === "INPUT" || a.tagName === "TEXTAREA" || a.isContentEditable)) {
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const st = S.get();
    if (!st.project) return;
    const ph = window.Player?.getPlayhead() ?? 0;
    const fps = st.project.frame_rate || 25;
    if (e.key === "j" || e.key === "J") { e.preventDefault(); window.Player?.seek(Math.max(0, ph - 1)); return; }
    if (e.key === "k" || e.key === "K") { e.preventDefault(); window.Player?.pause?.(); return; }
    if (e.key === "l" || e.key === "L") { e.preventDefault(); window.Player?.play?.(); return; }
    if (e.key === "+" || e.key === "=") { e.preventDefault(); setZoom(pxPerSec * 1.2, { manual: true }); return; }
    if (e.key === "-" || e.key === "_") { e.preventDefault(); setZoom(pxPerSec / 1.2, { manual: true }); return; }
    if (e.key === " " || e.code === "Space") {
      e.preventDefault();
      if (window.Player?.isPlaying?.()) window.Player.pause();
      else window.Player?.play?.();
      return;
    }
    if (e.key === "ArrowLeft") { e.preventDefault(); window.Player?.seek(Math.max(0, ph - 1 / fps)); return; }
    if (e.key === "ArrowRight") { e.preventDefault(); window.Player?.seek(ph + 1 / fps); return; }
    if (e.key === "Delete" || e.key === "Backspace") {
      if (st.selectedOverlayId) {
        e.preventDefault();
        S.removeSelectedOverlay();
        return;
      }
      if (st.selectedAudioTrackId) {
        e.preventDefault();
        S.removeAudioTrack(st.selectedAudioTrackId);
        return;
      }
      if (selectedIds(st).length) {
        e.preventDefault();
        S.removeSelectedScenes();
        return;
      }
      return;
    }
    if (st.selectedAudioTrackId) {
      const aud = (st.project.audio_tracks || []).find((t) => t.id === st.selectedAudioTrackId);
      if (aud) {
        if (e.key === "i" || e.key === "I") {
          e.preventDefault();
          window.Player?.seek(aud.start_sec || 0);
          return;
        }
        if (e.key === "o" || e.key === "O") {
          e.preventDefault();
          const dur = S.audioTrackDurSec ? S.audioTrackDurSec(aud) : (aud.source_dur || 0);
          window.Player?.seek((aud.start_sec || 0) + dur);
          return;
        }
        if (e.key === "s" || e.key === "S") { e.preventDefault(); splitSelectedAtPlayhead(); return; }
      }
      return;
    }
    if (!st.selectedSceneId) return;
    if (e.key === "i" || e.key === "I") {
      e.preventDefault();
      const seg = (S.buildPreviewSegments ? S.buildPreviewSegments() : []).find((s) => s.kind === "scene" && s.scene.id === st.selectedSceneId);
      if (seg) window.Player?.seek(seg.offsetSec);
      return;
    }
    if (e.key === "o" || e.key === "O") {
      e.preventDefault();
      const seg = (S.buildPreviewSegments ? S.buildPreviewSegments() : []).find((s) => s.kind === "scene" && s.scene.id === st.selectedSceneId);
      if (seg) window.Player?.seek(seg.offsetSec + seg.durationSec);
      return;
    }
    if (e.key === "s" || e.key === "S") { e.preventDefault(); splitSelectedAtPlayhead(); }
  });

  // Update only the playhead needle + timecode during video playback.
  // Runs at video's timeupdate rate (~4–25Hz) — no full re-render needed.
  if (window.Player) {
    window.Player.onPlayheadChanged((sec, playing) => {
      const clamped = Math.min(Math.max(0, sec), tlTotalSec);
      if (tlPhEl) tlPhEl.style.left = (clamped * pxPerSec) + "px";
      if (tlTcEl) tlTcEl.textContent = timecode(clamped, tlFps) + " / " + timecode(tlTotalSec, tlFps);
      if (!playing) _playbackScrollFollow = true;
      // Auto-pan during playback only; manual scrollbar / wheel disables until pause.
      if (tlScrollEl && playing && _playbackScrollFollow && !_isUserScrollingTimeline()) {
        const phPx = clamped * pxPerSec;
        const vis0 = tlScrollEl.scrollLeft;
        const vis1 = vis0 + tlScrollEl.clientWidth;
        if (phPx > vis1 - 60 || phPx < vis0 + 20) {
          _setTimelineScrollLeft(Math.max(0, phPx - tlScrollEl.clientWidth / 3));
        }
      }
    });
  }
})();
