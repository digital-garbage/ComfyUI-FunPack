// Bottom zone: a real NLE timeline. Clips are laid out on a time axis (width =
// duration × zoom), with a HH:MM:SS ruler, a playhead, drag-to-trim edges (which
// recompute frame counts from duration × fps), split, and per-seam crossfades.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const body = document.getElementById("timeline-body");
  const meta = document.getElementById("timeline-meta");

  const SRC_ICON = { empty: "▦", image: "◐", generated_frame: "⛶", carry: "⇥", mixed: "◑", video: "▶", v2v: "⟳" };

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
    if (t === "image" || t === "mixed" || t === "generated_frame" || t === "video" || t === "v2v") return src.media_ref || null;
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
    head.append(el("span", "clip-src", SRC_ICON[srcType] || "▦"));
  }

  function appendClipHeadBar(head, durSec, fps, onRemove, removeTitle) {
    const bar = el("div", "clip-head-bar");
    bar.append(el("span", "clip-dur", timecode(durSec, fps)));
    const rm = el("button", "clip-rm btn ghost tiny danger", "Remove");
    rm.type = "button";
    rm.title = removeTitle || "Remove clip (Delete / Backspace)";
    rm.onclick = (e) => { e.stopPropagation(); onRemove(); };
    bar.append(rm);
    head.append(bar);
  }

  const ZOOM_KEY = "funpack_tl_zoom";
  const GUTTER_W = 56;  // sticky lane labels (Video / Audio) — shared time origin in tl-content
  const AUDIO_LANE_H = 44;

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

  // Refs updated on each render so the Player playhead listener can update
  // just the needle and timecode without a full re-render.
  let tlPhEl = null;
  let tlTcEl = null;
  let tlScrollEl = null;
  let tlTotalSec = 0;
  let tlFps = 25;

  // ── helpers ──────────────────────────────────────────────────────────────────
  const sFps = (sc, p) => ((sc.fps_mode !== "project" && sc.fps != null) ? sc.fps : p.frame_rate) || 25;
  const sFrames = (sc, p) => ((sc.frames_mode !== "project" && sc.frames != null) ? sc.frames : p.num_frames_per_scene) || 1;
  const sDur = (sc, p) => sFrames(sc, p) / sFps(sc, p);

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

  function coalescedDrag(startEvt, onMove, onUp) {
    window.EditorHistory?.beginCoalesce(S);
    onDrag(startEvt, onMove, (dx) => {
      try { if (onUp) onUp(dx); } finally { window.EditorHistory?.endCoalesce(S); }
    });
  }

  function appendFilmstrip(clip, st, scene, widthPx) {
    if (!hasRender(st, scene.id)) return;
    if (clip.querySelector(".clip-filmstrip")) return;
    const r = st.sceneRenders[scene.id];
    if (!r?.media || widthPx < 40) return;
    const strip = el("div", "clip-filmstrip");
    clip.append(strip);
    const n = Math.min(10, Math.max(3, Math.floor(widthPx / 28)));
    const url = window.MovieEditorAPI.resultUrl(st.project.id, r.media);
    const vid = document.createElement("video");
    vid.muted = true; vid.preload = "auto"; vid.src = url;
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
      const dur = vid.duration || sDur(scene, st.project);
      vid.currentTime = Math.max(0, Math.min(time, dur - 0.01));
    });
    vid.onloadeddata = async () => {
      if (!strip.isConnected) return;
      const dur = vid.duration || sDur(scene, st.project);
      const inSec = (r.inSec || 0) + (scene.source_in || 0);
      for (let i = 0; i < n; i++) {
        if (!strip.isConnected) return;
        const t = inSec + (dur * (i + 0.5) / n);
        await captureAt(t);
      }
    };
  }

  const hasRender = (st, sceneId) => !!(st.sceneRenders && st.sceneRenders[sceneId] && st.sceneRenders[sceneId].media);
  const REORDER_PX = 12;

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
    S.selectScene(sceneId, { additive: e.metaKey || e.ctrlKey, range: e.shiftKey });
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
        const anchor = clipEl.querySelector(".clip-chars") || clipEl.querySelector(".clip-text");
        if (anchor && anchor.nextSibling) clipEl.insertBefore(badge, anchor.nextSibling);
        else clipEl.append(badge);
      }
      badge.title = "Rated for FunPack Studio on next generation";
      badge.textContent = "★ " + label;
    });
  }

  function toolbarConvertButton(st) {
    const id = st.selectedSceneId;
    if (!id) return null;
    const sc = S.scene(id);
    if (!sc) return null;
    if (S.isVideoClip(sc)) {
      const btn = el("button", "btn ghost tiny", "Convert to scene");
      btn.dataset.convertClip = "1";
      btn.title = sc.scene_archive
        ? "Restore prompt, source, guides, and settings from before this was locked as video"
        : "Make this a generative v2v scene (prompt + Generate)";
      btn.onclick = () => S.convertToScene(sc.id);
      return btn;
    }
    if (S.isGenerativeScene(sc)) {
      const btn = el("button", "btn ghost tiny", "Convert to video");
      btn.dataset.convertClip = "1";
      btn.title = "Lock as a plain video clip — skipped by Generate; settings saved for Convert back to scene";
      btn.onclick = () => S.convertToVideo(sc.id);
      return btn;
    }
    return null;
  }

  function syncToolbarSelection(st) {
    const bar = body.querySelector(".tl-toolbar");
    if (!bar || !st.project) return;
    const hasSel = !!st.selectedSceneId;
    bar.querySelectorAll("[data-needs-sel]").forEach((b) => { b.disabled = !hasSel; });
    const exp = bar.querySelector("[data-export-scene]");
    if (exp) exp.disabled = !(hasSel && S.clipSaveableToMediaBin?.(st.selectedSceneId));
    const saveBin = bar.querySelector("[data-save-mediabin]");
    if (saveBin) saveBin.disabled = !(hasSel && S.clipSaveableToMediaBin?.(st.selectedSceneId));
    const sep = bar.querySelector("[data-separate-audio]");
    if (sep) {
      const sc = hasSel ? S.scene(st.selectedSceneId) : null;
      sep.disabled = !(sc && S.isGenerativeScene(sc) && hasRender(st, st.selectedSceneId) && !sc?.audio_separated);
    }
    const rmSep = bar.querySelector("[data-remove-sep-audio]");
    if (rmSep) {
      const sc = hasSel ? S.scene(st.selectedSceneId) : null;
      const sepTrack = sc && S.separatedTrackForScene ? S.separatedTrackForScene(sc.id) : null;
      rmSep.disabled = !sepTrack;
    }
    const oldConv = bar.querySelector("[data-convert-clip]");
    const freshConv = toolbarConvertButton(st);
    if (oldConv) {
      if (freshConv) oldConv.replaceWith(freshConv);
      else oldConv.remove();
    } else if (freshConv) {
      const anchor = bar.querySelector("[data-separate-audio]") || bar.querySelector("[data-remove-sep-audio]") || bar.querySelector("[data-save-mediabin]") || bar.querySelector("[data-export-scene]");
      if (anchor?.nextSibling) bar.insertBefore(freshConv, anchor.nextSibling);
      else if (anchor) anchor.after(freshConv);
      else bar.append(freshConv);
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
      + (selN > 1 ? ` · ${selN} selected` : "")
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
    const move = (e) => onMove(e.clientX - x0);
    const up = (e) => { document.removeEventListener("mousemove", move); document.removeEventListener("mouseup", up); if (onUp) onUp(e.clientX - x0); };
    document.addEventListener("mousemove", move);
    document.addEventListener("mouseup", up);
  }

  // ── drag-reorder: where in the track does the cursor want to drop? ───────────────
  // Returns {idx, x}: idx = post-removal insertion index among non-dragged clips; x =
  // the boundary's x relative to the track (for the insertion line).
  function _dropTarget(track, dragged, clientX) {
    const others = [...track.querySelectorAll(".clip:not(.ghost)")].filter((c) => c !== dragged);
    const tr = track.getBoundingClientRect();
    let idx = 0;
    for (const c of others) {
      const r = c.getBoundingClientRect();
      if (clientX > r.left + r.width / 2) idx++; else break;
    }
    let x;
    if (!others.length) x = 0;
    else if (idx === 0) x = others[0].getBoundingClientRect().left - tr.left;
    else x = others[idx - 1].getBoundingClientRect().right - tr.left;
    return { idx, x };
  }

  // Removed scene — preview-only ghost (still plays last render until regen).
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
  function clipEl(st, p, scene, index, leftPx, widthPx) {
    const unitCuts = (p.scenes || []).filter((s) => (s.gen_unit_id || s.id) === (scene.gen_unit_id || scene.id)).length;
    const subclip = (scene.cut_offset_frames || 0) > 0;
    const src = sceneSourceForClip(scene, p);
    const srcType = src.type || "empty";
    const clip = el("div", "clip" + clipSelClass(st, scene.id)
      + (S.isVideoClip(scene) ? " clip-video" : "")
      + (scene.excluded ? " excluded" : "")
      + (hasRender(st, scene.id) ? " rendered" : (!scene.excluded && S.isGenerativeScene(scene) ? " pending" : ""))
      + (hasRender(st, scene.id) && S.renderIsStale?.(scene.id) ? " stale-render" : "")
      + (unitCuts > 1 ? " gen-cut" : "") + (subclip ? " subclip" : "")
      + (srcType === "mixed" ? " src-mixed" : srcType === "carry" ? " src-carry" : srcType === "image" ? " src-image" : srcType === "video" ? " src-video" : srcType === "v2v" ? " src-v2v" : ""));
    clip.dataset.sceneId = scene.id;
    clip.style.left = leftPx + "px";
    clip.style.width = Math.max(widthPx, 8) + "px";

    // drag the clip body left/right to reorder it on the timeline (a small threshold keeps
    // plain clicks = select; trim handle / action buttons opt out).
    clip.addEventListener("mousedown", (e) => {
      if (e.button !== 0 || e.target.closest(".clip-head-bar, .clip-rm, .clip-trim, .clip-vt-tail, .seam-cut, button")) return;
      const startX = e.clientX;
      let dragging = false, drop = null;
      const track = clip.parentNode;
      const onMove = (ev) => {
        const dx = ev.clientX - startX;
        if (!dragging) {
          if (Math.abs(dx) < REORDER_PX) return;
          dragging = true; clip.classList.add("reordering"); document.body.classList.add("tl-reordering");
          _reordering = true;
          window.EditorHistory?.beginCoalesce(S);
        }
        clip.style.transform = `translateX(${dx}px)`;
        drop = _dropTarget(track, clip, ev.clientX);
        const stNow = S.get();
        const pNow = stNow.project;
        if (pNow) drop.x = snapPx(drop.x, timelineSnapAnchorsPx(stNow, pNow), 10);
        let line = track.querySelector(".tl-dropline");
        if (!line) { line = el("div", "tl-dropline"); track.append(line); }
        line.style.left = drop.x + "px";
      };
      const onUp = (ev) => {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        document.body.classList.remove("tl-reordering");
        const line = track.querySelector(".tl-dropline"); if (line) line.remove();
        const dx = ev.clientX - startX;
        if (!dragging) return;
        clip.style.transform = ""; clip.classList.remove("reordering");
        _reordering = false;
        if (Math.abs(dx) < REORDER_PX) {
          window.EditorHistory?.endCoalesce(S);
          return;
        }
        ev.preventDefault();
        if (drop) S.moveSceneTo(scene.id, drop.idx);
        window.EditorHistory?.endCoalesce(S);
        track.addEventListener("click", (ce) => { ce.stopPropagation(); ce.preventDefault(); }, { capture: true, once: true });
      };
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    });

    // accept a media asset dragged from the bin → sets this clip's source
    clip.addEventListener("dragover", (e) => { if (e.dataTransfer.types.includes("application/funpack-media")) { e.preventDefault(); clip.classList.add("drop-target"); } });
    clip.addEventListener("dragleave", () => clip.classList.remove("drop-target"));
    clip.addEventListener("drop", (e) => {
      const id = e.dataTransfer.getData("application/funpack-media");
      clip.classList.remove("drop-target");
      if (id) { e.preventDefault(); S.assignMediaToScene(scene.id, id); }
    });

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
    appendClipHeadBar(head, sDur(scene, p), sFps(scene, p), () => S.removeScene(scene.id));
    clip.append(head);

    const root = unitCuts > 1
      ? (p.scenes || []).find((s) => (s.gen_unit_id || s.id) === (scene.gen_unit_id || scene.id) && !(s.cut_offset_frames > 0))
      : null;
    const label = S.isVideoClip(scene)
      ? ((mref && (st.mediaBin || []).find((m) => m.id === mref)?.name) || "Video clip")
      : (scene.text || (root && root.text) || (subclip ? "cut" : "empty scene"));
    clip.append(el("div", "clip-text" + (label && label !== "empty scene" && label !== "cut" && label !== "Video clip" ? "" : " empty"), label));

    const charIds = S.sceneCharacterIds(scene.id);
    if (charIds.length) {
      const chars = el("div", "clip-chars");
      charIds.forEach((cid) => {
        const c = (st.characters || []).find((x) => x.id === cid);
        chars.append(el("span", "clip-char", c?.name || cid));
      });
      clip.append(chars);
    }

    const rating = sceneRatingDisplay(st, scene);
    if (rating) {
      const rated = el("div", "clip-rated");
      rated.title = "Rated for FunPack Studio conditioning on next generation";
      rated.textContent = "★ " + rating;
      clip.append(rated);
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
          const snappedLeft = snapPx(leftPx + dx, anchors, 10);
          finalDelta = (snappedLeft - leftPx) / pxPerSec;
          tip.textContent = e.altKey && hasRender(st, scene.id)
            ? `slip ${finalDelta >= 0 ? "+" : ""}${finalDelta.toFixed(2)}s`
            : `trim ${(-finalDelta).toFixed(2)}s`;
        }, () => {
          clip.classList.remove("trimming"); tip.remove();
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
      ? "Length is Custom — change it in the inspector (set Frames to Inherit timeline to trim here)"
      : "Drag to trim · length = duration × fps";
    const baseDur = sDur(scene, p);
    if (locked) { clip.append(handle); return clip; }
    handle.addEventListener("mousedown", (e) => {
      clip.classList.add("trimming");
      const tip = el("div", "trim-tip"); clip.append(tip);
      const anchors = timelineSnapAnchorsPx(st, p);
      let finalDur = baseDur;
      coalescedDrag(e, (dx) => {
        const rightPx = snapPx(leftPx + baseDur * pxPerSec + dx, anchors, 10);
        finalDur = Math.max(0.1, (rightPx - leftPx) / pxPerSec);
        const w = finalDur * pxPerSec;
        clip.style.width = Math.max(w, 8) + "px";
        const fps = sFps(scene, p);
        tip.textContent = `${timecode(finalDur, fps)} · ${S.snapFrames(finalDur * fps)}f`;
      }, () => {
        clip.classList.remove("trimming"); tip.remove();
        S.resizeScene(scene.id, finalDur);
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
    addRow("Text", "Coming soon", null, true);
    addRow("Image", "Coming soon", null, true);
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
    const clip = el("div", "tl-aud-clip scene-aud" + clipSelClass(st, scene.id));
    clip.dataset.sceneId = scene.id;
    clip.style.left = leftPx + "px";
    clip.style.width = w + "px";
    clip.style.maxWidth = w + "px";
    clip.onclick = (e) => { if (e.target.closest(".tl-aud-controls")) return; e.stopPropagation(); onClipSelect(e, scene.id); };

    const wave = el("div", "tl-aud-wave");
    const canvas = el("canvas");
    wave.append(canvas);
    clip.append(wave);

    const r = (st.sceneRenders || {})[scene.id];
    if (r?.media && st.project?.id) {
      _attachWaveform(canvas, `scene-aud:${scene.id}`, window.MovieEditorAPI.resultUrl(st.project.id, r.media), {
        width: w,
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
    const body = el("div", "tl-audio-lane-body");
    const startSec = track.start_sec || 0;
    let durSec = isSep
      ? (S.separatedTrackDurSec ? S.separatedTrackDurSec(track) : (track.pinned_dur || track.source_dur || 0))
      : (track.source_dur || asset?.duration_sec || 0);
    const w = Math.max((durSec || 2) * pxPerSec, 48);
    const block = el("div", "tl-aud-clip ins" + (isSep ? " sep" : " overlay"));
    block.style.left = (startSec * pxPerSec) + "px";
    block.style.width = w + "px";

    const wave = el("div", "tl-aud-wave");
    const canvas = el("canvas");
    wave.append(canvas);
    block.append(wave);
    if (isSep && sepScene) {
      const pinned = track.pinned_media;
      const r = (st.sceneRenders || {})[track.scene_id];
      const media = pinned || r?.media;
      if (media && st.project?.id) {
        const wfKey = `sep-aud:${track.id}:${media.filename || ""}:${track.pinned_in_sec ?? track.source_in_sec ?? 0}`;
        _attachWaveform(canvas, wfKey, window.MovieEditorAPI.resultUrl(st.project.id, media), {
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
      _attachWaveform(canvas, `aud:${asset.id}`, window.MovieEditorAPI.mediaUrl(asset.id), {
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
      _attachWaveform(canvas, `aud-render:${track.id}`, window.MovieEditorAPI.resultUrl(st.project.id, track.render_media), {
        width: w,
        color: "rgba(96, 165, 250, 0.55)",
      });
    }

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
      S.updateAudioTrack(track.id, { start_sec: v }, true);
      block.style.left = (v * pxPerSec) + "px";
    };
    startIn.onclick = (e) => e.stopPropagation();
    controls.append(startIn);
    const rm = el("button", "ic-btn danger tl-aud-rm", "✕");
    rm.title = isSep
      ? "Remove separated audio (restores embedded audio on the clip)"
      : "Remove overlay audio track";
    rm.onclick = (e) => { e.stopPropagation(); S.removeAudioTrack(track.id); };
    controls.append(rm);
    block.append(controls);

    block.addEventListener("mousedown", (e) => {
      if (e.target.closest("input,button,select")) return;
      e.stopPropagation();
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

  function gutterLane(label, title, kind) {
    const lane = el("div", "tl-gutter-lane " + kind, label);
    lane.title = title || label;
    return lane;
  }

  function timelineGutter(st, p) {
    const gutter = el("div", "tl-gutter");
    gutter.style.width = GUTTER_W + "px";
    gutter.append(el("div", "tl-gutter-ruler"));
    const gTracks = el("div", "tl-gutter-tracks");
    gTracks.append(gutterLane("Video", "Video", "video"));
    const gAud = el("div", "tl-gutter-aud");
    gAud.append(gutterLane("Original", "Original audio from generated clips and video", "audio orig"));
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
      gAud.append(gutterLane(short, title, "audio" + (isOverlay ? " overlay" : isSep ? " sep" : "")));
    });
    gTracks.append(gAud);
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

  // ── toolbar ─────────────────────────────────────────────────────────────────────
  function toolbarRatingBlock(st, p) {
    const wrap = el("div", "tl-rating-block");
    const studioCond = !st.project.conditioning_slot || st.project.conditioning_slot === "funpack";
    const hasSel = !!st.selectedSceneId;
    const Picker = window.MovieRatingPicker;
    if (!studioCond || !hasSel || !hasRender(st, st.selectedSceneId) || !Picker) return wrap;
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
    const selIds = selectedIds(st);
    const hasSel = selIds.length > 0;
    const split = el("button", "btn ghost tiny", "Split");
    split.dataset.needsSel = "1";
    split.title = "Split the selected clip at the playhead (S)"; split.disabled = !hasSel;
    split.onclick = () => splitSelectedAtPlayhead();
    const del = el("button", "btn ghost tiny danger", "Remove");
    del.dataset.needsSel = "1";
    del.title = "Remove selected clip(s) (Delete / Backspace)"; del.disabled = !hasSel;
    del.onclick = () => S.removeSelectedScenes();
    const exp = el("button", "btn ghost tiny", "⤓ Export");
    exp.dataset.exportScene = "1";
    exp.title = "Save the selected clip's rendered video to disk (renders are temporary)";
    exp.disabled = !(hasSel && S.clipSaveableToMediaBin?.(st.selectedSceneId));
    exp.onclick = () => S.exportSelected();
    const saveBin = el("button", "btn ghost tiny", "Save to media bin");
    saveBin.dataset.saveMediabin = "1";
    saveBin.title = "Copy the selected clip into the Media bin — add it back as a plain video clip";
    saveBin.disabled = !(hasSel && S.clipSaveableToMediaBin?.(st.selectedSceneId));
    saveBin.onclick = () => S.saveSelectedToMediaBin();
    const sepAud = el("button", "btn ghost tiny", "⊟ Separate audio");
    sepAud.dataset.needsSel = "1";
    sepAud.dataset.separateAudio = "1";
    sepAud.title = "Detach this clip's audio onto its own track (video keeps picture only)";
    const selSc = hasSel ? S.scene(st.selectedSceneId) : null;
    sepAud.disabled = !(selSc && hasRender(st, st.selectedSceneId) && !selSc?.audio_separated && S.isGenerativeScene(selSc));
    sepAud.onclick = () => S.separateSceneAudio(st.selectedSceneId);
    const sepTrack = selSc && S.separatedTrackForScene ? S.separatedTrackForScene(selSc.id) : null;
    const rmSepAud = el("button", "btn ghost tiny danger", "Remove audio");
    rmSepAud.dataset.removeSepAudio = "1";
    rmSepAud.title = "Remove the separated audio track for this clip";
    rmSepAud.disabled = !sepTrack;
    rmSepAud.onclick = () => { if (sepTrack) S.removeAudioTrack(sepTrack.id); };
    bar.append(split); bar.append(del); bar.append(exp); bar.append(saveBin); bar.append(sepAud); bar.append(rmSepAud);
    const conv = toolbarConvertButton(st);
    if (conv) bar.append(conv);
    bar.append(toolbarRatingBlock(st, p));

    const spacer = el("div", "tl-spacer"); bar.append(spacer);
    const keys = el("span", "tl-keys", "J/K/L · S split · I/O in/out · +/- zoom");
    keys.title = "⌘/Ctrl-click toggles selection · Shift-click extends range · S splits at playhead · Delete removes focus clip";
    bar.append(keys);
    const zlabel = el("span", "tl-zlabel", "zoom"); bar.append(zlabel);
    const zout = el("button", "btn ghost tiny", "−"); zout.onclick = () => setZoom(pxPerSec / 1.4);
    const zin = el("button", "btn ghost tiny", "＋"); zin.onclick = () => setZoom(pxPerSec * 1.4);
    const zfit = el("button", "btn ghost tiny", "fit"); zfit.onclick = () => fit(totalSec);
    bar.append(zout); bar.append(zin); bar.append(zfit);
    return bar;
  }
  function setZoom(v) { pxPerSec = Math.min(600, Math.max(8, v)); localStorage.setItem(ZOOM_KEY, pxPerSec); render(S.get()); }
  function fit(totalSec) {
    const w = (body.querySelector(".tl-scroll")?.clientWidth || 800) - GUTTER_W - 8;
    if (totalSec > 0) setZoom(w / totalSec);
  }

  let _pendingAutoFit = false;
  window.Timeline = {
    requestAutoFit() { _pendingAutoFit = true; },
    fit: () => fit(S.previewTotalSec ? S.previewTotalSec() : tlTotalSec),
    syncClipRatings,
  };

  // ── render ───────────────────────────────────────────────────────────────────────
  // Don't rebuild the timeline while the user is interacting with one of its controls
  // (e.g. the rating dropdown) — a store notify (autosave/progress) would close it.
  let _tlEditing = false;
  let _reordering = false;  // a clip is being drag-reordered — never rebuild mid-drag
  let _seamDragging = false; // adjusting a video transition — never rebuild mid-drag

  function render(st) {
    if (_reordering || _seamDragging) return false;
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
    scroll.addEventListener("scroll", () => { scrollLeft = scroll.scrollLeft; });
    // Click empty timeline space (not a clip/seam) to clear the selection.
    scroll.addEventListener("click", (e) => {
      const cur = S.get();
      if ((cur.selectedSceneId || (cur.selectedSceneIds || []).length) && !e.target.closest(".clip") && !e.target.closest(".seam-cut") && !e.target.closest(".tl-ruler2") && !e.target.closest(".tl-aud-clip"))
        S.selectScene(null);
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
      const scrub = (ev) => {
        const r = ruler.getBoundingClientRect();
        const sec = Math.max(0, Math.min((ev.clientX - r.left) / pxPerSec, totalSec));
        window.Player?.seek(sec);
      };
      scrub(e);
      const move = (ev) => scrub(ev);
      const up = () => { document.removeEventListener("mousemove", move); document.removeEventListener("mouseup", up); };
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", up);
    });
    content.append(ruler);

    const tracks = el("div", "tl-tracks");
    const track = el("div", "tl-track2");
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
        S.addVideoClip(id);
      }
    });
    track.addEventListener("click", (e) => {
      const clip = e.target.closest(".clip:not(.ghost)[data-scene-id]");
      if (!clip) return;
      if (e.target.closest(".clip-head-bar, .clip-rm, .clip-trim, .clip-vt-tail, .seam-cut, button")) return;
      onClipSelect(e, clip.dataset.sceneId);
    });
    lay.forEach(({ seg, o, d }) => {
      if (seg.kind === "ghost") track.append(ghostClipEl(st, p, seg.ghost, o * pxPerSec, d * pxPerSec));
      else track.append(clipEl(st, p, seg.scene, scenes.indexOf(seg.scene), o * pxPerSec, d * pxPerSec));
    });
    for (let i = 0; i < scenes.length - 1; i++) {
      const nextSeg = segs.find((s) => s.kind === "scene" && s.scene.id === scenes[i + 1].id);
      if (nextSeg) track.append(seamEl(st, p, scenes[i], nextSeg.offsetSec * pxPerSec));
    }
    if (!lay.length) track.append(el("div", "tl-emptyhint", "No clips yet — add one from the toolbar."));
    tracks.append(track);
    tracks.append(audioLanes(st, p, sceneLay));
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
    const metaTxt = `${scenes.length} clips · ${active} active` + (ghosts ? ` · ${ghosts} ghost${ghosts > 1 ? "s" : ""}` : "") + (selN > 1 ? ` · ${selN} selected` : "") + ` · ${timecode(totalSec, p.frame_rate)}`;
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
      _lastDataFp = fpData;
      _lastSelFp = fpSel;
      _lastRatFp = fpRat;
      const ok = render(st);
      if (_pendingAutoFit && ok && st.project) {
        _pendingAutoFit = false;
        const total = S.previewTotalSec ? S.previewTotalSec() : tlTotalSec;
        requestAnimationFrame(() => { if (total > 0) fit(total); });
      }
      return;
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
    if (!st.project || !st.selectedSceneId) return;
    const p = st.project;
    const seg = (S.buildPreviewSegments ? S.buildPreviewSegments() : []).find((s) => s.kind === "scene" && s.scene.id === st.selectedSceneId);
    if (!seg) return;
    const target = seg.scene;
    const off = seg.offsetSec;
    const ph = window.Player?.getPlayhead() ?? 0;
    const fps = sFps(target, p), dur = sDur(target, p);
    // Split at the playhead when it's inside the clip; otherwise at the midpoint.
    const at = (ph > off + 0.05 && ph < off + dur - 0.05) ? Math.round((ph - off) * fps) : null;
    S.splitScene(target.id, at);
  }

  window.addEventListener("keydown", (e) => {
    const a = document.activeElement;
    if (a && (a.tagName === "INPUT" || a.tagName === "TEXTAREA" || a.isContentEditable)) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const st = S.get();
    if (!st.project) return;
    const ph = window.Player?.getPlayhead() ?? 0;
    const fps = st.project.frame_rate || 25;
    if (e.key === "j" || e.key === "J") { e.preventDefault(); window.Player?.seek(Math.max(0, ph - 1)); return; }
    if (e.key === "k" || e.key === "K") { e.preventDefault(); window.Player?.pause?.(); return; }
    if (e.key === "l" || e.key === "L") { e.preventDefault(); window.Player?.play?.(); return; }
    if (e.key === "+" || e.key === "=") { e.preventDefault(); setZoom(pxPerSec * 1.2); return; }
    if (e.key === "-" || e.key === "_") { e.preventDefault(); setZoom(pxPerSec / 1.2); return; }
    if (e.key === "ArrowLeft") { e.preventDefault(); window.Player?.seek(Math.max(0, ph - 1 / fps)); return; }
    if (e.key === "ArrowRight") { e.preventDefault(); window.Player?.seek(ph + 1 / fps); return; }
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
    else if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault();
      if (selectedIds(st).length) S.removeSelectedScenes();
    }
  });

  // Update only the playhead needle + timecode during video playback.
  // Runs at video's timeupdate rate (~4–25Hz) — no full re-render needed.
  if (window.Player) {
    window.Player.onPlayheadChanged((sec) => {
      const clamped = Math.min(Math.max(0, sec), tlTotalSec);
      if (tlPhEl) tlPhEl.style.left = (clamped * pxPerSec) + "px";
      if (tlTcEl) tlTcEl.textContent = timecode(clamped, tlFps) + " / " + timecode(tlTotalSec, tlFps);
      // Auto-scroll to keep the playhead visible during playback
      if (tlScrollEl && window.Player.isPlaying()) {
        const phPx = clamped * pxPerSec;
        const vis0 = tlScrollEl.scrollLeft;
        const vis1 = vis0 + tlScrollEl.clientWidth;
        if (phPx > vis1 - 60 || phPx < vis0 + 20) {
          tlScrollEl.scrollLeft = Math.max(0, phPx - tlScrollEl.clientWidth / 3);
        }
      }
    });
  }
})();
