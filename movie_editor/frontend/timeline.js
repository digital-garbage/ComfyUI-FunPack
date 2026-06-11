// Bottom zone: a real NLE timeline. Clips are laid out on a time axis (width =
// duration × zoom), with a HH:MM:SS ruler, a playhead, drag-to-trim edges (which
// recompute frame counts from duration × fps), split, and per-seam crossfades.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const body = document.getElementById("timeline-body");
  const meta = document.getElementById("timeline-meta");

  const SRC_ICON = { empty: "▦", image: "◐", generated_frame: "⛶", carry: "⇥", mixed: "◑" };

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
    if (t === "image" || t === "mixed" || t === "generated_frame") return src.media_ref || null;
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
  const ZOOM_KEY = "funpack_tl_zoom";
  const GUTTER_W = 56;  // sticky lane labels (Video / Audio) — shared time origin in tl-content
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

  function appendFilmstrip(clip, st, scene, widthPx) {
    if (!hasRender(st, scene.id)) return;
    const r = st.sceneRenders[scene.id];
    if (!r?.media || widthPx < 40) return;
    const strip = el("div", "clip-filmstrip");
    const n = Math.min(10, Math.max(3, Math.floor(widthPx / 28)));
    const url = window.MovieEditorAPI.resultUrl(st.project.id, r.media);
    const vid = document.createElement("video");
    vid.muted = true; vid.preload = "auto"; vid.src = url;
    vid.onloadeddata = () => {
      const dur = vid.duration || sDur(scene, st.project);
      const inSec = (r.inSec || 0) + (scene.source_in || 0);
      for (let i = 0; i < n; i++) {
        const cell = el("div", "clip-fs-cell");
        const t = inSec + (dur * (i + 0.5) / n);
        const c = document.createElement("canvas");
        c.width = 48; c.height = 27;
        const ctx = c.getContext("2d");
        const thumb = el("img", "clip-fs-thumb");
        thumb.alt = "";
        const capture = (time) => {
          vid.currentTime = Math.min(time, dur - 0.01);
          vid.onseeked = () => {
            try { ctx.drawImage(vid, 0, 0, 48, 27); thumb.src = c.toDataURL("image/jpeg", 0.55); } catch (_) {}
            cell.append(thumb);
            strip.append(cell);
          };
        };
        capture(t);
      }
    };
    clip.append(strip);
  }

  const hasRender = (st, sceneId) => !!(st.sceneRenders && st.sceneRenders[sceneId] && st.sceneRenders[sceneId].media);

  function selectedIds(st) {
    return st.selectedSceneIds?.length ? st.selectedSceneIds : (st.selectedSceneId ? [st.selectedSceneId] : []);
  }
  function clipSelClass(st, sceneId) {
    const ids = selectedIds(st);
    let cls = "";
    if (ids.includes(sceneId)) cls += " selected";
    if (sceneId === st.selectedSceneId) cls += " focus";
    return cls;
  }
  function onClipSelect(e, sceneId) {
    S.selectScene(sceneId, { additive: e.metaKey || e.ctrlKey, range: e.shiftKey });
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

  const VT_TYPES = [
    ["", "Hard cut"],
    ["crossfade", "Dissolve"],
    ["fadeblack", "Fade black"],
    ["wipeleft", "Wipe left"],
    ["wiperight", "Wipe right"],
  ];
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

  function syncVtVisual(seam, bridge, type, frames) {
    const on = frames > 0 && !!type;
    bridge.classList.toggle("on", on);
    seam.classList.toggle("has-vt", on);
    ["crossfade", "fadeblack", "wipeleft", "wiperight"].forEach((t) => {
      seam.classList.remove("vt-" + t);
      bridge.classList.remove("vt-" + t);
    });
    if (on) {
      seam.classList.add("vt-" + type);
      bridge.classList.add("vt-" + type);
    }
  }

  // Prompt transition (Studio split marker) — separate from rendered video transition.
  function promptTransitionSelect(value, onChange) {
    const sel = el("select", "seam-prompt-type");
    const none = el("option", null, "default cut"); none.value = ""; sel.append(none);
    (S.get().transitions || []).forEach((t) => {
      const name = t.trigger || t.name || t.key; if (!name) return;
      const o = el("option", null, name); o.value = name; if (name === value) o.selected = true; sel.append(o);
    });
    if (value && ![...sel.options].some((o) => o.value === value)) { const o = el("option", null, value); o.value = value; o.selected = true; sel.append(o); }
    sel.onchange = (e) => { e.stopPropagation(); onChange(sel.value); };
    sel.onclick = (e) => e.stopPropagation();
    sel.title = "How generation divides before the next scene (prompt split marker — not a video dissolve). Edit the global prompt to set splits in bulk.";
    return sel;
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
    head.append(el("span", "clip-dur", timecode((frames || 1) / (fps || 25), fps || 25)));
    clip.append(head);
    const label = ghost.pendingGen ? "removed · generating…" : (ghost.text || "removed scene");
    clip.append(el("div", "clip-text ghost-label", label));
    const actions = el("div", "clip-actions");
    const dismiss = el("button", "ic-btn", "✕");
    dismiss.title = "Dismiss ghost from preview";
    dismiss.onclick = (e) => { e.stopPropagation(); S.dismissGhost(ghost.id); };
    actions.append(dismiss);
    clip.append(actions);
    return clip;
  }

  // ── clip ───────────────────────────────────────────────────────────────────────
  function clipEl(st, p, scene, index, leftPx, widthPx) {
    const unitCuts = (p.scenes || []).filter((s) => (s.gen_unit_id || s.id) === (scene.gen_unit_id || scene.id)).length;
    const subclip = (scene.cut_offset_frames || 0) > 0;
    const src = sceneSourceForClip(scene, p);
    const srcType = src.type || "empty";
    const clip = el("div", "clip" + clipSelClass(st, scene.id)
      + (scene.excluded ? " excluded" : "")
      + (hasRender(st, scene.id) ? " rendered" : (!scene.excluded ? " pending" : ""))
      + (unitCuts > 1 ? " gen-cut" : "") + (subclip ? " subclip" : "")
      + (srcType === "mixed" ? " src-mixed" : srcType === "carry" ? " src-carry" : srcType === "image" ? " src-image" : ""));
    clip.style.left = leftPx + "px";
    clip.style.width = Math.max(widthPx, 8) + "px";
    clip.onclick = (e) => onClipSelect(e, scene.id);

    // drag the clip body left/right to reorder it on the timeline (a small threshold keeps
    // plain clicks = select; trim handle / action buttons opt out).
    clip.addEventListener("mousedown", (e) => {
      if (e.button !== 0 || e.target.closest(".clip-actions, .clip-trim")) return;
      const startX = e.clientX;
      let dragging = false, drop = null;
      const track = clip.parentNode;
      const onMove = (ev) => {
        const dx = ev.clientX - startX;
        if (!dragging) {
          if (Math.abs(dx) < 5) return;
          dragging = true; clip.classList.add("reordering"); document.body.classList.add("tl-reordering");
          _reordering = true;  // hard-block store-driven rebuilds while dragging
        }
        clip.style.transform = `translateX(${dx}px)`;
        drop = _dropTarget(track, clip, ev.clientX);
        let line = track.querySelector(".tl-dropline");
        if (!line) { line = el("div", "tl-dropline"); track.append(line); }
        line.style.left = drop.x + "px";
      };
      const onUp = (ev) => {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        document.body.classList.remove("tl-reordering");
        const line = track.querySelector(".tl-dropline"); if (line) line.remove();
        if (dragging) {
          ev.preventDefault();
          clip.style.transform = ""; clip.classList.remove("reordering");
          _reordering = false;
          if (drop) S.moveSceneTo(scene.id, drop.idx);
          // swallow the click that fires right after the drag so it doesn't re-select
          clip.addEventListener("click", (ce) => { ce.stopPropagation(); ce.preventDefault(); }, { capture: true, once: true });
        }
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
    head.append(el("span", "clip-no", p2(index + 1)));
    appendSrcBadge(head, srcType);
    head.append(el("span", "clip-dur", timecode(sDur(scene, p), sFps(scene, p))));
    clip.append(head);

    const root = unitCuts > 1
      ? (p.scenes || []).find((s) => (s.gen_unit_id || s.id) === (scene.gen_unit_id || scene.id) && !(s.cut_offset_frames > 0))
      : null;
    const label = scene.text || (root && root.text) || (subclip ? "cut" : "empty scene");
    clip.append(el("div", "clip-text" + (label && label !== "empty scene" && label !== "cut" ? "" : " empty"), label));

    const charIds = S.sceneCharacterIds(scene.id);
    if (charIds.length) {
      const chars = el("div", "clip-chars");
      charIds.forEach((cid) => {
        const c = (st.characters || []).find((x) => x.id === cid);
        chars.append(el("span", "clip-char", c?.name || cid));
      });
      clip.append(chars);
    }

    const rating = ((root && root.rating) || scene.rating || "").trim();
    if (rating) {
      const rated = el("div", "clip-rated");
      rated.title = "Rated for FunPack Studio conditioning on next generation";
      rated.textContent = "Rated: " + rating;
      clip.append(rated);
    }

    const vt = videoTransitionState(scene, p);
    if (vt.active) {
      const tail = el("div", "clip-vt-tail vt-" + vt.type);
      tail.style.width = Math.max(10, vt.sec * pxPerSec) + "px";
      tail.title = `${VT_SHORT[vt.type] || vt.type} → next · ${vt.frames}f (${vt.sec.toFixed(2)}s)`;
      clip.append(tail);
    }

    if (hasRender(st, scene.id) && S.renderPromptMismatch) {
      const mismatch = S.renderPromptMismatch(scene.id);
      if (mismatch) {
        clip.classList.add("prompt-mismatch");
        const gen = el("div", "clip-gen-prompt");
        gen.title = "Timeline prompt was edited after generation — rate against this text";
        gen.append(el("span", "clip-gen-prompt-label", "Generated with"));
        gen.append(el("span", "clip-gen-prompt-text", mismatch.rendered || "(empty)"));
        clip.append(gen);
      }
    }

    const actions = el("div", "clip-actions");
    const mk = (label, title, cls, fn) => { const b = el("button", "ic-btn" + (cls ? " " + cls : ""), label); b.title = title; b.onclick = (e) => { e.stopPropagation(); fn(); }; return b; };
    actions.append(mk("⟈", "Split clip at playhead", "", () => {
      const offsetSec = leftPx / pxPerSec, durSec = sDur(scene, p), fps = sFps(scene, p);
      const ph = window.Player?.getPlayhead() ?? 0;
      let at = null;
      if (ph > offsetSec + 0.05 && ph < offsetSec + durSec - 0.05)
        at = Math.round((ph - offsetSec) * fps);
      S.splitScene(scene.id, at);
    }));
    actions.append(mk("‹", "Move left", "", () => S.moveScene(scene.id, -1)));
    actions.append(mk("›", "Move right", "", () => S.moveScene(scene.id, 1)));
    actions.append(mk("✕", "Delete clip", "danger", () => S.removeScene(scene.id)));
    clip.append(actions);

    // right-edge trim → new duration → frames recomputed (duration × fps).
    const locked = scene.frames_mode === "custom";
    const leftHandle = el("div", "clip-trim clip-trim-left" + (locked ? " locked" : ""));
    leftHandle.title = locked ? "Length locked (custom mode)" : "Drag to trim start · Alt+drag to slip source when rendered";
    if (!locked) {
      leftHandle.addEventListener("mousedown", (e) => {
        e.stopPropagation();
        clip.classList.add("trimming");
        const tip = el("div", "trim-tip"); clip.append(tip);
        let finalDelta = 0;
        onDrag(e, (dx) => {
          finalDelta = dx / pxPerSec;
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
      let finalDur = baseDur;
      onDrag(e, (dx) => {
        finalDur = Math.max(0.1, baseDur + dx / pxPerSec);
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

  // ── video transition bridge at each scene seam ─────────────────────────────────
  function seamEl(st, p, scene, seamPx) {
    const vt = videoTransitionState(scene, p);
    const fps = vt.fps;
    const seam = el("div", "seam" + (vt.active ? " has-vt vt-" + vt.type : ""));
    seam.style.left = seamPx + "px";

    const bridge = el("div", "vt-bridge" + (vt.active ? " on vt-" + vt.type : ""));
    const wPx = Math.max(18, (vt.active ? vt.sec : 0.35) * pxPerSec);
    bridge.style.width = wPx + "px";
    bridge.style.marginLeft = (-wPx / 2) + "px";

    const label = el("div", "vt-label");
    const syncLabel = (type, frames) => {
      if (!type || !frames) label.textContent = "drag → add dissolve";
      else label.textContent = `${VT_SHORT[type] || type} · ${frames}f · ${(frames / fps).toFixed(2)}s`;
    };
    syncLabel(vt.type, vt.frames);
    bridge.append(label);

    const typeSel = el("select", "vt-type");
    VT_TYPES.forEach(([v, lbl]) => {
      const o = el("option", null, lbl); o.value = v;
      if (v === (vt.type || "")) o.selected = true;
      typeSel.append(o);
    });
    typeSel.title = "Video blend at this seam (pixel dissolve — separate from generation split markers above)";
    typeSel.onclick = (e) => e.stopPropagation();
    typeSel.onchange = (e) => {
      e.stopPropagation();
      const next = typeSel.value;
      applyVideoTransition(scene.id, next, next ? (vt.frames || 16) : 0);
    };
    bridge.append(typeSel);

    const handle = el("div", "vt-drag-handle");
    handle.title = "Drag to set transition length (frames)";
    bridge.title = "Drag to set transition length (frames)";
    bridge.addEventListener("mousedown", (e) => {
      if (e.target.closest("select")) return;
      e.stopPropagation();
      _seamDragging = true;
      const baseType = vt.type || "crossfade";
      const baseFrames = vt.frames || 0;
      const basePx = (baseFrames > 0 ? baseFrames / fps : 0.35) * pxPerSec;
      const tip = el("div", "trim-tip"); bridge.append(tip);
      let type = baseType;
      let frames = baseFrames;
      onDrag(e, (dx) => {
        const sec = Math.max(0, (basePx + dx) / pxPerSec);
        frames = Math.min(120, Math.round(sec * fps));
        if (frames > 0 && !type) type = "crossfade";
        const nextW = Math.max(18, (frames > 0 ? frames / fps : 0.35) * pxPerSec);
        bridge.style.width = nextW + "px";
        bridge.style.marginLeft = (-nextW / 2) + "px";
        syncVtVisual(seam, bridge, frames > 0 ? type : "", frames);
        syncLabel(frames > 0 ? type : "", frames);
        tip.textContent = frames > 0 ? `${(frames / fps).toFixed(2)}s · ${frames}f` : "release to clear";
      }, () => {
        tip.remove();
        _seamDragging = false;
        applyVideoTransition(scene.id, frames > 0 ? (type || "crossfade") : "", frames);
      });
    });
    bridge.append(handle);
    seam.append(bridge);

    const promptRow = el("div", "seam-prompt-row");
    promptRow.title = "Generation split marker (prompt) — how Studio divides scenes in a long montage";
    promptRow.append(el("span", "seam-split-lbl", "Split"));
    promptRow.append(promptTransitionSelect(scene.transition_to_next || "", (v) => S.patchScene(scene.id, { transition_to_next: v })));
    seam.append(promptRow);
    return seam;
  }

  // ── effects / transitions menu (timeline toolbar) ───────────────────────────────
  // Each writes onto the selected scene, so the inspector's per-scene controls stay the
  // single source of truth and reflect whatever is added here.
  const FX_DEFS = [
    { id: "zoom_in",  label: "Zoom in (push)",      apply: (sc) => ({ effects: { ...(sc.effects || {}), zoom: "in" } }) },
    { id: "zoom_out", label: "Zoom out (pull back)", apply: (sc) => ({ effects: { ...(sc.effects || {}), zoom: "out" } }) },
    { id: "blur",     label: "Gaussian blur",       val: { label: "Strength (0–1)", def: 0.3, min: 0, max: 1, step: 0.05 }, apply: (sc, v) => ({ effects: { ...(sc.effects || {}), blur: v } }) },
    { id: "fade_in",  label: "Fade in",             val: { label: "Seconds", def: 0.5, min: 0, max: 10, step: 0.1 }, apply: (sc, v) => ({ effects: { ...(sc.effects || {}), fade_in: v } }) },
    { id: "fade_out", label: "Fade out",            val: { label: "Seconds", def: 0.5, min: 0, max: 10, step: 0.1 }, apply: (sc, v) => ({ effects: { ...(sc.effects || {}), fade_out: v } }) },
    { id: "crossfade", label: "Crossfade → next",   val: { label: "Frames", def: 16, min: 1, max: 120, step: 1 }, apply: (sc, v) => ({ video_transition: "crossfade", transition_frames: Math.round(v) }) },
    { id: "fadeblack", label: "Fade to black → next", val: { label: "Frames", def: 16, min: 1, max: 120, step: 1 }, apply: (sc, v) => ({ video_transition: "fadeblack", transition_frames: Math.round(v) }) },
    { id: "wipeleft",  label: "Wipe left → next",   val: { label: "Frames", def: 16, min: 1, max: 120, step: 1 }, apply: (sc, v) => ({ video_transition: "wipeleft", transition_frames: Math.round(v) }) },
    { id: "wiperight", label: "Wipe right → next",  val: { label: "Frames", def: 16, min: 1, max: 120, step: 1 }, apply: (sc, v) => ({ video_transition: "wiperight", transition_frames: Math.round(v) }) },
  ];

  function effectsDropdown(st, inline) {
    const wrap = el("div", "tl-dd");
    const hasSel = !!st.selectedSceneId;
    const btn = inline ? null : el("button", "btn ghost tiny", "✨ Effects ▾");
    if (btn) {
      btn.disabled = !hasSel;
      btn.title = hasSel ? "Add a video effect or transition to the selected clip" : "Select a clip first";
      wrap.append(btn);
    }

    const panel = el("div", "tl-dd-panel" + (inline ? " inline" : ""));
    if (!inline) panel.hidden = true;
    panel.append(el("div", "tl-dd-head", "Add to selected clip"));
    const sel = el("select", "tl-dd-sel");
    FX_DEFS.forEach((d) => sel.append(new Option(d.label, d.id)));
    const valRow = el("label", "tl-dd-val");
    const valLbl = el("span", null, ""); const valIn = el("input"); valIn.type = "number";
    valRow.append(valLbl, valIn);
    const addBtn = el("button", "btn primary tiny", "Add");
    const sync = () => {
      const d = FX_DEFS.find((x) => x.id === sel.value);
      if (d && d.val) {
        valRow.hidden = false; valLbl.textContent = d.val.label;
        valIn.min = d.val.min; valIn.max = d.val.max; valIn.step = d.val.step;
        if (valIn.value === "") valIn.value = d.val.def;
      } else valRow.hidden = true;
    };
    sel.onchange = () => { valIn.value = ""; sync(); };
    addBtn.onclick = () => {
      const d = FX_DEFS.find((x) => x.id === sel.value); if (!d) return;
      const sc = S.scene(st.selectedSceneId); if (!sc) { alert("Select a clip first."); return; }
      const v = d.val ? parseFloat(valIn.value || d.val.def) : null;
      S.patchScene(sc.id, d.apply(sc, v));
      panel.hidden = true;
    };
    panel.append(sel, valRow, addBtn);
    sync();
    if (inline) return panel;
    wrap.append(panel);

    btn.onclick = (e) => {
      e.stopPropagation();
      const show = panel.hidden;
      panel.hidden = !show;
      if (show) {
        const off = (ev) => { if (!wrap.contains(ev.target)) { panel.hidden = true; document.removeEventListener("mousedown", off, true); } };
        setTimeout(() => document.addEventListener("mousedown", off, true), 0);
      }
    };
    return wrap;
  }

  // ── audio (NLE lanes below video) ───────────────────────────────────────────────
  function audioToolbar(st, p) {
    const wrap = el("div", "tl-audio-dd");
    const keepLbl = el("label", "chk");
    const keepCb = el("input"); keepCb.type = "checkbox";
    keepCb.checked = p.keep_original_audio !== false;
    keepCb.title = "Mix generated (LTXAV) audio from each clip at render";
    keepCb.onchange = () => S.patchProject({ keep_original_audio: keepCb.checked });
    keepLbl.append(keepCb); keepLbl.append(el("span", null, "Original audio"));
    wrap.append(keepLbl);

    const audioAssets = (st.mediaBin || []).filter((m) => m.kind === "audio" || /\.(mp3|wav|m4a|aac|ogg|flac)$/i.test(m.name || ""));
    const addSel = el("select", "tl-audio-add");
    addSel.append(new Option(audioAssets.length ? "+ Audio track…" : "+ Upload audio first", ""));
    audioAssets.forEach((m) => addSel.append(new Option(m.name || m.id, m.id)));
    addSel.onchange = () => { if (addSel.value) { S.addAudioTrack(addSel.value, 0); addSel.value = ""; } };
    wrap.append(addSel);
    return wrap;
  }

  function sceneAudioClip(st, p, scene, index, leftPx, widthPx) {
    const vol = scene.audio_volume != null ? scene.audio_volume : 1;
    const w = Math.max(widthPx, 8);  // match video clip min width — never grow past scene duration
    const clip = el("div", "tl-aud-clip scene-aud" + clipSelClass(st, scene.id));
    clip.style.left = leftPx + "px";
    clip.style.width = w + "px";
    clip.style.maxWidth = w + "px";
    clip.onclick = (e) => { e.stopPropagation(); onClipSelect(e, scene.id); };
    clip.append(el("span", "tl-aud-name", `S${index + 1}`));
    const slider = el("input", "tl-aud-vol"); slider.type = "range"; slider.min = "0"; slider.max = "2"; slider.step = "0.05";
    slider.value = vol; slider.title = `Clip ${index + 1} volume`;
    slider.oninput = (e) => { e.stopPropagation(); S.patchSceneQuiet(scene.id, { audio_volume: parseFloat(slider.value) }); };
    slider.onclick = (e) => e.stopPropagation();
    clip.append(slider);
    return clip;
  }

  function insertedAudioLane(st, p, track, laneH) {
    const lane = el("div", "tl-audio-lane"); lane.style.height = laneH + "px";
    const asset = (st.mediaBin || []).find((m) => m.id === track.media_ref);
    const body = el("div", "tl-audio-lane-body");
    const startSec = track.start_sec || 0;
    const dur = asset && asset.duration_sec ? asset.duration_sec : Math.max(2, (p.scenes || []).reduce((a, sc) => a + sDur(sc, p), 0) - startSec);
    const w = Math.max(dur * pxPerSec, 48);
    const block = el("div", "tl-aud-clip ins");
    block.style.left = (startSec * pxPerSec) + "px";
    block.style.width = w + "px";
    block.append(el("span", "tl-aud-name", (asset && asset.name) || "audio"));
    const slider = el("input", "tl-aud-vol"); slider.type = "range"; slider.min = "0"; slider.max = "2"; slider.step = "0.05";
    slider.value = track.volume != null ? track.volume : 1;
    slider.oninput = () => S.updateAudioTrack(track.id, { volume: parseFloat(slider.value) }, true);
    block.append(slider);
    const startIn = el("input"); startIn.type = "number"; startIn.min = "0"; startIn.step = "0.1";
    startIn.value = startSec; startIn.title = "Start (s)"; startIn.style.width = "42px";
    startIn.oninput = (e) => {
      e.stopPropagation();
      const v = parseFloat(startIn.value || "0");
      S.updateAudioTrack(track.id, { start_sec: v }, true);
      block.style.left = (v * pxPerSec) + "px";
    };
    startIn.onclick = (e) => e.stopPropagation();
    block.append(startIn);
    const rm = el("button", "ic-btn danger tl-aud-rm", "✕"); rm.title = "Remove track";
    rm.onclick = (e) => { e.stopPropagation(); S.removeAudioTrack(track.id); };
    block.append(rm);
    block.addEventListener("mousedown", (e) => {
      if (e.target.closest("input,button,select")) return;
      e.stopPropagation();
      const baseLeft = startSec * pxPerSec;
      onDrag(e, (dx) => {
        const snapped = snapPx(baseLeft + dx, seamAnchorsPx([]), 10);
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
    gAud.append(gutterLane("Audio", "Audio — per-scene volume from generated clips", "audio"));
    (p.audio_tracks || []).forEach((t) => {
      const asset = (st.mediaBin || []).find((m) => m.id === t.media_ref);
      const name = (asset && asset.name) || t.label || "Audio";
      const short = name.length > 9 ? name.slice(0, 8) + "…" : name;
      gAud.append(gutterLane(short, name, "audio"));
    });
    gTracks.append(gAud);
    gutter.append(gTracks);
    return gutter;
  }

  function audioLanes(st, p, lay) {
    const wrap = el("div", "tl-audio-lanes");
    const origLane = el("div", "tl-audio-lane"); origLane.style.height = "36px";
    const origBody = el("div", "tl-audio-lane-body");
    lay.forEach(({ sc, o, d }, i) => origBody.append(sceneAudioClip(st, p, sc, i, o * pxPerSec, d * pxPerSec)));
    origLane.append(origBody);
    wrap.append(origLane);
    (p.audio_tracks || []).forEach((t) => wrap.append(insertedAudioLane(st, p, t, 36)));
    return wrap;
  }

  function editToolsDropdown(st, p) {
    const wrap = el("div", "tl-dd");
    const btn = el("button", "btn ghost tiny", "Edit tools ▾");
    const panel = el("div", "tl-dd-panel");
    panel.hidden = true;
    btn.onclick = (e) => { e.stopPropagation(); panel.hidden = !panel.hidden; };
    document.addEventListener("click", () => { panel.hidden = true; }, { once: true });
    panel.append(effectsDropdown(st, true));
    const hasSel = !!st.selectedSceneId;
    const studioCond = !st.project.conditioning_slot || st.project.conditioning_slot === "funpack";
    if (studioCond && hasSel && hasRender(st, st.selectedSceneId) && (st.ratingLabels || []).length) {
      const sc = S.scene(st.selectedSceneId);
      const sceneNo = p.scenes.indexOf(sc) + 1;
      const row = el("div", "tl-dd-row");
      row.append(el("span", "tl-keys", `★ Scene ${sceneNo}`));
      const rsel = el("select", "tl-rating");
      rsel.append(new Option("— rate —", ""));
      (st.ratingLabels || []).forEach((l) => { const o = new Option(l, l); if (l === (sc.rating || "")) o.selected = true; rsel.append(o); });
      rsel.onchange = () => S.setSceneRating(sc.id, rsel.value);
      row.append(rsel);
      panel.append(row);
    }
    wrap.append(btn); wrap.append(panel);
    return wrap;
  }

  // ── toolbar ─────────────────────────────────────────────────────────────────────
  function toolbar(st, p, totalSec) {
    const bar = el("div", "tl-toolbar");
    const add = el("button", "btn ghost tiny", "＋ Clip"); add.onclick = () => S.addScene();
    bar.append(add);
    const ph = window.Player?.getPlayhead() ?? 0;
    const tc = el("span", "tl-tc", timecode(Math.min(ph, totalSec), p.frame_rate) + " / " + timecode(totalSec, p.frame_rate));
    tlTcEl = tc;  // keep ref for Player's onPlayheadChanged updates
    bar.append(tc);

    // Clip actions on the selected clip (also bound to S / Delete).
    const hasSel = !!st.selectedSceneId;
    const split = el("button", "btn ghost tiny", "⧅ Split");
    split.title = "Split the selected clip at the playhead (S)"; split.disabled = !hasSel;
    split.onclick = () => splitSelectedAtPlayhead();
    const del = el("button", "btn ghost tiny danger", "✕ Remove");
    del.title = "Remove the selected clip (Delete / Backspace)"; del.disabled = !hasSel;
    del.onclick = () => { if (st.selectedSceneId) S.removeScene(st.selectedSceneId); };
    const exp = el("button", "btn ghost tiny", "⤓ Export");
    exp.title = "Save the selected clip's rendered video to disk (renders are temporary)";
    exp.disabled = !(hasSel && hasRender(st, st.selectedSceneId));
    exp.onclick = () => S.exportSelected();
    bar.append(split); bar.append(del); bar.append(exp);
    bar.append(editToolsDropdown(st, p));
    bar.append(audioToolbar(st, p));

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
      if ((st.selectedSceneId || (st.selectedSceneIds || []).length) && !e.target.closest(".clip") && !e.target.closest(".seam") && !e.target.closest(".tl-ruler2") && !e.target.closest(".tl-aud-clip"))
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
    return true;
  }

  function onStore(st) {
    const ok = render(st);
    if (_pendingAutoFit && ok && st.project) {
      _pendingAutoFit = false;
      const total = S.previewTotalSec ? S.previewTotalSec() : tlTotalSec;
      requestAnimationFrame(() => { if (total > 0) fit(total); });
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
    else if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); S.removeScene(st.selectedSceneId); }
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
