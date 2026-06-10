// Bottom zone: a real NLE timeline. Clips are laid out on a time axis (width =
// duration × zoom), with a HH:MM:SS ruler, a playhead, drag-to-trim edges (which
// recompute frame counts from duration × fps), split, and per-seam crossfades.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const body = document.getElementById("timeline-body");
  const meta = document.getElementById("timeline-meta");

  const SRC_ICON = { empty: "▦", image: "◐", generated_frame: "⛶", carry: "⇥" };
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

  const hasRender = (st, sceneId) => !!(st.sceneRenders && st.sceneRenders[sceneId] && st.sceneRenders[sceneId].media);

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

  // ── transition (seam) picker ─────────────────────────────────────────────────
  function transitionSelect(value, onChange) {
    const sel = el("select", "seam-type");
    const none = el("option", null, "cut"); none.value = ""; sel.append(none);
    (S.get().transitions || []).forEach((t) => {
      const name = t.trigger || t.name || t.key; if (!name) return;
      const o = el("option", null, name); o.value = name; if (name === value) o.selected = true; sel.append(o);
    });
    if (value && ![...sel.options].some((o) => o.value === value)) { const o = el("option", null, value); o.value = value; o.selected = true; sel.append(o); }
    sel.onchange = (e) => { e.stopPropagation(); onChange(sel.value); };
    sel.onclick = (e) => e.stopPropagation();
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
    clip.append(el("div", "clip-text ghost-label", ghost.text || "removed scene"));
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
    const clip = el("div", "clip" + (scene.id === st.selectedSceneId ? " selected" : "") + (scene.excluded ? " excluded" : "") + (hasRender(st, scene.id) ? " rendered" : (!scene.excluded ? " pending" : "")) + (unitCuts > 1 ? " gen-cut" : "") + (subclip ? " subclip" : ""));
    clip.style.left = leftPx + "px";
    clip.style.width = Math.max(widthPx, 8) + "px";
    clip.onclick = () => S.selectScene(scene.id);

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

    // media thumbnail badge when an image asset is assigned
    const mref = scene.source?.type === "image" ? scene.source.media_ref : null;
    const asset = mref ? (st.mediaBin || []).find((m) => m.id === mref) : null;
    if (asset) {
      const th = el("div", "clip-thumb");
      if (asset.kind === "image") { const img = el("img"); img.src = window.MovieEditorAPI.mediaUrl(asset.id); img.loading = "lazy"; th.append(img); }
      else th.append(el("span", null, "▶"));
      clip.append(th);
      clip.classList.add("has-media");
    }

    const head = el("div", "clip-head");
    head.append(el("span", "clip-no", p2(index + 1)));
    head.append(el("span", "clip-src", SRC_ICON[scene.source?.type] || "▦"));
    head.append(el("span", "clip-dur", timecode(sDur(scene, p), sFps(scene, p))));
    clip.append(head);

    const root = unitCuts > 1
      ? (p.scenes || []).find((s) => (s.gen_unit_id || s.id) === (scene.gen_unit_id || scene.id) && !(s.cut_offset_frames > 0))
      : null;
    const label = scene.text || (root && root.text) || (subclip ? "cut" : "empty scene");
    clip.append(el("div", "clip-text" + (label && label !== "empty scene" && label !== "cut" ? "" : " empty"), label));

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
    // Locked when length is "custom" (the inspector value wins over the timeline).
    const locked = scene.frames_mode === "custom";
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
    return clip;
  }

  // ── crossfade chip in the seam ──────────────────────────────────────────────────
  function seamEl(st, p, scene, seamPx) {
    const seam = el("div", "seam");
    seam.style.left = seamPx + "px";
    const xfFrames = scene.transition_frames || 0;
    const fps = sFps(scene, p);
    const w = Math.max(14, (xfFrames / fps) * pxPerSec);
    const fade = el("div", "xfade" + (xfFrames ? " on" : ""));
    fade.style.width = w + "px";
    fade.title = xfFrames ? `Crossfade ${(xfFrames / fps).toFixed(2)}s (${xfFrames}f) — drag to adjust` : "Drag right to add a crossfade";
    // drag the fade width → transition_frames
    fade.addEventListener("mousedown", (e) => {
      const base = (xfFrames / fps) * pxPerSec;
      const tip = el("div", "trim-tip"); fade.append(tip);
      let frames = xfFrames;
      onDrag(e, (dx) => {
        const sec = Math.max(0, (base + dx) / pxPerSec);
        frames = Math.round(sec * fps);
        fade.style.width = Math.max(14, sec * pxPerSec) + "px";
        tip.textContent = frames ? `${(frames / fps).toFixed(2)}s · ${frames}f` : "no crossfade";
      }, () => { tip.remove(); S.patchScene(scene.id, { transition_frames: frames || null }); });
    });
    seam.append(fade);
    seam.append(transitionSelect(scene.transition_to_next || "", (v) => S.patchScene(scene.id, { transition_to_next: v })));
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

  function effectsDropdown(st) {
    const wrap = el("div", "tl-dd");
    const hasSel = !!st.selectedSceneId;
    const btn = el("button", "btn ghost tiny", "✨ Effects ▾");
    btn.disabled = !hasSel;
    btn.title = hasSel ? "Add a video effect or transition to the selected clip" : "Select a clip first";
    wrap.append(btn);

    const panel = el("div", "tl-dd-panel"); panel.hidden = true;
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
    const clip = el("div", "tl-aud-clip scene-aud" + (scene.id === st.selectedSceneId ? " selected" : ""));
    clip.style.left = leftPx + "px";
    clip.style.width = w + "px";
    clip.style.maxWidth = w + "px";
    clip.onclick = (e) => { e.stopPropagation(); S.selectScene(scene.id); };
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
    bar.append(effectsDropdown(st));
    bar.append(audioToolbar(st, p));

    // Scene rating — only when FunPack Studio is the conditioning provider AND the
    // selected clip has a render. Rates that clip's own scene (cut-aware by scene id);
    // fed into Studio at the next generation of its run.
    const studioCond = !st.project.conditioning_slot || st.project.conditioning_slot === "funpack";
    if (studioCond && hasSel && hasRender(st, st.selectedSceneId) && (st.ratingLabels || []).length) {
      const sc = S.scene(st.selectedSceneId);
      const sceneNo = p.scenes.indexOf(sc) + 1;
      const rlabel = el("span", "tl-keys", `★ Scene ${sceneNo}`);
      const rsel = el("select", "tl-rating");
      rsel.title = "Rate this scene's render — FunPack Studio refines from it next generation";
      rsel.append(new Option("— rate —", ""));
      (st.ratingLabels || []).forEach((l) => { const o = new Option(l, l); if (l === (sc.rating || "")) o.selected = true; rsel.append(o); });
      rsel.onchange = () => S.setSceneRating(sc.id, rsel.value);
      bar.append(rlabel); bar.append(rsel);
    }

    const spacer = el("div", "tl-spacer"); bar.append(spacer);
    const keys = el("span", "tl-keys", "S split · ⌫ remove"); keys.title = "Select a clip, then: S splits it at the playhead · Delete/Backspace removes it";
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

  function render(st) {
    if (_reordering) return false;
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
      if (st.selectedSceneId && !e.target.closest(".clip") && !e.target.closest(".seam") && !e.target.closest(".tl-ruler2") && !e.target.closest(".tl-aud-clip"))
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
    const metaTxt = `${scenes.length} clips · ${active} active` + (ghosts ? ` · ${ghosts} ghost${ghosts > 1 ? "s" : ""}` : "") + ` · ${timecode(totalSec, p.frame_rate)}`;
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

  S.subscribe(onStore);

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
    if (!st.project || !st.selectedSceneId) return;
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
