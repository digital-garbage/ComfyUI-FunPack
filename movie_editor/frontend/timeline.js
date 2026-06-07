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

  const hasRender = (st, sceneId) =>
    (st.renderedSegments || []).some((s) => s.media && (s.sceneIds || []).includes(sceneId));

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

  // ── clip ───────────────────────────────────────────────────────────────────────
  function clipEl(st, p, scene, index, leftPx, widthPx) {
    const clip = el("div", "clip" + (scene.id === st.selectedSceneId ? " selected" : "") + (scene.excluded ? " excluded" : "") + (hasRender(st, scene.id) ? " rendered" : ""));
    clip.style.left = leftPx + "px";
    clip.style.width = Math.max(widthPx, 8) + "px";
    clip.onclick = () => S.selectScene(scene.id);

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

    clip.append(el("div", "clip-text" + (scene.text ? "" : " empty"), scene.text || "empty scene"));

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
    const w = (body.querySelector(".tl-scroll")?.clientWidth || 800) - 40;
    if (totalSec > 0) setZoom(w / totalSec);
  }

  // ── render ───────────────────────────────────────────────────────────────────────
  function render(st) {
    clear(body); clear(meta);
    if (!st.project) { body.append(el("div", "empty-stage", "Open a project to start cutting.")); return; }
    const p = st.project;
    const scenes = p.scenes || [];

    // layout: cumulative offsets in seconds
    let acc = 0;
    const lay = scenes.map((sc) => { const d = sDur(sc, p); const o = acc; acc += d; return { sc, o, d }; });
    const totalSec = acc;
    tlTotalSec = totalSec;
    tlFps = p.frame_rate;
    const contentW = Math.max(totalSec * pxPerSec + 40, 480);

    body.append(toolbar(st, p, totalSec));

    const scroll = el("div", "tl-scroll");
    scroll.addEventListener("scroll", () => { scrollLeft = scroll.scrollLeft; });
    // Click empty timeline space (not a clip/seam) to clear the selection.
    scroll.addEventListener("click", (e) => {
      if (st.selectedSceneId && !e.target.closest(".clip") && !e.target.closest(".seam") && !e.target.closest(".tl-ruler2"))
        S.selectScene(null);
    });
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

    // track
    const track = el("div", "tl-track2");
    lay.forEach(({ sc, o, d }, i) => track.append(clipEl(st, p, sc, i, o * pxPerSec, d * pxPerSec)));
    // seams between clips
    for (let i = 0; i < lay.length - 1; i++) track.append(seamEl(st, p, lay[i].sc, (lay[i].o + lay[i].d) * pxPerSec));
    if (!lay.length) track.append(el("div", "tl-emptyhint", "No clips yet — add one from the toolbar."));
    // playhead — read from Player; keep a ref so the onPlayheadChanged handler
    // can update just the left position without a full re-render.
    const phSec = Math.min(window.Player?.getPlayhead() ?? 0, totalSec);
    tlPhEl = el("div", "tl-playhead"); tlPhEl.style.left = (phSec * pxPerSec) + "px"; track.append(tlPhEl);
    content.append(track);

    scroll.append(content);
    body.append(scroll);
    scroll.scrollLeft = scrollLeft;
    tlScrollEl = scroll;

    const active = scenes.filter((s) => !s.excluded).length;
    meta.append(el("span", null, `${scenes.length} clips · ${active} active · ${timecode(totalSec, p.frame_rate)}`));
  }

  S.subscribe(render);

  // ── keyboard: S = split selected clip at playhead, Del/Backspace = remove it ──
  function splitSelectedAtPlayhead() {
    const st = S.get();
    if (!st.project || !st.selectedSceneId) return;
    const p = st.project;
    let off = 0, target = null;
    for (const sc of (p.scenes || [])) { if (sc.id === st.selectedSceneId) { target = sc; break; } off += sDur(sc, p); }
    if (!target) return;
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
