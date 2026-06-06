// Bottom zone: a real NLE timeline. Clips are laid out on a time axis (width =
// duration × zoom), with a HH:MM:SS ruler, a playhead, drag-to-trim edges (which
// recompute frame counts from duration × fps), split, and per-seam crossfades.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const body = document.getElementById("timeline-body");
  const meta = document.getElementById("timeline-meta");

  const SRC_ICON = { empty: "▦", image: "◐", generated_frame: "⛶" };
  const ZOOM_KEY = "funpack_tl_zoom";
  let pxPerSec = parseFloat(localStorage.getItem(ZOOM_KEY)) || 80;  // zoom
  let playheadSec = 0;
  let scrollLeft = 0;

  // ── helpers ──────────────────────────────────────────────────────────────────
  const sFps = (sc, p) => (sc.fps != null ? sc.fps : p.frame_rate) || 25;
  const sFrames = (sc, p) => (sc.frames != null ? sc.frames : p.num_frames_per_scene) || 1;
  const sDur = (sc, p) => sFrames(sc, p) / sFps(sc, p);

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
    const clip = el("div", "clip" + (scene.id === st.selectedSceneId ? " selected" : "") + (scene.excluded ? " excluded" : ""));
    clip.style.left = leftPx + "px";
    clip.style.width = Math.max(widthPx, 8) + "px";
    clip.onclick = () => S.selectScene(scene.id);

    const head = el("div", "clip-head");
    head.append(el("span", "clip-no", p2(index + 1)));
    head.append(el("span", "clip-src", SRC_ICON[scene.source?.type] || "▦"));
    head.append(el("span", "clip-dur", timecode(sDur(scene, p), sFps(scene, p))));
    clip.append(head);

    clip.append(el("div", "clip-text" + (scene.text ? "" : " empty"), scene.text || "empty scene"));

    const actions = el("div", "clip-actions");
    const mk = (label, title, cls, fn) => { const b = el("button", "ic-btn" + (cls ? " " + cls : ""), label); b.title = title; b.onclick = (e) => { e.stopPropagation(); fn(); }; return b; };
    actions.append(mk("⟈", "Split clip", "", () => S.splitScene(scene.id)));
    actions.append(mk("‹", "Move left", "", () => S.moveScene(scene.id, -1)));
    actions.append(mk("›", "Move right", "", () => S.moveScene(scene.id, 1)));
    actions.append(mk("✕", "Delete clip", "danger", () => S.removeScene(scene.id)));
    clip.append(actions);

    // right-edge trim → new duration → frames recomputed (duration × fps)
    const handle = el("div", "clip-trim");
    handle.title = "Drag to trim · length = duration × fps";
    const baseDur = sDur(scene, p);
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
    const tc = el("span", "tl-tc", timecode(playheadSec, p.frame_rate) + " / " + timecode(totalSec, p.frame_rate));
    bar.append(tc);
    const spacer = el("div", "tl-spacer"); bar.append(spacer);
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
    playheadSec = Math.min(playheadSec, totalSec);
    const contentW = Math.max(totalSec * pxPerSec + 40, 480);

    body.append(toolbar(st, p, totalSec));

    const scroll = el("div", "tl-scroll");
    scroll.addEventListener("scroll", () => { scrollLeft = scroll.scrollLeft; });
    const content = el("div", "tl-content"); content.style.width = contentW + "px";

    // ruler
    const ruler = el("div", "tl-ruler2");
    const iv = tickInterval();
    for (let t = 0; t <= totalSec + iv; t += iv) {
      const tick = el("div", "tl-tick"); tick.style.left = (t * pxPerSec) + "px";
      tick.append(el("span", "tl-tick-label", rulerLabel(t, iv)));
      ruler.append(tick);
    }
    ruler.onclick = (e) => { const r = ruler.getBoundingClientRect(); playheadSec = Math.max(0, (e.clientX - r.left) / pxPerSec); render(st); };
    content.append(ruler);

    // track
    const track = el("div", "tl-track2");
    lay.forEach(({ sc, o, d }, i) => track.append(clipEl(st, p, sc, i, o * pxPerSec, d * pxPerSec)));
    // seams between clips
    for (let i = 0; i < lay.length - 1; i++) track.append(seamEl(st, p, lay[i].sc, (lay[i].o + lay[i].d) * pxPerSec));
    if (!lay.length) track.append(el("div", "tl-emptyhint", "No clips yet — add one from the toolbar."));
    // playhead
    const ph = el("div", "tl-playhead"); ph.style.left = (playheadSec * pxPerSec) + "px"; track.append(ph);
    content.append(track);

    scroll.append(content);
    body.append(scroll);
    scroll.scrollLeft = scrollLeft;

    const active = scenes.filter((s) => !s.excluded).length;
    meta.append(el("span", null, `${scenes.length} clips · ${active} active · ${timecode(totalSec, p.frame_rate)}`));
  }

  S.subscribe(render);
})();
