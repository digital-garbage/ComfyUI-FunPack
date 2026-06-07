// Right-top zone: NLE program monitor.
//
// Architecture:
//   - A POOL of <video>, one per clip file, lives in a persistent stage; only the
//     active one is shown. Switching clips swaps which is visible — every clip is
//     preloaded, so scrubbing across cut boundaries never blacks out (real-NLE feel).
//   - The pool is rebuilt as the timeline changes (cuts/regens are dynamic); segment
//     positions are recomputed live from store state.
//   - window.Player mediates the playhead between this module and timeline.js.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const API = window.MovieEditorAPI;
  const body = document.getElementById("preview-body");
  const statusEl = document.getElementById("preview-status");

  // ── preloaded video pool ─────────────────────────────────────────────────────
  const stage = document.createElement("div"); stage.className = "pm-stage";
  const pool = new Map();    // url -> <video>
  let _active = null;        // visible <video>
  let _seekPending = null;   // offset to apply once the active video has metadata
  let _playPending = false;

  // ── playhead state ─────────────────────────────────────────────────────────
  let _phSec = 0;
  let _playing = false;
  const _phListeners = new Set();

  function _notifyPh() {
    _phListeners.forEach((fn) => { try { fn(_phSec, _playing); } catch (_) {} });
  }

  // ── clips ────────────────────────────────────────────────────────────────────
  // Per-scene clips: { media, sceneId, startSec (timeline), durationSec, inSec (source
  // in-point) }. Built from store.sceneRenders so splits/deletes play correct portions.
  let _clips = [];
  let _currentClip = null;

  const _urlFor = (media) => API.resultUrl(S.get().project?.id, media);
  const _clampT = (v, t) => Math.max(0, Math.min(t, (v.duration || (t + 1)) - 0.02));

  function _buildClips(st) {
    if (!st.project) return [];
    const p = st.project;
    const sr = st.sceneRenders || {};
    const sFps = (sc) => ((sc.fps_mode !== "project" && sc.fps != null) ? sc.fps : p.frame_rate) || 25;
    const sFrames = (sc) => ((sc.frames_mode !== "project" && sc.frames != null) ? sc.frames : p.num_frames_per_scene) || 1;
    const out = [];
    let acc = 0;
    for (const sc of (p.scenes || [])) {
      const dur = sFrames(sc) / sFps(sc);
      const r = sr[sc.id];
      if (r && r.media) out.push({ media: r.media, sceneId: sc.id, startSec: acc, durationSec: dur, inSec: r.inSec || 0, fx: sc.effects || {}, vol: sc.audio_volume != null ? sc.audio_volume : 1 });
      acc += dur;  // timeline advances even over ungenerated scenes (gaps)
    }
    return out;
  }

  function _clipAt(sec) {
    return _clips.find((c) => sec >= c.startSec - 0.05 && sec < c.startSec + c.durationSec - 0.001) || null;
  }
  function _clipFrom(sec) {  // first clip at/after sec (for play/skip-gap)
    return _clipAt(sec) || _clips.find((c) => c.startSec >= sec - 0.05) || null;
  }

  // ── video pool management ─────────────────────────────────────────────────────
  function _ensureVideo(url) {
    let v = pool.get(url);
    if (v) return v;
    v = document.createElement("video");
    v.className = "pm-video"; v.preload = "auto"; v.playsInline = true; v.src = url;
    v.addEventListener("loadedmetadata", () => {
      if (v !== _active) return;
      if (_seekPending != null) { v.currentTime = _clampT(v, _seekPending); _seekPending = null; }
      if (_playPending) { _playPending = false; v.play().catch(() => {}); }
    });
    v.addEventListener("ended", () => { if (v === _active) _advance(); });
    stage.append(v); pool.set(url, v);
    return v;
  }

  function _syncPool() {
    const urls = new Set(_clips.map((c) => _urlFor(c.media)));
    urls.forEach((u) => _ensureVideo(u));
    for (const [u, v] of [...pool]) {
      if (urls.has(u)) continue;
      v.pause(); v.remove(); pool.delete(u);
      if (_active === v) { _active = null; }
    }
  }

  function _resetPool() {
    _stopTick();
    for (const [, v] of pool) { v.pause(); v.remove(); }
    pool.clear(); _active = null; _currentClip = null; _seekPending = null; _playPending = false;
  }

  function _setActive(v) {
    if (_active === v) return;
    if (_active) {
      _active.pause(); _active.classList.remove("active");
      _active.style.filter = ""; _active.style.transform = ""; _active.style.opacity = "";  // drop stale fx
    }
    _active = v; if (v) v.classList.add("active");
  }

  // Live preview approximation of a clip's video effects (the render does the real thing).
  // within = seconds into the clip. Blur -> CSS blur; zoom is VIRTUAL — the frame stays the
  // canvas size and only the content scales (overflow clipped by the canvas), matching the
  // render's zoompan: in 1.0->1.2 (push in), out 1.2->1.0 (pull back). fade -> opacity.
  // Computed from the playhead so it tracks scrubbing too.
  function _applyFx(clip, within) {
    const v = _active; if (!v) return;
    const fx = (clip && clip.fx) || {};
    const dur = (clip && clip.durationSec) || 0;
    const t = dur > 0 ? Math.max(0, Math.min(1, within / dur)) : 0;
    const blur = +fx.blur || 0;
    v.style.filter = blur > 0 ? `blur(${(blur * 8).toFixed(1)}px)` : "";
    let scale = 1;
    if (fx.zoom === "in") scale = 1 + 0.2 * t;
    else if (fx.zoom === "out") scale = 1.2 - 0.2 * t;
    v.style.transformOrigin = "center";
    v.style.transform = scale !== 1 ? `scale(${scale.toFixed(4)})` : "";
    let op = 1;
    const fi = +fx.fade_in || 0, fo = +fx.fade_out || 0;
    if (fi > 0 && within < fi) op = Math.max(0, within / fi);
    if (fo > 0 && dur > 0 && within > dur - fo) op = Math.min(op, Math.max(0, (dur - within) / fo));
    v.style.opacity = op < 1 ? op.toFixed(3) : "";
    // audio: per-clip original volume, muted when the project drops original audio. Guarded —
    // a non-finite volume throws on assignment, which (in the rAF tick) would freeze playback.
    const keepOrig = (S.get().project || {}).keep_original_audio !== false;
    let vol = (clip && clip.vol != null) ? +clip.vol : 1;
    if (!isFinite(vol)) vol = 1;
    const nv = keepOrig ? Math.max(0, Math.min(1, vol)) : 0;
    if (v.volume !== nv) { try { v.volume = nv; } catch (_) {} }
  }

  // Show `clip` at `offset` seconds into the clip; play if requested. Seeks to the clip's
  // in-point + offset within the (preloaded) source video.
  function _goto(clip, offset, play) {
    if (!clip || !clip.media) return;
    const v = _ensureVideo(_urlFor(clip.media));
    _currentClip = clip; _setActive(v);
    const target = (clip.inSec || 0) + Math.max(0, offset);
    if (v.readyState >= 1) {
      v.currentTime = _clampT(v, target);
      if (play) { v.play().catch(() => {}); _startTick(); } else _playPending = false;
    } else {
      _seekPending = target; _playPending = !!play;
      if (play) _startTick();
    }
    _applyFx(clip, Math.max(0, offset));
  }

  // Advance to the next clip at a clip's out-point (or stop). Same-source contiguous
  // clips (a split seam) continue without reseeking; otherwise seek; no next → stop.
  function _advance() {
    if (!_currentClip) { _pause(); return; }
    const end = _currentClip.startSec + _currentClip.durationSec;
    const next = _clips.find((c) => c.startSec >= end - 0.02 && c !== _currentClip);
    if (!next) { _phSec = end; _pause(); return; }
    const contiguous = _urlFor(next.media) === _urlFor(_currentClip.media)
      && Math.abs((next.inSec || 0) - ((_currentClip.inSec || 0) + _currentClip.durationSec)) < 0.05;
    _phSec = next.startSec;
    _currentClip = next;
    if (!contiguous) _goto(next, 0, _playing);  // jump (different source / non-contiguous)
    _notifyPh();
  }

  // ── playback tick (precise out-points, so deleted parts aren't played) ─────────
  let _raf = null;
  function _startTick() { if (!_raf) _raf = requestAnimationFrame(_tick); }
  function _stopTick() { if (_raf) { cancelAnimationFrame(_raf); _raf = null; } }
  function _tick() {
    _raf = null;
    if (!_playing || !_active || !_currentClip) return;
    const within = _active.currentTime - (_currentClip.inSec || 0);
    if (within >= _currentClip.durationSec - 0.001) { _advance(); }
    else { _phSec = _currentClip.startSec + Math.max(0, within); try { _applyFx(_currentClip, within); } catch (_) {} _notifyPh(); }
    if (_playing) _raf = requestAnimationFrame(_tick);
  }

  // ── transport actions ──────────────────────────────────────────────────────
  function _seek(sec) {
    _phSec = Math.max(0, sec);
    const clip = _clipAt(_phSec);
    if (clip) _goto(clip, _phSec - clip.startSec, false);
    _notifyPh();
  }

  function _play() {
    // At (or past) the end of the timeline → loop the playhead back to the start.
    if (_phSec >= (_totalSecCur || 0) - 0.05 || !_clipFrom(_phSec)) _phSec = 0;
    const clip = _clipFrom(_phSec);
    if (!clip) return;
    if (_phSec < clip.startSec) _phSec = clip.startSec;
    _playing = true;
    _goto(clip, _phSec - clip.startSec, true);
    _notifyPh();
    _renderTransport();
  }

  function _pause() {
    if (_active) _active.pause();
    _playing = false;
    _playPending = false;
    _stopTick();
    _notifyPh();
    _renderTransport();
  }

  function _stop() {
    if (_active) _active.pause();
    _phSec = 0;
    _playing = false;
    _playPending = false;
    _stopTick();
    const first = _clips[0];
    if (first) _goto(first, 0, false);
    _notifyPh();
    _renderTransport();
  }

  // ── public Player API (consumed by timeline.js + inspector) ──────────────
  window.Player = {
    seek: _seek,
    getPlayhead: () => _phSec,
    play: _play,
    pause: _pause,
    stop: _stop,
    isPlaying: () => _playing,
    onPlayheadChanged: (fn) => { _phListeners.add(fn); return () => _phListeners.delete(fn); },
    // Capture the current video frame as a PNG Blob.  Returns null if no video
    // is loaded or the frame isn't ready.
    captureFrame: () => {
      const v = _active;
      if (!v || !v.videoWidth || v.readyState < 2) return null;
      const c = document.createElement("canvas");
      c.width = v.videoWidth; c.height = v.videoHeight;
      c.getContext("2d").drawImage(v, 0, 0);
      return new Promise((resolve) => c.toBlob((b) => resolve(b), "image/png"));
    },
  };

  // ── timecode formatter ─────────────────────────────────────────────────────
  const _p2 = (n) => String(Math.floor(Math.max(0, n))).padStart(2, "0");
  function _tc(sec, fps) {
    fps = fps || 25; sec = Math.max(0, sec);
    const f = Math.round(sec * fps), ff = f % fps, t = Math.floor(f / fps);
    return _p2(t / 3600) + ":" + _p2((t % 3600) / 60) + ":" + _p2(t % 60) + ":" + _p2(ff);
  }

  // ── transport DOM (re-rendered without touching the video) ─────────────────
  let _transportEl = null;
  let _minibarEl = null;
  let _totalSecCur = 0;   // total timeline duration for the current project
  let _fpsCur = 25;
  let _tcEl = null;       // timecode span (updated live, not rebuilt)
  let _playBtnEl = null;  // play/pause button (glyph updated live, not rebuilt)
  let _genMsgEl = null, _genBarEl = null, _genFillEl = null;  // gen readout, updated live

  // Update the generation readout in place on progress ticks (no store re-render, so the
  // editor stays interactive — frames can be saved, effects added — while a job runs).
  window.addEventListener("funpack-gen-progress", (e) => {
    const g = (e && e.detail) || {};
    if (_genMsgEl && g.msg != null) _genMsgEl.textContent = g.msg || g.state || "";
    if (_genFillEl && _genBarEl) {
      if (g.maxStep > 0) {
        _genBarEl.style.visibility = "";
        _genFillEl.style.width = Math.min(100, Math.round((g.step / g.maxStep) * 100)) + "%";
      }
    }
  });

  // Lightweight per-tick update: never rebuilds the buttons (rebuilding mid-click ate the
  // mousedown/mouseup, so play/pause/stop stopped responding during playback).
  function _updateTransportLive() {
    if (_tcEl) _tcEl.textContent = _tc(_phSec, _fpsCur);
    if (_playBtnEl) {
      _playBtnEl.textContent = _playing ? "⏸" : "▶";
      _playBtnEl.title = _playing ? "Pause" : "Play";
      _playBtnEl.classList.toggle("active", _playing);
    }
  }

  function _renderTransport() {
    if (!_transportEl) return;
    clear(_transportEl);
    const stopBtn = el("button", "ic-btn", "⏹"); stopBtn.title = "Stop"; stopBtn.onclick = _stop;
    const playBtn = el("button", "ic-btn" + (_playing ? " active" : ""), _playing ? "⏸" : "▶");
    playBtn.title = _playing ? "Pause" : "Play"; playBtn.onclick = () => _playing ? _pause() : _play();
    _playBtnEl = playBtn;
    const tc = el("span", "pm-tc", _tc(_phSec, _fpsCur));
    _tcEl = tc;
    _transportEl.append(stopBtn, playBtn, tc);

    // Anchor capture — only when a rendered frame is under the playhead.
    if (_currentClip) {
      const scenes = (S.get().project?.scenes || []).filter((s) => !s.excluded);
      if (scenes.length) {
        const sep = el("div", "pm-sep"); _transportEl.append(sep);

        const anchorSel = el("select", "pm-anchor-sel");
        anchorSel.title = "Scene to anchor — or leave blank to just save the frame to the Media bin";
        // No default scene: blank = just save the frame (never silently overwrites a scene).
        anchorSel.append(new Option("— save to Media bin —", ""));
        scenes.forEach((s, i) => {
          const label = `anchor → Scene ${i + 1}` + (s.text ? ": " + s.text.substring(0, 18) : "");
          anchorSel.append(new Option(label, s.id));
        });

        const anchorBtn = el("button", "btn ghost tiny pm-anchor-btn", "📌 Save frame");
        const _btnLabel = () => (anchorSel.value ? "📌 Use as anchor" : "📌 Save frame");
        anchorSel.onchange = () => { anchorBtn.textContent = _btnLabel(); };
        anchorBtn.title = "Capture this frame — to the Media bin, or as the chosen scene's i2v anchor";
        anchorBtn.onclick = async () => {
          const sceneId = anchorSel.value;
          anchorBtn.disabled = true; anchorBtn.textContent = "Capturing…";
          try {
            const blob = await window.Player.captureFrame();
            if (!blob) {
              anchorBtn.textContent = _btnLabel();
              alert("No video frame available — make sure the playhead is on a rendered segment.");
              return;
            }
            const sc = sceneId ? scenes.find((s) => s.id === sceneId) : null;
            const name = `frame_${sc ? "scene" + (S.get().project.scenes.indexOf(sc) + 1) + "_" : ""}${Date.now()}.png`;
            const file = new File([blob], name, { type: "image/png" });
            await S.uploadMedia([file]);
            if (sceneId) {
              const bin = S.get().mediaBin;
              const asset = bin[bin.length - 1];
              if (!asset) { alert("Upload failed."); return; }
              // Store as generated_frame type (builder accepts it like "image")
              S.patchScene(sceneId, { source: { type: "generated_frame", media_ref: asset.id } });
              anchorBtn.textContent = "✓ Applied";
            } else {
              anchorBtn.textContent = "✓ Saved";
            }
            setTimeout(() => _renderTransport(), 1600);
          } catch (e) {
            alert("Capture failed: " + e.message);
            anchorBtn.textContent = _btnLabel();
          } finally {
            anchorBtn.disabled = false;
          }
        };

        _transportEl.append(anchorSel, anchorBtn);
      }
    }
  }

  function _renderNeedle() {
    if (!_minibarEl || !_totalSecCur) return;
    let needle = _minibarEl.querySelector(".pm-needle");
    if (!needle) { needle = el("div", "pm-needle"); _minibarEl.append(needle); }
    needle.style.left = (Math.min(_phSec, _totalSecCur) / _totalSecCur * 100).toFixed(3) + "%";
  }

  // Subscribe to Player changes to update transport + needle without full re-render
  // (a full _renderTransport here would recreate the buttons every tick and swallow clicks).
  window.Player.onPlayheadChanged(() => {
    _updateTransportLive();
    _renderNeedle();
  });

  // ── segment change detection ───────────────────────────────────────────────
  let _lastSegHash = "";
  let _lastProjectId = null;

  // ── full render (store subscription) ──────────────────────────────────────
  function render(st) {
    // Project switch: drop the whole pool and reset.
    if (st.project?.id !== _lastProjectId) {
      _lastProjectId = st.project?.id || null;
      _resetPool();
      _phSec = 0; _playing = false; _lastSegHash = "";
    }

    _clips = _buildClips(st);
    _syncPool();  // preload current clips, drop stale ones (timeline is dynamic)

    // Re-resolve the clip under the playhead each render (positions change with edits).
    const hash = _clips.map((c) => c.media?.filename + "@" + c.startSec.toFixed(2) + "+" + c.inSec.toFixed(2)).join("|");
    if (hash !== _lastSegHash) {
      _lastSegHash = hash;
      const clip = _clipAt(_phSec);
      if (clip) _goto(clip, _phSec - clip.startSec, _playing);
      else if (_clips.length && !_active) _goto(_clips[0], 0, false);
    }

    const p = st.project;
    const gen = st.gen || { state: "idle", media: [] };
    _fpsCur = p?.frame_rate || 25;

    // Compute total duration
    const sFps = (sc) => ((sc.fps_mode !== "project" && sc.fps != null) ? sc.fps : _fpsCur) || 25;
    const sFrames = (sc) => ((sc.frames_mode !== "project" && sc.frames != null) ? sc.frames : p?.num_frames_per_scene || 1) || 1;
    _totalSecCur = 0;
    for (const sc of (p?.scenes || [])) _totalSecCur += sFrames(sc) / sFps(sc);

    clear(body);

    // ── canvas (video + placeholder) ────────────────────────────────────────
    const canvas = el("div", "pm-canvas");
    if (_clips.length) {
      canvas.append(stage);  // pooled videos live here; only the active one is visible
    } else if (["queuing", "running", "pending"].includes(gen.state)) {
      const splash = el("div", "pm-gen-splash");
      splash.append(el("div", "pm-gen-icon", "⚙"));
      splash.append(el("div", "pm-gen-label", gen.msg || "Generating…"));
      canvas.append(splash);
    } else {
      const empty = el("div", "program-empty");
      empty.append(el("div", "reel", "🎬"));
      empty.append(el("div", null, p ? "No render yet" : "Open or create a project"));
      empty.append(el("div", "pj-meta", "Generate from the menu or timeline"));
      canvas.append(empty);
    }
    // generation progress overlay (visible even when there's already media). Built once per
    // store render; live progress ticks update it in place (see the gen-progress listener),
    // so the rest of the editor isn't rebuilt while a generation runs.
    _genMsgEl = _genBarEl = _genFillEl = null;
    if (["queuing", "running", "pending", "error"].includes(gen.state)) {
      const busy = gen.state !== "error";
      const ro = el("div", "gen-readout" + (gen.state === "error" ? " error" : ""));
      ro.append(el("span", "pulse"));
      _genMsgEl = el("span", null, gen.msg || gen.state); ro.append(_genMsgEl);
      if (busy) {
        const bar = el("div", "gen-bar"); _genBarEl = bar;
        _genFillEl = el("div", "gen-bar-fill");
        _genFillEl.style.width = gen.maxStep > 0 ? Math.min(100, Math.round((gen.step / gen.maxStep) * 100)) + "%" : "0%";
        if (!(gen.maxStep > 0)) bar.style.visibility = "hidden";
        bar.append(_genFillEl); ro.append(bar);
        const stop = el("button", "gen-interrupt", "■ Interrupt");
        stop.title = "Stop the current generation";
        stop.onclick = () => S.interrupt();
        ro.append(stop);
      }
      canvas.append(ro);
    }
    body.append(canvas);

    // ── minibar (segment strips + scrub needle) ──────────────────────────────
    const minibar = el("div", "pm-minibar");
    _minibarEl = minibar;
    if (p && _totalSecCur > 0) {
      let acc = 0;
      (p.scenes || []).forEach((sc) => {
        const d = sFrames(sc) / sFps(sc);
        const o = acc; acc += d;
        const rendered = !!((st.sceneRenders || {})[sc.id] || {}).media;
        const chip = el("div", "pm-chip" + (rendered ? " rendered" : ""));
        chip.style.width = (d / _totalSecCur * 100).toFixed(3) + "%";
        chip.title = `Scene ${(p.scenes || []).indexOf(sc) + 1}${rendered ? " (rendered)" : " (not rendered)"}`;
        // Drag-to-scrub on minibar
        chip.addEventListener("mousedown", (e) => {
          e.preventDefault();
          const scrub = (ev) => {
            const r = minibar.getBoundingClientRect();
            const frac = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width));
            _seek(frac * _totalSecCur);
          };
          scrub(e);
          const move = (ev) => scrub(ev);
          const up = () => { document.removeEventListener("mousemove", move); document.removeEventListener("mouseup", up); };
          document.addEventListener("mousemove", move);
          document.addEventListener("mouseup", up);
        });
        minibar.append(chip);
      });
      const needle = el("div", "pm-needle");
      needle.style.left = (Math.min(_phSec, _totalSecCur) / _totalSecCur * 100).toFixed(3) + "%";
      minibar.append(needle);
    }
    body.append(minibar);

    // ── transport bar ────────────────────────────────────────────────────────
    const transport = el("div", "pm-transport");
    _transportEl = transport;
    _renderTransport();
    body.append(transport);

    // header status
    clear(statusEl);
    if (gen.promptId) statusEl.append(el("span", null, gen.state.toUpperCase()));
  }

  S.subscribe(render);
})();
