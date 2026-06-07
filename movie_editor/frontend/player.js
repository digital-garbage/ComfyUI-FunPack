// Right-top zone: NLE program monitor.
//
// Architecture:
//   - <video> is created ONCE and re-appended on each store render (never destroyed),
//     so playback survives editor state changes without restarting.
//   - window.Player mediates the playhead between this module and timeline.js.
//   - renderedSegments (set by the store when generation completes) maps scene ids
//     to video files + time offsets; the player seeks within the correct segment.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const API = window.MovieEditorAPI;
  const body = document.getElementById("preview-body");
  const statusEl = document.getElementById("preview-status");

  // ── persistent video element ───────────────────────────────────────────────
  const video = document.createElement("video");
  video.className = "pm-video";
  video.preload = "auto";
  video.playsInline = true;

  // ── playhead state ─────────────────────────────────────────────────────────
  let _phSec = 0;
  let _playing = false;
  const _phListeners = new Set();

  function _notifyPh() {
    _phListeners.forEach((fn) => { try { fn(_phSec, _playing); } catch (_) {} });
  }

  // ── segment helpers ────────────────────────────────────────────────────────
  // Segments are [{sceneIds, media, startSec, durationSec}], computed from store state.
  let _segs = [];

  function _buildSegs(st) {
    if (!st.project) return [];
    const p = st.project;
    const sFps = (sc) => ((sc.fps_mode !== "project" && sc.fps != null) ? sc.fps : p.frame_rate) || 25;
    const sFrames = (sc) => ((sc.frames_mode !== "project" && sc.frames != null) ? sc.frames : p.num_frames_per_scene) || 1;

    // Map sceneId → timeline start time
    const startOf = {};
    let acc = 0;
    for (const sc of (p.scenes || [])) {
      startOf[sc.id] = acc;
      acc += sFrames(sc) / sFps(sc);
    }

    return (st.renderedSegments || []).flatMap((seg) => {
      const ids = seg.sceneIds || [];
      if (!ids.length || !seg.media) return [];
      const starts = ids.map((id) => startOf[id]).filter((s) => s != null);
      if (!starts.length) return [];
      const startSec = Math.min(...starts);
      const endSec = Math.max(...ids.map((id) => {
        const sc = (p.scenes || []).find((s) => s.id === id);
        return sc ? startOf[id] + sFrames(sc) / sFps(sc) : 0;
      }));
      return [{ ...seg, startSec, durationSec: Math.max(0, endSec - startSec) }];
    });
  }

  function _segAt(sec) {
    return _segs.find((s) => sec >= s.startSec - 0.05 && sec < s.startSec + s.durationSec + 0.05) || null;
  }

  // ── video src management ───────────────────────────────────────────────────
  let _currentUrl = null;
  let _loadedSeg = null;
  let _seekPending = null;   // time offset to apply after loadedmetadata
  let _playPending = false;  // start playback once the (newly loaded) source is ready

  // Load `seg` at `offsetSec`. If `play` is set, begin playback as soon as it's ready
  // (immediately for an already-loaded file, else after loadedmetadata — calling play()
  // right after a fresh load() rejects and would stall the chain at one clip).
  function _loadAndSeek(seg, offsetSec, play) {
    if (!seg?.media) return;
    const pid = S.get().project?.id;
    const url = API.resultUrl(pid, seg.media);
    _loadedSeg = seg;
    if (url === _currentUrl) {
      video.currentTime = Math.max(0, Math.min(offsetSec, video.duration - 0.02 || offsetSec));
      if (play) video.play().catch(() => {});
    } else {
      _currentUrl = url;
      _seekPending = offsetSec;
      _playPending = !!play;
      video.src = url;
      video.load();
    }
  }

  video.addEventListener("loadedmetadata", () => {
    if (_seekPending != null) {
      video.currentTime = Math.max(0, Math.min(_seekPending, video.duration - 0.02));
      _seekPending = null;
    }
    if (_playPending) { _playPending = false; video.play().catch(() => {}); }
  });

  video.addEventListener("timeupdate", () => {
    if (!_loadedSeg) return;
    _phSec = _loadedSeg.startSec + video.currentTime;
    _notifyPh();
  });

  // At end of segment: jump to next rendered segment or stop.
  video.addEventListener("ended", () => {
    _playing = false;
    if (_loadedSeg) {
      const endSec = _loadedSeg.startSec + _loadedSeg.durationSec;
      const next = _segs.find((s) => s.startSec >= endSec - 0.05 && s !== _loadedSeg);
      if (next) {
        _phSec = next.startSec;
        _playing = true;
        _loadAndSeek(next, 0, true);  // play once the next clip is loaded
      } else {
        _phSec = endSec;
      }
    }
    _notifyPh();
    _renderTransport();
  });

  // ── transport actions ──────────────────────────────────────────────────────
  function _seek(sec) {
    _phSec = Math.max(0, sec);
    const seg = _segAt(_phSec);
    if (seg) _loadAndSeek(seg, _phSec - seg.startSec);
    _notifyPh();
  }

  function _play() {
    let seg = _segAt(_phSec);
    if (!seg) seg = _segs.find((s) => s.startSec >= _phSec - 0.05);
    if (!seg) return;
    if (_phSec < seg.startSec) _phSec = seg.startSec;
    _playing = true;
    _loadAndSeek(seg, _phSec - seg.startSec, true);
    _notifyPh();
    _renderTransport();
  }

  function _pause() {
    video.pause();
    _playing = false;
    _playPending = false;  // cancel any queued autoplay
    _notifyPh();
    _renderTransport();
  }

  function _stop() {
    video.pause();
    _phSec = 0;
    _playing = false;
    const first = _segs[0];
    if (first) _loadAndSeek(first, 0);
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
      if (!video.videoWidth || video.readyState < 2) return null;
      const c = document.createElement("canvas");
      c.width = video.videoWidth; c.height = video.videoHeight;
      c.getContext("2d").drawImage(video, 0, 0);
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

  function _renderTransport() {
    if (!_transportEl) return;
    clear(_transportEl);
    const stopBtn = el("button", "ic-btn", "⏹"); stopBtn.title = "Stop"; stopBtn.onclick = _stop;
    const playBtn = el("button", "ic-btn" + (_playing ? " active" : ""), _playing ? "⏸" : "▶");
    playBtn.title = _playing ? "Pause" : "Play"; playBtn.onclick = () => _playing ? _pause() : _play();
    const tc = el("span", "pm-tc", _tc(_phSec, _fpsCur));
    _transportEl.append(stopBtn, playBtn, tc);

    // Anchor capture — only when a rendered frame is under the playhead.
    if (_loadedSeg) {
      const scenes = (S.get().project?.scenes || []).filter((s) => !s.excluded);
      if (scenes.length) {
        const sep = el("div", "pm-sep"); _transportEl.append(sep);

        const anchorSel = el("select", "pm-anchor-sel");
        anchorSel.title = "Scene that will receive this frame as its i2v anchor";
        // Default: currently selected scene, or first scene
        const curId = S.get().selectedSceneId;
        scenes.forEach((s, i) => {
          const label = `Scene ${i + 1}` + (s.text ? ": " + s.text.substring(0, 22) : "");
          const o = new Option(label, s.id);
          if (s.id === curId) o.selected = true;
          anchorSel.append(o);
        });

        const anchorBtn = el("button", "btn ghost tiny pm-anchor-btn", "📌 Use as anchor");
        anchorBtn.title = "Capture this frame and set it as the i2v starting image for the selected scene";
        anchorBtn.onclick = async () => {
          anchorBtn.disabled = true; anchorBtn.textContent = "Capturing…";
          try {
            const blob = await window.Player.captureFrame();
            if (!blob) {
              anchorBtn.textContent = "📌 Use as anchor";
              alert("No video frame available — make sure the playhead is on a rendered segment.");
              return;
            }
            const sceneId = anchorSel.value;
            const sc = scenes.find((s) => s.id === sceneId);
            const name = `anchor_scene${sc ? (S.get().project.scenes.indexOf(sc) + 1) : ""}_${Date.now()}.png`;
            const file = new File([blob], name, { type: "image/png" });
            await S.uploadMedia([file]);
            const bin = S.get().mediaBin;
            const asset = bin[bin.length - 1];
            if (!asset) { alert("Upload failed."); return; }
            // Store as generated_frame type (builder accepts it like "image")
            S.patchScene(sceneId, {
              source: { type: "generated_frame", media_ref: asset.id, target: "port:FunPackStudio.source_image" },
            });
            anchorBtn.textContent = "✓ Applied";
            setTimeout(() => _renderTransport(), 1600);
          } catch (e) {
            alert("Capture failed: " + e.message);
            anchorBtn.textContent = "📌 Use as anchor";
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
  window.Player.onPlayheadChanged(() => {
    _renderTransport();
    _renderNeedle();
  });

  // ── segment change detection ───────────────────────────────────────────────
  let _lastSegHash = "";
  let _lastProjectId = null;

  // ── full render (store subscription) ──────────────────────────────────────
  function render(st) {
    // Project switch: reset video state
    if (st.project?.id !== _lastProjectId) {
      _lastProjectId = st.project?.id || null;
      video.pause(); video.src = "";
      _currentUrl = null; _loadedSeg = null; _seekPending = null;
      _phSec = 0; _playing = false;
    }

    _segs = _buildSegs(st);
    const hash = _segs.map((s) => s.media?.filename || "").join("|");
    if (hash !== _lastSegHash) {
      _lastSegHash = hash;
      // Segments changed — re-resolve current position
      const seg = _segAt(_phSec);
      if (seg) _loadAndSeek(seg, _phSec - seg.startSec);
      else if (_segs.length && !_loadedSeg) _loadAndSeek(_segs[0], 0);
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
    if (_segs.length) {
      canvas.append(video);  // re-appending the same element — playback continues
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
    // generation progress overlay (visible even when there's already media)
    if (["queuing", "running", "pending", "error"].includes(gen.state)) {
      const busy = gen.state !== "error";
      const ro = el("div", "gen-readout" + (gen.state === "error" ? " error" : ""));
      ro.append(el("span", "pulse"));
      ro.append(el("span", null, gen.msg || gen.state));
      // step progress bar
      if (busy && gen.maxStep > 0) {
        const bar = el("div", "gen-bar");
        const fill = el("div", "gen-bar-fill");
        fill.style.width = Math.min(100, Math.round((gen.step / gen.maxStep) * 100)) + "%";
        bar.append(fill);
        ro.append(bar);
      }
      // interrupt
      if (busy) {
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
        const rendered = _segs.some((s) => (s.sceneIds || []).includes(sc.id));
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
