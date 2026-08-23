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
  const _urlClips = new Map(); // url -> Set<clip> (for segment load fallback)
  let _active = null;        // visible <video>
  let _seekPending = null;   // offset to apply once the active video has metadata
  let _playPending = false;
  let _seekRaf = null;       // coalesce rapid scrub seeks to one per frame
  let _pendingSeekSec = null;
  let _videoSeekTimer = null;
  let _pendingVideoTarget = null;
  let _pendingFxClip = null;
  let _pendingFxOffset = 0;
  let _seekToken = 0;
  let _scrubDepth = 0;       // suppress render()-driven re-seek while dragging playhead
  let _clipBoundaryToken = 0; // suppress double-advance while a boundary seek is in flight
  const _insPool = new Map(); // track id -> <audio>
  let _insTracks = [];

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
  const _clipUrl = (clip) => (clip && clip.streamUrl) || clip.binUrl || (clip.media ? _urlFor(clip.media) : "");
  const _clipInSec = (clip) => ((clip && clip.streamUrl) ? 0 : ((clip && clip.inSec) || 0));
  const _mediaChainKey = (clip) => {
    if (!clip?.media?.filename) return null;
    const m = clip.media;
    return `${m.type || "output"}:${m.subfolder || ""}:${m.filename}`;
  };
  const _clampT = (v, t) => Math.max(0, Math.min(t, (v.duration || (t + 1)) - 0.02));

  function _applyVideoTime(v, target, fast) {
    const t = _clampT(v, target);
    try {
      if (fast && typeof v.fastSeek === "function") v.fastSeek(t);
      else v.currentTime = t;
    } catch (_) {
      try { v.currentTime = t; } catch (_) {}
    }
  }

  function _flushVideoSeek(force) {
    if (_videoSeekTimer) {
      clearTimeout(_videoSeekTimer);
      _videoSeekTimer = null;
    }
    if (_pendingVideoTarget == null || !_active || !_pendingFxClip) return;
    const v = _active;
    const target = _pendingVideoTarget;
    const clip = _pendingFxClip;
    const offset = _pendingFxOffset;
    _pendingVideoTarget = null;
    _pendingFxClip = null;
    _pendingFxOffset = 0;
    const token = ++_seekToken;
    if (v.readyState >= 1) {
      _applyVideoTime(v, target, force !== true && !_playing);
      if (token === _seekToken) _applyFx(clip, offset);
    } else {
      v._pmSeekPending = target;
      _seekPending = target;
    }
  }

  function _scheduleVideoSeek(target, clip, offset) {
    _pendingVideoTarget = target;
    _pendingFxClip = clip;
    _pendingFxOffset = offset;
    if (_videoSeekTimer) return;
    _videoSeekTimer = setTimeout(() => _flushVideoSeek(false), _playing ? 0 : 24);
  }

  function _isSeeking() {
    if (_seekPending != null || _playPending) return true;
    const v = _active;
    if (!v) return false;
    if (v._pmSeekPending != null) return true;
    // Mid-reset/reload (no metadata yet) currentTime reads 0 — meaningless; treating it
    // as a seek keeps the tick from stomping the playhead back to the clip start.
    if (v.readyState < 1) return true;
    try { if (v.seeking) return true; } catch (_) {}
    return false;
  }

  function _clipWithinSec() {
    if (!_active || !_currentClip || _currentClip.pending) return 0;
    return _active.currentTime - _clipInSec(_currentClip);
  }

  function _registerUrlClip(url, clip) {
    if (!url || !clip) return;
    if (!_urlClips.has(url)) _urlClips.set(url, new Set());
    _urlClips.get(url).add(clip);
  }

  function _unregisterUrlClip(url, clip) {
    const set = _urlClips.get(url);
    if (!set) return;
    set.delete(clip);
    if (!set.size) _urlClips.delete(url);
  }

  function _fallbackSegmentClip(clip) {
    if (!clip?.streamUrl || clip._segFallback) return;
    clip._segFallback = true;
    const badUrl = clip.streamUrl;
    clip.streamUrl = null;
    _unregisterUrlClip(badUrl, clip);
    if (_currentClip === clip) {
      const offset = Math.max(0, _phSec - clip.startSec);
      _goto(clip, offset, _playing);
    }
    _lastSegHash = "";
  }

  // Multi-scene chain outputs share one ComfyUI file; deep seeks fail in the browser
  // unless the MP4 is faststart-trimmed like export. Serve per-scene segments instead.
  // Ghosts (removed scenes still previewing their render) and lone clips with a deep
  // in-point take the same path — they'd otherwise deep-seek the moov-at-end source
  // directly, which stalls/blacks the monitor when scrubbing across them.
  function _assignChainPreviewSegments(st, clips) {
    const pid = st.project?.id;
    if (!pid || !API.previewSegmentUrl) return;
    const groups = new Map();
    for (const c of clips) {
      if (c.pending || c.binUrl || !c.media) continue;
      if (!c.sceneId && !c.ghostId) continue;
      const k = _mediaChainKey(c);
      if (!k) continue;
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(c);
    }
    for (const list of groups.values()) {
      const shared = list.length >= 2;
      for (const c of list) {
        // Reverse is the one effect with no preview-side equivalent — the server has to
        // produce the reversed bytes, so the shallow-clip shortcut cannot apply to it.
        const wantsReverse = !!(c.sceneId && S.scene(c.sceneId)?.effects?.reverse);
        if (!wantsReverse && !shared && (c.inSec || 0) <= 0.5) continue;  // shallow single clip: direct URL is fine
        if (c.ghost) {
          // No scene server-side — the trim window travels entirely in the query.
          c.streamUrl = API.previewSegmentUrl(pid, c.ghostId, {
            media: c.media, renderIn: c.inSec || 0, dur: c.durationSec,
          });
          continue;
        }
        const sc = S.scene(c.sceneId);
        const sourceIn = sc?.source_in || 0;
        const renderIn = Math.max(0, (c.inSec || 0) - sourceIn);
        c.streamUrl = API.previewSegmentUrl(pid, c.sceneId, {
          media: c.media, renderIn, dur: c.durationSec, reverse: wantsReverse,
        });
      }
    }
  }

  function _onVideoSeeked(v) {
    if (v !== _active || !_currentClip) return;
    _clipBoundaryToken = 0;
    const within = Math.max(0, _clipWithinSec());
    try { _applyFx(_currentClip, within); } catch (_) {}
    if (_playing && v.paused) v.play().catch(() => {});
  }

  function _segmentSourceBadge(sceneId) {
    if (!sceneId) return "";
    const sc = S.scene(sceneId);
    if (S.isVideoClip?.(sc)) return "Video clip";
    const t = sc?.source?.type;
    if (t === "mixed") return "Mixed · ◐ anchor + ⇥ guides";
    return "";
  }

  function _binVideoAsset(st, mediaRef) {
    if (!mediaRef) return null;
    const asset = (st.mediaBin || []).find((m) => m.id === mediaRef);
    return asset?.kind === "video" ? asset : null;
  }

  function _binVideoUrl(st, mediaRef) {
    return _binVideoAsset(st, mediaRef) ? API.mediaUrl(mediaRef) : null;
  }

  function _clipDurationSec(sc, segDur) {
    if (sc.source_dur != null) return sc.source_dur;
    return segDur;
  }

  function _buildClips(st) {
    if (!st.project || !S.buildPreviewSegments) return [];
    const sr = st.sceneRenders || {};
    const out = [];
    for (const seg of S.buildPreviewSegments()) {
      const dur = seg.durationSec;
      const startSec = seg.offsetSec;
      if (seg.kind === "gap") {
        out.push({
          blank: true, gap: true, startSec, durationSec: dur,
        });
        continue;
      }
      if (seg.kind === "ghost") {
        const g = seg.ghost;
        if (!g.media) {
          if (g.pendingGen) {
            out.push({
              pending: true, ghost: true, ghostId: g.id, startSec, durationSec: dur,
              slate: "Removed — generating…",
              slateSub: "Preview updates when the in-flight run finishes",
            });
          }
          continue;
        }
        out.push({
          media: g.media, ghost: true, ghostId: g.id, startSec, durationSec: dur,
          inSec: g.inSec || 0, fx: g.effects || {}, vol: g.audio_volume != null ? g.audio_volume : 1,
          slate: "Removed — preview only",
        });
        continue;
      }
      const sc = seg.scene;
      const r = sr[sc.id];
      const fx = sc.effects || {};
      const vol = sc.audio_separated ? 0 : (sc.audio_volume != null ? sc.audio_volume : 1);
      const clipDur = _clipDurationSec(sc, dur);
      const srcType = sc.source?.type || "carry";
      const mediaRef = sc.source?.media_ref || null;
      const binUrl = _binVideoUrl(st, mediaRef);

      // Locked / imported video clips always play from the media bin (playhead-driven pool).
      if (S.isVideoClip?.(sc) && binUrl) {
        out.push({
          binUrl,
          sceneId: sc.id,
          startSec,
          durationSec: clipDur,
          inSec: sc.source_in || 0,
          fx,
          vol,
        });
        continue;
      }
      // Converted-back v2v scenes preview their source file until a render exists.
      if (srcType === "v2v" && binUrl && !r?.media) {
        out.push({
          binUrl,
          sceneId: sc.id,
          startSec,
          durationSec: clipDur,
          inSec: sc.source_in || 0,
          fx,
          vol,
        });
        continue;
      }
      if (S.isVideoClip?.(sc) && r?.media) {
        out.push({
          media: r.media,
          sceneId: sc.id,
          startSec,
          durationSec: clipDur,
          inSec: S.clipSourceInSec ? S.clipSourceInSec(sc, r) : (sc.source_in || 0),
          fx,
          vol,
        });
        continue;
      }

      const inSec = S.clipSourceInSec ? S.clipSourceInSec(sc, r) : ((sc.source_in || 0) + (r?.inSec || 0));
      if (r && r.media) {
        out.push({
          media: r.media, sceneId: sc.id, startSec, durationSec: dur, inSec,
          fx, vol,
        });
      } else if (!sc.excluded && S.isGenerativeScene && S.isGenerativeScene(sc)) {
        out.push({
          pending: true, sceneId: sc.id, startSec, durationSec: dur,
          slate: "Not generated yet",
          slateSub: "Generate this clip to include it in the next run",
        });
      }
    }
    _assignChainPreviewSegments(st, out);
    return out;
  }

  function _buildInsTracks(st) {
    if (!st.project) return [];
    const sr = st.sceneRenders || {};
    const pid = st.project.id;
    const bin = st.mediaBin || [];
    const out = [];
    for (const t of (st.project.audio_tracks || [])) {
      if (S.isSeparatedAudioTrack && S.isSeparatedAudioTrack(t)) {
        const url = S.separatedTrackAudioUrl ? S.separatedTrackAudioUrl(st, t) : null;
        if (!url) {
          const media = S.separatedTrackMedia ? S.separatedTrackMedia(t) : null;
          if (media && pid) {
            out.push({
              id: t.id,
              url: _urlFor(media),
              startSec: t.start_sec || 0,
              durSec: S.separatedTrackDurSec ? S.separatedTrackDurSec(t) : (t.source_dur || 1),
              inSec: S.separatedTrackInSec ? S.separatedTrackInSec(t) : (t.source_in_sec || 0),
              vol: t.volume != null ? t.volume : 1,
            });
          }
          continue;
        }
        out.push({
          id: t.id,
          url,
          startSec: t.start_sec || 0,
          durSec: S.separatedTrackDurSec ? S.separatedTrackDurSec(t) : (t.source_dur || 1),
          inSec: S.separatedTrackInSec ? S.separatedTrackInSec(t) : (t.source_in_sec || 0),
          vol: t.volume != null ? t.volume : 1,
        });
      } else if (S.isOverlayAudioTrack ? S.isOverlayAudioTrack(t) : (t.media_ref && t.kind !== "separated")) {
        const asset = t.media_ref ? bin.find((m) => m.id === t.media_ref) : null;
        const renderMedia = t.render_media;
        let url = null;
        let durSec = t.source_dur || 86400;
        if (asset) {
          url = API.mediaUrl(t.media_ref);
          durSec = t.source_dur || asset.duration_sec || 86400;
        } else if (renderMedia?.filename && pid) {
          url = _urlFor(renderMedia);
          durSec = t.source_dur || 86400;
        }
        if (!url) continue;
        out.push({
          id: t.id,
          url,
          startSec: t.start_sec || 0,
          durSec,
          inSec: t.source_in_sec || 0,
          vol: t.volume != null ? t.volume : 1,
        });
      }
    }
    return out;
  }

  function _syncInsPool() {
    const active = new Set(_insTracks.map((t) => t.id));
    for (const [id, a] of [..._insPool]) {
      if (active.has(id)) continue;
      a.pause(); _insPool.delete(id);
    }
  }

  function _pauseInsAudio() {
    for (const a of _insPool.values()) a.pause();
  }

  function _syncInsAudio() {
    for (const tr of _insTracks) {
      let a = _insPool.get(tr.id);
      if (!a) {
        a = document.createElement("audio");
        a.preload = "auto";
        _insPool.set(tr.id, a);
      }
      if (a.src !== tr.url) a.src = tr.url;
      const inRange = _phSec >= tr.startSec - 0.02 && _phSec < tr.startSec + tr.durSec;
      if (!_playing || !inRange) { a.pause(); continue; }
      const offset = tr.inSec + (_phSec - tr.startSec);
      const vol = Math.max(0, Math.min(1, +tr.vol || 0));
      if (Math.abs(a.currentTime - offset) > 0.12) {
        try { a.currentTime = Math.max(0, offset); } catch (_) {}
      }
      if (a.volume !== vol) a.volume = vol;
      if (a.paused) a.play().catch(() => {});
    }
  }

  function _hasPlayableClipMedia(clip) {
    return !!(clip && !clip.pending && (clip.media || clip.binUrl || clip.streamUrl));
  }

  function _isBlankProgram() {
    const total = S.previewTotalSec ? S.previewTotalSec() : 0;
    if (total <= 0 || !S.hasOverlayOrAudioContent?.()) return false;
    return !_clips.some(_hasPlayableClipMedia);
  }

  function _blankClip() {
    const dur = Math.max(0.01, S.previewTotalSec ? S.previewTotalSec() : _totalSecCur || 0.01);
    return { blank: true, startSec: 0, durationSec: dur };
  }

  function _clipAt(sec) {
    const clip = _clips.find((c) => sec >= c.startSec - 0.05 && sec < c.startSec + c.durationSec - 0.001);
    if (clip) return clip;
    if (_isBlankProgram() && sec >= 0 && sec < (_totalSecCur || S.previewTotalSec?.() || 0)) return _blankClip();
    return null;
  }
  function _clipFrom(sec) {  // first playable clip at/after sec (skip pending gaps)
    const at = _clipAt(sec);
    if (at && !at.pending) return at;
    const next = _clips.find((c) => !c.pending && c.startSec >= sec - 0.05);
    if (next) return next;
    if (_isBlankProgram()) return _blankClip();
    return null;
  }

  let _slateEl = null;
  let _badgeEl = null;
  function _showSlate(title, sub) {
    if (!_slateEl) return;
    _slateEl.style.display = title ? "" : "none";
    if (!title) return;
    const t = _slateEl.querySelector(".pm-slate-title");
    const s = _slateEl.querySelector(".pm-slate-sub");
    if (t) t.textContent = title;
    if (s) { s.textContent = sub || ""; s.style.display = sub ? "" : "none"; }
  }
  function _showBadge(text) {
    if (!_badgeEl) return;
    _badgeEl.textContent = text || "";
    _badgeEl.style.display = text ? "" : "none";
  }

  // ── video pool management ─────────────────────────────────────────────────────
  function _clearFxStyles(v) {
    if (!v) return;
    v.style.filter = "";
    const fx = v._pmFx;
    if (fx) {
      fx.zoom.style.transform = "";
      fx.viewport.style.opacity = "";
    }
  }

  // A failed load must not leave a permanently dead element in the pool: the URL never
  // changes, so without a retry the clip stays unplayable until a full page reload (seen
  // when previewing run 1's video while ComfyUI was GIL-bound saving run 2). Bounded
  // backoff, cache-busted so a poisoned browser cache entry can't be replayed.
  const MAX_VIDEO_RETRIES = 4;
  function _scheduleVideoRetry(url, v, delayMs) {
    const n = (v._pmRetries || 0) + 1;
    if (n > MAX_VIDEO_RETRIES) return;
    v._pmRetries = n;
    clearTimeout(v._pmRetryTimer);
    v._pmRetryTimer = setTimeout(() => {
      v._pmRetryTimer = null;
      if (pool.get(url) !== v) return; // dropped from the pool meanwhile
      const sep = url.includes("?") ? "&" : "?";
      v.src = url + sep + "r=" + Date.now();
      v.load();
    }, delayMs != null ? delayMs : 1500 * n);
  }

  // A media reset (browser dropping an aborted/poisoned stream, a src swap that never
  // started, the resource being reclaimed) leaves an element with no metadata AND no load
  // in flight. Nothing will ever fire `loadedmetadata` for it, and because `error` never
  // fired it has v.error === null — which is what every other recovery path here keys on.
  // Left alone it sits behind "Video is loading…" forever and the transport freezes, since
  // _isSeeking() treats readyState < 1 as a seek in progress and the tick refuses to
  // advance the playhead. networkState 2 is NETWORK_LOADING: bytes are on the way, leave
  // it be.
  function _isStalledEmpty(v) {
    if (!v || v.readyState >= 1) return false;
    try { return v.networkState !== 2; } catch (_) { return false; }
  }

  // Confirm the stall before acting on it: a fresh `v.src = …` also reads as EMPTY for a
  // beat before the resource selection algorithm starts, so an immediate check would
  // cancel loads that were about to succeed.
  const STALL_WATCHDOG_MS = 1200;
  function _armStallWatchdog(url, v) {
    clearTimeout(v._pmStallTimer);
    v._pmStallTimer = setTimeout(() => {
      v._pmStallTimer = null;
      if (pool.get(url) !== v || !_urlClips.has(url)) return;
      if (!_isStalledEmpty(v)) return;   // loading, or metadata already arrived
      if (v._pmRetryTimer) return;       // an error-driven retry already owns recovery
      _scheduleVideoRetry(url, v, 0);
    }, STALL_WATCHDOG_MS);
  }

  // If the active clip's element changed readiness, repaint the slate/badge over it.
  function _refreshClipStatusUi(url) {
    if (_currentClip && !_currentClip.pending && _clipUrl(_currentClip) === url) {
      _applyClipUi(_currentClip);
    }
  }

  // Failed element → ask the server WHY (HEAD /result now answers 503 while the render is
  // still being written to disk) so the slate can say "processing" vs "loading", and so a
  // long GIL-bound save can't exhaust the retry budget: server-confirmed "processing"
  // retries on a steady cadence with the budget refunded, everything else backs off.
  function _classifyVideoError(url, v) {
    const probe = url.split("&r=")[0].split("?r=")[0];
    fetch(probe, { method: "HEAD", cache: "no-store" })
      .then((r) => (r.status === 503 ? "processing" : "loading"))
      .catch(() => "loading")
      .then((status) => {
        if (pool.get(url) !== v || !_urlClips.has(url)) return;
        v._pmStatus = status;
        if (status === "processing") {
          v._pmRetries = 0;
          _scheduleVideoRetry(url, v, 3000);
        } else {
          if ((v._pmRetries || 0) >= MAX_VIDEO_RETRIES) v._pmStatus = "failed";
          _scheduleVideoRetry(url, v);
        }
        _refreshClipStatusUi(url);
      });
  }

  function _ensureVideo(url, clip) {
    let v = pool.get(url);
    if (v) {
      if (clip) _registerUrlClip(url, clip);
      return v;
    }
    const viewport = document.createElement("div");
    viewport.className = "pm-fx-viewport";
    const zoom = document.createElement("div");
    zoom.className = "pm-fx-zoom";
    v = document.createElement("video");
    // preload starts as metadata: with preload=auto EVERY pooled clip downloaded at full
    // tilt simultaneously — on a tunneled rental instance that starves the browser's
    // per-origin connection pool and aborts the stream actually being watched.
    // _updatePreloadHints() promotes the current clip and its neighbours to auto.
    v.className = "pm-video"; v.preload = "metadata"; v.playsInline = true; v.src = url;
    v._pmFx = { viewport, zoom };
    v._pmStatus = "loading"; // → "ready" on loadedmetadata; "processing"/"failed" from _classifyVideoError
    zoom.append(v);
    viewport.append(zoom);
    v.addEventListener("loadedmetadata", () => {
      v._pmRetries = 0; // healthy again — restore the full retry budget
      clearTimeout(v._pmStallTimer); v._pmStallTimer = null;
      v._pmStatus = "ready";
      _refreshClipStatusUi(url);
      const pending = v._pmSeekPending != null ? v._pmSeekPending : (v === _active ? _seekPending : null);
      if (pending != null) {
        _applyVideoTime(v, pending, !_playing);
        v._pmSeekPending = null;
        if (v === _active) _seekPending = null;
      }
      if (v !== _active) return;
      // Resume on _playing too: after a mid-playback media reset there is no _playPending,
      // but the element came back paused at the restored position and must keep going.
      if (_playPending || (_playing && v.paused)) { _playPending = false; v.play().catch(() => {}); }
      if (_currentClip) _applyFx(_currentClip, Math.max(0, _phSec - _currentClip.startSec));
    });
    v.addEventListener("seeked", () => { _onVideoSeeked(v); });
    // A media reset — load()/src swap from the retry path, or the browser dropping an
    // aborted/poisoned stream (rapid play/pause over the network) — rewinds the element
    // to 0 and would restart the clip from its beginning. Stash the playhead position so
    // loadedmetadata seeks back instead; _onVideoSeeked then resumes playback if playing.
    v.addEventListener("emptied", () => {
      // A media reset means the bytes are gone — downgrade so the slate covers the element
      // instead of it blinking black while the retry reloads.
      if (v._pmStatus === "ready") { v._pmStatus = "loading"; _refreshClipStatusUi(url); }
      // The reset may or may not be followed by a load. Check back and restart it if not.
      _armStallWatchdog(url, v);
      if (v !== _active || !_currentClip || _currentClip.pending) return;
      if (_clipUrl(_currentClip) !== url) return;
      if (v._pmSeekPending == null) {
        v._pmSeekPending = _clipInSec(_currentClip) + Math.max(0, _phSec - _currentClip.startSec);
      }
    });
    v.addEventListener("error", () => {
      const clips = _urlClips.get(url);
      if (clips) [...clips].forEach((c) => _fallbackSegmentClip(c));
      // Segment clips fell back to their base URL above (and unregistered); anything still
      // registered here has no alternative URL — classify the failure (processing vs dead)
      // and retry this one instead of staying dead.
      if (_urlClips.has(url)) {
        v._pmStatus = "loading";
        _refreshClipStatusUi(url);
        _classifyVideoError(url, v);
      }
    });
    v.addEventListener("ended", () => {
      if (v !== _active || _isSeeking() || _clipBoundaryToken) return;
      _advance();
    });
    stage.append(viewport); pool.set(url, v);
    if (clip) _registerUrlClip(url, clip);
    return v;
  }

  // Materializing a <video> per clip URL for the WHOLE timeline breaks big (montage)
  // timelines: Chrome refuses new media players past ~75 per page, and every element's
  // preload=metadata fetch of a per-clip preview segment (each a server-side ffmpeg
  // encode) fights over the browser's ~6 per-origin connections on a tunneled instance —
  // so clips in the back half of the timeline sit "loading" forever while the front half
  // plays. Only clips in a window around the playhead get elements; the window keeps cut
  // boundaries instant (upcoming clips are materialized ahead) while the element count
  // stays flat no matter how long the timeline is.
  const POOL_BEHIND = 2, POOL_AHEAD = 8, POOL_MAX = 16;

  function _windowUrls() {
    const urls = new Set();
    let idx = _currentClip ? _clips.indexOf(_currentClip) : -1;
    if (idx < 0) {
      const at = _clipAt(_phSec);
      idx = at ? _clips.indexOf(at) : -1;
    }
    if (idx < 0) idx = _clips.findIndex(_hasPlayableClipMedia);
    if (idx >= 0) {
      for (let i = Math.max(0, idx - POOL_BEHIND); i < _clips.length && i <= idx + POOL_AHEAD; i++) {
        const c = _clips[i];
        if (!c || c.pending || c.blank) continue;
        const u = _clipUrl(c);
        if (u) urls.add(u);
      }
    }
    return urls;
  }

  function _releasePooledVideo(u, v) {
    pool.delete(u);
    if (_active === v) _active = null;
    v.pause();
    clearTimeout(v._pmRetryTimer);
    clearTimeout(v._pmStallTimer); v._pmStallTimer = null;
    // Release the socket BEFORE detaching: a detached media element that still has a
    // src holds its per-origin connection until GC (media-connection-pool class).
    try { v.removeAttribute("src"); v.load(); } catch (_) {}
    v._pmFx?.viewport.remove();
  }

  // Materialize window members that are missing; evict far elements once the pool
  // outgrows POOL_MAX (the slack keeps recently-visited clips warm for back-and-forth
  // scrubbing instead of re-fetching on every jump). Runs at every playhead-affecting
  // step via _updatePreloadHints, and is idempotent/cheap when nothing changed.
  function _syncPoolWindow() {
    const want = _windowUrls();
    for (const u of want) {
      if (pool.has(u)) continue;
      const clip = _clips.find((c) => _clipUrl(c) === u);
      if (clip) _ensureVideo(u, clip);
    }
    if (pool.size <= POOL_MAX) return;
    for (const [u, v] of [...pool]) {  // Map order = insertion order ⇒ oldest first
      if (pool.size <= POOL_MAX) break;
      if (want.has(u) || v === _active) continue;
      _releasePooledVideo(u, v);
    }
  }

  function _syncPool() {
    _urlClips.clear();
    // Timeline rebuild: drop elements whose URL no longer exists on any clip at all
    // (deleted/regenerated clips) regardless of the window.
    const live = new Set(_clips.filter((c) => c.media || c.binUrl || c.streamUrl).map((c) => _clipUrl(c)));
    for (const [u, v] of [...pool]) {
      if (live.has(u)) continue;
      _releasePooledVideo(u, v);
    }
    _syncPoolWindow();
    for (const [u, v] of pool) {
      const clip = _clips.find((c) => _clipUrl(c) === u);
      if (clip) _registerUrlClip(u, clip);
      // Clip rebuild (e.g. a run just completed): give exhausted-but-errored elements a
      // fresh recovery attempt — the failure was likely the server being busy back then.
      if ((v.error || _isStalledEmpty(v)) && !v._pmRetryTimer) {
        v._pmRetries = 0; _scheduleVideoRetry(u, v);
      }
    }
  }

  function _resetPool() {
    _stopTick();
    for (const [u, v] of [...pool]) _releasePooledVideo(u, v);
    pool.clear(); _active = null; _currentClip = null; _seekPending = null; _playPending = false;
    _urlClips.clear();
    _clipBoundaryToken = 0;
    _pendingSeekSec = null;
    _scrubDepth = 0;
    if (_seekRaf) { cancelAnimationFrame(_seekRaf); _seekRaf = null; }
    _flushVideoSeek(true);
    _pauseInsAudio();
    for (const a of _insPool.values()) a.removeAttribute("src");
    _insPool.clear();
    _insTracks = [];
  }

  function _setActive(v) {
    if (_active === v) return;
    if (_active) {
      _active.pause();
      _active.classList.remove("active");
      _active._pmFx?.viewport.classList.remove("active");
      _clearFxStyles(_active);
    }
    _active = v;
    if (v) {
      v.classList.add("active");
      v._pmFx?.viewport.classList.add("active");
      if (v._pmSeekPending != null && v.readyState >= 1) {
        _applyVideoTime(v, v._pmSeekPending, !_playing);
        v._pmSeekPending = null;
        _seekPending = null;
      }
    }
  }

  function _zoomEffectParams(fx, durSec, fps) {
    const nframes = Math.max(1, Math.round(durSec * fps));
    const ratio = Math.max(0.01, Math.min(0.5, +(fx.zoom_ratio ?? 0.15)));
    let start = Math.max(0, Math.min(Math.round(+(fx.zoom_start_frame ?? 0)), Math.max(0, nframes - 1)));
    const defaultLen = Math.min(Math.max(1, Math.floor(nframes / 4)), 25);
    let length = Math.round(+(fx.zoom_frames ?? defaultLen));
    length = Math.max(1, Math.min(length, Math.max(1, nframes - start)));
    return { ratio, start, length, nframes };
  }

  function _zoomScale(fx, withinSec, durSec, fps) {
    const mode = fx.zoom;
    if (!mode || mode === "none") return 1;
    const { ratio, start, length, nframes } = _zoomEffectParams(fx, durSec, fps);
    const frame = Math.max(0, Math.min(nframes - 1, Math.round(withinSec * fps)));
    const end = 1 + ratio;
    if (frame < start) return mode === "in" ? 1 : end;
    if (frame >= start + length) return mode === "in" ? end : 1;
    const t = (frame - start) / length;
    return mode === "in" ? 1 + ratio * t : end - ratio * t;
  }

  // Live preview approximation of a clip's video effects (the render does the real thing).
  // within = seconds into the clip. Blur -> CSS blur on the video; zoom uses a clipped
  // viewport + inner scale layer (matches ffmpeg zoompan: fixed frame, centered crop).
  // fade -> viewport opacity. Computed from the playhead so it tracks scrubbing too.
  function _applyFx(clip, within) {
    const v = _active; if (!v) return;
    const fxLayer = v._pmFx;
    const fx = (clip && clip.fx) || {};
    const dur = (clip && clip.durationSec) || 0;
    const blur = +fx.blur || 0;
    v.style.filter = blur > 0 ? `blur(${(blur * 8).toFixed(1)}px)` : "";
    // Flips: on the element itself, so they compose with the zoom layer's transform rather
    // than overwrite it. Mirrors ffmpeg hflip/vflip exactly.
    const flips = [];
    if (fx.flip_h) flips.push("scaleX(-1)");
    if (fx.flip_v) flips.push("scaleY(-1)");
    v.style.transform = flips.join(" ");
    // Fit: `contain` letterboxes (ffmpeg scale=decrease + pad), `cover` fills the frame and
    // crops the overflow (ffmpeg scale=increase + crop). Same geometry either way.
    v.style.objectFit = fx.fit === "fill" ? "cover" : "contain";
    // Crop-inset punches in by trimming each edge, then rescaling to the frame — which is a
    // static scale of 1/(1-2f) on the clipped viewport. Multiplied with Ken Burns, so the
    // two compose instead of one silently winning.
    const inset = Math.max(0, Math.min(0.4, +fx.crop_inset || 0));
    const cropScale = inset > 0 ? 1 / (1 - 2 * inset) : 1;
    const scale = _zoomScale(fx, within, dur, _fpsCur || 25) * cropScale;
    if (fxLayer?.zoom) {
      fxLayer.zoom.style.transform = scale !== 1 ? `scale(${scale.toFixed(4)})` : "";
    }
    let op = 1;
    const fi = +fx.fade_in || 0, fo = +fx.fade_out || 0;
    if (fi > 0 && within < fi) op = Math.max(0, within / fi);
    if (fo > 0 && dur > 0 && within > dur - fo) op = Math.min(op, Math.max(0, (dur - within) / fo));
    if (fxLayer?.viewport) {
      fxLayer.viewport.style.opacity = op < 1 ? op.toFixed(3) : "";
    }
    // audio: per-clip original volume, muted when the project drops original audio. Guarded —
    // a non-finite volume throws on assignment, which (in the rAF tick) would freeze playback.
    const keepOrig = (S.get().project || {}).keep_original_audio !== false;
    let vol = (clip && clip.vol != null) ? +clip.vol : 1;
    // Read the scene's volume LIVE: the timeline slider changes audio_volume via
    // patchSceneQuiet (no re-render), so the cached clip.vol would stay stale —
    // a mute that never un-mutes when dragged back up. _applyFx runs every frame
    // during playback, so a live read tracks the slider immediately.
    if (clip && clip.sceneId && S.scene) {
      const sc = S.scene(clip.sceneId);
      if (sc) vol = sc.audio_separated ? 0 : (sc.audio_volume != null ? sc.audio_volume : 1);
    }
    if (!isFinite(vol)) vol = 1;
    const nv = keepOrig ? Math.max(0, Math.min(1, vol)) : 0;
    if (v.volume !== nv) { try { v.volume = nv; } catch (_) {} }
  }

  function _applyClipUi(clip) {
    if (!clip || clip.pending) {
      _showBadge("");
      _showSlate(clip?.slate || "Not generated yet", clip?.slateSub);
      return;
    }
    // A clip whose element hasn't decoded yet must say so instead of showing a black or
    // half-loaded <video>: "processing" = the server answered 503 (render still being
    // written to disk), "loading" = bytes on the way / retrying, "failed" = retry budget
    // exhausted (re-armed on the next clip rebuild). Playback through _playPending already
    // waits for loadedmetadata, so the preview only becomes live once the video really is.
    const v = clip.blank ? null : pool.get(_clipUrl(clip));
    if (v && v._pmStatus !== "ready") {
      _showBadge("");
      if (v._pmStatus === "processing") {
        _showSlate("Video is processing…", "Preview isn't available yet");
      } else if (v._pmStatus === "failed") {
        _showSlate("Preview unavailable", "The video failed to load — it will retry when the timeline updates");
      } else {
        _showSlate("Video is loading…", "Preview isn't available yet");
      }
      return;
    }
    _showSlate("");
    const badge = clip.ghost ? (clip.slate || "Removed — preview only")
      : _segmentSourceBadge(clip.sceneId);
    _showBadge(badge);
    if (_badgeEl && badge.startsWith("Mixed")) _badgeEl.classList.add("mixed-mode");
    else if (_badgeEl) _badgeEl.classList.remove("mixed-mode");
  }

  // Promote the clip under the playhead and its neighbours to preload=auto (cut boundaries
  // stay instant); everything else idles at metadata so the pool doesn't saturate the
  // browser's per-origin connections. A hint change never interrupts an in-flight load.
  function _updatePreloadHints() {
    _syncPoolWindow();  // playhead moved — materialize upcoming clips, evict far ones
    if (!pool.size) return;
    const hot = new Set();
    const add = (c) => {
      if (!c || c.pending || c.blank) return;
      const u = _clipUrl(c);
      if (u) hot.add(u);
    };
    const idx = _currentClip ? _clips.indexOf(_currentClip) : -1;
    if (idx >= 0) {
      add(_clips[idx - 1]); add(_clips[idx]); add(_clips[idx + 1]);
    } else {
      add(_clips.find(_hasPlayableClipMedia));
    }
    for (const [u, v] of pool) {
      const want = hot.has(u) ? "auto" : "metadata";
      if (v.preload !== want) v.preload = want;
    }
  }

  function _sameVideoUrl(a, b) {
    if (!a || !b) return false;
    const ua = _clipUrl(a), ub = _clipUrl(b);
    return !!ua && ua === ub;
  }

  // Show `clip` at `offset` seconds into the clip; play if requested. Seeks to the clip's
  // in-point + offset within the (preloaded) source video.
  function _goto(clip, offset, play) {
    if (!clip) return;
    if (clip.blank) {
      _currentClip = clip;
      _setActive(null);
      _showSlate("");
      _showBadge("");
      if (play) _startTick();
      return;
    }
    if (clip.pending) {
      _currentClip = clip;
      _setActive(null);
      _applyClipUi(clip);
      if (play) _startTick();
      return;
    }
    if (!clip.media && !clip.binUrl && !clip.streamUrl) return;
    const v = _ensureVideo(_clipUrl(clip), clip);
    _currentClip = clip; _setActive(v);
    _applyClipUi(clip);
    const target = _clipInSec(clip) + Math.max(0, offset);
    if (v.readyState >= 1) {
      if (play && Math.abs(v.currentTime - _clampT(v, target)) > 0.04) _clipBoundaryToken++;
      _applyVideoTime(v, target, !play);
      v._pmSeekPending = null;
      _seekPending = null;
      if (play) { v.play().catch(() => {}); _startTick(); } else { _playPending = false; _clipBoundaryToken = 0; }
    } else {
      // Waiting on loadedmetadata is only correct while a load is actually running. On a
      // stalled-empty element it waits forever — and this is the branch Play goes through,
      // which is why play/pause reads as dead. Restart it, with a fresh budget: the user
      // asking again is a better reason to retry than any timer.
      if (_isStalledEmpty(v) && !v._pmRetryTimer) {
        v._pmRetries = 0;
        _scheduleVideoRetry(_clipUrl(clip), v, 0);
      }
      v._pmSeekPending = target;
      _seekPending = target;
      _playPending = !!play;
      if (play) { _clipBoundaryToken++; _startTick(); }
      else _clipBoundaryToken = 0;
    }
    _applyFx(clip, Math.max(0, offset));
    _updatePreloadHints();
  }

  // Advance to the next clip at a clip's out-point (or stop). Same-source contiguous
  // clips (a split seam) continue without reseeking; otherwise seek; no next → stop.
  function _advance() {
    if (!_currentClip) { _pause(); return; }
    const end = _currentClip.startSec + _currentClip.durationSec;
    const next = _clips.find((c) => !c.pending && c.startSec >= end - 0.02 && c !== _currentClip);
    if (!next) { _phSec = end; _pause(); return; }
    const prev = _currentClip;
    const sameSource = next.media && prev.media
      && _urlFor(next.media) === _urlFor(prev.media);
    const sameBin = next.binUrl && prev.binUrl && next.binUrl === prev.binUrl;
    const sharedSource = sameSource || sameBin;
    const contiguous = sharedSource && !prev.streamUrl && !next.streamUrl
      && Math.abs(_clipInSec(next) - (_clipInSec(prev) + prev.durationSec)) < 0.05;
    _phSec = next.startSec;
    _currentClip = next;
    const crossClip = next.sceneId !== prev.sceneId || next.ghostId !== prev.ghostId;
    const needSeek = !sharedSource || !contiguous || crossClip;
    if (needSeek) _goto(next, 0, _playing);
    else {
      _clipBoundaryToken = 0;
      _applyClipUi(next);
      if (_active) {
        const within = Math.max(0, _clipWithinSec());
        try { _applyFx(next, within); } catch (_) {}
      }
      if (_playing) _startTick();
      _updatePreloadHints();
    }
    _notifyPh();
  }

  // ── playback tick (precise out-points, so deleted parts aren't played) ─────────
  let _raf = null;
  function _startTick() { if (!_raf) _raf = requestAnimationFrame(_tick); }
  function _stopTick() { if (_raf) { cancelAnimationFrame(_raf); _raf = null; } }
  function _tick() {
    _raf = null;
    if (!_playing || !_currentClip) return;
    if (_currentClip.blank) {
      const within = _phSec - _currentClip.startSec + (1 / 60);
      if (within >= _currentClip.durationSec - 0.001) {
        _phSec = _currentClip.startSec + _currentClip.durationSec;
        if (_currentClip.gap) _advance();
        else _pause();
      } else {
        _phSec = _currentClip.startSec + within;
        _syncInsAudio();
        _notifyPh();
      }
      if (_playing) _raf = requestAnimationFrame(_tick);
      return;
    }
    if (_currentClip.pending) {
      const within = _phSec - _currentClip.startSec + (1 / 60);
      if (within >= _currentClip.durationSec - 0.001) _advance();
      else { _phSec = _currentClip.startSec + within; _syncInsAudio(); _notifyPh(); }
      if (_playing) _raf = requestAnimationFrame(_tick);
      return;
    }
    if (_isSeeking()) {
      if (_playing) _raf = requestAnimationFrame(_tick);
      return;
    }
    if (!_active) {
      if (_currentClip.media || _currentClip.binUrl || _currentClip.streamUrl) {
        _goto(_currentClip, Math.max(0, _phSec - _currentClip.startSec), true);
        if (_playing) _raf = requestAnimationFrame(_tick);
        return;
      }
      if (_playing) _pause();
      return;
    }
    const within = _clipWithinSec();
    if (within >= _currentClip.durationSec - 0.001) { _advance(); }
    else { _phSec = _currentClip.startSec + Math.max(0, within); try { _applyFx(_currentClip, within); } catch (_) {} _syncInsAudio(); _notifyPh(); }
    if (_playing) _raf = requestAnimationFrame(_tick);
  }

  // ── transport actions ──────────────────────────────────────────────────────
  function _seekNow(sec) {
    // Any transport scrub/seek exits the transient media-bin preview so the program
    // resumes following the playhead (otherwise it stays frozen on the previewed image).
    if (S.get().mediaPreviewId) S.clearMediaPreview();
    _phSec = Math.max(0, sec);
    const clip = _clipAt(_phSec);
    if (!clip) {
      _flushVideoSeek(true);
      _currentClip = null;
      _setActive(null);
      _showSlate("");
      _showBadge("");
      _syncInsAudio();
      _notifyPh();
      return;
    }
    const offset = _phSec - clip.startSec;
    const prevClip = _currentClip;
    const sameClip = prevClip === clip
      || (prevClip && clip && prevClip.sceneId === clip.sceneId && _sameVideoUrl(prevClip, clip));
    const sameVideo = !clip.pending && _active && _sameVideoUrl(prevClip || clip, clip)
      && (clip.media || clip.binUrl);
    if (sameVideo && _active.readyState >= 1) {
      const crossScene = prevClip && clip && prevClip !== clip
        && (prevClip.sceneId !== clip.sceneId || Math.abs(_clipInSec(prevClip) - _clipInSec(clip)) > 0.05);
      _currentClip = clip;
      _applyClipUi(clip);
      const target = _clipInSec(clip) + Math.max(0, offset);
      if (crossScene) _flushVideoSeek(true);
      _scheduleVideoSeek(target, clip, offset);
    } else if (!sameClip || clip.pending) {
      _flushVideoSeek(true);
      _goto(clip, offset, false);
    } else if (!clip.pending && (clip.media || clip.binUrl)) {
      const target = _clipInSec(clip) + Math.max(0, offset);
      const v = _active;
      if (v && v.readyState >= 1) _scheduleVideoSeek(target, clip, offset);
      else _goto(clip, offset, false);
    }
    _syncInsAudio();
    _updatePreloadHints();
    _notifyPh();
  }

  function _seek(sec) {
    _pendingSeekSec = Math.max(0, sec);
    if (_seekRaf) return;
    _seekRaf = requestAnimationFrame(() => {
      _seekRaf = null;
      const s = _pendingSeekSec;
      _pendingSeekSec = null;
      if (s == null) return;
      _seekNow(s);
    });
  }

  function _flushSeek() {
    if (_seekRaf) {
      cancelAnimationFrame(_seekRaf);
      _seekRaf = null;
    }
    if (_pendingSeekSec != null) {
      const s = _pendingSeekSec;
      _pendingSeekSec = null;
      _seekNow(s);
    }
    _flushVideoSeek(true);
  }

  function _beginScrub() { _scrubDepth++; }
  function _endScrub() {
    _scrubDepth = Math.max(0, _scrubDepth - 1);
    _flushSeek();
  }

  function _play() {
    if (S.get().mediaPreviewId) S.clearMediaPreview();
    // At (or past) the end of the timeline → loop the playhead back to the start.
    if (_phSec >= (_totalSecCur || 0) - 0.05 || !_clipFrom(_phSec)) _phSec = 0;
    const clip = _clipFrom(_phSec);
    if (!clip) return;
    if (_phSec < clip.startSec) _phSec = clip.startSec;
    _playing = true;
    _goto(clip, _phSec - clip.startSec, true);
    _syncInsAudio();
    _notifyPh();
    _renderTransport();
  }

  function _pause() {
    if (_active) _active.pause();
    const wasPlaying = _playing;
    _playing = false;
    _playPending = false;
    _clipBoundaryToken = 0;
    _stopTick();
    _pauseInsAudio();
    _renderTransport();
    if (wasPlaying) render(S.get());
    _notifyPh();
  }

  function _stop() {
    if (_active) _active.pause();
    _phSec = 0;
    _playing = false;
    _playPending = false;
    _stopTick();
    const first = _clips.find(_hasPlayableClipMedia) || (_isBlankProgram() ? _blankClip() : _clips[0]);
    if (first) _goto(first, 0, false);
    _notifyPh();
    _renderTransport();
  }

  // ── public Player API (consumed by timeline.js + inspector) ──────────────
  window.Player = {
    seek: _seek,
    flushSeek: _flushSeek,
    beginScrub: _beginScrub,
    endScrub: _endScrub,
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

  function _ensureBlankFrame(p) {
    if (!p) return;
    const pw = p.width || 768, ph = p.height || 512;
    let blank = stage.querySelector(".pm-blank-frame");
    if (!blank) {
      blank = document.createElement("div");
      blank.className = "pm-blank-frame";
      stage.appendChild(blank);
    }
    blank.style.aspectRatio = `${pw} / ${ph}`;
  }

  function _ensureProgramStage(st) {
    if (!body) return;
    if (!_clips.some(_hasPlayableClipMedia) && !_isBlankProgram()) return;
    if (body.contains(stage)) {
      if (_isBlankProgram()) _ensureBlankFrame(st?.project || S.get().project);
      return;
    }
    let canvas = body.querySelector(".pm-canvas");
    if (!canvas) {
      canvas = el("div", "pm-canvas");
      body.append(canvas);
    }
    if (!canvas.contains(stage)) canvas.insertBefore(stage, canvas.firstChild);
    if (_isBlankProgram()) _ensureBlankFrame(st?.project || S.get().project);
  }

  function _ensureStageMounted() { _ensureProgramStage(S.get()); }

  // ── full render (store subscription) ──────────────────────────────────────
  function render(st) {
    // Project switch: drop the whole pool and reset.
    if (st.project?.id !== _lastProjectId) {
      _lastProjectId = st.project?.id || null;
      _resetPool();
      _phSec = 0; _playing = false; _lastSegHash = "";
    }

    _clips = _buildClips(st);
    _insTracks = _buildInsTracks(st);
    _syncInsPool();
    _syncPool();  // preload current clips, drop stale ones (timeline is dynamic)

    // Re-resolve the clip under the playhead each render (positions change with edits).
    const hash = _clips.map((c) => {
      if (c.pending) return "pending:" + (c.sceneId || c.ghostId || "");
      if (c.ghost) return "ghost:" + c.ghostId;
      return (_clipUrl(c) || "x") + "@" + c.startSec.toFixed(2) + "+" + _clipInSec(c).toFixed(2);
    }).join("|");
    if (hash !== _lastSegHash) {
      _lastSegHash = hash;
      if (!st.mediaPreviewId && _scrubDepth === 0) {
        const clip = _clipAt(_phSec);
        if (clip) _goto(clip, _phSec - clip.startSec, _playing);
        else if (_clips.length && !_active) _goto(_clips[0], 0, false);
        else if (_isBlankProgram()) _goto(_blankClip(), _phSec, _playing);
      }
    }
    _updatePreloadHints();

    const p = st.project;
    const gen = st.gen || { state: "idle", media: [] };
    _fpsCur = p?.frame_rate || 25;
    _totalSecCur = S.previewTotalSec ? S.previewTotalSec() : 0;

    // Rebuilding the monitor DOM during playback detaches <video> nodes and kills multi-scene
    // chain preview mid-stream (often visible from scene 3 onward once store ticks catch up).
    if (_scrubDepth > 0 || _playing) {
      _ensureStageMounted();
      return;
    }

    clear(body);

    const previewAsset = st.mediaPreviewId
      ? (st.mediaBin || []).find((m) => m.id === st.mediaPreviewId)
      : null;
    const previewOverlay = previewAsset && (previewAsset.kind === "image" || previewAsset.kind === "audio" || previewAsset.kind === "video");

    // ── canvas (video + placeholder) ────────────────────────────────────────
    const canvas = el("div", "pm-canvas" + (previewOverlay ? " media-previewing" : ""));
    _slateEl = el("div", "pm-slate");
    _slateEl.style.display = "none";
    _slateEl.append(el("div", "pm-slate-title"));
    _slateEl.append(el("div", "pm-slate-sub"));
    _badgeEl = el("div", "pm-segment-badge");
    _badgeEl.style.display = "none";

    if (_clips.some(_hasPlayableClipMedia)) {
      canvas.append(stage);  // pooled videos live here; only the active one is visible
      canvas.append(_slateEl);
      canvas.append(_badgeEl);
    } else if (_isBlankProgram()) {
      canvas.append(stage);
      _ensureBlankFrame(p);
      canvas.append(_slateEl);
      canvas.append(_badgeEl);
    } else if (previewOverlay) {
      // image/audio still use a lightweight overlay; video uses the pool above
    } else if (["queuing", "running", "pending"].includes(gen.state)) {
      const splash = el("div", "pm-gen-splash");
      splash.append(el("div", "pm-gen-icon", "⚙"));
      splash.append(el("div", "pm-gen-label", gen.msg || "Generating…"));
      canvas.append(splash);
    } else {
      const empty = el("div", "program-empty");
      empty.append(el("div", "reel", "🎬"));
      empty.append(el("div", null, p ? "No render yet" : "Open or create a project"));
      empty.append(el("div", "pj-meta", window.FunPackMode?.isSimple()
        ? "Write a prompt, then press Generate"
        : "Use Generate in the timeline header"));
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
      if (!busy) {
        // A blocking message names a node, an input and a fix. Being able to take it
        // somewhere — a note, a bug report — beats retyping it from a screenshot.
        const copy = el("button", "btn ghost tiny gen-copy", "⧉");
        copy.title = "Copy this message";
        copy.onclick = async (e) => {
          e.stopPropagation();
          try {
            await navigator.clipboard.writeText(gen.msg || "");
            copy.textContent = "✓";
            setTimeout(() => { copy.textContent = "⧉"; }, 1200);
          } catch (_) { copy.textContent = "✕"; }
        };
        ro.append(copy);
      }
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
    if (previewAsset?.kind === "image") {
      const layer = el("div", "pm-media-preview");
      const label = el("div", "pm-media-preview-label", previewAsset.name || "Media preview");
      const img = el("img", "pm-media-preview-img");
      img.src = API.mediaUrl(previewAsset.id);
      img.draggable = false;
      img.onclick = (e) => e.stopPropagation();
      layer.append(label, img);
      canvas.append(layer);
      canvas.onclick = (e) => {
        if (!e.target.closest(".pm-media-preview-img")) S.clearMediaPreview();
      };
    } else if (previewAsset?.kind === "audio") {
      const layer = el("div", "pm-media-preview");
      const label = el("div", "pm-media-preview-label", previewAsset.name || "Audio preview");
      const aud = el("audio", "pm-media-preview-aud");
      aud.src = API.mediaUrl(previewAsset.id);
      aud.controls = true;
      aud.onclick = (e) => e.stopPropagation();
      layer.append(label, aud);
      canvas.append(layer);
      canvas.onclick = (e) => {
        if (!e.target.closest(".pm-media-preview-aud")) S.clearMediaPreview();
      };
    } else if (previewAsset?.kind === "video") {
      const layer = el("div", "pm-media-preview");
      const label = el("div", "pm-media-preview-label", previewAsset.name || "Video preview");
      const vid = el("video", "pm-media-preview-vid");
      vid.src = API.mediaUrl(previewAsset.id);
      vid.controls = true;
      vid.playsInline = true;
      vid.preload = "metadata";
      vid.onclick = (e) => e.stopPropagation();
      layer.append(label, vid);
      canvas.append(layer);
      canvas.onclick = (e) => {
        if (!e.target.closest(".pm-media-preview-vid")) S.clearMediaPreview();
      };
    }
    body.append(canvas);

    // ── minibar (segment strips + scrub needle) ──────────────────────────────
    const minibar = el("div", "pm-minibar");
    _minibarEl = minibar;
    _totalSecCur = S.previewTotalSec ? S.previewTotalSec() : 0;
    if (p && _totalSecCur > 0 && S.buildPreviewSegments) {
      const previewSegs = S.buildPreviewSegments();
      if (previewSegs.length) {
        previewSegs.forEach((seg) => {
        const d = seg.durationSec;
        let cls = "pm-chip";
        let title = "";
        if (seg.kind === "ghost") {
          cls += " ghost" + (seg.ghost.pendingGen ? " pending" : "");
          title = seg.ghost.pendingGen
            ? "Removed scene — generation in flight"
            : "Removed scene (preview only — not in next run)";
        } else if (seg.kind === "gap") {
          cls += " gap";
          title = `Pause (${d.toFixed(2)}s)`;
        } else {
          const sc = seg.scene;
          if (!sc) return;
          const stype = sc.source?.type || "empty";
          const isVideo = S.isVideoClip && S.isVideoClip(sc);
          const rendered = isVideo
            ? !!(sc.source?.media_ref || ((st.sceneRenders || {})[sc.id] || {}).media)
            : !!((st.sceneRenders || {})[sc.id] || {}).media;
          const stale = !isVideo && rendered && S.renderIsStale?.(sc.id);
          cls += rendered ? " rendered" : " pending";
          if (isVideo) { cls += " video"; title = "Video clip (locked)"; }
          else if (stale) cls += " stale";
          else if (stype === "mixed") { cls += " mixed"; title = "Mixed · anchor + prior guides"; }
          else if (stype === "carry") { cls += " carry"; title = "Carry · prior guides only"; }
          const idx = (p.scenes || []).indexOf(sc) + 1;
          if (!title) title = `Scene ${idx}${rendered ? (stale ? " (previous generation)" : " (rendered)") : " (not rendered)"}`;
          else if (!isVideo) title = `Scene ${idx} · ${title}${rendered ? (stale ? " · previous generation" : "") : " · not rendered"}`;
        }
        const chip = el("div", cls);
        chip.style.width = (d / _totalSecCur * 100).toFixed(3) + "%";
        chip.title = title;
        // Drag-to-scrub on minibar
        chip.addEventListener("mousedown", (e) => {
          e.preventDefault();
          _beginScrub();
          const scrub = (ev) => {
            const r = minibar.getBoundingClientRect();
            const frac = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width));
            _seek(frac * _totalSecCur);
          };
          scrub(e);
          const move = (ev) => scrub(ev);
          const up = () => {
            document.removeEventListener("mousemove", move);
            document.removeEventListener("mouseup", up);
            _endScrub();
          };
          document.addEventListener("mousemove", move);
          document.addEventListener("mouseup", up);
        });
        minibar.append(chip);
        });
      } else {
        const chip = el("div", "pm-chip blank-program");
        chip.style.width = "100%";
        chip.title = "Overlays / audio";
        chip.addEventListener("mousedown", (e) => {
          e.preventDefault();
          _beginScrub();
          const scrub = (ev) => {
            const r = minibar.getBoundingClientRect();
            const frac = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width));
            _seek(frac * _totalSecCur);
          };
          scrub(e);
          const move = (ev) => scrub(ev);
          const up = () => {
            document.removeEventListener("mousemove", move);
            document.removeEventListener("mouseup", up);
            _endScrub();
          };
          document.addEventListener("mousemove", move);
          document.addEventListener("mouseup", up);
        });
        minibar.append(chip);
      }
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

    try { window.OverlayPreview?.refresh?.(); } catch (_) {}
  }

  if (window.ViewBus) window.ViewBus.subscribePlayer(render);
  else S.subscribe(render);
})();
