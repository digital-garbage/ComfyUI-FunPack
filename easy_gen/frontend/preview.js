// Big preview pane: plain <video>/<img>, not the Editor's timeline-oriented
// player.js. Polls /api/progress (step count) while a run is in flight and
// /api/projects/{id}/status/{prompt_id} for the final result.
//
// Display priority: generated result > uploaded source > empty message. The
// corner ✕ closes whichever of those two media layers is currently showing —
// closing a generated result reveals the uploaded source underneath (if any),
// closing the uploaded source clears the upload itself (via onClearUpload).
(function () {
  const { el, clear } = window.dom;
  const API = window.MovieEditorAPI;

  function create(root) {
    const inner = el("div", "easy-preview-inner");
    const closeBtn = el("button", "easy-preview-close", "✕");
    closeBtn.type = "button";
    closeBtn.title = "Close";
    closeBtn.hidden = true;
    const badge = el("div", "easy-preview-badge", "Uploaded source");
    badge.hidden = true;
    const progressWrap = el("div", "easy-progress");
    progressWrap.hidden = true;
    const bar = el("div", "easy-progress-bar");
    const label = el("div", "easy-progress-label");
    // Same "■ Interrupt" affordance as the Cutting Room's player readout, and it lives in the
    // progress panel so it only exists while there's something to stop.
    const stopBtn = el("button", "gen-interrupt easy-progress-stop", "■ Interrupt");
    stopBtn.type = "button";
    stopBtn.title = "Stop the current generation";
    const progressRow = el("div", "easy-progress-row");
    progressRow.append(label, stopBtn);
    progressWrap.append(bar, progressRow);
    root.append(inner, closeBtn, badge, progressWrap);

    let pollTimer = null;
    let progressTimer = null;

    // generated/upload hold the last-drawn payload so the ✕ handler knows what
    // layer it's dismissing; emptyMsg is what to fall back to once both are gone.
    let generated = null;   // { media, projectId } | null
    let upload = null;      // { url, kind } | null
    let errorMsg = null;
    let emptyMsg = "No generation yet";
    let onClearUpload = null;

    function stopPolling() {
      if (pollTimer) clearTimeout(pollTimer);
      if (progressTimer) clearInterval(progressTimer);
      pollTimer = progressTimer = null;
    }

    function mediaEl(url, kind, opts = {}) {
      if (kind === "video") {
        const video = el("video");
        video.src = url;
        video.controls = true;
        // Not the `autoplay` attribute: that makes the browser call .play() internally, and
        // if this element gets replaced/removed before the fetch starts playing (e.g. the
        // user immediately picks a different Gallery item), that internal promise rejects
        // with "fetching process...aborted by the user agent" — surfacing as an unhandled
        // rejection banner, since we never had a handle on it to catch. Call .play()
        // ourselves so we own the promise.
        if (opts.autoplay) {
          video.loop = true;
          video.play().catch(() => {});
        }
        return video;
      }
      const img = el("img");
      img.src = url;
      return img;
    }

    // The store notifies on every health poll / prompt keystroke, and each of those calls
    // setUpload()/render() again — without this, draw() rebuilt the <video> from scratch
    // every time (new element = autoplay/loop reapplied), silently resuming anything the
    // user had paused. Skip the rebuild when the content actually shown hasn't changed.
    let _lastSig = null;
    function _sig() {
      if (errorMsg) return "error:" + errorMsg;
      if (generated) return "generated:" + generated.projectId + ":" + generated.media.filename;
      if (upload) return "upload:" + upload.url + ":" + upload.kind;
      return "empty:" + emptyMsg;
    }

    function draw() {
      const sig = _sig();
      if (sig === _lastSig) return;
      _lastSig = sig;
      clear(inner);
      badge.hidden = true;
      if (errorMsg) {
        closeBtn.hidden = true;
        inner.append(el("div", "easy-preview-empty easy-preview-error", errorMsg));
      } else if (generated) {
        closeBtn.hidden = false;
        const url = API.resultUrl(generated.projectId, generated.media);
        inner.append(mediaEl(url, generated.media.kind === "images" ? "image" : "video", { autoplay: true }));
      } else if (upload) {
        closeBtn.hidden = false;
        badge.hidden = false;
        inner.append(mediaEl(upload.url, upload.kind));
      } else {
        closeBtn.hidden = true;
        inner.append(el("div", "easy-preview-empty", emptyMsg));
      }
    }

    closeBtn.onclick = () => {
      if (generated) {
        generated = null;
        draw();
      } else if (upload) {
        upload = null;
        draw();
        if (onClearUpload) onClearUpload();
      }
    };

    function setUpload(url, kind) {
      upload = url ? { url, kind } : null;
      draw();
    }

    function setGenerated(media, projectId) {
      generated = media ? { media, projectId } : null;
      draw();
    }

    function showEmpty(msg) {
      errorMsg = null;
      generated = null;
      emptyMsg = msg || "No generation yet";
      draw();
    }

    function showError(msg) {
      errorMsg = msg;
      draw();
    }

    async function watch(projectId, promptId, { onDone } = {}) {
      stopPolling();
      errorMsg = null;
      progressWrap.hidden = false;
      label.textContent = "Starting…";
      bar.style.width = "0%";
      // An interrupted job lands in ComfyUI's history as status_str "error" with no media —
      // indistinguishable from a real failure at the API level. Remember that the user asked
      // for it so the poll below reports "stopped", not "Generation failed".
      let interrupted = false;
      stopBtn.disabled = false;
      stopBtn.textContent = "■ Interrupt";
      stopBtn.onclick = async () => {
        interrupted = true;
        stopBtn.disabled = true;
        stopBtn.textContent = "Interrupting…";
        label.textContent = "Interrupting…";
        try { await API.interrupt(); }
        catch (e) {
          // The run may well still be going — say so instead of leaving a dead button.
          interrupted = false;
          stopBtn.disabled = false;
          stopBtn.textContent = "■ Interrupt";
          label.textContent = "Could not interrupt: " + (e.message || e);
        }
      };

      progressTimer = setInterval(async () => {
        if (interrupted) return; // don't paint "Sampling…" over "Interrupting…"
        try {
          const pr = await API.progress();
          if (pr.max > 0) {
            const pct = Math.min(100, Math.round((pr.value / pr.max) * 100));
            bar.style.width = pct + "%";
            label.textContent = `Sampling… ${pr.value}/${pr.max}`;
          }
        } catch (_) { /* transient */ }
      }, 700);

      const poll = async () => {
        let res;
        try {
          res = await API.status(projectId, promptId);
        } catch (e) {
          stopPolling();
          progressWrap.hidden = true;
          showError("Status check failed: " + (e.message || e));
          if (onDone) onDone(false);
          return;
        }
        if (res.state === "completed") {
          stopPolling();
          progressWrap.hidden = true;
          const media = res.media && res.media.length ? res.media[res.media.length - 1] : null;
          // A run stopped mid-sampling can still have written partial media — show it if so,
          // it's the whole point of interrupting a long run you can already tell is wrong.
          if (media) setGenerated(media, projectId);
          else if (interrupted) showEmpty("Generation stopped.");
          else showEmpty("Generation finished but produced no media.");
          if (onDone) onDone(!interrupted, res);
          return;
        }
        if (res.state === "error") {
          stopPolling();
          progressWrap.hidden = true;
          if (interrupted) showEmpty("Generation stopped.");
          else showError(res.error || "Generation failed.");
          if (onDone) onDone(false, res);
          return;
        }
        pollTimer = setTimeout(poll, 1200);
      };
      poll();
    }

    return {
      setUpload,
      setGenerated,
      setOnClearUpload(fn) { onClearUpload = fn; },
      showEmpty,
      showError,
      watch,
      stopPolling,
    };
  }

  window.EasyPreview = { create };
})();
