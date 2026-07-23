// Big preview pane: plain <video>/<img>, not the Editor's timeline-oriented
// player.js. Polls /api/progress (step count) while a run is in flight and
// /api/projects/{id}/status/{prompt_id} for the final result.
(function () {
  const { el, clear } = window.dom;
  const API = window.MovieEditorAPI;

  function create(root) {
    const inner = el("div", "easy-preview-inner");
    const empty = el("div", "easy-preview-empty", "No generation yet");
    inner.append(empty);
    const progressWrap = el("div", "easy-progress");
    progressWrap.hidden = true;
    const bar = el("div", "easy-progress-bar");
    const label = el("div", "easy-progress-label");
    progressWrap.append(bar, label);
    root.append(inner, progressWrap);

    let pollTimer = null;
    let progressTimer = null;

    function stopPolling() {
      if (pollTimer) clearTimeout(pollTimer);
      if (progressTimer) clearInterval(progressTimer);
      pollTimer = progressTimer = null;
    }

    function showMedia(m, projectId) {
      clear(inner);
      const url = API.resultUrl(projectId, m);
      if (m.kind === "images") {
        const img = el("img");
        img.src = url;
        inner.append(img);
      } else {
        const video = el("video");
        video.src = url;
        video.controls = true;
        video.autoplay = true;
        video.loop = true;
        inner.append(video);
      }
    }

    function showEmpty(msg) {
      clear(inner);
      inner.append(el("div", "easy-preview-empty", msg || "No generation yet"));
    }

    function showError(msg) {
      clear(inner);
      inner.append(el("div", "easy-preview-empty easy-preview-error", msg));
    }

    async function watch(projectId, promptId, { onDone } = {}) {
      stopPolling();
      progressWrap.hidden = false;
      label.textContent = "Starting…";
      bar.style.width = "0%";

      progressTimer = setInterval(async () => {
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
          if (media) showMedia(media, projectId); else showEmpty("Generation finished but produced no media.");
          if (onDone) onDone(true, res);
          return;
        }
        if (res.state === "error") {
          stopPolling();
          progressWrap.hidden = true;
          showError(res.error || "Generation failed.");
          if (onDone) onDone(false, res);
          return;
        }
        pollTimer = setTimeout(poll, 1200);
      };
      poll();
    }

    return { showMedia, showEmpty, showError, watch, stopPolling };
  }

  window.EasyPreview = { create };
})();
