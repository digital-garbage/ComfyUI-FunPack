// Generate + poll status + render result media.
(function () {
  const API = window.MovieEditorAPI;
  let polling = null;

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, txt) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  };

  function status(msg, kind) {
    const box = $("player");
    box.replaceChildren();
    box.append(el("div", "player-status " + (kind || ""), msg));
  }

  function renderMedia(projectId, media) {
    const box = $("player");
    box.replaceChildren();
    if (!media.length) {
      box.append(el("div", "player-status warn", "Completed, but no video/image output found in history."));
      return;
    }
    for (const m of media) {
      const url = API.resultUrl(projectId, m);
      const fmt = (m.format || "").toLowerCase();
      const isVideo = m.kind !== "images" || fmt.includes("mp4") || fmt.includes("webm") ||
        /\.(mp4|webm|mov)$/i.test(m.filename);
      if (isVideo && !/\.gif$/i.test(m.filename)) {
        const v = el("video"); v.src = url; v.controls = true; v.autoplay = true; v.loop = true;
        v.className = "result-media"; box.append(v);
      } else {
        const img = el("img"); img.src = url; img.className = "result-media"; box.append(img);
      }
      const cap = el("div", "result-cap", m.filename);
      const a = el("a", "ghost", "open"); a.href = url; a.target = "_blank";
      cap.append(document.createTextNode("  ")); cap.append(a);
      box.append(cap);
    }
  }

  async function poll(projectId, promptId) {
    clearInterval(polling);
    polling = setInterval(async () => {
      try {
        const s = await API.status(projectId, promptId);
        if (s.state === "completed") {
          clearInterval(polling);
          renderMedia(projectId, s.media);
        } else {
          status(`Generating… (${s.state})`, "");
        }
      } catch (e) {
        clearInterval(polling);
        status("Status error: " + e.message, "warn");
      }
    }, 2000);
  }

  async function generate(projectId, onlyScene) {
    status(onlyScene ? "Queuing scene…" : "Queuing montage…", "");
    try {
      const r = await API.generate(projectId, onlyScene);
      if (!r.prompt_id) {
        status("Queued but no prompt id returned.", "warn");
        return;
      }
      status("Queued. Waiting for ComfyUI…", "");
      poll(projectId, r.prompt_id);
    } catch (e) {
      status("Generate failed: " + e.message, "warn");
    }
  }

  window.MoviePlayer = { generate };
})();
