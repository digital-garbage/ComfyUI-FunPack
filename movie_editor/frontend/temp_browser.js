// Temp files browser: a lightweight media-bin view over ComfyUI's temp directory
// (scene previews & other transient outputs that are wiped on restart). Opened from the
// Settings menu. Each item can be opened in a new tab or saved to disk — no in-app player.
(function () {
  const { el, clear } = window.dom;
  const API = window.MovieEditorAPI;

  let overlay = null;
  let files = [];
  // Probe <video> elements still holding a network connection. Chrome keeps stalled media
  // connections open and allows only ~6 per origin — a grid of live <video> thumbnails
  // starved the pool so /view tabs wouldn't load and even generate/API calls hung until a
  // ComfyUI reboot. Thumbnails are snapshotted to a canvas and the probe released at once.
  const _probes = new Set();

  function _releaseProbe(vid) {
    _probes.delete(vid);
    try { vid.removeAttribute("src"); vid.load(); } catch (_) {}
  }

  function _releaseAllProbes() { [..._probes].forEach(_releaseProbe); }

  function close() {
    _releaseAllProbes();
    if (overlay) { overlay.remove(); overlay = null; }
  }

  function _fmtSize(n) {
    if (!n && n !== 0) return "";
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  }

  function _appendThumb(thumb, f) {
    const url = API.tempFileUrl(f);
    if (f.kind === "image") {
      const img = el("img");
      img.src = url; img.loading = "lazy";
      thumb.append(img);
      return;
    }
    if (f.kind === "video") {
      const ph = el("span", "media-icon media-vid-ph", "▶");
      thumb.append(ph);
      const vid = document.createElement("video");
      vid.muted = true; vid.preload = "metadata"; vid.playsInline = true;
      vid.src = url;
      _probes.add(vid);
      vid.onerror = () => _releaseProbe(vid);
      vid.onloadeddata = () => {
        if (thumb.isConnected && !thumb.querySelector("canvas")) {
          const canvas = document.createElement("canvas");
          canvas.className = "media-vid-thumb";
          canvas.width = vid.videoWidth || 320;
          canvas.height = vid.videoHeight || 180;
          try {
            canvas.getContext("2d").drawImage(vid, 0, 0, canvas.width, canvas.height);
            ph.remove();
            thumb.append(canvas);
          } catch (_) {}
        }
        _releaseProbe(vid); // frame captured — free the media connection immediately
      };
      return;
    }
    thumb.append(el("span", "media-icon media-aud-ph", "♪"));
  }

  function _card(f) {
    const card = el("div", "media-card tmp-card");
    const thumb = el("div", "media-thumb");
    _appendThumb(thumb, f);
    card.append(thumb);

    const nameRow = el("div", "media-name-row");
    const nameEl = el("div", "media-name", f.filename);
    nameEl.title = f.subfolder ? `${f.subfolder}/${f.filename}` : f.filename;
    nameRow.append(nameEl);
    card.append(nameRow);

    const meta = el("div", "pj-meta tmp-meta", _fmtSize(f.size));
    card.append(meta);

    const acts = el("div", "tmp-card-acts");
    if (f.kind === "video" || f.kind === "image" || f.kind === "audio") {
      const open = el("button", "btn ghost tiny", f.kind === "video" ? "↗ Open video" : "↗ Open");
      open.title = "Open in a new browser tab";
      open.onclick = () => window.open(API.tempFileUrl(f), "_blank");
      acts.append(open);
    }
    const save = el("button", "btn ghost tiny", "⤓ Save");
    save.title = "Download to your computer";
    save.onclick = async () => {
      save.disabled = true;
      try { await API.downloadTempFile(f); }
      catch (e) { alert("Save failed: " + (e.message || e)); }
      finally { save.disabled = false; }
    };
    acts.append(save);
    card.append(acts);
    return card;
  }

  function renderBody(body) {
    clear(body);
    if (!files.length) {
      body.append(el("div", "pj-meta media-grid-empty", "No temp media right now. Scene previews and other transient outputs appear here while ComfyUI is running."));
      return;
    }
    const grid = el("div", "media-grid tmp-grid");
    files.forEach((f) => grid.append(_card(f)));
    body.append(grid);
  }

  async function refresh(body, countEl) {
    _releaseAllProbes(); // drop probes from the previous grid before rebuilding
    body.append(el("div", "pj-meta", "Loading…"));
    try {
      files = (await API.listTemp()).files || [];
    } catch (e) {
      clear(body);
      body.append(el("div", "pj-meta media-grid-empty", "Could not list temp files: " + (e.message || e)));
      return;
    }
    if (countEl) countEl.textContent = files.length ? `${files.length} file(s)` : "";
    clear(body);
    renderBody(body);
  }

  function open() {
    close();
    overlay = el("div", "modal-overlay");
    const box = el("div", "modal");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", "Temp files"));
    const hr = el("div", "modal-head-right");
    const countEl = el("span", "pj-meta tmp-count");
    const refreshBtn = el("button", "btn ghost tiny", "⟳ Refresh");
    const x = el("button", "btn ghost tiny", "✕"); x.onclick = close;
    hr.append(countEl, refreshBtn, x);
    head.append(hr); box.append(head);

    const content = el("div", "modal-content");
    content.append(el("div", "es-hint",
      "Transient ComfyUI outputs — scene previews and the like. These are wiped when ComfyUI restarts, so save anything worth keeping."));
    const body = el("div", "tmp-body");
    content.append(body);
    box.append(content);
    overlay.append(box);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.body.append(overlay);

    refreshBtn.onclick = () => refresh(body, countEl);
    refresh(body, countEl);
  }

  window.TempBrowserModal = { open, close };
})();
