// "Gallery" picker: choose an already-uploaded media-bin file as the i2v anchor,
// without re-uploading it. Standalone modal (same modal-overlay/.modal shell as
// project_menu.js) since Easy Gen has no Media Bin panel of its own.
(function () {
  const { el, clear } = window.dom;
  const API = window.MovieEditorAPI;
  const S = window.Store;
  let overlay = null;

  function close() {
    if (overlay) { overlay.remove(); overlay = null; }
  }

  function cellFor(m) {
    const cell = el("div", "gal-cell");
    cell.title = m.name || m.id;
    const thumb = el("div", "gal-thumb");
    const url = API.mediaUrl(m.id);
    if (m.kind === "video") {
      const v = el("video");
      v.src = url; v.muted = true; v.preload = "metadata";
      thumb.append(v);
    } else {
      const img = el("img");
      img.src = url; img.loading = "lazy";
      thumb.append(img);
    }
    cell.append(thumb);
    cell.append(el("div", "gal-name", m.name || m.id));
    cell.onclick = async () => {
      S.setSceneMedia(m.id, m.kind === "video" ? "video" : "image");
      await S.save();
      close();
    };
    return cell;
  }

  async function open() {
    if (!S.get().project) return;
    close();
    overlay = el("div", "modal-overlay");
    const box = el("div", "modal modal-wide");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", "Gallery — choose an anchor"));
    const x = el("button", "btn ghost tiny", "✕");
    x.onclick = close;
    head.append(el("div", "modal-head-right"), x);
    box.append(head);

    const content = el("div", "modal-content gal-grid");
    content.append(el("div", "sw-hint", "Loading…"));
    box.append(content);

    overlay.append(box);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.addEventListener("keydown", function onKey(e) {
      if (e.key === "Escape") { close(); document.removeEventListener("keydown", onKey); }
    });
    document.body.append(overlay);

    try {
      const r = await API.listMedia();
      const items = (r.media || []).filter((m) => m.kind === "image" || m.kind === "video");
      clear(content);
      if (!items.length) {
        content.append(el("div", "sw-hint", "No uploaded media yet — use Upload first."));
        return;
      }
      items.forEach((m) => content.append(cellFor(m)));
    } catch (e) {
      clear(content);
      content.append(el("div", "sw-hint", "Could not load media: " + (e.message || e)));
    }
  }

  window.EasyGallery = { open, close };
})();
