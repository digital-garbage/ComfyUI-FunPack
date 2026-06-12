// Program-monitor graphics overlays (text + images): preview compositing + drag positioning.
(function () {
  const S = window.Store;
  const API = window.MovieEditorAPI;
  let layer = null;
  let dragging = null;

  function canvasEl() {
    return document.querySelector("#preview-body .pm-canvas");
  }

  function ensureLayer() {
    const canvas = canvasEl();
    if (!canvas) return null;
    if (!layer || !layer.isConnected) {
      layer = document.createElement("div");
      layer.className = "pm-overlays";
      canvas.appendChild(layer);
    } else if (!canvas.contains(layer)) {
      canvas.appendChild(layer);
    }
    return layer;
  }

  function overlayVisible(ov, t) {
    const start = ov.start_sec || 0;
    const dur = ov.duration_sec || 0;
    return dur > 0 && t >= start && t < start + dur;
  }

  function placeBox(box, ov, rect) {
    const nx = ov.x != null ? ov.x : 0.5;
    const ny = ov.y != null ? ov.y : 0.5;
    const bw = box.offsetWidth || rect.width * (ov.scale || 0.35);
    const bh = box.offsetHeight || 40;
    const left = nx * rect.width - bw / 2;
    const top = ny * rect.height - bh / 2;
    box.style.left = left + "px";
    box.style.top = top + "px";
    box.style.opacity = ov.opacity != null ? ov.opacity : 1;
  }

  function startDrag(e, ov, box, rect) {
    if (S.get().selectedOverlayId !== ov.id) {
      S.selectOverlay(ov.id);
      return;
    }
    e.preventDefault();
    e.stopPropagation();
    const startX = e.clientX;
    const startY = e.clientY;
    const ox = ov.x != null ? ov.x : 0.5;
    const oy = ov.y != null ? ov.y : 0.5;
    dragging = { id: ov.id, startX, startY, ox, oy, rect };
    box.classList.add("dragging");

    const onMove = (ev) => {
      if (!dragging) return;
      const dx = ev.clientX - dragging.startX;
      const dy = ev.clientY - dragging.startY;
      const nx = Math.max(0, Math.min(1, dragging.ox + dx / dragging.rect.width));
      const ny = Math.max(0, Math.min(1, dragging.oy + dy / dragging.rect.height));
      box.style.left = (nx * dragging.rect.width - box.offsetWidth / 2) + "px";
      box.style.top = (ny * dragging.rect.height - box.offsetHeight / 2) + "px";
    };
    const onUp = (ev) => {
      if (!dragging) return;
      const dx = ev.clientX - dragging.startX;
      const dy = ev.clientY - dragging.startY;
      const nx = Math.max(0, Math.min(1, dragging.ox + dx / dragging.rect.width));
      const ny = Math.max(0, Math.min(1, dragging.oy + dy / dragging.rect.height));
      S.updateOverlayTrack(dragging.id, { x: nx, y: ny }, true);
      dragging = null;
      box.classList.remove("dragging");
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
    };
    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
  }

  function render() {
    const st = S.get();
    const host = ensureLayer();
    if (!host) return;
    if (st.mediaPreviewId) {
      host.replaceChildren();
      host.style.display = "none";
      return;
    }
    host.style.display = "";
    const tracks = st.project?.overlay_tracks || [];
    const t = window.Player?.getPlayhead?.() ?? 0;
    const sel = st.selectedOverlayId;
    const rect = host.getBoundingClientRect();
    host.replaceChildren();

    tracks.forEach((ov) => {
      if (!overlayVisible(ov, t)) return;
      const box = document.createElement("div");
      box.className = "pm-ov-box" + (ov.kind === "text" ? " text" : " image") + (sel === ov.id ? " selected" : "");
      box.title = ov.label || ov.text || "Overlay";

      if (ov.kind === "text") {
        box.textContent = ov.text || "Text";
        box.style.fontSize = (ov.font_size || 42) + "px";
        box.style.color = ov.color || "#ffffff";
      } else {
        const asset = (st.mediaBin || []).find((m) => m.id === ov.media_ref);
        if (asset?.kind === "image") {
          const img = document.createElement("img");
          img.src = API.mediaUrl(asset.id);
          img.draggable = false;
          box.appendChild(img);
          const scale = ov.scale != null ? ov.scale : 0.35;
          img.onload = () => {
            const w = Math.max(24, rect.width * scale);
            img.style.width = w + "px";
            img.style.height = "auto";
            placeBox(box, ov, rect);
          };
        }
      }

      placeBox(box, ov, rect);
      box.addEventListener("pointerdown", (e) => startDrag(e, ov, box, rect));
      box.addEventListener("click", (e) => { e.stopPropagation(); S.selectOverlay(ov.id); });
      host.append(box);
    });
  }

  if (window.ViewBus) {
    window.ViewBus.subscribePlayer(render);
  } else {
    S.subscribe(render);
  }
  if (window.Player) window.Player.onPlayheadChanged(render);
  render();
})();
