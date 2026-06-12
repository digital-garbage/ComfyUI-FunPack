// Program-monitor graphics overlays (text + images): preview compositing + drag positioning.
(function () {
  const S = window.Store;
  const API = window.MovieEditorAPI;
  const OUI = window.OverlayUI;
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
    box.style.left = (nx * rect.width) + "px";
    box.style.top = (ny * rect.height) + "px";
    box.style.opacity = ov.opacity != null ? ov.opacity : 1;
  }

  function applyTextStyle(box, ov) {
    box.textContent = ov.text || "Text";
    box.style.fontSize = (ov.font_size || 42) + "px";
    box.style.color = ov.color || "#ffffff";
    if (OUI) box.style.fontFamily = OUI.cssFamily(ov.font_family);
  }

  function startDrag(e, ov, box, rect) {
    e.preventDefault();
    e.stopPropagation();
    S.selectOverlay(ov.id);

    const boxRect = box.getBoundingClientRect();
    const grabOffX = e.clientX - (boxRect.left + boxRect.width / 2);
    const grabOffY = e.clientY - (boxRect.top + boxRect.height / 2);

    const posAt = (clientX, clientY) => {
      const cx = clientX - rect.left - grabOffX;
      const cy = clientY - rect.top - grabOffY;
      return {
        nx: Math.max(0, Math.min(1, cx / rect.width)),
        ny: Math.max(0, Math.min(1, cy / rect.height)),
      };
    };

    dragging = { id: ov.id, grabOffX, grabOffY, rect, box };
    box.classList.add("dragging");

    const onMove = (ev) => {
      if (!dragging) return;
      const { nx, ny } = posAt(ev.clientX, ev.clientY);
      dragging.box.style.left = (nx * dragging.rect.width) + "px";
      dragging.box.style.top = (ny * dragging.rect.height) + "px";
    };
    const onUp = (ev) => {
      if (!dragging) return;
      const { nx, ny } = posAt(ev.clientX, ev.clientY);
      const id = dragging.id;
      dragging.box.classList.remove("dragging");
      dragging = null;
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
      S.updateOverlayTrack(id, { x: nx, y: ny }, true);
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
    if (dragging) return;
    host.style.display = "";
    const tracks = S.sortedOverlayTracks ? S.sortedOverlayTracks() : (st.project?.overlay_tracks || []);
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
        applyTextStyle(box, ov);
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
      host.append(box);
    });
  }

  if (window.ViewBus?.subscribeOverlays) {
    window.ViewBus.subscribeOverlays(render);
  } else {
    S.subscribe(render);
  }
  if (window.Player) window.Player.onPlayheadChanged(render);
  render();
})();
