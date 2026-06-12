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

  function clamp01(n, fallback) {
    if (!Number.isFinite(n)) return fallback;
    return Math.max(0, Math.min(1, n));
  }

  function layerRect() {
    const host = ensureLayer();
    if (!host) return null;
    const rect = host.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return null;
    return rect;
  }

  function placeBox(box, ov, rect) {
    const nx = ov.x != null ? ov.x : 0.5;
    const ny = ov.y != null ? ov.y : 0.5;
    box.style.left = (clamp01(nx, 0.5) * rect.width) + "px";
    box.style.top = (clamp01(ny, 0.5) * rect.height) + "px";
    box.style.opacity = ov.opacity != null ? ov.opacity : 1;
  }

  function applyTextStyle(box, ov) {
    box.textContent = ov.text || "Text";
    box.style.fontSize = (ov.font_size || 42) + "px";
    box.style.color = ov.color || "#ffffff";
    if (OUI) box.style.fontFamily = OUI.cssFamily(ov.font_family);
  }

  function openSettings(ov) {
    const fresh = S.overlayTrack(ov.id);
    if (fresh && window.Timeline?.openOverlaySettings) window.Timeline.openOverlaySettings(fresh);
  }

  function startDrag(e, ov, box, rect) {
    if (e.button !== 0 || e.detail > 1) return;
    e.preventDefault();
    e.stopPropagation();

    const startX = e.clientX;
    const startY = e.clientY;
    const boxRect = box.getBoundingClientRect();
    const grabOffX = e.clientX - (boxRect.left + boxRect.width / 2);
    const grabOffY = e.clientY - (boxRect.top + boxRect.height / 2);

    const posAt = (clientX, clientY, r) => {
      const cx = clientX - r.left - grabOffX;
      const cy = clientY - r.top - grabOffY;
      return {
        nx: clamp01(cx / r.width, ov.x != null ? ov.x : 0.5),
        ny: clamp01(cy / r.height, ov.y != null ? ov.y : 0.5),
      };
    };

    dragging = { id: ov.id, grabOffX, grabOffY, rect, box, startX, startY, moved: false };
    box.classList.add("dragging");
    if (S.get().selectedOverlayId !== ov.id) S.selectOverlay(ov.id);

    const onMove = (ev) => {
      if (!dragging) return;
      if (Math.hypot(ev.clientX - dragging.startX, ev.clientY - dragging.startY) > 4) {
        dragging.moved = true;
      }
      const r = layerRect() || dragging.rect;
      dragging.rect = r;
      const { nx, ny } = posAt(ev.clientX, ev.clientY, r);
      dragging.box.style.left = (nx * r.width) + "px";
      dragging.box.style.top = (ny * r.height) + "px";
    };
    const onUp = (ev) => {
      if (!dragging) return;
      const { id, moved } = dragging;
      const r = layerRect() || dragging.rect;
      const { nx, ny } = posAt(ev.clientX, ev.clientY, r);
      dragging.box.classList.remove("dragging");
      dragging = null;
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
      if (moved) S.updateOverlayTrack(id, { x: nx, y: ny }, true);
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
    const rect = layerRect();
    if (!rect) return;
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
            const r = layerRect();
            if (!r) return;
            const w = Math.max(24, r.width * scale);
            img.style.width = w + "px";
            img.style.height = "auto";
            placeBox(box, ov, r);
          };
        }
      }

      placeBox(box, ov, rect);
      box.addEventListener("pointerdown", (e) => startDrag(e, ov, box, rect));
      box.addEventListener("dblclick", (e) => {
        e.preventDefault();
        e.stopPropagation();
        dragging = null;
        S.selectOverlay(ov.id);
        openSettings(ov);
      });
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
