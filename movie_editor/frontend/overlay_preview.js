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

  function canvasSize() {
    return S.projectCanvasSize ? S.projectCanvasSize() : { w: 768, h: 512 };
  }

  function previewScale(rect) {
    const { w } = canvasSize();
    return rect.width / Math.max(1, w);
  }

  function imageWidthPx(ov) {
    const { w } = canvasSize();
    return OUI?.overlayWidthPx ? OUI.overlayWidthPx(ov, w) : Math.max(8, Math.round((ov.scale ?? 0.35) * w));
  }

  function imageHeightPx(ov) {
    const { w, h } = canvasSize();
    return OUI?.overlayHeightPx ? OUI.overlayHeightPx(ov, w, h) : imageWidthPx(ov);
  }

  function boxTransform(ov) {
    const sx = ov.flip_h ? -1 : 1;
    const sy = ov.flip_v ? -1 : 1;
    let t = "translate(-50%, -50%)";
    if (sx !== 1 || sy !== 1) t += ` scale(${sx}, ${sy})`;
    return t;
  }

  function placeBox(box, ov, rect) {
    const nx = ov.x != null ? ov.x : 0.5;
    const ny = ov.y != null ? ov.y : 0.5;
    box.style.left = (clamp01(nx, 0.5) * rect.width) + "px";
    box.style.top = (clamp01(ny, 0.5) * rect.height) + "px";
    box.style.opacity = ov.opacity != null ? ov.opacity : 1;
    box.style.transform = boxTransform(ov);
  }

  function applyTextStyle(box, ov) {
    box.textContent = ov.text || "Text";
    box.style.fontSize = (ov.font_size || 42) + "px";
    box.style.color = ov.color || "#ffffff";
    if (OUI) box.style.fontFamily = OUI.cssFamily(ov.font_family);
  }

  function applyImageSize(img, ov, rect) {
    const scale = previewScale(rect);
    const wPx = imageWidthPx(ov);
    if (ov.keep_aspect !== false) {
      img.style.width = Math.max(8, wPx * scale) + "px";
      img.style.height = "auto";
    } else {
      const hPx = imageHeightPx(ov);
      img.style.width = Math.max(8, wPx * scale) + "px";
      img.style.height = Math.max(8, hPx * scale) + "px";
      img.style.objectFit = "fill";
    }
  }

  function openSettings(ov) {
    const fresh = S.overlayTrack(ov.id);
    if (fresh && window.Timeline?.openOverlaySettings) window.Timeline.openOverlaySettings(fresh);
  }

  function addSelectionChrome(box, ov) {
    const handles = document.createElement("div");
    handles.className = "pm-ov-handles";
    ["nw", "ne", "sw", "se"].forEach((corner) => {
      const h = document.createElement("div");
      h.className = "pm-ov-handle " + corner;
      h.addEventListener("pointerdown", (e) => startResize(e, ov, box, corner));
      handles.append(h);
    });
    box.append(handles);

    const bar = document.createElement("div");
    bar.className = "pm-ov-flip-bar";
    const mkFlip = (label, key) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pm-ov-flip-btn" + (ov[key] ? " active" : "");
      btn.textContent = label;
      btn.title = key === "flip_h" ? "Flip horizontal" : "Flip vertical";
      btn.onmousedown = (e) => e.stopPropagation();
      btn.onclick = (e) => {
        e.stopPropagation();
        S.updateOverlayTrack(ov.id, { [key]: !ov[key] }, true);
      };
      return btn;
    };
    bar.append(mkFlip("↔", "flip_h"), mkFlip("↕", "flip_v"));
    box.append(bar);
  }

  function startResize(e, ov, box, corner) {
    if (e.button !== 0 || e.detail > 1) return;
    e.preventDefault();
    e.stopPropagation();

    const rect = layerRect();
    if (!rect) return;
    const boxRect = box.getBoundingClientRect();
    const cx = boxRect.left + boxRect.width / 2;
    const cy = boxRect.top + boxRect.height / 2;
    const startDist = Math.max(12, Math.hypot(e.clientX - cx, e.clientY - cy));
    const startW = ov.kind === "text" ? (ov.font_size || 42) : imageWidthPx(ov);
    const startH = ov.kind === "image" ? imageHeightPx(ov) : null;

    dragging = {
      mode: "resize",
      id: ov.id,
      kind: ov.kind,
      keepAspect: ov.kind === "text" || ov.keep_aspect !== false,
      startDist,
      startW,
      startH,
      cx,
      cy,
      box,
      rect,
      img: box.querySelector("img"),
      moved: false,
    };
    box.classList.add("dragging");
    if (S.get().selectedOverlayId !== ov.id) S.selectOverlay(ov.id);

    const onMove = (ev) => {
      if (!dragging || dragging.mode !== "resize") return;
      dragging.moved = true;
      const dist = Math.max(12, Math.hypot(ev.clientX - dragging.cx, ev.clientY - dragging.cy));
      const ratio = dist / dragging.startDist;
      if (dragging.kind === "text") {
        const fontSize = Math.round(Math.max(8, Math.min(400, dragging.startW * ratio)));
        dragging.box.style.fontSize = fontSize + "px";
      } else if (dragging.img) {
        const scale = previewScale(dragging.rect);
        const wPx = Math.round(Math.max(8, dragging.startW * ratio));
        if (dragging.keepAspect) {
          dragging.img.style.width = Math.max(8, wPx * scale) + "px";
          dragging.img.style.height = "auto";
        } else {
          const hPx = Math.round(Math.max(8, (dragging.startH || wPx) * ratio));
          dragging.img.style.width = Math.max(8, wPx * scale) + "px";
          dragging.img.style.height = Math.max(8, hPx * scale) + "px";
        }
      }
    };

    const onUp = (ev) => {
      if (!dragging || dragging.mode !== "resize") return;
      const { id, kind, startDist, startW, startH, keepAspect, cx, cy, moved } = dragging;
      const dist = Math.max(12, Math.hypot(ev.clientX - cx, ev.clientY - cy));
      const ratio = dist / startDist;
      dragging.box.classList.remove("dragging");
      dragging = null;
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
      if (!moved) return;
      if (kind === "text") {
        S.updateOverlayTrack(id, { font_size: Math.round(Math.max(8, Math.min(400, startW * ratio))) }, true);
      } else {
        const patch = { width_px: Math.round(Math.max(8, startW * ratio)) };
        if (!keepAspect) patch.height_px = Math.round(Math.max(8, (startH || startW) * ratio));
        S.updateOverlayTrack(id, patch, true);
      }
    };

    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
  }

  function startDrag(e, ov, box, rect) {
    if (e.button !== 0 || e.detail > 1) return;
    if (e.target.closest(".pm-ov-handle") || e.target.closest(".pm-ov-flip-bar")) return;
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

    dragging = { mode: "move", id: ov.id, grabOffX, grabOffY, rect, box, startX, startY, moved: false };
    box.classList.add("dragging");
    if (S.get().selectedOverlayId !== ov.id) S.selectOverlay(ov.id);

    const onMove = (ev) => {
      if (!dragging || dragging.mode !== "move") return;
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
      if (!dragging || dragging.mode !== "move") return;
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
      const fresh = S.overlayTrack ? S.overlayTrack(ov.id) : ov;
      const item = fresh || ov;
      const box = document.createElement("div");
      box.className = "pm-ov-box" + (item.kind === "text" ? " text" : " image") + (sel === item.id ? " selected" : "");
      box.title = item.label || item.text || "Overlay";

      if (item.kind === "text") {
        applyTextStyle(box, item);
      } else {
        const asset = (st.mediaBin || []).find((m) => m.id === item.media_ref);
        if (asset?.kind === "image") {
          const img = document.createElement("img");
          img.src = API.mediaUrl(asset.id);
          img.draggable = false;
          box.appendChild(img);
          img.onload = () => {
            const r = layerRect();
            if (!r) return;
            applyImageSize(img, item, r);
            placeBox(box, item, r);
          };
          applyImageSize(img, item, rect);
        }
      }

      placeBox(box, item, rect);
      if (sel === item.id) addSelectionChrome(box, item);
      box.addEventListener("pointerdown", (e) => startDrag(e, item, box, rect));
      box.addEventListener("dblclick", (e) => {
        e.preventDefault();
        e.stopPropagation();
        dragging = null;
        S.selectOverlay(item.id);
        openSettings(item);
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
  window.OverlayPreview = { refresh: render };
  render();
})();
