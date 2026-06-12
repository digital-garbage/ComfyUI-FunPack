// Shared overlay fonts + modal shell for text/image overlay dialogs.
(function () {
  const FONTS = [
    { id: "system-ui", label: "System default", css: "system-ui, -apple-system, sans-serif" },
    { id: "arial", label: "Arial", css: "Arial, Helvetica, sans-serif" },
    { id: "helvetica", label: "Helvetica Neue", css: "'Helvetica Neue', Helvetica, Arial, sans-serif" },
    { id: "georgia", label: "Georgia", css: "Georgia, 'Times New Roman', serif" },
    { id: "times", label: "Times New Roman", css: "'Times New Roman', Times, serif" },
    { id: "courier", label: "Courier New", css: "'Courier New', Courier, monospace" },
    { id: "verdana", label: "Verdana", css: "Verdana, Geneva, sans-serif" },
    { id: "impact", label: "Impact", css: "Impact, Haettenschweiler, sans-serif" },
  ];

  function cssFamily(id) {
    const f = FONTS.find((x) => x.id === id) || FONTS[0];
    return f.css;
  }

  function fontSelect(value, onChange) {
    const sel = document.createElement("select");
    sel.className = "ov-input";
    FONTS.forEach((f) => {
      const o = document.createElement("option");
      o.value = f.id;
      o.textContent = f.label;
      o.style.fontFamily = f.css;
      if (f.id === (value || "system-ui")) o.selected = true;
      sel.appendChild(o);
    });
    sel.onchange = () => onChange(sel.value);
    return sel;
  }

  function openModal({ title, subtitle, widthClass, onClose }) {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay ov-modal-overlay";
    const box = document.createElement("div");
    box.className = "modal ov-modal" + (widthClass ? " " + widthClass : "");
    const head = document.createElement("div");
    head.className = "ov-modal-head";
    const headText = document.createElement("div");
    headText.className = "ov-modal-head-text";
    const titleEl = document.createElement("div");
    titleEl.className = "ov-modal-title";
    titleEl.textContent = title;
    headText.appendChild(titleEl);
    if (subtitle) {
      const sub = document.createElement("div");
      sub.className = "ov-modal-sub";
      sub.textContent = subtitle;
      headText.appendChild(sub);
    }
    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "ov-modal-close";
    closeBtn.title = "Close";
    closeBtn.textContent = "×";
    head.append(headText, closeBtn);
    const body = document.createElement("div");
    body.className = "ov-modal-body";
    const foot = document.createElement("div");
    foot.className = "ov-modal-foot";
    box.append(head, body, foot);
    overlay.append(box);
    const close = () => {
      overlay.remove();
      if (onClose) onClose();
    };
    closeBtn.onclick = close;
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.body.append(overlay);
    return { overlay, body, foot, close };
  }

  function field(label, control, opts) {
    opts = opts || {};
    const row = document.createElement("label");
    row.className = "ov-field" + (opts.full ? " full" : "");
    const lab = document.createElement("span");
    lab.className = "ov-label";
    lab.textContent = label;
    row.append(lab, control);
    return row;
  }

  function rangeField(label, input, valueEl) {
    const wrap = document.createElement("div");
    wrap.className = "ov-field full ov-range-field";
    const top = document.createElement("div");
    top.className = "ov-range-top";
    const lab = document.createElement("span");
    lab.className = "ov-label";
    lab.textContent = label;
    top.append(lab, valueEl || null);
    input.className = "ov-range";
    wrap.append(top, input);
    return wrap;
  }

  function overlayWidthPx(ov, canvasW) {
    if (ov.width_px != null) return Math.max(8, parseInt(ov.width_px, 10) || 8);
    if (ov.scale != null) return Math.max(8, Math.round(+ov.scale * canvasW));
    return Math.max(8, Math.round(canvasW * 0.35));
  }

  function overlayHeightPx(ov, canvasW, canvasH) {
    if (ov.height_px != null) return Math.max(8, parseInt(ov.height_px, 10) || 8);
    return overlayWidthPx(ov, canvasW);
  }

  function pixelSizeFields(ov, canvasW, canvasH, onPatch) {
    const wrap = document.createElement("div");
    wrap.className = "ov-size-fields";
    const row = document.createElement("div");
    row.className = "ov-form-row";

    const wIn = document.createElement("input");
    wIn.type = "number";
    wIn.min = "8";
    wIn.max = String(Math.max(canvasW * 4, 4096));
    wIn.step = "1";
    wIn.className = "ov-input";
    wIn.value = overlayWidthPx(ov, canvasW);

    const hIn = document.createElement("input");
    hIn.type = "number";
    hIn.min = "8";
    hIn.max = String(Math.max(canvasH * 4, 4096));
    hIn.step = "1";
    hIn.className = "ov-input";
    hIn.value = overlayHeightPx(ov, canvasW, canvasH);

    const keep = document.createElement("input");
    keep.type = "checkbox";
    keep.checked = ov.keep_aspect !== false;
    keep.id = "ov-keep-" + (ov.id || "new");

    const syncHeight = () => {
      hIn.disabled = keep.checked;
      hIn.style.opacity = keep.checked ? "0.45" : "1";
    };
    syncHeight();

    wIn.oninput = () => onPatch({ width_px: Math.max(8, parseInt(wIn.value || "8", 10)) });
    hIn.oninput = () => onPatch({ height_px: Math.max(8, parseInt(hIn.value || "8", 10)) });
    keep.onchange = () => {
      syncHeight();
      onPatch({ keep_aspect: keep.checked });
    };

    row.append(field("Width (px)", wIn), field("Height (px)", hIn));
    wrap.append(row);

    const keepRow = document.createElement("label");
    keepRow.className = "ov-keep-row";
    const keepLab = document.createElement("span");
    keepLab.textContent = "Keep proportion";
    keepRow.append(keep, keepLab);
    wrap.append(keepRow);

    const flipRow = document.createElement("div");
    flipRow.className = "ov-flip-row";
    let flipH = !!ov.flip_h;
    let flipV = !!ov.flip_v;
    const mkFlip = (label, get, set) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn ghost tiny" + (get() ? " active" : "");
      btn.textContent = label;
      btn.onclick = (e) => {
        e.preventDefault();
        set(!get());
        btn.classList.toggle("active", get());
        onPatch({ flip_h: flipH, flip_v: flipV });
      };
      return btn;
    };
    flipRow.append(
      mkFlip("Flip horizontal", () => flipH, (v) => { flipH = v; }),
      mkFlip("Flip vertical", () => flipV, (v) => { flipV = v; }),
    );
    wrap.append(flipRow);

    return wrap;
  }

  window.OverlayUI = {
    FONTS, cssFamily, fontSelect, openModal, field, rangeField,
    overlayWidthPx, overlayHeightPx, pixelSizeFields,
  };
})();
