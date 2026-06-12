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

  window.OverlayUI = { FONTS, cssFamily, fontSelect, openModal, field, rangeField };
})();
