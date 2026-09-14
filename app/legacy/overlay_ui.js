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

  // ── shared text styling (preview canvas + modal preview + editors) ──────────────
  const TEXT_STYLE_DEFAULTS = {
    bold: false, italic: false, text_align: "center", line_spacing: 1.2,
    stroke_width: 0, stroke_color: "#000000",
    shadow: false, shadow_color: "#000000",
    bg_enabled: false, bg_color: "#000000", bg_opacity: 0.5,
  };
  function textStyle(ov, key) {
    const v = ov ? ov[key] : undefined;
    return v != null ? v : TEXT_STYLE_DEFAULTS[key];
  }
  function _hexA(hex, alpha) {
    const c = String(hex || "#000000").replace("#", "");
    const n = c.length === 3 ? c.split("").map((x) => x + x).join("") : c;
    const r = parseInt(n.slice(0, 2), 16) || 0, g = parseInt(n.slice(2, 4), 16) || 0, b = parseInt(n.slice(4, 6), 16) || 0;
    return `rgba(${r}, ${g}, ${b}, ${Math.max(0, Math.min(1, alpha))})`;
  }

  // Apply every text style to a DOM box. `fontPx` is the already-scaled render size;
  // stroke / shadow / padding scale off it so preview and program-monitor match.
  function applyTextCss(box, ov, fontPx) {
    const base = (ov.font_size || 42) || 42;
    fontPx = fontPx || base;
    const k = fontPx / base;
    box.style.fontSize = Math.max(8, fontPx) + "px";
    box.style.color = ov.color || "#ffffff";
    box.style.fontFamily = cssFamily(ov.font_family);
    box.style.fontWeight = textStyle(ov, "bold") ? "700" : "400";
    box.style.fontStyle = textStyle(ov, "italic") ? "italic" : "normal";
    box.style.textAlign = textStyle(ov, "text_align");
    box.style.lineHeight = String(textStyle(ov, "line_spacing"));
    box.style.whiteSpace = "pre-wrap";
    const sw = +textStyle(ov, "stroke_width") || 0;
    if (sw > 0) {
      box.style.webkitTextStroke = (sw * k).toFixed(2) + "px " + textStyle(ov, "stroke_color");
      box.style.paintOrder = "stroke fill";
    } else { box.style.webkitTextStroke = ""; box.style.paintOrder = ""; }
    if (textStyle(ov, "shadow")) {
      const off = Math.max(1, base * 0.06) * k;
      box.style.textShadow = `${off.toFixed(1)}px ${off.toFixed(1)}px ${(off * 1.2).toFixed(1)}px ${textStyle(ov, "shadow_color")}`;
    } else box.style.textShadow = "";
    if (textStyle(ov, "bg_enabled")) {
      box.style.background = _hexA(textStyle(ov, "bg_color"), textStyle(ov, "bg_opacity"));
      box.style.padding = `${(base * 0.22 * k).toFixed(1)}px ${(base * 0.4 * k).toFixed(1)}px`;
      box.style.borderRadius = (base * 0.1 * k).toFixed(1) + "px";
    } else { box.style.background = ""; box.style.padding = ""; box.style.borderRadius = ""; }
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

  // Build the extra text-style controls (alignment, bold/italic, outline, shadow,
  // background, line spacing). `get(key)` reads the current value, `set(patch)` commits.
  // Shared by the Add-text modal and the inspector so they never drift apart.
  function textStyleControls(get, set) {
    const wrap = document.createElement("div");
    wrap.className = "ov-textstyle";

    // Bold / Italic / Alignment — a single row of toggle buttons.
    const tRow = document.createElement("div");
    tRow.className = "ov-style-row";
    const toggleBtn = (label, title, isOn, onClick) => {
      const b = document.createElement("button");
      b.type = "button"; b.title = title;
      b.className = "ov-style-btn" + (isOn() ? " active" : "");
      b.textContent = label;
      b.onclick = (e) => { e.preventDefault(); onClick(); b.classList.toggle("active", isOn()); };
      return b;
    };
    tRow.append(
      toggleBtn("B", "Bold", () => !!get("bold"), () => set({ bold: !get("bold") })),
      toggleBtn("I", "Italic", () => !!get("italic"), () => set({ italic: !get("italic") })),
    );
    const alignGroup = document.createElement("div");
    alignGroup.className = "ov-align-group";
    const alignBtns = {};
    const setAlign = (v) => { set({ text_align: v }); Object.entries(alignBtns).forEach(([k, b]) => b.classList.toggle("active", k === v)); };
    [["left", "⇤"], ["center", "↔"], ["right", "⇥"]].forEach(([v, gly]) => {
      const b = document.createElement("button");
      b.type = "button"; b.title = "Align " + v; b.textContent = gly;
      b.className = "ov-style-btn" + ((get("text_align") || "center") === v ? " active" : "");
      b.onclick = (e) => { e.preventDefault(); setAlign(v); };
      alignBtns[v] = b; alignGroup.append(b);
    });
    tRow.append(alignGroup);
    wrap.append(field("Style", tRow, { full: true }));

    // Line spacing.
    const ls = document.createElement("input");
    ls.type = "range"; ls.min = "0.8"; ls.max = "2.5"; ls.step = "0.05";
    ls.value = get("line_spacing") != null ? get("line_spacing") : 1.2;
    const lsVal = document.createElement("span");
    lsVal.className = "ov-range-val"; lsVal.textContent = (+ls.value).toFixed(2);
    ls.oninput = () => { lsVal.textContent = (+ls.value).toFixed(2); set({ line_spacing: parseFloat(ls.value) }); };
    wrap.append(rangeField("Line spacing", ls, lsVal));

    // Outline (stroke width + color).
    const oRow = document.createElement("div");
    oRow.className = "ov-form-row";
    const sw = document.createElement("input");
    sw.type = "number"; sw.min = "0"; sw.max = "40"; sw.step = "0.5"; sw.className = "ov-input";
    sw.value = get("stroke_width") != null ? get("stroke_width") : 0;
    sw.oninput = () => set({ stroke_width: Math.max(0, parseFloat(sw.value || "0")) });
    const sc = document.createElement("input");
    sc.type = "color"; sc.className = "ov-color";
    sc.value = get("stroke_color") || "#000000";
    sc.oninput = () => set({ stroke_color: sc.value });
    oRow.append(field("Outline (px)", sw), field("Outline color", sc));
    wrap.append(oRow);

    // Shadow toggle + color.
    const shRow = document.createElement("div");
    shRow.className = "ov-form-row";
    const shWrap = document.createElement("label");
    shWrap.className = "ov-check";
    const sh = document.createElement("input"); sh.type = "checkbox"; sh.checked = !!get("shadow");
    const shLab = document.createElement("span"); shLab.textContent = "Drop shadow";
    shWrap.append(sh, shLab);
    const shc = document.createElement("input");
    shc.type = "color"; shc.className = "ov-color";
    shc.value = get("shadow_color") || "#000000";
    shc.oninput = () => set({ shadow_color: shc.value });
    sh.onchange = () => set({ shadow: sh.checked });
    shRow.append(field("Shadow", shWrap), field("Shadow color", shc));
    wrap.append(shRow);

    // Background box: toggle + color + opacity.
    const bgRow = document.createElement("div");
    bgRow.className = "ov-form-row";
    const bgWrap = document.createElement("label");
    bgWrap.className = "ov-check";
    const bg = document.createElement("input"); bg.type = "checkbox"; bg.checked = !!get("bg_enabled");
    const bgLab = document.createElement("span"); bgLab.textContent = "Background box";
    bgWrap.append(bg, bgLab);
    bg.onchange = () => set({ bg_enabled: bg.checked });
    const bgc = document.createElement("input");
    bgc.type = "color"; bgc.className = "ov-color";
    bgc.value = get("bg_color") || "#000000";
    bgc.oninput = () => set({ bg_color: bgc.value });
    bgRow.append(field("Box", bgWrap), field("Box color", bgc));
    wrap.append(bgRow);
    const bgo = document.createElement("input");
    bgo.type = "range"; bgo.min = "0"; bgo.max = "1"; bgo.step = "0.05";
    bgo.value = get("bg_opacity") != null ? get("bg_opacity") : 0.5;
    const bgoVal = document.createElement("span");
    bgoVal.className = "ov-range-val"; bgoVal.textContent = Math.round((+bgo.value) * 100) + "%";
    bgo.oninput = () => { bgoVal.textContent = Math.round((+bgo.value) * 100) + "%"; set({ bg_opacity: parseFloat(bgo.value) }); };
    wrap.append(rangeField("Box opacity", bgo, bgoVal));

    return wrap;
  }

  window.OverlayUI = {
    FONTS, cssFamily, fontSelect, openModal, field, rangeField,
    overlayWidthPx, overlayHeightPx, pixelSizeFields,
    TEXT_STYLE_DEFAULTS, textStyle, applyTextCss, textStyleControls,
  };
})();
