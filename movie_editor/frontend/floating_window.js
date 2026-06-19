// Reusable OS-style floating window: drag, close, minimize (to a dock pill), maximize/restore.
// Foundation for Composer now; Models / Engine settings can adopt it later.
//
//   const win = window.FloatingWindow.create({ id, title, subtitle, width, height, onClose });
//   win.open();  win.body  // mount your content into win.body
//
// One shared minimized-dock at bottom-left; one shared z-counter so click-to-front works.
(function () {
  const { el, clear } = window.dom;
  let zTop = 4000;

  // Lazily-created strip that holds minimized window pills.
  let dock = null;
  function dockEl() {
    if (!dock) {
      dock = el("div", "fw-dock");
      dock.id = "fw-dock";
      document.body.append(dock);
    }
    return dock;
  }

  function create(opts) {
    opts = opts || {};
    const W = Math.max(opts.minWidth || 360, opts.width || 560);
    const H = Math.max(opts.minHeight || 240, opts.height || 480);

    const root = el("div", "fw");
    root.style.width = W + "px";
    root.style.height = H + "px";
    root.hidden = true;

    // ── chrome ────────────────────────────────────────────────────────────────
    const bar = el("div", "fw-bar");
    const titleWrap = el("div", "fw-title");
    const dot = el("span", "fw-dot");
    const titleText = el("span", "fw-title-text", opts.title || "Window");
    titleWrap.append(dot, titleText);
    if (opts.subtitle) titleWrap.append(el("span", "fw-sub", opts.subtitle));

    const ctrls = el("div", "fw-ctrls");
    const bMin = el("button", "fw-btn fw-min", "—"); bMin.title = "Minimize";
    const bMax = el("button", "fw-btn fw-max", "▢"); bMax.title = "Maximize";
    const bClose = el("button", "fw-btn fw-close", "✕"); bClose.title = "Close";
    ctrls.append(bMin, bMax, bClose);
    bar.append(titleWrap, ctrls);

    const body = el("div", "fw-body");
    const grip = el("div", "fw-grip"); grip.title = "Resize";
    root.append(bar, body, grip);
    document.body.append(root);

    // ── state ─────────────────────────────────────────────────────────────────
    let placed = false;       // has it been centred / positioned yet
    let maximized = false;
    let pre = null;           // geometry saved before maximize
    let pill = null;          // minimized dock pill

    function front() { root.style.zIndex = ++zTop; }

    function center() {
      const w = root.offsetWidth, h = root.offsetHeight;
      const x = Math.max(8, Math.round((window.innerWidth - w) / 2));
      const y = Math.max(48, Math.round((window.innerHeight - h) / 2.3));
      root.style.left = x + "px"; root.style.top = y + "px";
    }

    function clampIntoView() {
      const w = root.offsetWidth, h = root.offsetHeight;
      let x = parseInt(root.style.left || "0", 10);
      let y = parseInt(root.style.top || "0", 10);
      x = Math.min(Math.max(8 - w + 80, x), window.innerWidth - 80);
      y = Math.min(Math.max(44, y), window.innerHeight - 44);
      root.style.left = x + "px"; root.style.top = y + "px";
    }

    function isOpen() { return !root.hidden && !pill; }

    function open() {
      removePill();
      root.hidden = false;
      if (!placed) { center(); placed = true; }
      else clampIntoView();
      front();
      if (opts.onOpen) opts.onOpen();
    }

    function close() {
      removePill();
      root.hidden = true;
      if (opts.onClose) opts.onClose();
    }

    function toggle() { isOpen() ? close() : open(); }

    function removePill() {
      if (pill) { pill.remove(); pill = null; }
    }

    function minimize() {
      if (pill) return;
      root.hidden = true;
      pill = el("div", "fw-pill");
      pill.append(el("span", "fw-pill-dot"), el("span", null, opts.title || "Window"));
      const x = el("button", "fw-pill-x", "✕");
      x.title = "Close"; x.onclick = (e) => { e.stopPropagation(); close(); };
      pill.append(x);
      pill.onclick = () => open();
      dockEl().append(pill);
    }

    function maximize() {
      if (maximized) { restore(); return; }
      pre = { left: root.style.left, top: root.style.top, w: root.style.width, h: root.style.height };
      maximized = true;
      root.classList.add("fw-maxed");
      front();
    }

    function restore() {
      if (!maximized) return;
      maximized = false;
      root.classList.remove("fw-maxed");
      if (pre) { root.style.left = pre.left; root.style.top = pre.top; root.style.width = pre.w; root.style.height = pre.h; }
    }

    bMin.onclick = minimize;
    bMax.onclick = maximize;
    bClose.onclick = close;
    bar.ondblclick = (e) => { if (e.target.closest(".fw-btn")) return; maximize(); };
    root.addEventListener("mousedown", front, true);

    // ── drag by titlebar ────────────────────────────────────────────────────────
    bar.addEventListener("mousedown", (e) => {
      if (e.target.closest(".fw-btn") || maximized) return;
      e.preventDefault();
      const sx = e.clientX, sy = e.clientY;
      const ox = parseInt(root.style.left || "0", 10);
      const oy = parseInt(root.style.top || "0", 10);
      document.body.classList.add("fw-dragging");
      const move = (ev) => {
        root.style.left = (ox + ev.clientX - sx) + "px";
        root.style.top = Math.max(44, oy + ev.clientY - sy) + "px";
      };
      const up = () => {
        document.removeEventListener("mousemove", move);
        document.removeEventListener("mouseup", up);
        document.body.classList.remove("fw-dragging");
      };
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", up);
    });

    // ── resize by grip ────────────────────────────────────────────────────────
    grip.addEventListener("mousedown", (e) => {
      if (maximized) return;
      e.preventDefault(); e.stopPropagation();
      const sx = e.clientX, sy = e.clientY;
      const ow = root.offsetWidth, oh = root.offsetHeight;
      document.body.classList.add("fw-dragging");
      const move = (ev) => {
        root.style.width = Math.max(opts.minWidth || 360, ow + ev.clientX - sx) + "px";
        root.style.height = Math.max(opts.minHeight || 240, oh + ev.clientY - sy) + "px";
      };
      const up = () => {
        document.removeEventListener("mousemove", move);
        document.removeEventListener("mouseup", up);
        document.body.classList.remove("fw-dragging");
      };
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", up);
    });

    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && isOpen()) close();
    });

    return {
      root, body, titleEl: titleText,
      open, close, toggle, minimize, maximize, restore, isOpen,
      setTitle(t) { titleText.textContent = t; },
    };
  }

  window.FloatingWindow = { create };
})();
