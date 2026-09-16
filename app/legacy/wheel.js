// The picker wheel: any offered action, under the pointer.
//
// [[project_v5_ui_requirements]]: "any action in the UI can be assigned to
// a slot -- opening a menu, changing a value, pressing a button, anything.
// Two shapes, the user's choice: full wheel or half wheel. Half wheel is
// positioned left or right. Both draggable... default: middle mouse
// button, anywhere." Supersedes v4's pinned_buttons.js (three fixed
// toolbar slots) -- kept installed alongside this for now rather than
// removed in the same change, since replacing a working control is a
// bigger decision than adding a new one.
//
// Reads window.FunPackActions.offered() -- whatever announced itself is
// what the wheel can show. Which of those are actually ON the wheel, and
// the shape/side preference, is a per-browser preference (like
// timeline_peek.js's own reasoning: how you want this gadget to behave is
// a property of the screen you're at, not the project), stored in
// localStorage.
(function () {
  const { el } = window.dom;
  const A = window.FunPackActions;
  if (!A) return;

  const LS_KEY = "funpack_wheel_prefs";
  const RADIUS = 92;
  const ITEM_SIZE = 56;
  const MIDDLE_BUTTON = 1;
  const DRAG_THRESHOLD = 6; // px before a hub press counts as a drag, not a click

  function loadPrefs() {
    try {
      const raw = localStorage.getItem(LS_KEY);
      const p = raw ? JSON.parse(raw) : {};
      return {
        shape: p.shape === "half" ? "half" : "full",
        side: p.side === "left" ? "left" : "right",
        enabled: Array.isArray(p.enabled) ? p.enabled : null, // null = not chosen yet
      };
    } catch (_) {
      return { shape: "full", side: "right", enabled: null };
    }
  }
  function savePrefs(p) {
    try { localStorage.setItem(LS_KEY, JSON.stringify(p)); } catch (_) { /* private mode */ }
  }
  let prefs = loadPrefs();

  // Default set, the first time this ever opens: everything offered by
  // the time the page has FINISHED loading. After that, an action offered
  // later still (a module that loaded after this ran) does not silently
  // appear on the wheel uninvited -- the user adds it from Customize, same
  // as any other action they didn't ask for yet.
  //
  // Snapshotting on the FIRST CALL rather than on `load` would let an
  // unlucky fast middle-click, fired in the gap between this script and a
  // later one that still has actions to register, permanently persist an
  // empty or partial default -- there would be nothing wrong to notice,
  // just a wheel that looks broken forever until someone thinks to open
  // Customize. `defaultsReady` closes that window: before `load`, this
  // shows whatever is offered SO FAR without writing it down as the
  // permanent choice.
  let defaultsReady = false;
  window.addEventListener("load", () => {
    defaultsReady = true;
    if (prefs.enabled === null) {
      prefs.enabled = A.offered().map((a) => a.id);
      savePrefs(prefs);
    }
  });

  function enabledIds() {
    if (prefs.enabled) return prefs.enabled;
    if (!defaultsReady) return A.offered().map((a) => a.id);
    prefs.enabled = A.offered().map((a) => a.id);
    savePrefs(prefs);
    return prefs.enabled;
  }

  function wheelItems() {
    const byId = new Map(A.offered().map((a) => [a.id, a]));
    return enabledIds().map((id) => byId.get(id)).filter(Boolean);
  }

  // ── geometry ──────────────────────────────────────────────────────────
  // 0deg = right, 90deg = down (screen y-down), matching CSS/canvas convention.
  function anglesFor(n, shape, side) {
    if (n <= 0) return [];
    if (shape === "full") {
      return Array.from({ length: n }, (_, i) => -90 + i * (360 / n));
    }
    if (n === 1) return [side === "left" ? 180 : 0];
    // Left and right arcs both sweep in the same rotational direction --
    // it's the START angle (90 vs -90) that mirrors them, not the sweep.
    const start = side === "left" ? 90 : -90;
    const step = 180 / (n - 1);
    return Array.from({ length: n }, (_, i) => start + i * step);
  }

  // ── the wheel itself ─────────────────────────────────────────────────
  let wheelEl = null;
  let cleanup = null;

  function close() {
    if (!wheelEl) return;
    wheelEl.remove();
    wheelEl = null;
    if (cleanup) { cleanup(); cleanup = null; }
  }

  function openCustomize() {
    close();
    openCustomizeDialog();
  }

  function open(cx, cy) {
    if (wheelEl) { close(); return; }
    const items = wheelItems();
    const angles = anglesFor(items.length, prefs.shape, prefs.side);

    wheelEl = el("div", "fp-wheel");
    wheelEl.style.left = cx + "px";
    wheelEl.style.top = cy + "px";

    const hub = el("button", "fp-wheel-hub", "⚙");
    hub.type = "button";
    hub.title = "Drag to move — click to customize";
    wheelEl.append(hub);

    // A half wheel packs the same item count into HALF the angular room a
    // full wheel gets -- the same fixed radius that looks fine for a full
    // wheel of 9 puts adjacent half-wheel items closer together than their
    // own diameter. Grow the radius only as far as needed to keep adjacent
    // items from overlapping, using the tightest gap in the layout (the
    // last-to-first wraparound for a full wheel, the actual last step for
    // a half wheel, which never wraps).
    const stepDeg = items.length > 1
      ? (prefs.shape === "full" ? 360 / items.length : 180 / (items.length - 1))
      : 360;
    const minRadius = ITEM_SIZE / (2 * Math.sin((stepDeg / 2) * Math.PI / 180));
    const r = Math.max(RADIUS, minRadius + 8);

    items.forEach((action, i) => {
      const a = (angles[i] * Math.PI) / 180;
      const x = Math.cos(a) * r;
      const y = Math.sin(a) * r;
      const btn = el("button", "fp-wheel-item");
      btn.type = "button";
      btn.title = action.label;
      btn.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%)`;
      btn.append(el("span", "fp-wheel-item-icon", action.icon));
      btn.append(el("span", "fp-wheel-item-label", action.label));
      btn.onclick = (e) => { e.stopPropagation(); close(); A.run(action.id); };
      wheelEl.append(btn);
    });

    document.body.append(wheelEl);

    // Hub drag: reposition THIS open only (not persisted -- "draggable" per
    // the spec is about the wheel not being stuck somewhere awkward once
    // it's up, not a remembered dock position). A press that never moves
    // past the threshold is a click, opening Customize instead.
    let dragging = false;
    let moved = false;
    let startX = 0, startY = 0, originX = cx, originY = cy;
    const axis = prefs.shape === "half" ? "y" : "xy";

    const onMove = (e) => {
      if (!dragging) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      if (!moved && Math.hypot(dx, dy) > DRAG_THRESHOLD) moved = true;
      const nx = axis === "xy" ? originX + dx : originX;
      const ny = originY + dy;
      wheelEl.style.left = nx + "px";
      wheelEl.style.top = ny + "px";
    };
    const onUp = () => {
      dragging = false;
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      if (!moved) openCustomize();
    };
    hub.onmousedown = (e) => {
      e.preventDefault(); e.stopPropagation();
      dragging = true; moved = false;
      startX = e.clientX; startY = e.clientY;
      originX = parseFloat(wheelEl.style.left) || cx;
      originY = parseFloat(wheelEl.style.top) || cy;
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    };

    const onOutside = (e) => { if (!wheelEl.contains(e.target)) close(); };
    const onKey = (e) => { if (e.key === "Escape") close(); };
    setTimeout(() => document.addEventListener("mousedown", onOutside, true), 0);
    document.addEventListener("keydown", onKey);
    cleanup = () => {
      document.removeEventListener("mousedown", onOutside, true);
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
  }

  function toggleAt(cx, cy) { if (wheelEl) close(); else open(cx, cy); }

  // ── trigger: middle mouse, anywhere ──────────────────────────────────
  // Middle click is otherwise autoscroll, which does nothing useful over
  // this app -- safe to claim everywhere, not just inside one region.
  document.addEventListener("mousedown", (e) => {
    if (e.button !== MIDDLE_BUTTON) return;
    e.preventDefault();
    toggleAt(e.clientX, e.clientY);
  });

  // ── customize dialog: which offered actions are on the wheel, and its shape ──
  let dialog = null;
  function closeDialog() { dialog?.remove(); dialog = null; }

  function openCustomizeDialog() {
    closeDialog();
    dialog = el("div", "modal-overlay pin-overlay");
    const box = el("div", "modal pin-modal");

    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", "Customize the wheel"));
    const closeBtn = el("button", "btn ghost tiny", "✕");
    closeBtn.type = "button";
    closeBtn.onclick = closeDialog;
    const hr = el("div", "modal-head-right");
    hr.append(closeBtn);
    head.append(hr);
    box.append(head);

    const content = el("div", "modal-content");

    const shapeRow = el("div", "pin-clear-row");
    ["full", "half"].forEach((shape) => {
      const b = el("button", "btn ghost tiny" + (prefs.shape === shape ? " occupied" : ""),
        shape === "full" ? "Full wheel" : "Half wheel");
      b.type = "button";
      b.onclick = () => { prefs.shape = shape; savePrefs(prefs); openCustomizeDialog(); };
      shapeRow.append(b);
    });
    if (prefs.shape === "half") {
      ["left", "right"].forEach((side) => {
        const b = el("button", "btn ghost tiny" + (prefs.side === side ? " occupied" : ""),
          side === "left" ? "Opens left" : "Opens right");
        b.type = "button";
        b.onclick = () => { prefs.side = side; savePrefs(prefs); openCustomizeDialog(); };
        shapeRow.append(b);
      });
    }
    content.append(shapeRow);
    content.append(el("div", "pin-hint",
      "Middle mouse button opens the wheel anywhere. Check the actions you want on it."));

    const list = el("div", "pin-slots");
    const enabled = new Set(enabledIds());
    A.offered().forEach((action) => {
      const row = el("button", "pin-slot" + (enabled.has(action.id) ? " occupied" : ""));
      row.type = "button";
      const left = el("div", "pin-slot-text");
      left.append(el("div", "pin-slot-name", `${action.icon} ${action.label}`));
      row.append(left);
      row.append(el("span", "pin-slot-pos", enabled.has(action.id) ? "on the wheel" : "not shown"));
      row.onclick = () => {
        if (enabled.has(action.id)) enabled.delete(action.id); else enabled.add(action.id);
        prefs.enabled = A.offered().map((a) => a.id).filter((id) => enabled.has(id));
        savePrefs(prefs);
        openCustomizeDialog();
      };
      list.append(row);
    });
    content.append(list);

    box.append(content);
    dialog.append(box);
    dialog.addEventListener("click", (e) => { if (e.target === dialog) closeDialog(); });
    document.body.append(dialog);
  }

  window.FunPackWheel = { open, close, toggleAt, openCustomizeDialog };
})();
