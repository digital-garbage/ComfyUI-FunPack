// Up to three user-chosen shortcuts, sitting left of the workspace toggles.
//
// Settings is a window with a sidebar, and the useful places inside it — a specific node's
// page, one Engine category — are several clicks deep. That is fine to walk once and
// tedious to walk twenty times while dialling something in. A pin puts any of them one
// click away.
//
// Slot number is the address: slot 1 is always the leftmost button, slot 3 always the one
// nearest Assets, whether or not the slots between them are filled. So the position of a
// shortcut never moves as others come and go, and muscle memory survives.
(function () {
  const SLOTS = 3;
  const KEY = "pinnedButtons";           // stored in editor settings: travels with the project

  // ── slot rules (pure — the part with actual rules in it) ──────────────────────
  /** Exactly SLOTS entries, so slot N is index N-1 and a hole stays a hole.
   *  Anything malformed becomes an empty slot rather than a button that cannot work. */
  function normalizePins(raw) {
    const out = Array.isArray(raw) ? raw.slice(0, SLOTS) : [];
    while (out.length < SLOTS) out.push(null);
    return out.map((p) => (p && typeof p === "object" && p.kind && p.id ? p : null));
  }

  /** Buttons to draw, left to right: holes close up, but each surviving button keeps the
   *  NUMBER of the slot it occupies. Slots 1+3 filled draws "Custom 1, Custom 3" — so a
   *  shortcut's position never shifts because a neighbour came or went. */
  function visibleSlots(list) {
    return normalizePins(list)
      .map((target, i) => ({ slotNo: i + 1, target }))
      .filter((e) => e.target);
  }

  /** Identity of a destination. `sub` is part of it: two Engine categories are the same
   *  section but different places, and treating them as one would make the second pin
   *  silently evict the first. */
  function pinKey(t) {
    return t ? `${t.kind}:${t.id}:${t.sub || ""}` : "";
  }

  /** The same destination twice is a wasted slot. Placing it in `index` clears any other
   *  slot pointing at the same thing. */
  function placePin(list, index, target) {
    const next = normalizePins(list);
    const key = pinKey(target);
    next.forEach((p, j) => { if (p && j !== index && pinKey(p) === key) next[j] = null; });
    next[index] = target;
    return next;
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { SLOTS, normalizePins, visibleSlots, placePin, pinKey };
  }
  if (typeof window === "undefined" || !window.dom || !window.Store) return;

  const { el, clear } = window.dom;
  const S = window.Store;
  const host = document.getElementById("pinned-tabs");
  if (!host) return;

  function pins() {
    return normalizePins(S.getEditorSetting ? S.getEditorSetting(KEY) : null);
  }

  function setPins(list) {
    const next = normalizePins(list);
    // Node ids are project-scoped, so this belongs with the project's own settings rather
    // than in the browser alone — the same mirror that carries the rest of the editor
    // preferences onto a freshly rented machine.
    S.setEditorSetting(KEY, next);
    render();
  }

  // ── describing a target ───────────────────────────────────────────────────────
  function describe(t) {
    if (!t) return "";
    if (t.kind === "node") return t.label || "a node";
    return t.label || t.id || "a settings page";
  }

  function tooltip(t, slotNo) {
    if (!t) return `Custom ${slotNo} — nothing pinned yet`;
    return `Opens up ${describe(t)}`;
  }

  // ── opening ───────────────────────────────────────────────────────────────────
  function openTarget(t) {
    if (!t) return;
    if (t.kind === "node") {
      // ModelsModal handles the "not in this pipeline any more" case, including the wait
      // for the section to load before it can know.
      window.ModelsModal?.openNode?.(t.id);
      return;
    }
    if (!window.SettingsWindow?.hasSection?.(t.id)) {
      alert(`"${describe(t)}" is not available in this build any more.\n\n`
            + "Pin something else to that slot to replace it.");
      return;
    }
    window.SettingsWindow.open(t.id, t.sub || null);
  }

  // ── the toolbar ───────────────────────────────────────────────────────────────
  function render() {
    if (!host) return;
    clear(host);
    visibleSlots(pins()).forEach(({ slotNo, target }) => {
      // Deliberately NOT .on: that state means "this panel is showing", and a shortcut is
      // not a toggle. .pinned-tab carries its own look.
      const btn = el("button", "dock-tab pinned-tab", `Custom ${slotNo}`);
      btn.type = "button";
      btn.title = tooltip(target, slotNo);
      btn.onclick = () => openTarget(target);
      host.append(btn);
    });
    host.hidden = !host.children.length;
  }

  // ── the pin dialog ────────────────────────────────────────────────────────────
  let dialog = null;

  function closeDialog() { dialog?.remove(); dialog = null; }

  function slotRow(i, target, current) {
    const occupied = pins()[i];
    const row = el("button", "pin-slot" + (occupied ? " occupied" : ""));
    row.type = "button";
    const left = el("div", "pin-slot-text");
    left.append(el("div", "pin-slot-name", `Custom ${i + 1}`));
    left.append(el("div", "pin-slot-sub", occupied ? `Opens up ${describe(occupied)}` : "Empty"));
    row.append(left);
    row.append(el("span", "pin-slot-pos", i === 0 ? "leftmost" : i === SLOTS - 1 ? "nearest Assets" : "middle"));
    row.onclick = () => {
      // Overwriting is the one destructive thing this dialog does, so it is the one thing
      // it asks about.
      if (occupied && !confirm(
        `Custom ${i + 1} already opens ${describe(occupied)}.\n\n`
        + `Replace it with ${describe(current)}?`
      )) return;
      setPins(placePin(pins(), i, current));
      closeDialog();
    };
    return row;
  }

  function pinCurrent() {
    const current = window.SettingsWindow?.currentTarget?.();
    if (!current) {
      alert("Open the page or node you want to pin first.");
      return;
    }
    openDialog(current);
  }

  function openDialog(current) {
    closeDialog();
    dialog = el("div", "modal-overlay pin-overlay");
    const box = el("div", "modal pin-modal");

    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", "Pin to a button"));
    const closeBtn = el("button", "btn ghost tiny", "✕");
    closeBtn.type = "button";
    closeBtn.onclick = closeDialog;
    const hr = el("div", "modal-head-right");
    hr.append(closeBtn);
    head.append(hr);
    box.append(head);

    const content = el("div", "modal-content");
    content.append(el("div", "pin-lead", `Put ${describe(current)} on the timeline toolbar.`));
    content.append(el("div", "pin-hint",
      "Slot 1 is the leftmost button and slot 3 the one nearest Assets — an empty slot "
      + "closes up, and the buttons that remain keep their own numbers and places."));

    const list = el("div", "pin-slots");
    for (let i = 0; i < SLOTS; i++) list.append(slotRow(i, current, current));
    content.append(list);

    // Clearing has to live somewhere, and this is the only screen that shows the slots.
    const filled = pins().filter(Boolean).length;
    if (filled) {
      const clearRow = el("div", "pin-clear-row");
      pins().forEach((p, i) => {
        if (!p) return;
        const b = el("button", "btn ghost tiny", `Clear Custom ${i + 1}`);
        b.type = "button";
        b.onclick = () => {
          const l = pins(); l[i] = null; setPins(l);
          closeDialog();
        };
        clearRow.append(b);
      });
      content.append(clearRow);
    }

    box.append(content);
    dialog.append(box);
    dialog.addEventListener("click", (e) => { if (e.target === dialog) closeDialog(); });
    document.body.append(dialog);
  }

  // Pins live in editor settings, which a project restores on open — so the toolbar has to
  // follow the store rather than being built once at load.
  let _last = "";
  function syncFromStore() {
    const now = JSON.stringify(pins());
    if (now === _last) return;
    _last = now;
    render();
  }
  S.subscribe(syncFromStore);

  render();
  _last = JSON.stringify(pins());

  window.PinnedButtons = { pinCurrent, render, openDialog };
})();
