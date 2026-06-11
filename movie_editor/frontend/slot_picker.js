// Small modal for picking a models slot (replaces window.prompt in menubar).
(function () {
  const { el, clear } = window.dom;
  let overlay = null;

  function close() {
    if (overlay) { overlay.remove(); overlay = null; }
  }

  function open({ title, options, onPick }) {
    close();
    overlay = el("div", "modal-overlay slot-picker-overlay");
    const box = el("div", "modal slot-picker-modal");
    box.append(el("div", "modal-title", title || "Choose"));
    const list = el("div", "slot-picker-list");
    options.forEach((opt) => {
      const row = el("button", "slot-picker-row", opt.label);
      if (opt.hint) {
        const h = el("span", "slot-picker-hint", opt.hint);
        row.append(h);
      }
      row.onclick = () => { close(); onPick(opt.value); };
      list.append(row);
    });
    box.append(list);
    const cancel = el("button", "btn ghost", "Cancel");
    cancel.onclick = close;
    box.append(cancel);
    overlay.append(box);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.body.append(overlay);
  }

  window.SlotPicker = { open, close };
})();
