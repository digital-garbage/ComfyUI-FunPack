// Editor Settings modal: per-browser editor preferences (autocomplete, anchor).
// Opened from the Settings menu. Distinct from Engine settings (which configures
// Studio / Chain Sampler behavior on the project).
(function () {
  const { el } = window.dom;
  const S = window.Store;

  let overlay = null;
  function close() { if (overlay) { overlay.remove(); overlay = null; } }

  function toggleRow(label, hint, key) {
    const row = el("div", "es-row");
    const lbl = el("label", "chk es-toggle");
    const cb = el("input"); cb.type = "checkbox"; cb.checked = !!S.getEditorSetting(key);
    cb.onchange = () => S.setEditorSetting(key, cb.checked);
    lbl.append(cb, el("span", null, label));
    row.append(lbl);
    if (hint) row.append(el("div", "es-hint", hint));
    return row;
  }

  function open() {
    close();
    overlay = el("div", "modal-overlay");
    const box = el("div", "modal");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", "Editor settings"));
    const hr = el("div", "modal-head-right");
    const x = el("button", "btn ghost tiny", "✕"); x.onclick = close;
    hr.append(x); head.append(hr); box.append(head);

    const content = el("div", "modal-content es-content");
    content.append(toggleRow(
      "Prompt autocomplete",
      "Suggest matching shortcuts while you type in the global prompt and scene prompts. Shows the trigger, its prompt, and category.",
      "autocomplete"));
    content.append(toggleRow(
      "Use anchor",
      "Text before the first split trigger is a shared anchor prepended to every scene. Turn off to make that leading text Scene 1 instead.",
      "anchorEnabled"));

    box.append(content); overlay.append(box);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.body.append(overlay);
  }

  window.EditorSettingsModal = { open, close };
})();
