// Simple mode's main surface: one big prompt under the preview, and two buttons that
// slide the Assets and Properties panels in over it. The panels are the SAME zones the
// Editor pins — nothing is rebuilt or duplicated, they just stop taking up the window.
//
// The prompt edits the global prompt, the same text the Composer's Compose tab edits, so
// split markers keep working and the two never disagree.
(function () {
  const { el } = window.dom;
  const S = window.Store;
  const M = () => window.FunPackMode;

  const host = document.getElementById("simple-bar");
  const workspace = document.getElementById("workspace");
  if (!host || !workspace) return;

  let ta = null;
  let draft = null;      // in-progress text; cleared once the store catches up

  function liveText() {
    const st = S.get();
    const pv = st.preview;
    return (st.project && st.project.global_prompt)
      || (pv && (pv.display_prompt != null ? pv.display_prompt : pv.combined_prompt)) || "";
  }

  function togglePanel(which) {
    const cls = "show-" + which;
    const other = which === "assets" ? "show-props" : "show-assets";
    workspace.classList.remove(other);
    workspace.classList.toggle(cls);
  }

  function closePanels() {
    workspace.classList.remove("show-assets", "show-props");
  }

  // A panel slid over the preview covers the button that opened it, so each one carries
  // its own close. Added to the zone head once; CSS shows it in Simple mode only.
  function installCloseButtons() {
    [["media-zone", "assets"], ["inspector-zone", "props"]].forEach(([id, which]) => {
      const head = document.getElementById(id)?.querySelector(".zone-head");
      if (!head || head.querySelector(".panel-close")) return;
      const b = el("button", "btn ghost tiny panel-close", "✕");
      b.type = "button";
      b.title = "Close";
      b.onclick = () => workspace.classList.remove("show-" + which);
      head.append(b);
    });
  }
  installCloseButtons();

  // Escape closes whichever is open, and so does a click on the preview behind it.
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && M()?.isSimple()) closePanels();
  });
  document.getElementById("preview-body")?.addEventListener("mousedown", () => {
    if (M()?.isSimple()) closePanels();
  });

  function build() {
    host.replaceChildren();
    if (!M()?.isSimple() || !S.get().project) { host.hidden = true; closePanels(); return; }
    host.hidden = false;

    ta = el("textarea", "simple-prompt");
    ta.rows = 3;
    ta.placeholder = "Describe the shot you want…";
    ta.value = draft != null ? draft : liveText();
    ta.dataset.k = "simple-prompt";
    ta.oninput = () => { draft = ta.value; S.scheduleGlobalPromptApply(ta.value); };
    host.append(ta);

    const row = el("div", "simple-bar-row");
    const mk = (label, fn) => { const b = el("button", "btn ghost tiny", label); b.type = "button"; b.onclick = fn; return b; };
    row.append(mk("Media", () => togglePanel("assets")));
    row.append(mk("Advanced settings", () => togglePanel("props")));
    host.append(row);
  }

  // Typing must not be interrupted by the repaint the store fires while distributing the
  // prompt into scenes; the draft is dropped once what is stored matches what is shown.
  function refresh() {
    if (!M()?.isSimple()) { build(); return; }
    if (document.activeElement === ta) return;
    if (draft != null && draft.trim() === liveText().trim()) draft = null;
    build();
  }

  S.subscribe(refresh);
  window.addEventListener("funpack-ui-mode", () => { draft = null; refresh(); });
  refresh();

  window.SimpleBar = { closePanels };
})();
