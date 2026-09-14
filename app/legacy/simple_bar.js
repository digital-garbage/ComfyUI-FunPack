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

  const PANEL_ZONES = { assets: "media-zone", props: "inspector-zone" };
  function panelZone(which) { return document.getElementById(PANEL_ZONES[which]); }

  function togglePanel(which) {
    const cls = "show-" + which;
    const other = which === "assets" ? "show-props" : "show-assets";
    workspace.classList.remove(other);
    const opening = !workspace.classList.contains(cls);
    workspace.classList.toggle(cls);
    // These are the Editor's own zones, and layout.js marks a collapsed column `hidden`.
    // Simple mode hides the dock tabs, so that state is a leftover Editor choice the user
    // cannot see or change here — and `[hidden]` is display:none, which no transform can
    // slide into view. The button did nothing at all, silently. Uncollapse on the way in;
    // the saved dock state is restored when the mode changes back.
    if (opening) panelZone(which)?.removeAttribute("hidden");
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
  window.addEventListener("funpack-ui-mode", () => {
    draft = null;
    closePanels();
    // Hand the columns back to the dock: whatever the user collapsed in Editor mode is
    // theirs, and Simple mode borrowing a zone must not permanently un-collapse it.
    window.DockLayout?.set({});
    refresh();
  });
  refresh();

  window.SimpleBar = { closePanels };
})();
