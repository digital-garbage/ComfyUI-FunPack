// Dock panel visibility: Media / Preview / Settings toggles above the timeline.
// Preserves the canonical layout (media left, preview top-right, settings bottom-right)
// when all three are on; collapsing a tab gives its space to the remaining panels.
(function () {
  const KEY = "funpack_dock_panels";
  const PANELS = [
    { id: "media", label: "Media" },
    { id: "preview", label: "Preview" },
    { id: "settings", label: "Settings" },
  ];

  const dock = document.getElementById("dock");
  const tabsEl = document.getElementById("dock-tabs");
  if (!dock || !tabsEl) return;

  function load() {
    try {
      const raw = localStorage.getItem(KEY);
      if (raw) {
        const p = JSON.parse(raw);
        return {
          media: p.media !== false,
          preview: p.preview !== false,
          settings: p.settings !== false,
        };
      }
    } catch (_) {}
    return { media: true, preview: true, settings: true };
  }

  let state = load();

  function save() {
    localStorage.setItem(KEY, JSON.stringify(state));
  }

  const panelEls = {
    media: document.getElementById("media-zone"),
    preview: document.getElementById("preview-zone"),
    settings: document.getElementById("inspector-zone"),
  };
  const splitLeft = document.getElementById("split-left");
  const splitMid = document.getElementById("split-mid");
  const rightStack = document.getElementById("right-stack");

  function setVis(el, show) {
    if (!el) return;
    if (show) el.removeAttribute("hidden"); else el.setAttribute("hidden", "");
  }

  function apply() {
    dock.dataset.media = state.media ? "1" : "0";
    dock.dataset.preview = state.preview ? "1" : "0";
    dock.dataset.settings = state.settings ? "1" : "0";
    setVis(panelEls.media, state.media);
    setVis(panelEls.preview, state.preview);
    setVis(panelEls.settings, state.settings);
    const rs = state.preview || state.settings;
    setVis(rightStack, rs);
    setVis(splitLeft, state.media && rs);
    setVis(splitMid, state.preview && state.settings);
    tabsEl.querySelectorAll(".dock-tab").forEach((btn) => {
      const id = btn.dataset.panel;
      const on = !!state[id];
      btn.classList.toggle("on", on);
      btn.setAttribute("aria-pressed", on ? "true" : "false");
    });
  }

  function toggle(id) {
    const on = Object.values(state).filter(Boolean).length;
    if (state[id] && on <= 1) return; // keep at least one panel visible
    state[id] = !state[id];
    save();
    apply();
  }

  PANELS.forEach(({ id, label }) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "dock-tab";
    btn.dataset.panel = id;
    btn.textContent = label;
    btn.title = `Show or hide ${label}`;
    btn.onclick = () => toggle(id);
    tabsEl.append(btn);
  });

  apply();
  window.DockLayout = { get: () => ({ ...state }), set: (patch) => { state = { ...state, ...patch }; save(); apply(); } };
})();