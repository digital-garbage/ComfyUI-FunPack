// Two faces of one app. "simple" is what Easy Gen used to be — prompt, Generate, result,
// and nothing that needs explaining. "editor" is the full cutting room. Same store, same
// pipeline, same project file: only what is on screen differs.
//
// Stamped on <html> as data-ui-mode so most of the hiding is CSS, and read at render time
// by the panels that have to build different content rather than hide some of it.
(function () {
  const KEY = "funpack_ui_mode";
  const MODES = ["simple", "editor"];
  const listeners = new Set();

  function stored() {
    try {
      const v = localStorage.getItem(KEY);
      return MODES.includes(v) ? v : null;
    } catch (_) { return null; }
  }

  // No stored choice: the URL decides once, so an Easy Gen bookmark still lands in the
  // mode it promised. Everything after that is the toggle.
  function initial() {
    const s = stored();
    if (s) return s;
    try {
      const q = new URLSearchParams(window.location.search).get("mode");
      if (MODES.includes(q)) return q;
      if (/\/easy\/?$/.test(window.location.pathname)) return "simple";
    } catch (_) {}
    return "editor";
  }

  let mode = initial();

  function stamp() { document.documentElement.setAttribute("data-ui-mode", mode); }

  function get() { return mode; }
  function is(m) { return mode === m; }
  function isSimple() { return mode === "simple"; }

  function set(next) {
    if (!MODES.includes(next) || next === mode) return mode;
    mode = next;
    stamp();
    try { localStorage.setItem(KEY, mode); } catch (_) {}
    listeners.forEach((fn) => { try { fn(mode); } catch (e) { console.error(e); } });
    try { window.dispatchEvent(new CustomEvent("funpack-ui-mode", { detail: mode })); } catch (_) {}
    // Panels that BUILD different content (rather than hide some of it) redraw off the
    // store, so one notify covers the inspector, timeline and action bar together.
    try { window.Store?.notify?.(); } catch (_) {}
    return mode;
  }

  function onChange(fn) { listeners.add(fn); return () => listeners.delete(fn); }

  const WARN_KEY = "funpack_ui_mode_warned";
  function warned() {
    try { return localStorage.getItem(WARN_KEY) === "1"; } catch (_) { return true; }
  }
  function markWarned() {
    try { localStorage.setItem(WARN_KEY, "1"); } catch (_) {}
  }

  stamp();
  window.FunPackMode = { get, set, is, isSimple, onChange, warned, markWarned, MODES };
})();
