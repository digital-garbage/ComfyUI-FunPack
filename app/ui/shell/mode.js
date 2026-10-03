// Simple or Editor: how much of the screen is showing. One word, kept in this browser, written onto the page
// (`data-ui-mode`) where the stylesheet reads it.
const KEY = "funpack_ui_mode";

export function createMode(root = document.documentElement) {
  const heard = new Set();
  const stored = () => { try { return localStorage.getItem(KEY) === "simple" ? "simple" : "editor"; } catch { return "editor"; } };
  let now = stored();
  root.dataset.uiMode = now;
  return {
    get now() { return now; },
    set(next) {
      if (next !== "simple" && next !== "editor") return;
      if (next === now) return;
      now = next;
      root.dataset.uiMode = now;
      try { localStorage.setItem(KEY, now); } catch { /* private window: lasts until reload */ }
      heard.forEach((fn) => fn(now));
    },
    on(fn) { heard.add(fn); return () => heard.delete(fn); },
  };
}
