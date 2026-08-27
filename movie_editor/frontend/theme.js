// Colour scheme: dark (the original cutting-room palette), light, or auto.
//
// "auto" is resolved HERE rather than in CSS: a @media (prefers-color-scheme)
// block cannot be reused across the two selectors it would need, so the palette
// would have to be written twice and would drift. The stylesheet only ever sees
// data-theme="dark" or "light"; this file decides which.
//
// Loaded before every other script (and re-applied inline in <head>) so the
// first paint is already in the right theme — a dark flash on a light setup is
// the one bug a theme switcher cannot apologise for.
(function () {
  const LS_KEY = "funpack_theme";
  const CHOICES = ["dark", "light", "auto"];
  const listeners = new Set();
  let mq = null;

  function stored() {
    try {
      const v = localStorage.getItem(LS_KEY);
      return CHOICES.includes(v) ? v : "dark";
    } catch (_) {
      return "dark";
    }
  }

  function systemIsLight() {
    return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches);
  }

  function resolve(choice) {
    return choice === "auto" ? (systemIsLight() ? "light" : "dark") : choice;
  }

  // data-theme is what the stylesheet reads; data-theme-pref is the choice behind it
  // ("auto" resolves to one of the other two). The theme changes on an explicit click
  // and nothing else, so these two always move together.
  function stamp(choice) {
    const resolved = resolve(choice);
    document.documentElement.setAttribute("data-theme", resolved);
    document.documentElement.setAttribute("data-theme-pref", choice);
    return resolved;
  }

  // Only while the preference is "auto" does the OS get a vote; a user who picked
  // light explicitly should not flip at sunset.
  function watchSystem(choice) {
    if (!window.matchMedia) return;
    if (!mq) {
      mq = window.matchMedia("(prefers-color-scheme: light)");
      const onChange = () => { if (get() === "auto") apply("auto"); };
      if (mq.addEventListener) mq.addEventListener("change", onChange);
      else if (mq.addListener) mq.addListener(onChange);
    }
    void choice;
  }

  function get() {
    return document.documentElement.getAttribute("data-theme-pref") || stored();
  }

  function resolved() {
    return document.documentElement.getAttribute("data-theme") || "dark";
  }

  function apply(choice, { persist = true } = {}) {
    if (!CHOICES.includes(choice)) choice = "dark";
    const res = stamp(choice);
    if (persist) {
      try { localStorage.setItem(LS_KEY, choice); } catch (_) {}
    }
    watchSystem(choice);
    listeners.forEach((fn) => { try { fn(choice, res); } catch (_) {} });
    return res;
  }

  function onChange(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  }

  apply(stored(), { persist: false });

  window.FunPackTheme = { get, resolved, apply, onChange, CHOICES };
})();
