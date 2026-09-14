// Tiny DOM helpers shared by view modules.
(function () {
  function el(tag, cls, txt) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  }
  function $(sel) { return document.querySelector(sel); }
  function clear(node) { while (node && node.firstChild) node.removeChild(node.firstChild); return node; }
  function icon(name) {
    // minimal inline glyphs (keeps us font-icon-free)
    const map = {
      empty: "▦", image: "◐", frame: "⛶", play: "▶", add: "＋",
      del: "✕", up: "‹", down: "›", warn: "▲", dot: "●", film: "🎞",
    };
    return map[name] || "•";
  }
  window.dom = { el, $, clear, icon };
})();
