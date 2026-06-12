// Detect ?mode=tour before the store boots (sandbox welcome guide).
(function () {
  try {
    const q = new URLSearchParams(window.location.search);
    window.__FUNPACK_TOUR__ = q.get("mode") === "tour";
  } catch (_) {
    window.__FUNPACK_TOUR__ = false;
  }
})();
