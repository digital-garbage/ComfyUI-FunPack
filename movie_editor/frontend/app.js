// Entry: wire splitters, boot the store (view modules self-mount on load).
(function () {
  function makeSplitter(splitterId, targetVar, axis, min, max, invert) {
    const sp = document.getElementById(splitterId);
    if (!sp) return;
    let dragging = false;
    const root = document.documentElement;
    sp.addEventListener("pointerdown", (e) => {
      dragging = true; sp.setPointerCapture(e.pointerId);
      document.body.classList.add("dragging");
    });
    sp.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      let v = axis === "x"
        ? (invert ? window.innerWidth - e.clientX : e.clientX)
        : (invert ? window.innerHeight - e.clientY : e.clientY);
      v = Math.max(min, Math.min(max(), v));
      root.style.setProperty(targetVar, v + "px");
    });
    sp.addEventListener("pointerup", (e) => {
      dragging = false; try { sp.releasePointerCapture(e.pointerId); } catch (_) {}
      document.body.classList.remove("dragging");
    });
  }

  window.addEventListener("DOMContentLoaded", () => {
    makeSplitter("split-left", "--media-w", "x", 200, () => window.innerWidth * 0.5, false);
    makeSplitter("split-bottom", "--timeline-h", "y", 160, () => window.innerHeight * 0.6, true);
    window.Store.init();
  });
})();
