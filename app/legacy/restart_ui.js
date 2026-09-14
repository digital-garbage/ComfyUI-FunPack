// Shared ComfyUI restart overlay + health polling (Update FunPack, pipeline installs).
(function () {
  const { el } = window.dom;

  function showOverlay(message, options) {
    document.querySelector(".restart-overlay")?.remove();
    const ov = el("div", "restart-overlay");
    const card = el("div", "restart-card");
    card.append(el("div", "restart-spin"));
    const msg = el("div", "restart-msg", message);
    card.append(msg);
    if (options?.onCancel) {
      const btn = el("button", "btn ghost restart-cancel", options.cancelLabel || "Cancel install");
      btn.type = "button";
      btn.onclick = options.onCancel;
      card.append(btn);
    }
    ov.append(card);
    document.body.append(ov);
    return msg;
  }

  function waitForReload(msgEl, startMs) {
    const start = startMs || Date.now();
    const tick = async () => {
      try {
        const h = await window.MovieEditorAPI.health();
        if (h && h.ok) { location.reload(); return; }
      } catch (_) { /* still down */ }
      if (Date.now() - start > 90000) {
        msgEl.textContent = "Still waiting on ComfyUI…\nIt may need a manual restart - check the console.";
      }
      setTimeout(tick, 2000);
    };
    setTimeout(tick, 3500);
  }

  window.FunPackRestart = {
    showOverlay,
    waitForReload,
    removeOverlay() { document.querySelector(".restart-overlay")?.remove(); },
  };
})();
