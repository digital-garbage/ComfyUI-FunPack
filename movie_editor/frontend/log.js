// Menubar "Log" button + a collapsible panel showing ComfyUI's real backend log
// (tee'd stdout/stderr from the server). Polls only while open. Freezes during selection to preserve user interaction.
(function () {
  const API = window.MovieEditorAPI;
  const right = document.querySelector(".menubar-right");
  if (!right) return;

  const btn = document.createElement("button");
  btn.className = "log-btn"; btn.textContent = "Log"; btn.title = "Show the ComfyUI backend log";
  right.insertBefore(btn, right.firstChild);

  const panel = document.createElement("div");
  panel.className = "log-panel"; panel.hidden = true;
  const head = document.createElement("div"); head.className = "log-head";
  const title = document.createElement("span"); title.textContent = "ComfyUI log";
  const actions = document.createElement("div"); actions.className = "log-actions";
  head.append(title, actions);
  const bodyEl = document.createElement("pre"); bodyEl.className = "log-body";
  panel.append(head, bodyEl);
  document.body.append(panel);

  let timer = null, autoscroll = true, lastText = "", isFrozen = false, isManuallyPaused = false, pendingText = null;

  // Helper to check if a node is inside bodyEl (handles text nodes too).
  function nodeInBody(node) {
    return node === bodyEl || bodyEl.contains(node);
  }

  // Helper to check if user has a non-collapsed selection inside bodyEl.
  function selectionInBody() {
    const sel = window.getSelection();
    if (sel.isCollapsed) return false;
    return nodeInBody(sel.anchorNode) || nodeInBody(sel.focusNode);
  }

  async function refresh() {
    if (isManuallyPaused) return;

    try {
      const r = await API.log(800);
      const newText = (r.lines || []).join("\n");

      // Skip DOM write if text hasn't changed.
      if (newText === lastText) return;

      // If user is selecting, freeze updates and keep polling. textContent assignment
      // destroys the text node tree, which nukes the selection mid-copy, so we defer the
      // write until the selection is released or moves out of the log.
      if (selectionInBody()) {
        isFrozen = true;
        pendingText = newText;
        updateFreezeIndicator();
        return;
      }

      // Update is safe to apply.
      lastText = newText;
      bodyEl.textContent = newText;
      if (autoscroll) bodyEl.scrollTop = bodyEl.scrollHeight;
      if (isFrozen) {
        isFrozen = false;
        pendingText = null;
        updateFreezeIndicator();
      }
    } catch (e) {
      const errMsg = "(log unavailable: " + e.message + ")";
      lastText = errMsg;
      bodyEl.textContent = errMsg;
    }
  }

  bodyEl.addEventListener("scroll", () => {
    autoscroll = bodyEl.scrollHeight - bodyEl.scrollTop - bodyEl.clientHeight < 40;
  });

  function updateFreezeIndicator() {
    const existingIndicator = actions.querySelector(".log-freeze-indicator");
    if (isFrozen) {
      if (!existingIndicator) {
        const indicator = document.createElement("span");
        indicator.className = "log-freeze-indicator";
        indicator.textContent = "paused — selection held";
        actions.insertBefore(indicator, actions.firstChild);
      }
    } else {
      if (existingIndicator) existingIndicator.remove();
    }
  }

  function open() {
    panel.hidden = false;
    btn.classList.add("active");
    autoscroll = true;
    isFrozen = false;
    isManuallyPaused = false;
    pendingText = null;
    updatePauseButton();
    updateFreezeIndicator();
    refresh();
    timer = setInterval(refresh, 1500);
  }

  function close() {
    panel.hidden = true;
    btn.classList.remove("active");
    clearInterval(timer);
    timer = null;
    isFrozen = false;
    isManuallyPaused = false;
    pendingText = null;
    updateFreezeIndicator();
  }

  btn.onclick = () => (panel.hidden ? open() : close());

  // Helper to get the currently selected text within bodyEl.
  function getSelectedText() {
    const sel = window.getSelection();
    if (sel.isCollapsed) return null;
    if (!nodeInBody(sel.anchorNode) && !nodeInBody(sel.focusNode)) return null;
    return sel.toString();
  }

  const copy = document.createElement("button");
  copy.className = "btn ghost tiny";
  copy.textContent = "Copy";
  copy.onclick = () => {
    try {
      const selectedText = getSelectedText();
      const textToCopy = selectedText || lastText;
      navigator.clipboard.writeText(textToCopy);
      copy.textContent = selectedText ? "Copied selection" : "Copied";
      setTimeout(() => (copy.textContent = "Copy"), 1200);
    } catch (_) {}
  };

  const pauseBtn = document.createElement("button");
  pauseBtn.className = "btn ghost tiny";
  pauseBtn.textContent = "⏸ Pause";
  pauseBtn.onclick = () => {
    isManuallyPaused = !isManuallyPaused;
    updatePauseButton();
    // Refetch rather than flushing pendingText: nothing was polled while paused, so the
    // only text held is whatever a selection froze earlier and it may be minutes stale.
    if (!isManuallyPaused) { pendingText = null; refresh(); }
  };

  function updatePauseButton() {
    pauseBtn.textContent = isManuallyPaused ? "▶ Resume" : "⏸ Pause";
    pauseBtn.classList.toggle("on", isManuallyPaused);
  }

  const x = document.createElement("button");
  x.className = "btn ghost tiny";
  x.textContent = "✕";
  x.onclick = close;

  actions.append(copy, pauseBtn, x);
})();
