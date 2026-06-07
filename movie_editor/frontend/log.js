// Menubar "Log" button + a collapsible panel showing ComfyUI's real backend log
// (tee'd stdout/stderr from the server). Polls only while open.
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

  let timer = null, autoscroll = true;

  async function refresh() {
    try {
      const r = await API.log(800);
      bodyEl.textContent = (r.lines || []).join("\n");
      if (autoscroll) bodyEl.scrollTop = bodyEl.scrollHeight;
    } catch (e) {
      bodyEl.textContent = "(log unavailable: " + e.message + ")";
    }
  }
  bodyEl.addEventListener("scroll", () => {
    autoscroll = bodyEl.scrollHeight - bodyEl.scrollTop - bodyEl.clientHeight < 40;
  });

  function open() { panel.hidden = false; btn.classList.add("active"); autoscroll = true; refresh(); timer = setInterval(refresh, 1500); }
  function close() { panel.hidden = true; btn.classList.remove("active"); clearInterval(timer); timer = null; }
  btn.onclick = () => (panel.hidden ? open() : close());

  const copy = document.createElement("button"); copy.className = "btn ghost tiny"; copy.textContent = "Copy";
  copy.onclick = () => { try { navigator.clipboard.writeText(bodyEl.textContent); copy.textContent = "Copied"; setTimeout(() => (copy.textContent = "Copy"), 1200); } catch (_) {} };
  const x = document.createElement("button"); x.className = "btn ghost tiny"; x.textContent = "✕"; x.onclick = close;
  actions.append(copy, x);
})();
