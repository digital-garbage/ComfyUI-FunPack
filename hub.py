"""FunPack Hub — a small landing page at /funpack/ linking to the two FunPack
UIs (Movie Editor "Cutting Room" and "Easy Gen"), plus an About block (version/
branch/commit) and an Update/Restart action. Talks to the Movie Editor's
existing /funpack/movie/api/{health,git/status,git/update,restart} endpoints —
no new backend API, same git_update.py logic the Editor's own "Updates &
ComfyUI" settings section uses. Kept as a single self-contained inline page
(no shared JS files) since its surface is a handful of fetch calls, much
smaller than pulling in api.js/dom.js/git_actions.js/restart_ui.js would need.
"""
from __future__ import annotations

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover - only available inside ComfyUI
    web = None
    PromptServer = None

from .movie_editor.backend import config as movie_config

HUB_PREFIX = "/funpack"

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>FunPack</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap" rel="stylesheet" />
<script>
  (function () {
    var v; try { v = localStorage.getItem("funpack_theme"); } catch (e) {}
    if (["dark", "light", "auto"].indexOf(v) < 0) v = "dark";
    var res = v === "auto"
      ? (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark")
      : v;
    document.documentElement.setAttribute("data-theme", res);
  })();
</script>
<style>
  /* Same token vocabulary and the same funpack_theme key as the two apps, so the
     door you come in through is not the one page that ignores your choice. */
  :root, :root[data-theme="dark"] {
    color-scheme: dark;
    --ink-0: #0c0b09; --ink-1: #131210; --ink-2: #1a1814; --ink-3: #221f19; --line: #2a261e;
    --text: #ece7db; --muted: #918a7a; --faint: #645e51;
    --accent: #f3a93c; --accent-hi: #ffc36b; --accent-lo: #e0972a; --on-accent: #1a1206;
    --teal: #57d0c6; --danger: #e8607a; --good: #79c479;
    --surface-modal: #16140f; --backdrop: rgba(6,5,4,.75); --shadow-ink: rgba(0,0,0,.7);
  }
  :root[data-theme="light"] {
    color-scheme: light;
    --ink-0: #f5f7fa; --ink-1: #ffffff; --ink-2: #eef1f6; --ink-3: #e1e7ef; --line: #d3dae3;
    --text: #16202c; --muted: #59677a; --faint: #8d9aab;
    --accent: #2f7fd4; --accent-hi: #5aa4ec; --accent-lo: #2668b0; --on-accent: #ffffff;
    --teal: #12857c; --danger: #c62a48; --good: #2f8f4e;
    --surface-modal: #ffffff; --backdrop: rgba(24,34,48,.34); --shadow-ink: rgba(24,34,48,.30);
  }
  :root { --radius: 12px; --shadow: 0 10px 30px -12px var(--shadow-ink); }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; min-height: 100%; background: var(--ink-0); color: var(--text);
    font-family: "Inter", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
    display: flex; align-items: center; justify-content: center; padding: 32px 16px;
  }
  .hub { text-align: center; max-width: 640px; width: 100%; }
  .brand { font-weight: 800; font-size: 28px; letter-spacing: .2px; margin-bottom: 6px; }
  .brand-mark { color: var(--accent); margin-right: 8px; }
  .lead { color: var(--muted); margin: 0 0 20px; font-size: 14px; }

  .about {
    display: inline-flex; align-items: center; gap: 14px; flex-wrap: wrap; justify-content: center;
    background: var(--ink-1); border: 1px solid var(--line); border-radius: var(--radius);
    padding: 10px 16px; margin-bottom: 28px; font-size: 12.5px; color: var(--muted);
    font-family: "Roboto Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .about .led { width: 7px; height: 7px; border-radius: 50%; background: var(--faint); display: inline-block; margin-right: 5px; }
  .about .led.ok { background: var(--good); box-shadow: 0 0 8px var(--good); }
  .about .led.bad { background: var(--danger); box-shadow: 0 0 8px var(--danger); }
  .about b { color: var(--text); font-weight: 600; }
  .about .sep { color: var(--line); }

  .update-row { margin-bottom: 28px; display: flex; align-items: center; justify-content: center; gap: 10px; flex-wrap: wrap; }
  .btn {
    background: var(--ink-2); border: 1px solid var(--line); border-radius: 7px; padding: 7px 14px;
    cursor: pointer; font-size: 13px; color: var(--text); transition: border-color .12s, background .12s;
  }
  .btn:hover:not(:disabled) { border-color: var(--accent); }
  .btn:disabled { opacity: .45; cursor: default; }
  .btn.primary { background: linear-gradient(var(--accent), var(--accent-lo)); color: var(--on-accent); border-color: var(--accent); font-weight: 700; }
  .btn.primary:hover:not(:disabled) { filter: brightness(1.08); }
  .btn.ghost { background: transparent; color: var(--muted); }
  .btn.ghost:hover:not(:disabled) { color: var(--accent-hi); background: var(--ink-3); }
  .update-hint { color: var(--faint); font-size: 12px; }
  .branch-select {
    background: var(--ink-2); border: 1px solid var(--line); border-radius: 7px; padding: 7px 10px;
    font-size: 13px; color: var(--text); font-family: "Roboto Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
    cursor: pointer; max-width: 220px;
  }
  .branch-select:hover:not(:disabled) { border-color: var(--accent); }
  .branch-select:disabled { opacity: .45; cursor: default; }

  .cards { display: flex; gap: 18px; flex-wrap: wrap; justify-content: center; }
  a.card {
    display: block; width: 260px; padding: 22px 20px; text-align: left;
    background: var(--ink-1); border: 1px solid var(--line); border-radius: var(--radius);
    text-decoration: none; color: inherit; box-shadow: var(--shadow);
    transition: border-color .12s, transform .12s;
  }
  a.card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .card-title { font-weight: 700; font-size: 17px; margin-bottom: 6px; }
  .card-title.easy { color: var(--accent-hi); }
  .card-title.editor { color: var(--teal); }
  .card-sub { color: var(--faint); font-size: 12.5px; line-height: 1.5; }

  .restart-overlay {
    position: fixed; inset: 0; z-index: 500; background: var(--backdrop); backdrop-filter: blur(3px);
    display: flex; align-items: center; justify-content: center;
  }
  .restart-card {
    background: var(--surface-modal); border: 1px solid var(--line); border-radius: 14px; padding: 26px 30px;
    box-shadow: var(--shadow); text-align: center; max-width: 360px;
  }
  .restart-spin {
    width: 26px; height: 26px; margin: 0 auto 14px; border-radius: 50%;
    border: 3px solid var(--line); border-top-color: var(--accent); animation: spin .8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .restart-msg { color: var(--muted); font-size: 13px; white-space: pre-wrap; }
</style>
</head>
<body>
  <div class="hub">
    <div class="brand"><span class="brand-mark">◉</span>FunPack</div>
    <p class="lead">Pick a UI.</p>

    <div class="about" id="about">Loading…</div>
    <div class="update-row" id="update-row"></div>

    <div class="cards">
      <a class="card" href="/funpack/easy/">
        <div class="card-title easy">Easy Gen</div>
        <div class="card-sub">One prompt, one Generate button, one big preview. Advanced settings tucked away.</div>
      </a>
      <a class="card" href="/funpack/movie/">
        <div class="card-title editor">Cutting Room</div>
        <div class="card-sub">The full multi-scene NLE — timeline, composer, overlays, audio.</div>
      </a>
    </div>
  </div>

<script>
(function () {
  const API = (p) => "/funpack/movie/api" + p;
  const aboutEl = document.getElementById("about");
  const updateEl = document.getElementById("update-row");

  async function j(method, url, body) {
    const opts = { method, headers: {} };
    if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
    const res = await fetch(url, opts);
    if (!res.ok) {
      let payload = null; try { payload = await res.json(); } catch (_) {}
      throw new Error((payload && payload.detail) || res.statusText || "Request failed");
    }
    return res.status === 204 ? null : res.json();
  }

  function led(ok) { return `<span class="led ${ok ? "ok" : "bad"}"></span>`; }

  function renderAbout(health, git) {
    const parts = [];
    parts.push(`${led(!!(health && health.ok))}${health && health.ok ? "online" : "offline"}`);
    if (git && git.ok) {
      parts.push(`<b>v${git.version || "?"}</b>`);
      parts.push(`${git.branch}@${git.commit}${git.dirty ? " (local changes)" : ""}`);
    } else {
      parts.push(git && git.version ? `<b>v${git.version}</b>` : "");
      parts.push("git unavailable");
    }
    aboutEl.innerHTML = parts.filter(Boolean).join(' <span class="sep">·</span> ');
  }

  function renderUpdateRow(git) {
    updateEl.innerHTML = "";
    if (!git || !git.ok) return;
    const updateBtn = document.createElement("button");
    updateBtn.className = "btn primary";
    updateBtn.disabled = !!git.dirty;
    updateBtn.textContent = git.behind > 0 ? `${git.behind}↓ Update` : "⬇⟳ Update";
    updateBtn.onclick = doUpdate;
    updateEl.appendChild(updateBtn);

    // A <select> rather than a button: /git/status already carries the branch list,
    // so the choice and the current branch fit in one control instead of two.
    const branches = git.branches || [];
    if (branches.length) {
      const sel = document.createElement("select");
      sel.className = "branch-select";
      sel.title = "Switch the FunPack branch and restart ComfyUI";
      sel.disabled = !!git.dirty;
      branches.forEach((b) => {
        const o = document.createElement("option");
        o.value = b; o.textContent = b === git.branch ? b + "  (current)" : b;
        if (b === git.branch) o.selected = true;
        sel.appendChild(o);
      });
      sel.onchange = () => doSwitch(sel, git.branch);
      updateEl.appendChild(sel);
    }

    const restartBtn = document.createElement("button");
    restartBtn.className = "btn ghost";
    restartBtn.textContent = "⟳ Restart ComfyUI";
    restartBtn.onclick = doRestart;
    updateEl.appendChild(restartBtn);

    const hint = document.createElement("span");
    hint.className = "update-hint";
    hint.textContent = git.dirty
      ? "Local changes — commit or stash first"
      : (git.behind > 0 ? `${git.behind} commit(s) behind origin/${git.branch}` : `origin/${git.branch} up to date`);
    updateEl.appendChild(hint);
  }

  function showOverlay(message) {
    document.querySelector(".restart-overlay")?.remove();
    const ov = document.createElement("div"); ov.className = "restart-overlay";
    const card = document.createElement("div"); card.className = "restart-card";
    const spin = document.createElement("div"); spin.className = "restart-spin";
    const msg = document.createElement("div"); msg.className = "restart-msg"; msg.textContent = message;
    card.append(spin, msg); ov.append(card); document.body.append(ov);
    return msg;
  }

  function waitForReload(msgEl, startMs) {
    const start = startMs || Date.now();
    const tick = async () => {
      try {
        const h = await j("GET", API("/health"));
        if (h && h.ok) { location.reload(); return; }
      } catch (_) { /* still down */ }
      if (Date.now() - start > 90000) {
        msgEl.textContent = "Still waiting on ComfyUI…\\nIt may need a manual restart - check the console.";
      }
      setTimeout(tick, 2000);
    };
    setTimeout(tick, 3500);
  }

  async function doUpdate() {
    let git;
    try { git = await j("GET", API("/git/status")); } catch (e) { alert("Could not read git status: " + e.message); return; }
    if (!git.ok) { alert(git.detail || "Git unavailable for this install."); return; }
    if (git.dirty) { alert("Local changes detected in the FunPack folder.\\nCommit or stash them before updating."); return; }
    const behind = git.behind > 0 ? `\\n\\n${git.behind} commit(s) available on origin/${git.branch}.` : "";
    if (!confirm(`Pull latest "${git.branch}" from origin and restart ComfyUI?\\n\\nAny running generation will be lost.${behind}`)) return;
    const msg = showOverlay(`Pulling origin/${git.branch}…\\nComfyUI will restart when the pull finishes.`);
    try {
      const res = await j("POST", API("/git/update"), {});
      msg.textContent = res.updated ? `Updated ${res.before} → ${res.after}.\\nRestarting ComfyUI…` : "Already up to date.\\nRestarting ComfyUI…";
    } catch (e) {
      document.querySelector(".restart-overlay")?.remove();
      alert("Update failed: " + e.message);
      return;
    }
    waitForReload(msg, Date.now());
  }

  async function doSwitch(sel, current) {
    const branch = sel.value;
    if (branch === current) return;
    if (!confirm(`Switch to "${branch}", pull from origin, and restart ComfyUI?\n\nAny running generation will be lost.`)) {
      sel.value = current;
      return;
    }
    const msg = showOverlay(`Switching to ${branch}…\nComfyUI will restart when ready.`);
    try {
      const res = await j("POST", API("/git/checkout"), { branch });
      msg.textContent = res.updated
        ? `Switched to ${branch} (${res.before} → ${res.after}).\nRestarting ComfyUI…`
        : `On ${branch}, already up to date.\nRestarting ComfyUI…`;
    } catch (e) {
      document.querySelector(".restart-overlay")?.remove();
      sel.value = current;
      alert("Branch switch failed: " + e.message);
      return;
    }
    waitForReload(msg, Date.now());
  }

  async function doRestart() {
    if (!confirm("Restart ComfyUI now?\\n\\nThe server will be down for ~10-40s and any running generation will be lost. This page reloads automatically when it's back.")) return;
    const msg = showOverlay("Restarting ComfyUI…\\nThis page will reload when it's back.");
    try { await j("POST", API("/restart"), {}); } catch (_) { /* connection drops as it execv's - expected */ }
    waitForReload(msg, Date.now());
  }

  async function boot() {
    let health = null, git = null;
    try { health = await j("GET", API("/health")); } catch (_) {}
    try { git = await j("GET", API("/git/status")); } catch (_) {}
    renderAbout(health, git);
    renderUpdateRow(git);
  }
  boot();
})();
</script>
</body>
</html>
"""

if web is not None and PromptServer is not None:
    routes = PromptServer.instance.routes

    @routes.get(HUB_PREFIX)
    async def _hub_redirect(_req):
        raise web.HTTPFound(HUB_PREFIX + "/")

    @routes.get(HUB_PREFIX + "/")
    async def _hub_index(_req):
        return web.Response(body=_PAGE, content_type="text/html", headers={"Cache-Control": "no-store, max-age=0"})

    print(f"[FunPack] Hub available at {movie_config.comfy_display_url()}{HUB_PREFIX}/")
