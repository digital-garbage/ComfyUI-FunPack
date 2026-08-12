// Shared FunPack git actions (Switch branch / Update + reload). Used by both the
// FunPack menu and the welcome screen so neither has to duplicate the flow — or force
// the user to load a project first just to switch branches.
(function () {
  const API = () => window.MovieEditorAPI;

  let _status = null;  // cached git status (branch, dirty, behind, branches)

  async function refresh() {
    try { _status = await API().gitStatus(); }
    catch (_) { _status = { ok: false }; }
    return _status;
  }

  function get() { return _status; }

  // Ensure we have a usable status; surface the same alerts the menu used to.
  async function _ensureStatus() {
    let gs = _status;
    if (!gs?.ok) {
      try { gs = await API().gitStatus(); _status = gs; }
      catch (e) { alert(String(e.message || e)); return null; }
    }
    if (!gs?.ok) { alert(gs?.detail || "Git status unavailable for this install."); return null; }
    return gs;
  }

  function _restartOverlay(message) {
    return window.FunPackRestart?.showOverlay?.(message);
  }

  function _waitForComfyReload(msgEl, startMs) {
    window.FunPackRestart?.waitForReload?.(msgEl, startMs);
  }

  // All three actions below end in a server restart and a page reload, so anything the
  // store is still holding — a debounced field, or a whole settings session waiting on its
  // close — has to reach disk first. flushSave() ignores the suspension for exactly this.
  async function _flushPendingEdits() {
    try { await window.Store?.flushSave?.(); } catch (_) {}
  }

  async function update() {
    const gs = await _ensureStatus();
    if (!gs) return;
    if (gs.dirty) {
      alert("Local changes detected in the FunPack folder.\nCommit or stash them before updating from git.");
      return;
    }
    const branch = gs.branch || "dev";
    const behind = gs.behind > 0 ? `\n\n${gs.behind} commit(s) available on origin/${branch}.` : "";
    if (!confirm(`Pull latest "${branch}" from origin and restart ComfyUI?\n\nAny running generation will be lost.${behind}`)) return;
    await _flushPendingEdits();
    const msg = _restartOverlay(`Pulling origin/${branch}…\nComfyUI will restart when the pull finishes.`);
    try {
      const res = await API().gitUpdate(branch);
      msg.textContent = res.updated
        ? `Updated ${res.before} → ${res.after}.\nRestarting ComfyUI…`
        : "Already up to date.\nRestarting ComfyUI…";
    } catch (e) {
      window.FunPackRestart?.removeOverlay?.();
      alert("Update failed: " + (e.message || e));
      return;
    }
    _waitForComfyReload(msg, Date.now());
  }

  async function switchBranch() {
    const gs = await _ensureStatus();
    if (!gs) return;
    if (gs.dirty) {
      alert("Local changes detected in the FunPack folder.\nCommit or stash them before switching branches.");
      return;
    }
    const branches = gs.branches || [];
    if (!branches.length) { alert("No git branches found."); return; }
    const cur = gs.branch;
    if (!window.SlotPicker) return;
    window.SlotPicker.open({
      title: "Switch FunPack branch",
      options: branches.map((b) => ({ value: b, label: b, hint: b === cur ? "current" : "" })),
      onPick: async (branch) => {
        if (branch === cur) return;
        if (!confirm(`Switch to "${branch}", pull from origin, and restart ComfyUI?\n\nAny running generation will be lost.`)) return;
        await _flushPendingEdits();
        const msg = _restartOverlay(`Switching to ${branch}…\nComfyUI will restart when ready.`);
        try {
          const res = await API().gitCheckout(branch);
          msg.textContent = res.updated
            ? `Switched to ${branch} (${res.before} → ${res.after}).\nRestarting ComfyUI…`
            : `On ${branch}, already up to date.\nRestarting ComfyUI…`;
        } catch (e) {
          window.FunPackRestart?.removeOverlay?.();
          alert("Branch switch failed: " + (e.message || e));
          return;
        }
        _waitForComfyReload(msg, Date.now());
      },
    });
  }

  async function restartComfy() {
    if (!confirm(
      "Restart ComfyUI now?\n\nThe server will be down for ~10-40s and any running generation "
      + "will be lost. This page reloads automatically when it's back."
    )) return;
    await _flushPendingEdits();
    const msg = _restartOverlay("Restarting ComfyUI…\nThis page will reload when it's back.");
    if (!msg) return;
    try { await API().restart(); } catch (_) { /* the connection drops as it execv's */ }
    _waitForComfyReload(msg, Date.now());
  }

  // The three maintenance actions every FunPack splash screen offers. Shared because
  // they are the actions you need BEFORE opening a project — a branch swap should not
  // require first loading a montage you did not want.
  function maintenanceRow(cls) {
    const { el } = window.dom;
    const row = el("div", "maint-row" + (cls ? " " + cls : ""));

    const mk = (label, hint, fn) => {
      const b = el("button", "maint-btn");
      b.type = "button";
      b.append(el("span", "maint-btn-label", label));
      b.append(el("span", "maint-btn-hint", hint));
      b.onclick = fn;
      return b;
    };

    const updateBtn = mk("Update", "Checking…", () => update());
    const switchBtn = mk("Switch branch", "Checking…", () => switchBranch());
    const restartBtn = mk("Restart ComfyUI", "Reload the server without updating", () => restartComfy());
    row.append(updateBtn, switchBtn, restartBtn);

    const hint = (b) => b.querySelector(".maint-btn-hint");
    // No isConnected guard: callers build the row before appending their overlay, so
    // status can land while the row is still detached — bailing there would leave the
    // hints reading "Checking…" for good. Writing into a discarded row is harmless.
    refresh().then((gs) => {
      if (!gs?.ok) {
        updateBtn.classList.add("disabled");
        switchBtn.classList.add("disabled");
        hint(updateBtn).textContent = "Git unavailable for this install";
        hint(switchBtn).textContent = "Git unavailable for this install";
        return;
      }
      hint(updateBtn).textContent = gs.dirty
        ? "Local changes — commit first"
        : (gs.behind > 0 ? `${gs.behind} commit(s) behind origin/${gs.branch}` : `origin/${gs.branch} up to date`);
      hint(switchBtn).textContent = gs.dirty
        ? `On ${gs.branch} · local changes — commit first`
        : `On ${gs.branch} · pick another`;
    });

    return row;
  }

  window.FunPackGit = { refresh, get, update, switchBranch, restartComfy, maintenanceRow };
})();
