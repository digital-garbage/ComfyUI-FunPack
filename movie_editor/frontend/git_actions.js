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

  window.FunPackGit = { refresh, get, update, switchBranch };
})();
