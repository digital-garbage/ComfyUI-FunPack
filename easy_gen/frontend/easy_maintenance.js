// window.FunPackMaintenance shim — ports the five actions menubar.js exposes
// for the Settings window's built-in "Refinement & Taste" / "Updates & ComfyUI"
// sections (settings_window.js), without pulling in the other ~300 lines of
// menubar.js (menu rendering, keyboard shortcuts, timeline-specific state).
// Same API calls, same confirm/alert flows as the Editor.
(function () {
  const API = window.MovieEditorAPI;

  function projectRefinementKey() {
    return window.Store?.get().project?.refinement_key || "";
  }

  const refinementKeyImportInput = (function () {
    const inp = document.createElement("input");
    inp.type = "file"; inp.accept = ".json,application/json"; inp.style.display = "none";
    inp.onchange = async () => {
      const file = inp.files?.[0];
      inp.value = "";
      if (!file) return;
      try {
        const data = JSON.parse(await file.text());
        let res;
        try {
          res = await API.importRefinementKey(data);
        } catch (e) {
          if (!e.exists) throw e;
          if (!confirm(`A refinement key named "${e.key}" already exists. Overwrite it?`)) return;
          res = await API.importRefinementKey(data, { overwrite: true });
        }
        alert(`Imported refinement key "${res.key}".`);
      } catch (e) {
        alert("Refinement key import failed: " + (e.message || e));
      }
    };
    document.body.appendChild(inp);
    return inp;
  })();

  async function exportRefinementKey() {
    let payload;
    try { payload = await API.refinementKeys(); }
    catch (e) { alert("Could not list refinement keys: " + (e.message || e)); return; }
    const names = (payload.keys || []).filter((k) => k && k !== "-None-");
    if (!names.length) { alert("No refinement keys on disk yet."); return; }
    const cur = projectRefinementKey();
    if (!window.SlotPicker) return;
    window.SlotPicker.open({
      title: "Export refinement key",
      options: names.map((k) => ({ value: k, label: k, hint: k === cur ? "project session" : "" })),
      onPick: async (key) => {
        try { await API.exportRefinementKeyFile(key); }
        catch (e) { alert("Refinement key export failed: " + (e.message || e)); }
      },
    });
  }

  async function deleteRefinementKey() {
    let payload;
    try { payload = await API.refinementKeys(); }
    catch (e) { alert("Could not list refinement keys: " + (e.message || e)); return; }
    const names = (payload.keys || []).filter((k) => k && k !== "-None-");
    if (!names.length) { alert("No refinement keys on disk yet."); return; }
    const cur = projectRefinementKey();
    if (!window.SlotPicker) return;
    window.SlotPicker.open({
      title: "Delete refinement key",
      options: names.map((k) => ({ value: k, label: k, hint: k === cur ? "project session" : "" })),
      onPick: async (key) => {
        if (!confirm(`Delete refinement key "${key}"?\n\nThis removes its learned state AND all sidecars (value function, blessed attention/K-V banks, creativity latent, velocity memory). This cannot be undone.`)) {
          return;
        }
        try { await API.deleteRefinementKey(key); alert(`Deleted "${key}".`); }
        catch (e) { alert("Delete failed: " + (e.message || e)); }
      },
    });
  }

  async function clearGlobalTaste() {
    let info;
    try { info = await API.absoluteStoreInfo(); }
    catch (e) { alert("Could not read the global-taste store: " + (e.message || e)); return; }
    if (!info.exists) { alert("The Absolute global-taste store is already empty."); return; }
    if (!confirm(`Clear the Absolute global-taste store?\n\nIt has pooled ${info.total_iterations} rated generation(s) (${info.liked_count} liked / ${info.bad_count} disliked directions) across all prompts. It learns from every rated run and is applied only in absolute/both steer mode. This cannot be undone.`)) {
      return;
    }
    try { await API.clearAbsoluteStore(); alert("Cleared the Absolute global-taste store."); }
    catch (e) { alert("Clear global taste failed: " + (e.message || e)); }
  }

  async function restartComfy() {
    if (!confirm("Restart ComfyUI now?\n\nThe server will be down for ~10-40s and any running generation will be lost. This page reloads automatically when it's back.")) return;
    const msg = window.FunPackRestart?.showOverlay?.("Restarting ComfyUI…\nThis page will reload when it's back.");
    try { await API.restart(); } catch (_) { /* connection drops as it execv's - expected */ }
    window.FunPackRestart?.waitForReload?.(msg, Date.now());
  }

  window.FunPackMaintenance = {
    exportRefinementKey,
    importRefinementKeyFile: () => refinementKeyImportInput.click(),
    deleteRefinementKey,
    clearGlobalTaste,
    restartComfy,
  };
})();
