// Prompt to install missing custom-node packs for the built-in pipeline (via Manager).
(function () {
  const { el } = window.dom;
  const S = window.Store;
  const API = window.MovieEditorAPI;
  const LS_KEY = "funpack_pipeline_deps_dismissed";

  let overlay = null;
  let _pollTimer = null;
  let _activeJobId = null;

  function dismissed() {
    try { return localStorage.getItem(LS_KEY) === "1"; } catch (_) { return false; }
  }

  function setDismissed() {
    try { localStorage.setItem(LS_KEY, "1"); } catch (_) {}
  }

  function closeModal() {
    if (_pollTimer) { clearTimeout(_pollTimer); _pollTimer = null; }
    overlay?.remove();
    overlay = null;
  }

  function builtInPipelineActive(st) {
    const m = st.models || {};
    return !m.disable_core;
  }

  async function useOwnPipeline() {
    if (!S.get().project) return;
    const models = JSON.parse(JSON.stringify(S.get().models || { slots: [] }));
    models.disable_core = true;
    await API.saveModels(S.get().project.id, models);
    await S.loadModels();
    setDismissed();
    closeModal();
  }

  function packList(deps) {
    const ul = el("ul", "pipe-setup-list");
    (deps.missing_packs || []).forEach((p) => {
      const li = el("li", null, p.title);
      if (p.missing_classes?.length) {
        li.append(el("span", "pipe-setup-sub", p.missing_classes.join(", ")));
      }
      ul.append(li);
    });
    return ul;
  }

  function manualBlock(deps) {
    const box = el("div", "pipe-setup-manual");
    box.append(el("div", "pipe-setup-manual-title", "Or clone these repos into custom_nodes manually:"));
    const ul = el("ul", "pipe-setup-list");
    (deps.manual_urls || []).forEach((u) => {
      const li = el("li");
      const a = el("a", null, u.title);
      a.href = u.url;
      a.target = "_blank";
      a.rel = "noopener";
      li.append(a);
      ul.append(li);
    });
    box.append(ul);
    return box;
  }

  async function cancelInstall() {
    const jobId = _activeJobId;
    if (!jobId) {
      window.FunPackRestart?.removeOverlay?.();
      return;
    }
    try { await API.pipelineDepsInstallCancel(jobId); } catch (_) { /* server may already be gone */ }
    _activeJobId = null;
    if (_pollTimer) { clearTimeout(_pollTimer); _pollTimer = null; }
    window.FunPackRestart?.removeOverlay?.();
  }

  function progressMessage(st) {
    if (st.kind === "manager") {
      return "Cloning ComfyUI-Manager…\nRestart follows automatically when ready.";
    }
    const n = Math.min(st.done + 1, st.total || 1);
    const title = st.current_title || "…";
    let msg = `Downloading and installing missing nodes:\n${n} out of ${st.total || "?"} - ${title}`;
    if (st.stale) {
      msg += "\n\nNo progress for a while. You can cancel and try again from ComfyUI-Manager.";
    }
    return msg;
  }

  async function pollInstall(jobId, msgEl) {
    try {
      const st = await API.pipelineDepsInstallStatus(jobId);
      if (st.state === "installing" || st.state === "queued") {
        msgEl.textContent = progressMessage(st);
        _pollTimer = setTimeout(() => pollInstall(jobId, msgEl), 1200);
        return;
      }
      if (st.state === "cancelled") {
        _activeJobId = null;
        window.FunPackRestart?.removeOverlay?.();
        return;
      }
      if (st.state === "restarting") {
        msgEl.textContent = st.kind === "manager"
          ? "ComfyUI-Manager installed.\nRestarting ComfyUI…"
          : "Install complete.\nRestarting ComfyUI…";
        _activeJobId = null;
        window.FunPackRestart?.waitForReload?.(msgEl, Date.now());
        return;
      }
      if (st.state === "error") {
        _activeJobId = null;
        window.FunPackRestart?.removeOverlay?.();
        alert(st.error || "Install failed.");
        return;
      }
      _pollTimer = setTimeout(() => pollInstall(jobId, msgEl), 1200);
    } catch (e) {
      _activeJobId = null;
      window.FunPackRestart?.removeOverlay?.();
      alert(String(e.message || e));
    }
  }

  function beginInstallOverlay(initialMessage) {
    closeModal();
    const msg = window.FunPackRestart?.showOverlay?.(initialMessage, {
      cancelLabel: "Cancel install",
      onCancel: () => cancelInstall(),
    });
    return msg;
  }

  async function startManagerInstall() {
    const msg = beginInstallOverlay("Cloning ComfyUI-Manager…\nRestart follows automatically when ready.");
    if (!msg) return;
    try {
      const res = await API.pipelineDepsInstallManager();
      _activeJobId = res.job_id;
      pollInstall(res.job_id, msg);
    } catch (e) {
      _activeJobId = null;
      window.FunPackRestart?.removeOverlay?.();
      alert("Install failed: " + (e.message || e));
    }
  }

  async function restartComfyOnly() {
    closeModal();
    const msg = beginInstallOverlay("Restarting ComfyUI…\nComfyUI-Manager is on disk but not loaded yet.");
    if (!msg) return;
    try {
      await API.restart();
    } catch (_) { /* expected while server exits */ }
    window.FunPackRestart?.waitForReload?.(msg, Date.now());
  }

  async function startInstall(deps) {
    const ids = (deps.missing_packs || []).map((p) => p.id);
    if (!ids.length) return;
    const msg = beginInstallOverlay(
      "Downloading and installing missing nodes:\n1 out of " + ids.length + " - " + (deps.missing_packs[0]?.title || "…"),
    );
    if (!msg) return;
    try {
      const res = await API.pipelineDepsInstall(ids);
      _activeJobId = res.job_id;
      pollInstall(res.job_id, msg);
    } catch (e) {
      _activeJobId = null;
      window.FunPackRestart?.removeOverlay?.();
      alert("Install failed: " + (e.message || e));
    }
  }

  function openModal(deps) {
    closeModal();
    overlay = el("div", "modal-overlay pipe-setup-overlay");
    const modal = el("div", "modal pipe-setup-modal");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", "Pipeline setup"));
    modal.append(head);

    const content = el("div", "modal-content");
    if (deps.needs_manager_install) {
      content.append(el("p", "pipe-setup-lead",
        "Nodes required for the built-in pipeline are missing. Install ComfyUI-Manager first, then we can fetch the rest."));
    } else if (deps.needs_manager_restart) {
      content.append(el("p", "pipe-setup-lead",
        "ComfyUI-Manager is on disk but not running yet. Restart ComfyUI to load it, then install the missing node packs."));
    } else {
      content.append(el("p", "pipe-setup-lead",
        "Nodes required for the built-in pipeline are missing. Do you want to install them?"));
    }
    if ((deps.missing_packs || []).length) {
      content.append(packList(deps));
    }
    if (deps.needs_manager_install) {
      content.append(el("div", "pipe-setup-hint",
        "ComfyUI-Manager handles downloads and updates for custom node packs."));
      content.append(manualBlock(deps));
    } else if (!deps.manager_available) {
      content.append(manualBlock(deps));
    }
    if ((deps.unmapped_classes || []).length) {
      content.append(el("div", "pipe-setup-warn",
        "Unmapped missing classes: " + deps.unmapped_classes.join(", ")));
    }
    modal.append(content);

    const foot = el("div", "pipe-setup-foot");
    if (deps.needs_manager_install) {
      const mgrBtn = el("button", "btn primary", "Install ComfyUI-Manager");
      mgrBtn.type = "button";
      mgrBtn.onclick = () => startManagerInstall();
      foot.append(mgrBtn);
    } else if (deps.needs_manager_restart) {
      const restartBtn = el("button", "btn primary", "Restart ComfyUI");
      restartBtn.type = "button";
      restartBtn.onclick = () => restartComfyOnly();
      foot.append(restartBtn);
    } else if (deps.manager_available) {
      const installBtn = el("button", "btn primary", "Install missing nodes");
      installBtn.type = "button";
      installBtn.onclick = () => startInstall(deps);
      foot.append(installBtn);
    }
    const ownBtn = el("button", "btn", "No, I'll use my own pipeline");
    ownBtn.type = "button";
    // This PERSISTS disable_core on the project — without a warning it read like a
    // harmless "dismiss", then Generate failed ("no outputs") and Studio/Chain Sampler
    // settings vanished with no visible cause.
    ownBtn.onclick = () => {
      if (!confirm(
        "This DISABLES the built-in FunPack pipeline for this project and saves that choice.\n\n"
        + "Generation will then run ONLY the nodes you wire yourself in Models — until you wire "
        + "a final IMAGE to the 🌐 Global video output, Generate has nothing to run.\n\n"
        + "You can re-enable it any time: Models → Enable built-in pipeline.\n\nContinue?"
      )) return;
      useOwnPipeline();
    };
    foot.append(ownBtn);
    const dismissBtn = el("button", "btn ghost", "Close");
    dismissBtn.type = "button";
    dismissBtn.onclick = () => closeModal();
    foot.append(dismissBtn);
    modal.append(foot);

    overlay.append(modal);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) closeModal(); });
    document.body.append(overlay);
  }

  async function maybePrompt() {
    if (window.__FUNPACK_TOUR__) return;
    if (dismissed()) return;
    const st = S.get();
    if (!st.project || !builtInPipelineActive(st)) return;
    let deps;
    try { deps = await API.pipelineDeps(); } catch (_) { return; }
    if (!deps?.needs_setup) return;
    openModal(deps);
  }

  window.PipelineSetup = { maybePrompt, close: closeModal };
})();
