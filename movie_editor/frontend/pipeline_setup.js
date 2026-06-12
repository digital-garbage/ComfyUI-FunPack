// Prompt to install missing custom-node packs for the built-in pipeline (via Manager).
(function () {
  const { el } = window.dom;
  const S = window.Store;
  const API = window.MovieEditorAPI;
  const LS_KEY = "funpack_pipeline_deps_dismissed";

  let overlay = null;
  let _pollTimer = null;

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
    box.append(el("div", "pipe-setup-manual-title", "Install ComfyUI-Manager or add these repos under custom_nodes:"));
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

  async function pollInstall(jobId, msgEl) {
    try {
      const st = await API.pipelineDepsInstallStatus(jobId);
      if (st.state === "installing" || st.state === "queued") {
        const n = Math.min(st.done + 1, st.total || 1);
        const title = st.current_title || "…";
        msgEl.textContent = `Downloading and installing missing nodes:\n${n} out of ${st.total || "?"} - ${title}`;
        _pollTimer = setTimeout(() => pollInstall(jobId, msgEl), 1200);
        return;
      }
      if (st.state === "restarting") {
        msgEl.textContent = "Install complete.\nRestarting ComfyUI…";
        window.FunPackRestart?.waitForReload?.(msgEl, Date.now());
        return;
      }
      if (st.state === "error") {
        window.FunPackRestart?.removeOverlay?.();
        alert(st.error || "Install failed.");
        return;
      }
      _pollTimer = setTimeout(() => pollInstall(jobId, msgEl), 1200);
    } catch (e) {
      window.FunPackRestart?.removeOverlay?.();
      alert(String(e.message || e));
    }
  }

  async function startInstall(deps) {
    const ids = (deps.missing_packs || []).map((p) => p.id);
    if (!ids.length) return;
    closeModal();
    const msg = window.FunPackRestart?.showOverlay?.(
      "Downloading and installing missing nodes:\n1 out of " + ids.length + " - " + (deps.missing_packs[0]?.title || "…"),
    );
    if (!msg) return;
    try {
      const res = await API.pipelineDepsInstall(ids);
      pollInstall(res.job_id, msg);
    } catch (e) {
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
    const closeBtn = el("button", "btn ghost", "Close");
    closeBtn.type = "button";
    closeBtn.onclick = () => closeModal();
    const headRight = el("div", "modal-head-right");
    headRight.append(closeBtn);
    head.append(headRight);
    modal.append(head);

    const content = el("div", "modal-content");
    content.append(el("p", "pipe-setup-lead",
      "Nodes required for the built-in pipeline are missing. Do you want to install them?"));
    content.append(packList(deps));
    if (!deps.manager_available) {
      content.append(el("div", "pipe-setup-warn",
        "ComfyUI-Manager was not detected. Automatic install is unavailable."));
      content.append(manualBlock(deps));
    }
    if ((deps.unmapped_classes || []).length) {
      content.append(el("div", "pipe-setup-warn",
        "Unmapped missing classes: " + deps.unmapped_classes.join(", ")));
    }
    modal.append(content);

    const foot = el("div", "pipe-setup-foot");
    if (deps.manager_available) {
      const installBtn = el("button", "btn primary", "Install missing nodes");
      installBtn.type = "button";
      installBtn.onclick = () => startInstall(deps);
      foot.append(installBtn);
    }
    const ownBtn = el("button", "btn", "No, I'll use my own pipeline");
    ownBtn.type = "button";
    ownBtn.onclick = () => useOwnPipeline();
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
    if (!deps?.needs_install) return;
    openModal(deps);
  }

  window.PipelineSetup = { maybePrompt, close: closeModal };
})();
