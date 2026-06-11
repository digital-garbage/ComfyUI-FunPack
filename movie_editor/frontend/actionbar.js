// Main transport actions: Generate / Generate selection / Render (timeline header).
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const mount = document.getElementById("transport-actions");
  if (!mount) return;

  function selCount() { return S.selectedSceneCount ? S.selectedSceneCount() : (S.get().selectedSceneId ? 1 : 0); }
  function hasProject() { return !!S.get().project; }
  function busy(st) { return ["queuing", "running", "pending"].includes(st.gen?.state); }

  function render(st) {
    clear(mount);
    const genAll = el("button", "btn primary compact", "▶ Generate");
    genAll.title = "Generate the whole montage";
    genAll.disabled = !hasProject() || busy(st);
    genAll.onclick = () => S.generate(null);
    mount.append(genAll);

    const n = selCount();
    const selLabel = n > 1 ? `Selected (${n})` : "Selected";
    const genSel = el("button", "btn ghost compact", selLabel);
    genSel.title = n > 1
      ? `Generate ${n} selected scenes (one chain run per segment)`
      : "Generate the selected scene";
    genSel.disabled = n === 0 || busy(st);
    genSel.onclick = () => S.generateSelected();
    mount.append(genSel);

    const renderBtn = el("button", "btn render compact", "⧉ Render");
    renderBtn.title = "Stitch generated clips into a final video";
    renderBtn.disabled = !hasProject() || busy(st);
    renderBtn.onclick = () => S.renderFinal();
    mount.append(renderBtn);
  }

  if (window.ViewBus) window.ViewBus.subscribeActionbar(render);
  else S.subscribe(render);
  render(S.get());
  render(S.get());
})();