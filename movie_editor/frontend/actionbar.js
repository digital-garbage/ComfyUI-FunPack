// Main transport actions: Generate / Generate selection / Render (timeline header).
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const mount = document.getElementById("transport-actions");
  if (!mount) return;

  function selCount() { return S.selectedSceneCount ? S.selectedSceneCount() : (S.get().selectedSceneId ? 1 : 0); }
  function hasProject() { return !!S.get().project; }
  function busy(st) { return ["queuing", "running", "pending"].includes(st.gen?.state); }

  // Live elapsed readout on whichever button started the run. The tick writes the button's
  // text DIRECTLY and never calls render(): a re-render once a second would rebuild every
  // button in this zone, throwing away focus and hover for a number one label owns. For the
  // same reason the elapsed value is not in the store's state and not in fpActionbar — the
  // zone still repaints on gen.state, which is what starts and stops this ticker.
  let tick = null;
  let liveBtn = null;
  let liveBase = "";

  // Seconds are zero-padded on BOTH sides of the minute mark so the label's width only
  // changes once a minute instead of every time a digit is added.
  function fmt(ms) {
    const s = Math.max(0, Math.floor(ms / 1000));
    const pad = (v) => String(v).padStart(2, "0");
    return s < 60 ? `${pad(s)}s` : `${Math.floor(s / 60)}m ${pad(s % 60)}s`;
  }

  function paint() {
    if (!liveBtn) return;
    const e = S.genElapsed ? S.genElapsed() : null;
    if (!e) { stopTick(); return; }
    // "+" on a re-attached run: the press happened before this page load, so the number is
    // time-since-reconnect and claiming it as the total would be a lie.
    liveBtn.textContent = `${liveBase} (${fmt(e.ms)}${e.approx ? "+" : ""})`;
  }

  function stopTick() {
    if (tick) { clearInterval(tick); tick = null; }
    liveBtn = null;
  }

  function startTick(btn, base) {
    liveBtn = btn;
    liveBase = base;
    btn.classList.add("btn-timer");
    paint();
    if (!tick) tick = setInterval(paint, 1000);
  }

  function render(st) {
    clear(mount);
    stopTick();  // the buttons the old ticker pointed at were just discarded
    const e = busy(st) && S.genElapsed ? S.genElapsed() : null;

    const genAll = el("button", "btn primary compact", "▶ Generate");
    genAll.dataset.tour = "generate-all";
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

    if (e) {
      // A run started from the Selected button counts on that button; everything else
      // (whole montage, and a single scene fired from the inspector) counts on ▶ Generate,
      // which is the transport's main readout.
      if (e.source === "selected") startTick(genSel, "Generating");
      else startTick(genAll, e.source === "scene" ? "▶ Generating scene" : "▶ Generating");
      liveBtn.title = e.approx
        ? "Reconnected to a run already in progress — counting from the reconnect, not from the press."
        : "Time since you pressed this button.";
    }

    const renderBtn = el("button", "btn render compact", "⧉ Render");
    renderBtn.dataset.tour = "render-final";
    renderBtn.title = "Stitch generated clips into a final video";
    renderBtn.disabled = !hasProject() || busy(st);
    renderBtn.onclick = () => S.renderFinal();
    mount.append(renderBtn);

    const montageBtn = el("button", "btn ghost compact", "⚡ Auto Montage");
    montageBtn.title = "Build a trailer-style cut from already-rendered clips";
    montageBtn.disabled = !hasProject() || busy(st);
    montageBtn.onclick = () => window.MontageDialog?.open();
    mount.append(montageBtn);

    // Settings that are switched on but cannot do anything belong next to Generate,
    // not buried in the pane that switched them on — you would only see it there if
    // you already went looking.
    const issues = [];
    const idIssue = window.PipelineCaps?.identityTransferIssue(st);
    if (idIssue) issues.push(idIssue);
    // On MiniMax H3 several LTX-only sampler settings are switched off by the sampler
    // itself. It says so on the console, but only once the run has already started —
    // here it costs nothing to know beforehand.
    issues.push(...(window.PipelineCaps?.h3InertSettings(st) || []));
    issues.forEach((issue) => {
      const chip = el("button", "btn ghost compact action-warn", "⚠ " + issue.short);
      chip.title = issue.detail + "\n\nClick to open Engine settings.";
      chip.onclick = () => window.SettingsWindow?.open("engine");
      mount.append(chip);
    });
  }

  if (window.ViewBus) window.ViewBus.subscribeActionbar(render);
  else S.subscribe(render);
  render(S.get());
  render(S.get());
})();