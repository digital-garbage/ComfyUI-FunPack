// Auto Montage dialog: pick a "lead" clip (plays in order, shrinking segments) and a pool
// of other rendered clips to randomly cut away to between segments — trailer-style pacing.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const O = window.OverlayUI;

  // Any scene with a playable render — an explicit video clip OR a generative scene that's
  // already been generated (has a sceneRenders entry). Matches the same "has content" check
  // used elsewhere in the editor (e.g. removeFromPlan), not just clips converted to video.
  function renderedScenes() {
    const st = S.get();
    return (st.project?.scenes || []).filter((s) => !s.excluded && S.hasPlayableRender(s));
  }

  function clipLabel(s) {
    const dur = S.sceneDurationSec ? S.sceneDurationSec(s) : null;
    const frames = dur != null ? Math.round(dur * S.sceneEffFps(s)) : (s.frames || "?");
    const text = (s.text || "").trim().replace(/\s+/g, " ");
    const snippet = text ? (text.length > 36 ? text.slice(0, 36) + "…" : text) : "(no prompt)";
    return `${snippet} — ${frames}f`;
  }

  function openAutoMontageDialog() {
    const clips = renderedScenes();
    if (clips.length < 2) {
      alert("Auto Montage needs at least two already-rendered scenes on the timeline (generated or converted-to-video clips): one lead scene and at least one cutaway scene.");
      return;
    }
    const { body, foot, close } = O.openModal({
      title: "Auto Montage",
      subtitle: "Cuts the lead scene into shrinking segments and randomly inserts cutaways between them.",
      widthClass: "ov-modal-wide",
    });

    // Pre-fill from the current timeline selection when it gives us at least a lead + one
    // cutaway candidate — matches "select scenes, then click Auto Montage" expectations.
    const selectedIds = new Set(S.get().selectedSceneIds || []);
    const selectedEligible = clips.filter((c) => selectedIds.has(c.id));
    let leadId = (selectedEligible[0] || clips[0]).id;
    const poolIds = new Set(selectedEligible.slice(1).map((c) => c.id));

    const leadSel = document.createElement("select");
    leadSel.className = "ov-input";
    clips.forEach((c) => {
      const o = document.createElement("option");
      o.value = c.id;
      o.textContent = clipLabel(c);
      leadSel.appendChild(o);
    });
    leadSel.onchange = () => { leadId = leadSel.value; renderPoolList(); };
    body.appendChild(O.field("Lead clip (plays in order)", leadSel, { full: true }));

    const poolWrap = document.createElement("div");
    poolWrap.className = "ov-field full";
    body.appendChild(poolWrap);

    function renderPoolList() {
      clear(poolWrap);
      const lab = el("span", "ov-label", "Cutaway pool (random draw, in order within each clip)");
      poolWrap.appendChild(lab);
      const list = el("div", "ov-checklist");
      clips.filter((c) => c.id !== leadId).forEach((c) => {
        const row = document.createElement("label");
        row.className = "ov-check";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = poolIds.has(c.id);
        cb.onchange = () => { if (cb.checked) poolIds.add(c.id); else poolIds.delete(c.id); };
        const span = el("span", "", clipLabel(c));
        row.append(cb, span);
        list.appendChild(row);
      });
      poolWrap.appendChild(list);
    }
    renderPoolList();

    const segInput = document.createElement("input");
    segInput.type = "number"; segInput.className = "ov-input"; segInput.min = "9"; segInput.step = "1"; segInput.value = "100";
    body.appendChild(O.field("Segment length (frames)", segInput));

    const decayInput = document.createElement("input");
    decayInput.type = "range"; decayInput.min = "0.5"; decayInput.max = "1"; decayInput.step = "0.05"; decayInput.value = "0.85";
    const decayVal = el("span", "ov-range-val", "0.85");
    decayInput.oninput = () => { decayVal.textContent = decayInput.value; };
    body.appendChild(O.rangeField("Acceleration (1.0 = constant, lower = faster cuts toward the end)", decayInput, decayVal));

    const buildBtn = el("button", "btn primary", "Build montage");
    buildBtn.onclick = () => {
      if (!poolIds.size) { alert("Pick at least one cutaway scene."); return; }
      const n = S.autoMontage({
        leadId,
        poolIds: Array.from(poolIds),
        segmentFrames: parseInt(segInput.value, 10) || 100,
        decay: parseFloat(decayInput.value) || 1,
      });
      if (!n) alert("Couldn't build the montage — check that the clips have enough rendered length.");
      close();
    };
    foot.appendChild(buildBtn);
  }

  window.MontageDialog = { open: openAutoMontageDialog };
})();
