// PLAN strip — a compact board of scene RECIPE cards above the result timeline.
// Each card is one scene's plan (anchor + prompt + status); the timeline below shows the
// immutable rendered RESULT. Click a card to select it; the chevron opens a full-width
// inline editor (prompt, source/anchor, generate, remove) — "Advanced" hands off to the
// inspector for guides/effects/transitions. The plan is "what will generate"; editing it
// never touches the rendered clip below (see immutable renders).
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const PC = () => window.PipelineCaps;

  const zone = document.getElementById("plan-zone");
  if (!zone) return;
  const body = zone.querySelector(".plan-body");
  const collapseBtn = document.getElementById("plan-collapse");

  let expandedId = null;             // scene whose editor panel is open (one at a time)
  let collapsed = localStorage.getItem("funpack_plan_collapsed") === "1";
  let _editing = false;              // a plan field is focused — don't rebuild mid-edit

  const SRC = [
    ["empty", "Empty · text-to-video"],
    ["image", "Image · i2v anchor"],
    ["generated_frame", "From generated frame"],
    ["v2v", "Video · v2v source"],
    ["carry", "Carry · continue previous"],
    ["mixed", "Mixed · anchor + prior guides"],
  ];

  function availableSources(st) {
    if (PC() && PC().usesChainSampler(st)) return SRC;
    return SRC.filter(([v]) => v === "empty" || v === "image");
  }

  function rootOf(sc) { return (S.genUnitRoot && S.genUnitRoot(S.genUnitId(sc))) || sc; }

  function statusOf(st, sc) {
    const r = (st.sceneRenders || {})[sc.id];
    if (!r || !r.media) return { cls: "none", label: "not generated" };
    if (S.renderIsStale && S.renderIsStale(sc.id)) return { cls: "edited", label: "edited" };
    return { cls: "sync", label: "in sync" };
  }

  function anchorThumb(st, sc) {
    const root = rootOf(sc);
    const ref = root.source && root.source.media_ref;
    const asset = ref ? (st.mediaBin || []).find((m) => m.id === ref) : null;
    const wrap = el("div", "plan-thumb");
    if (asset && window.MovieEditorAPI) {
      const img = el("img"); img.src = window.MovieEditorAPI.mediaUrl(asset.id); img.loading = "lazy"; img.alt = "";
      wrap.append(img);
    } else {
      const t = (root.source && root.source.type) || "empty";
      wrap.append(el("span", "plan-thumb-glyph", t === "empty" ? "T" : (t === "carry" ? "⇥" : "▢")));
    }
    return wrap;
  }

  function field(labelText, control) {
    const l = el("label", "plan-field"); l.append(el("span", null, labelText)); l.append(control); return l;
  }

  // i2v anchors are dropped on the PLAN (never the timeline). Accept image media only.
  function attachAnchorDrop(node, st, onImage) {
    node.addEventListener("dragover", (e) => {
      if (!e.dataTransfer.types.includes("application/funpack-media")) return;
      e.preventDefault(); e.dataTransfer.dropEffect = "copy"; node.classList.add("drop-target");
    });
    node.addEventListener("dragleave", () => node.classList.remove("drop-target"));
    node.addEventListener("drop", (e) => {
      node.classList.remove("drop-target");
      const id = e.dataTransfer.getData("application/funpack-media");
      if (!id) return;
      const asset = (st.mediaBin || []).find((m) => m.id === id);
      if (!asset || asset.kind !== "image") return; // images = anchors; videos go to the timeline
      e.preventDefault(); e.stopPropagation();
      onImage(id);
    });
  }

  function cardEditor(st, sc) {
    const root = rootOf(sc);
    const ed = el("div", "plan-editor");
    const head = el("div", "plan-editor-head");
    head.append(el("span", "plan-editor-title", "Scene " + (st.project.scenes.indexOf(sc) + 1)));
    const close = el("button", "btn ghost tiny", "▾ Close"); close.onclick = () => { expandedId = null; render(st); };
    head.append(close);
    ed.append(head);

    const grid = el("div", "plan-editor-grid");

    const ta = el("textarea", "plan-edit-prompt"); ta.rows = 3; ta.value = root.text || "";
    ta.placeholder = "Describe this scene…"; ta.dataset.k = "plan-prompt";
    ta.oninput = () => S.patchSceneQuiet(sc.id, { text: ta.value });
    grid.append(field("Prompt", ta));

    const chain = PC() && PC().usesChainSampler(st);
    const stored = (root.source && root.source.type) || (chain ? "carry" : "empty");
    const opts = availableSources(st);
    const sel = el("select"); sel.dataset.k = "plan-src";
    opts.forEach(([v, label]) => { const o = new Option(label, v); if (v === stored) o.selected = true; sel.append(o); });
    if (!opts.some(([v]) => v === stored)) {
      const o = new Option((PC() ? PC().sourceLabel(stored) : stored) + " (needs Chain Sampler)", stored);
      o.selected = true; sel.append(o);
    }
    sel.onchange = () => {
      const next = sel.value;
      const patch = { source: { ...(root.source || {}), type: next } };
      if (next === "image" || next === "empty") patch.guides = [];
      S.patchScene(sc.id, patch);
    };
    grid.append(field("Source", sel));

    const eff = root.source && root.source.type;
    if ((eff === "image" || eff === "mixed") && window.MediaPicker) {
      const pick = window.MediaPicker.create({
        value: root.source && root.source.media_ref,
        mediaBin: st.mediaBin,
        noneLabel: "— choose anchor image —",
        onChange: (mediaRef) => {
          const patch = { source: { ...(root.source || {}), type: eff, media_ref: mediaRef } };
          if ((root.source && root.source.media_ref || null) !== mediaRef) patch.guides = [];
          S.patchScene(sc.id, patch);
        },
      });
      grid.append(field("Anchor image", pick));
    }
    ed.append(grid);

    const effFrames = S.sceneEffFrames ? S.sceneEffFrames(sc) : null;
    const effFps = (S.sceneEffFps ? S.sceneEffFps(sc) : null) || 1;
    const note = el("div", "plan-editor-note");
    note.textContent = effFrames
      ? `Plan length ≈ ${(effFrames / effFps).toFixed(2)}s · length, guides, effects & transitions in Advanced`
      : "Length, guides, effects & transitions in Advanced";
    ed.append(note);

    const actions = el("div", "plan-editor-actions");
    const gen = el("button", "btn primary tiny", "Generate this scene");
    gen.onclick = () => S.generate(sc.id);
    const mvL = el("button", "btn ghost tiny", "← Plan");
    mvL.title = "Move earlier in the plan (generation order). Does NOT move the timeline clip.";
    mvL.onclick = () => S.moveScene(sc.id, -1);
    const mvR = el("button", "btn ghost tiny", "Plan →");
    mvR.title = "Move later in the plan (generation order). Does NOT move the timeline clip.";
    mvR.onclick = () => S.moveScene(sc.id, 1);
    const adv = el("button", "btn ghost tiny", "Advanced ↗");
    adv.title = "Open full scene settings (guides, effects, transitions) in the inspector";
    adv.onclick = () => { S.selectScene(sc.id); if (window.DockLayout) window.DockLayout.set({ settings: true }); };
    const rm = el("button", "btn ghost tiny danger", "Remove from plan");
    rm.title = "Remove from the plan. A generated clip stays on the timeline; an ungenerated scene is deleted.";
    rm.onclick = () => { expandedId = null; S.removeFromPlan(sc.id); };
    actions.append(gen, mvL, mvR, adv, rm);
    ed.append(actions);
    return ed;
  }

  function cardEl(st, sc, idx) {
    const isSel = st.selectedSceneId === sc.id;
    const isOpen = expandedId === sc.id;
    const c = el("div", "plan-card" + (isSel ? " sel" : "") + (isOpen ? " open" : ""));
    c.dataset.sceneId = sc.id;
    c.append(anchorThumb(st, sc));
    const meta = el("div", "plan-card-meta");
    const row = el("div", "plan-card-row");
    row.append(el("span", "plan-card-no", "S" + (idx + 1)));
    const st2 = statusOf(st, sc);
    row.append(el("span", "plan-chip " + st2.cls, st2.label));
    meta.append(row);
    const txt = (rootOf(sc).text || "").trim() || (S.isVideoClip(sc) ? "Video clip" : "(empty)");
    meta.append(el("div", "plan-card-prompt", txt));
    c.append(meta);
    if (!S.isVideoClip(sc)) {
      const chev = el("button", "plan-chev", isOpen ? "▾" : "▸");
      chev.title = isOpen ? "Close editor" : "Edit scene";
      chev.onclick = (e) => { e.stopPropagation(); expandedId = isOpen ? null : sc.id; S.selectScene(sc.id); render(st); };
      c.append(chev);
    }
    if (!S.isVideoClip(sc)) attachAnchorDrop(c, st, (id) => S.assignMediaToScene(sc.id, id));
    c.onclick = () => { if (st.selectedSceneId !== sc.id) S.selectScene(sc.id); };
    return c;
  }

  function render(st) {
    st = st || S.get();
    zone.classList.toggle("collapsed", collapsed);
    if (collapseBtn) collapseBtn.textContent = collapsed ? "Show" : "Hide";
    clear(body);
    if (collapsed || !st.project) return;
    const scenes = (st.project.scenes || []).filter((s) => !s.removed_from_plan);
    const rowEl = el("div", "plan-row");
    scenes.forEach((sc, i) => rowEl.append(cardEl(st, sc, i)));
    const add = el("button", "plan-add", "+ Add scene");
    add.title = "Add a new scene to the plan · or drop an image here to anchor a new scene";
    add.onclick = () => S.addScene();
    rowEl.append(add);
    // Drop an image on empty plan space → a new scene anchored to it (cards stopPropagation).
    attachAnchorDrop(rowEl, st, (id) => S.addImageClip(id));
    body.append(rowEl);
    if (expandedId) {
      const sc = S.scene(expandedId);
      if (sc && !S.isVideoClip(sc)) body.append(cardEditor(st, sc));
      else expandedId = null;
    }
  }

  // ── re-render only on relevant change, and never while a plan field is being edited ──
  function fp(st) {
    if (!st.project) return "none";
    const p = st.project;
    return JSON.stringify({
      pid: p.id, sel: st.selectedSceneId, exp: expandedId, col: collapsed,
      scenes: (p.scenes || []).map((s) => {
        const r = rootOf(s);
        return [s.id, r.text, r.source && r.source.type, r.source && r.source.media_ref, s.excluded, S.isVideoClip(s)];
      }),
      renders: (p.scenes || []).map((s) => {
        const rr = (st.sceneRenders || {})[s.id];
        return [!!(rr && rr.media), S.renderIsStale && S.renderIsStale(s.id) ? 1 : 0];
      }),
      media: (st.mediaBin || []).length,
    });
  }

  let _lastFp = null;
  function isProtectedField(a) {
    if (!a || !a.dataset || !a.dataset.k || !body.contains(a)) return false;
    if (a.tagName === "TEXTAREA") return true;
    if (a.tagName !== "INPUT") return false; // SELECT rebuilds (reveals dependents)
    const t = (a.type || "text").toLowerCase();
    return t !== "checkbox" && t !== "radio";
  }

  body.addEventListener("focusin", (e) => { if (e.target && e.target.dataset && e.target.dataset.k) _editing = true; });
  body.addEventListener("focusout", (e) => {
    if (!(e.target && e.target.dataset && e.target.dataset.k)) return;
    _editing = false;
    setTimeout(() => { if (!_editing) { _lastFp = fp(S.get()); render(S.get()); } }, 60);
  });

  if (collapseBtn) {
    collapseBtn.onclick = () => {
      collapsed = !collapsed;
      localStorage.setItem("funpack_plan_collapsed", collapsed ? "1" : "0");
      render(S.get());
    };
  }

  S.subscribe((st) => {
    const f = fp(st);
    if (f === _lastFp) return;
    if (_editing && isProtectedField(document.activeElement)) return; // resyncs on blur
    _lastFp = f;
    render(st);
  });
})();
