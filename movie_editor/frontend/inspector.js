// Right-bottom zone: context inspector (selected scene OR project) + split preview.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const body = document.getElementById("inspector-body");
  const title = document.getElementById("inspector-title");

  const SRC = [
    ["empty", "Empty · text-to-video"],
    ["image", "Image · i2v anchor"],
    ["generated_frame", "From generated frame"],
    ["carry", "Carry i2v guide · continue previous"],
    ["mixed", "Mixed · Img2Video anchor + prior guides"],
  ];

  function splitMarkerSelect(value, onChange, opts) {
    opts = opts || {};
    const sel = el("select");
    const noneLabel = opts.noneLabel || "— default cut —";
    const none = el("option", null, noneLabel); none.value = ""; sel.append(none);
    (S.get().transitions || []).forEach((t) => {
      const name = t.trigger || t.name || t.key; if (!name) return;
      const o = el("option", null, name); o.value = name; if (name === value) o.selected = true; sel.append(o);
    });
    if (value && ![...sel.options].some((o) => o.value === value)) {
      const o = el("option", null, value); o.value = value; o.selected = true; sel.append(o);
    }
    sel.onchange = () => onChange(sel.value);
    sel.title = opts.title || "Prompt marker: how Studio splits the montage before the next scene when generating";
    return sel;
  }

  // Legacy alias used in a few call sites
  function transitionSelect(value, onChange) {
    return splitMarkerSelect(value, onChange);
  }

  function field(labelText, control) {
    const l = el("label", "field"); l.append(el("span", null, labelText)); l.append(control); return l;
  }

  function numberField(labelText, value, on, key) {
    const i = el("input"); i.type = "number"; i.value = value; if (key) i.dataset.k = key;
    i.oninput = () => on(parseInt(i.value || "0", 10));
    return field(labelText, i);
  }

  function renderImageSource(st, scene, parent) {
    parent = parent || body;
    const ref = scene.source?.media_ref;
    const isMixed = (scene.source?.type) === "mixed";
    const pick = window.MediaPicker.create({
      value: ref,
      mediaBin: st.mediaBin,
      noneLabel: "— choose anchor image —",
      onChange: (mediaRef) => {
        const patch = { source: { ...(scene.source || {}), type: isMixed ? "mixed" : "image", media_ref: mediaRef } };
        if ((scene.source?.media_ref || null) !== mediaRef) patch.guides = [];
        S.patchScene(scene.id, patch);
      },
    });
    parent.append(field(isMixed ? "Anchor image (mixed i2v)" : "Anchor image", pick));
    parent.append(el("div", "insp-hint", isMixed
      ? "Starting frame for this scene; prior-scene guides stay active (◐+⇥ on the timeline)."
      : "Image-to-video anchor for this scene. Drag from the Media bin onto the clip, or Browse here."));
  }

  function renderGeneratedFrameSource(st, scene) {
    const ref = scene.source?.media_ref;
    const asset = (st.mediaBin || []).find((m) => m.id === ref);

    if (asset) {
      // A frame has been captured — show it with a replace hint.
      const block = el("div", "insp-block img-src");
      const prev = el("div", "img-src-prev");
      const img = el("img"); img.src = window.MovieEditorAPI.mediaUrl(asset.id); img.loading = "lazy"; prev.append(img);
      const info = el("div", "img-src-info");
      info.append(el("div", "img-src-name", "Captured frame · " + (asset.name || asset.id)));
      info.append(el("div", "insp-hint", "To replace: scrub the player to the new frame, then press 📌 Use as anchor and pick this scene."));
      block.append(prev, info);
      body.append(block);
    } else {
      // No frame captured yet — show instructions.
      const hint = el("div", "insp-block");
      hint.append(el("div", "insp-hint",
        "Scrub the player to the desired frame (playhead must be on a rendered segment), then press 📌 Use as anchor in the player transport bar and select this scene."));
      body.append(hint);
    }
  }

  // Per-scene length / fps with a 3-way mode (project | timeline | custom).
  const LEN_META = {
    frames: { label: "Frames", projKey: "num_frames_per_scene", snap: (v) => S.snapFrames(v) },
    fps:    { label: "FPS",    projKey: "frame_rate",          snap: (v) => Math.max(1, v) },
  };
  const LEN_DEFAULT_MODE = { frames: "timeline", fps: "project" };

  function modeOf(scene, kind) { return scene[kind + "_mode"] || LEN_DEFAULT_MODE[kind]; }
  function effOf(scene, kind) {
    const m = LEN_META[kind], p = S.get().project;
    return modeOf(scene, kind) !== "project" && scene[kind] != null ? scene[kind] : p[m.projKey];
  }

  function lengthControl(scene, kind) {
    const m = LEN_META[kind], p = S.get().project;
    const mode = modeOf(scene, kind);
    const wrap = el("label", "field");
    wrap.append(el("span", null, m.label));

    const sel = el("select"); sel.dataset.k = "sc-" + kind + "mode";
    [["project", "Inherit project global"], ["timeline", "Inherit timeline (trim)"], ["custom", "Custom"]]
      .forEach(([v, l]) => { const o = el("option", null, l); o.value = v; if (v === mode) o.selected = true; sel.append(o); });
    sel.onchange = () => {
      const patch = { [kind + "_mode"]: sel.value };
      // Seed the per-scene value from the current effective value when leaving "project".
      if (sel.value !== "project" && scene[kind] == null) patch[kind] = effOf(scene, kind);
      S.patchScene(scene.id, patch);
    };
    wrap.append(sel);

    if (mode === "custom") {
      const i = el("input"); i.type = "number"; i.value = effOf(scene, kind); i.dataset.k = "sc-" + kind;
      i.oninput = () => S.patchSceneQuiet(scene.id, { [kind]: m.snap(parseInt(i.value || "0", 10)) });
      wrap.append(i);
    } else {
      const note = el("div", "len-readout");
      note.textContent = mode === "project"
        ? `${p[m.projKey]} · from project`
        : `${effOf(scene, kind)} · from timeline`;
      wrap.append(note);
    }
    return wrap;
  }

  function foldSection(title, openDefault, buildFn) {
    const det = el("details", "insp-fold");
    det.open = openDefault;
    det.append(el("summary", "insp-fold-sum", title));
    const inner = el("div", "insp-fold-body");
    buildFn(inner);
    det.append(inner);
    body.append(det);
    return inner;
  }

  function renderScene(st, scene) {
    const root = S.genUnitRoot(S.genUnitId(scene)) || scene;
    const unitScenes = (st.project.scenes || []).filter((s) => S.genUnitId(s) === S.genUnitId(scene))
      .sort((a, b) => (a.cut_offset_frames || 0) - (b.cut_offset_frames || 0));
    const cutNo = unitScenes.indexOf(scene) + 1;
    title.textContent = `Scene · ${st.project.scenes.indexOf(scene) + 1}`
      + (unitScenes.length > 1 ? ` · cut ${cutNo}/${unitScenes.length}` : "");
    const selN = S.selectedSceneCount ? S.selectedSceneCount() : 1;
    if (selN > 1) {
      body.append(el("div", "insp-hint",
        `${selN} clips selected — use Selected in the timeline header to generate each chain segment.`));
    }
    if (unitScenes.length > 1) {
      body.append(el("div", "insp-hint", S.isGenSubclip(scene)
        ? "Editorial cut — prompt and source are shared with the root clip."
        : "This scene has editorial cuts — Generate regens the whole uncut scene."));
    }

    const ta = el("textarea"); ta.rows = 4; ta.value = root.text || ""; ta.placeholder = "Describe this scene…"; ta.dataset.k = "sc-text";
    ta.oninput = () => S.patchSceneQuiet(scene.id, { text: ta.value });
    body.append(field("Prompt", ta));
    const promptMismatch = S.renderPromptMismatch ? S.renderPromptMismatch(scene.id) : null;
    if (promptMismatch) {
      const warn = el("div", "render-prompt-warn");
      warn.append(el("div", "render-prompt-warn-title", "Preview was generated with a different prompt"));
      warn.append(el("div", "render-prompt-text", promptMismatch.rendered || "(empty)"));
      body.append(warn);
    }
    const anchorMismatch = S.renderAnchorMismatch ? S.renderAnchorMismatch(scene.id) : null;
    if (anchorMismatch) {
      const warn = el("div", "render-prompt-warn render-anchor-warn");
      warn.append(el("div", "render-prompt-warn-title", "i2v image changed · showing previous generation"));
      warn.append(el("div", "render-prompt-text", anchorMismatch.renderedLabel || "(none)"));
      body.append(warn);
    }

    const src = el("select");
    SRC.forEach(([v, label]) => { const o = el("option", null, label); o.value = v; if ((root.source?.type) === v) o.selected = true; src.append(o); });
    src.onchange = () => {
      const prev = root.source?.type || "carry";
      const next = src.value;
      const patch = { source: { ...(root.source || {}), type: next } };
      if (next === "image" || next === "empty" || next === "generated_frame") patch.guides = [];
      else if (prev !== next && (prev === "carry" || next === "carry" || prev === "mixed" || next === "mixed")) patch.guides = [];
      S.patchScene(scene.id, patch);
    };
    body.append(field("Source", src));
    if ((root.source?.type) === "image" || (root.source?.type) === "mixed") renderImageSource(st, root, body);
    if ((root.source?.type) === "generated_frame") renderGeneratedFrameSource(st, root);

    const effFrames = effOf(scene, "frames"), effFps = effOf(scene, "fps") || 1;
    body.append(el("div", "insp-hint", `Duration ≈ ${(effFrames / effFps).toFixed(2)}s · trim on timeline · scene splits for generation: edit the global prompt or markers on timeline seams`));

    const actions = el("div", "insp-block");
    const genBtn = el("button", "btn primary", "Generate this scene");
    genBtn.onclick = () => S.generate(scene.id);
    actions.append(genBtn);
    const del = el("button", "btn danger", "Delete"); del.style.marginLeft = "8px"; del.onclick = () => S.removeScene(scene.id);
    actions.append(del);
    body.append(actions);

    foldSection("More editing", false, (more) => {
      renderSceneCharacters(st, scene, more);

      const fxTag = el("div", "insp-tag"); fxTag.textContent = "Video effects"; more.append(fxTag);
      const fx = scene.effects || {};
      const patchFx = (k, v, quiet) => {
        const next = { ...(scene.effects || {}), [k]: v };
        quiet ? S.patchSceneQuiet(scene.id, { effects: next }) : S.patchScene(scene.id, { effects: next });
      };
      const _num = (val, k, opts) => {
        const i = el("input"); i.type = "number"; i.value = val; i.dataset.k = k;
        if (opts) Object.assign(i, opts);
        return i;
      };
      const blur = el("input"); blur.type = "range"; blur.min = "0"; blur.max = "1"; blur.step = "0.05";
      blur.value = fx.blur || 0; blur.dataset.k = "sc-fx-blur";
      blur.oninput = () => patchFx("blur", parseFloat(blur.value), true);
      more.append(field(`Blur (${Math.round((fx.blur || 0) * 100)}%)`, blur));
      const fadeRow = el("div", "fields-row");
      const fi = _num(fx.fade_in || 0, "sc-fx-fi", { min: 0, max: 10, step: 0.1 });
      fi.oninput = () => patchFx("fade_in", parseFloat(fi.value || "0"), true);
      const fo = _num(fx.fade_out || 0, "sc-fx-fo", { min: 0, max: 10, step: 0.1 });
      fo.oninput = () => patchFx("fade_out", parseFloat(fo.value || "0"), true);
      fadeRow.append(field("Fade in (s)", fi)); fadeRow.append(field("Fade out (s)", fo));
      more.append(fadeRow);
      const zoom = el("select"); zoom.dataset.k = "sc-fx-zoom";
      [["none", "None"], ["in", "Zoom in"], ["out", "Zoom out"]].forEach(([v, label]) => {
        const o = el("option", null, label); o.value = v; if ((fx.zoom || "none") === v) o.selected = true; zoom.append(o);
      });
      zoom.onchange = () => patchFx("zoom", zoom.value);
      more.append(field("Ken Burns zoom", zoom));

      const lenRow = el("div", "fields-row");
      lenRow.append(lengthControl(scene, "frames"));
      lenRow.append(lengthControl(scene, "fps"));
      more.append(lenRow);

      const r = (st.sceneRenders || {})[scene.id];
      if (r && r.media) {
        const srcTag = el("div", "insp-tag"); srcTag.textContent = "Source trim (slip)"; more.append(srcTag);
        const inRow = el("div", "fields-row");
        const iIn = _num(scene.source_in || 0, "src-in", { min: 0, step: 0.05 });
        iIn.oninput = () => S.setSourceTrim(scene.id, { source_in: parseFloat(iIn.value || 0) });
        inRow.append(field("Source in (s)", iIn));
        const iDur = _num(scene.source_dur != null ? scene.source_dur : "", "src-dur", { min: 0.1, step: 0.05 });
        iDur.placeholder = "full";
        iDur.oninput = () => S.setSourceTrim(scene.id, { source_dur: iDur.value ? parseFloat(iDur.value) : null });
        inRow.append(field("Source dur (s)", iDur));
        more.append(inRow);
        const reset = el("button", "btn ghost tiny", "Reset source trim");
        reset.onclick = () => S.setSourceTrim(scene.id, { source_in: 0, source_dur: null });
        more.append(reset);
      }

      renderSceneGuides(st, scene, more);

      const row = el("div", "insp-block");
      const chk = el("label", "chk"); const cb = el("input"); cb.type = "checkbox"; cb.checked = !!scene.excluded;
      cb.onchange = () => S.patchScene(scene.id, { excluded: cb.checked });
      chk.append(cb); chk.append(el("span", null, "Exclude from full generation"));
      row.append(chk); more.append(row);
    });
  }

  function renderSceneCharacters(st, scene, parent) {
    parent = parent || body;
    const ids = S.sceneCharacterIds(scene.id);
    const tag = el("div", "insp-tag"); tag.textContent = "Characters"; parent.append(tag);
    const chips = el("div", "char-chips");
    if (!ids.length) {
      chips.append(el("span", "char-chip empty", "None — assign in Characters bin"));
    } else {
      ids.forEach((cid) => {
        const c = (st.characters || []).find((x) => x.id === cid);
        const chip = el("span", "char-chip");
        chip.textContent = c?.name || cid;
        const rm = el("button", "char-chip-rm", "✕");
        rm.onclick = () => S.toggleSceneCharacter(scene.id, cid);
        chip.append(rm);
        chips.append(chip);
      });
    }
    parent.append(chips);
  }

  function slotLabelFor(st, slotId) {
    if (!slotId || slotId === "funpack") return null;
    const slot = (st.models?.slots || []).find((s) => s.id === slotId);
    return slot ? (slot.label || slot.node_class || slot.id) : slotId;
  }

  function renderEngineStrip(st) {
    const p = st.project;
    const studioOn = !p.conditioning_slot || p.conditioning_slot === "funpack";
    const chainOn = !p.sampler_slot || p.sampler_slot === "funpack";
    const parts = [];
    parts.push(studioOn ? "FunPack Studio" : (slotLabelFor(st, p.conditioning_slot) || p.conditioning_slot));
    parts.push(chainOn ? "Chain Sampler" : (slotLabelFor(st, p.sampler_slot) || p.sampler_slot));
    const strip = el("div", "insp-engine-strip");
    strip.append(el("span", "insp-engine-strip-txt", "Generating with " + parts.join(" + ")));
    const btn = el("button", "btn ghost tiny", "Engine settings →");
    btn.onclick = () => window.EngineSettingsModal.open();
    strip.append(btn);
    body.append(strip);
  }

  function renderProject(st) {
    const p = st.project;
    title.textContent = "Project";

    const outTag = el("div", "insp-tag"); outTag.textContent = "Output"; body.append(outTag);
    const name = el("input"); name.value = p.name || ""; name.dataset.k = "pj-name";
    name.oninput = () => S.patchProjectQuiet({ name: name.value });
    body.append(field("Project name", name));

    const row1 = el("div", "fields-row");
    row1.append(numberField("Frames / scene", p.num_frames_per_scene, (v) => S.patchProjectQuiet({ num_frames_per_scene: v }), "pj-frames"));
    body.append(row1);
    const row2 = el("div", "fields-row");
    row2.append(numberField("FPS", p.frame_rate, (v) => S.patchProjectQuiet({ frame_rate: v }), "pj-fps"));
    body.append(row2);

    const promptTag = el("div", "insp-tag"); promptTag.textContent = "Prompt"; body.append(promptTag);
    const anchor = el("textarea"); anchor.rows = 2; anchor.value = p.anchor || ""; anchor.dataset.k = "pj-anchor";
    anchor.placeholder = "World / setting context prepended to every scene";
    anchor.oninput = () => S.patchProjectQuiet({ anchor: anchor.value });
    body.append(field("Anchor", anchor));
    body.append(field("Split before scene 1 (generation prompt)", splitMarkerSelect(p.intro_transition || "",
      (v) => S.patchProject({ intro_transition: v }),
      { noneLabel: "— default cut —", title: "Prompt marker between anchor and scene 1 when Studio splits a long montage" })));
    const neg = el("textarea"); neg.rows = 2; neg.value = p.negative_prompt || ""; neg.dataset.k = "pj-neg";
    neg.placeholder = "What to avoid in every scene";
    neg.oninput = () => S.patchProjectQuiet({ negative_prompt: neg.value });
    body.append(field("Negative prompt", neg));

    foldSection("Advanced project settings", false, (adv) => {
      adv.append(numberField("Max scenes", p.max_scenes, (v) => S.patchProjectQuiet({ max_scenes: v }), "pj-max"));
      const row3 = el("div", "fields-row");
      row3.append(numberField("Width", p.width != null ? p.width : 768, (v) => S.patchProjectQuiet({ width: v }), "pj-w"));
      row3.append(numberField("Height", p.height != null ? p.height : 512, (v) => S.patchProjectQuiet({ height: v }), "pj-h"));
      adv.append(row3);
    });
  }

  const STUDIO_DEFAULT_GUIDE = { enabled: true, source: "template", frame_idx: 0, apply_at: 0, strength: 0.35 };

  function normGuideSettings(p) {
    const gs = (p && p.guide_settings) || {};
    return { stack_enabled: !!gs.stack_enabled, accumulate_prior: !!gs.accumulate_prior };
  }

  function renderSceneGuides(st, scene, parent) {
    parent = parent || body;
    if (!normGuideSettings(st.project).stack_enabled) return;
    const sceneNo = st.project.scenes.indexOf(scene) + 1;
    const tag = el("div", "insp-tag"); tag.textContent = "i2v guides"; parent.append(tag);
    if (sceneNo <= 1) {
      parent.append(el("div", "insp-hint", "Scene 1 is the i2v anchor — guides apply from scene 2 onward."));
      return;
    }
    const guides = [...(scene.guides || [])];
    const wrap = el("div", "insp-block");

    const list = el("div", "guide-list");
    const persist = (next, quiet) => {
      quiet ? S.patchSceneQuiet(scene.id, { guides: next }) : S.patchScene(scene.id, { guides: next });
    };
    const redraw = () => { /* full inspector rebuild on notify */ };

    const rowFor = (g, idx) => {
      const row = el("div", "fields-row guide-row");
      const en = el("input"); en.type = "checkbox"; en.checked = g.enabled !== false; en.title = "Enabled";
      en.onchange = () => { guides[idx] = { ...g, enabled: en.checked }; persist(guides, true); };
      row.append(en);
      const src = el("select");
      [["template", "Scene 1 template"], ["scene", "Prior scene"], ["image", "Image (bin)"]].forEach(([v, lbl]) => {
        const o = new Option(lbl, v); if ((g.source || "template") === v) o.selected = true; src.append(o);
      });
      src.onchange = () => { guides[idx] = { ...g, source: src.value }; persist(guides); };
      row.append(src);
      if ((g.source || "template") === "scene") {
        const sel = el("select");
        (st.project.scenes || []).forEach((s, i) => {
          if (i >= sceneNo - 1) return;
          const o = new Option(`Scene ${i + 1}`, s.id); if (g.scene_id === s.id) o.selected = true; sel.append(o);
        });
        sel.onchange = () => { guides[idx] = { ...g, scene_id: sel.value }; persist(guides); };
        row.append(sel);
      }
      if ((g.source || "template") === "image") {
        const pick = window.MediaPicker.create({
          value: g.media_ref,
          mediaBin: st.mediaBin,
          compact: true,
          onChange: (mediaRef) => { guides[idx] = { ...g, media_ref: mediaRef }; persist(guides); },
        });
        row.append(pick);
      }
      const fi = el("input"); fi.type = "number"; fi.value = g.frame_idx != null ? g.frame_idx : 0; fi.title = "Source frame_idx";
      fi.oninput = () => { guides[idx] = { ...g, frame_idx: parseInt(fi.value || "0", 10) }; persist(guides, true); };
      const ai = el("input"); ai.type = "number"; ai.value = g.apply_at != null ? g.apply_at : 0; ai.title = "apply_at";
      ai.oninput = () => { guides[idx] = { ...g, apply_at: parseInt(ai.value || "0", 10) }; persist(guides, true); };
      const si = el("input"); si.type = "number"; si.min = "0.25"; si.max = "0.5"; si.step = "0.05";
      si.value = g.strength != null ? g.strength : 0.35; si.title = "Strength";
      si.oninput = () => { guides[idx] = { ...g, strength: parseFloat(si.value || "0.35") }; persist(guides, true); };
      row.append(field("frame", fi)); row.append(field("apply", ai)); row.append(field("str", si));
      const rm = el("button", "btn ghost tiny danger", "✕");
      rm.onclick = () => { guides.splice(idx, 1); persist(guides); };
      row.append(rm);
      return row;
    };

    guides.forEach((g, i) => list.append(rowFor(g, i)));
    wrap.append(list);

    const btns = el("div", "fields-row");
    const addDef = el("button", "btn ghost tiny", "+ Studio default");
    addDef.title = "Template from scene 1 at frame 0";
    addDef.onclick = () => { guides.push({ ...STUDIO_DEFAULT_GUIDE }); persist(guides); };
    const addScene = el("button", "btn ghost tiny", "+ Prior scene");
    addScene.onclick = () => {
      const prior = st.project.scenes[sceneNo - 2];
      guides.push({ enabled: true, source: "scene", scene_id: prior?.id, frame_idx: 0, apply_at: 0, strength: 0.35 });
      persist(guides);
    };
    const addSel = el("button", "btn ghost tiny", "+ From selected");
    addSel.disabled = !st.selectedSceneId;
    addSel.onclick = () => {
      const sid = st.selectedSceneId;
      const idx = st.project.scenes.findIndex((s) => s.id === sid);
      if (idx < 0 || idx >= sceneNo - 1) return;
      guides.push({ enabled: true, source: "scene", scene_id: sid, frame_idx: 0, apply_at: 0, strength: 0.35 });
      persist(guides);
    };
    btns.append(addDef, addScene, addSel);
    wrap.append(btns);
    if (!guides.length) {
      wrap.append(el("div", "insp-hint", "No custom entries — generation uses the Studio default for this scene."));
    }
    parent.append(wrap);
  }

  function renderSplit(st) {
    const pv = st.preview; if (!pv) return;
    const wrap = el("div", "insp-block");
    const tag = el("div", "insp-tag"); tag.textContent = "Split preview (generation prompt)"; wrap.append(tag);
    const box = el("div", "split-pv");
    const val = pv.validation || {};
    if (pv.warning) { const w = el("div", "pv-warn"); w.append(el("span", null, "▲")); w.append(el("span", null, pv.warning)); box.append(w); }
    if (val.prompt_changed_since_last_queue) {
      const w = el("div", "pv-warn");
      w.append(el("span", null, "▲"));
      const why = val.anchors_changed_since_last_queue && !val.text_changed_since_last_queue
        ? "Anchor or source mode changed since last generate"
        : "Prompt or anchors changed since last generate";
      w.append(el("span", null, `${why} — next run rebuilds guides and clears stale action/detail repairs; Studio training history is kept.`));
      box.append(w);
    }
    if (pv.parse_error) { const w = el("div", "pv-warn"); w.append(el("span", null, "▲")); w.append(el("span", null, "ComfyUI offline — preview paused")); box.append(w); }
    const parsed = pv.parsed || {};
    if (parsed.anchor) { const l = el("div", "pv-line"); l.append(el("span", "pv-badge anchor", "anchor")); l.append(el("span", null, parsed.anchor)); box.append(l); }
    (parsed.scenes || []).forEach((s, i) => {
      const l = el("div", "pv-line");
      l.append(el("span", "pv-badge", "S" + (i + 1)));
      l.append(el("span", null, s.text || "(empty)"));
      // Show detected transition after this scene
      const t = (parsed.transitions || []).find((tr) => tr.after_scene === i);
      if (t) l.append(el("span", "pv-badge trans", "→ split"));
      box.append(l);
    });
    const raw = el("details", "pv-raw"); raw.append(el("summary", null, "generation prompt (sent to Studio)")); raw.append(el("pre", null, pv.combined_prompt || "")); box.append(raw);
    if (pv.display_prompt && pv.display_prompt !== pv.combined_prompt) {
      const disp = el("details", "pv-raw"); disp.append(el("summary", null, "display prompt (timeline view)")); disp.append(el("pre", null, pv.display_prompt)); box.append(disp);
    }
    wrap.append(box);

    // Sync button — lets the user push what was parsed back into the scene data
    if ((parsed.scenes || []).length > 0) {
      const syncBtn = el("button", "btn ghost tiny sync-preview-btn", "↺ Sync scenes from preview");
      syncBtn.title = "Distribute the parsed anchor / scene texts / split markers back into the timeline";
      syncBtn.onclick = () => {
        if (confirm("This will overwrite scene texts and split markers with what the parser detected. Continue?"))
          S.syncFromPreview();
      };
      wrap.append(syncBtn);
    }
    body.append(wrap);
  }

  function exposedControl(desc, value, on) {
    let ctrl;
    if (desc.kind === "combo") {
      ctrl = el("select");
      (desc.choices || []).forEach((c) => { const o = el("option", null, String(c)); o.value = c; if (c === value) o.selected = true; ctrl.append(o); });
      if (!(desc.choices || []).length) { ctrl.append(el("option", null, "(none)")); ctrl.disabled = true; }
      ctrl.onchange = () => on(ctrl.value);
    } else if (desc.kind === "boolean") {
      ctrl = el("input"); ctrl.type = "checkbox"; ctrl.checked = !!value; ctrl.style.width = "auto";
      ctrl.onchange = () => on(ctrl.checked);
    } else if (desc.kind === "int" || desc.kind === "float") {
      ctrl = el("input"); ctrl.type = "number"; if (desc.kind === "float") ctrl.step = "any";
      ctrl.value = value != null ? value : "";
      ctrl.oninput = () => on(desc.kind === "int" ? parseInt(ctrl.value || "0", 10) : parseFloat(ctrl.value || "0"));
    } else {
      ctrl = el("input"); ctrl.type = "text"; ctrl.value = value != null ? value : "";
      ctrl.oninput = () => on(ctrl.value);
    }
    return ctrl;
  }

  function slotLabel(slot) {
    return slot.role && slot.role !== "custom" ? slot.role : (slot.node_class || "node");
  }

  function renderExposed(st) {
    const m = st.models || {};
    const slotItems = [];
    (m.slots || []).forEach((slot) => (slot.exposed || []).forEach((d) => slotItems.push([slot, d])));
    const linkItems = (m.links || []).filter((l) => l.exposed);
    if (!slotItems.length && !linkItems.length) return;
    const wrap = el("div", "insp-block");
    const tag = el("div", "insp-tag"); tag.textContent = "Exposed controls"; wrap.append(tag);
    const SRC_LBL = { frame_rate: "Project FPS", num_frames_per_scene: "Project Frames", width: "Project Width", height: "Project Height" };
    linkItems.forEach((l) => {
      if (l.source === "editor") {
        const note = el("div", "lib-sub"); note.textContent = `← ${SRC_LBL[l.editor_key] || l.editor_key}`;
        wrap.append(field(`🔗 ${l.name} (${(l.members || []).length})`, note));
      } else {
        const ctrl = exposedControl({ kind: l.kind, choices: l.choices }, l.value, (v) => S.setModelLink(l.id, v));
        wrap.append(field(`🔗 ${l.name} (${(l.members || []).length})`, ctrl));
      }
    });
    slotItems.forEach(([slot, d]) => {
      const ctrl = exposedControl(d, (slot.inputs || {})[d.name], (v) => S.setModelInput(slot.id, d.name, v));
      wrap.append(field(`${slotLabel(slot)} · ${d.label || d.name}`, ctrl));
    });
    body.append(wrap);
  }

  // Pinned global-prompt editor (always on top). Backed by the live combined prompt
  // so editing any scene flows into it; `gpDraft` holds unapplied direct edits.
  let gpDraft = null;
  let gpProjectId = null;
  function renderGlobalPrompt(st) {
    if (st.project.id !== gpProjectId) { gpProjectId = st.project.id; gpDraft = null; }  // drop stale edits on project switch
    // display_prompt = verbatim timeline text; combined_prompt may include injected
    // "scene N" split markers used only when sending to Studio (for_generation).
    const pv = st.preview;
    const live = (pv && (pv.display_prompt != null ? pv.display_prompt : pv.combined_prompt)) || st.project.global_prompt || "";
    const dirty = gpDraft != null && gpDraft !== live;
    const val = gpDraft != null ? gpDraft : live;

    const sec = el("div", "insp-global");
    const head = el("div", "insp-global-head");
    head.append(el("span", "insp-global-title", "Global prompt"));
    const apply = el("button", "btn primary tiny", "Apply →");
    apply.title = "Split this prompt into anchor, scenes, and split markers on the timeline";
    apply.disabled = !val.trim();  // clickable whenever there's a prompt to (re)split — not only after edits
    apply.onclick = async () => {
      _editing = false;
      ta.blur();
      apply.disabled = true;
      apply.textContent = "Applying…";
      try {
        const ok = await S.applyGlobalPrompt(gpDraft != null ? gpDraft : live);
        if (ok) gpDraft = null;
      } finally {
        apply.disabled = false;
        apply.textContent = "Apply →";
        render(S.get());
      }
    };
    head.append(apply);
    sec.append(head);

    const ta = el("textarea", "insp-global-ta"); ta.rows = 3; ta.value = val; ta.dataset.k = "global-prompt";
    ta.placeholder = "Anchor, scene texts, and split markers — one combined montage prompt for generation.";
    ta.oninput = () => { gpDraft = ta.value; apply.disabled = !ta.value.trim(); };
    sec.append(ta);
    sec.append(el("div", "insp-hint", dirty
      ? "Edited — press Apply to (re)split onto the timeline."
      : "Primary way to control how long videos divide into scenes. Per-seam tweaks: Split dropdown on timeline seams (video blends are separate, on the same seam)."));
    body.append(sec);
  }

  function renderSwitch(st, scene) {
    const sw = el("div", "insp-switch");
    const mk = (label, active, on, disabled) => {
      const b = el("button", "insp-seg" + (active ? " active" : ""), label);
      if (disabled) b.disabled = true; else b.onclick = on;
      return b;
    };
    sw.append(mk("Project", !scene, () => S.selectScene(null)));
    sw.append(mk("Scene", !!scene, () => {
      const id = st.selectedSceneId || (st.project.scenes[0] && st.project.scenes[0].id);
      if (id) S.selectScene(id);
    }, !st.project.scenes.length));
    body.append(sw);
  }

  // While the user is actively editing one of our fields, DON'T rebuild the inspector —
  // autosave fires ~1s and a rebuild would yank the field out, drop the selection and
  // make typing append (e.g. "512" + "640" -> "512640"). We defer and re-sync on blur.
  let _editing = false;

  function render(st) {
    if (_editing) {
      // Only skip if a field is actually still focused; else the flag got stuck (focused
      // element removed without focusout) — clear it so the inspector resumes updating.
      const a = document.activeElement;
      if (a && body.contains(a) && (a.tagName === "INPUT" || a.tagName === "TEXTAREA" || a.tagName === "SELECT") && a.dataset.k) return;
      _editing = false;
    }

    const scrollTop = body.scrollTop;  // preserve scroll across the rebuild
    clear(body);
    if (!st.project) { title.textContent = "Inspector"; body.append(el("div", "pj-meta", "No project open.")); return; }
    const scene = st.selectedSceneId ? S.scene(st.selectedSceneId) : null;
    renderGlobalPrompt(st);
    renderSwitch(st, scene);
    renderEngineStrip(st);
    if (scene) renderScene(st, scene);
    else renderProject(st);
    renderExposed(st);
    renderSplit(st);
    body.scrollTop = scrollTop;  // restore so editing doesn't jump to the top
  }

  // Track edit state so autosave re-renders never interrupt the focused field.
  body.addEventListener("focusin", (e) => {
    const t = e.target;
    if (t && t.dataset && t.dataset.k) _editing = true;
  });
  body.addEventListener("focusout", (e) => {
    const t = e.target;
    if (!(t && t.dataset && t.dataset.k)) return;
    _editing = false;
    // Persist the edit now (no more per-second autosave), then re-sync if focus didn't
    // move to another field. flushSave commits + notifies (which renders); fall back to a
    // manual render when there was nothing pending.
    setTimeout(() => {
      if (_editing) return;
      if (!S.flushSave()) render(S.get());
    }, 60);
  });

  window.addEventListener("funpack-invalidate-global-prompt", () => { gpDraft = null; });

  if (window.ViewBus) window.ViewBus.subscribeInspector(render);
  else S.subscribe(render);
})();
