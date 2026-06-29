// Right-bottom zone: context inspector (selected scene OR project) + split preview.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const body = document.getElementById("inspector-body");
  const title = document.getElementById("inspector-title");

  // t2v ("empty") is intentionally not selectable — scenes are i2v/v2v/carry. An
  // anchorless scene still falls back to t2v in the engine (effectiveSourceType).
  const SRC = [
    ["image", "Image · i2v anchor"],
    ["generated_frame", "From generated frame"],
    ["v2v", "Video · v2v source"],
    ["carry", "Carry i2v guide · continue previous"],
    ["mixed", "Mixed · Img2Video anchor + prior guides"],
    ["anchor_guide", "Anchor as guide · image steers, empty latent"],
  ];

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

  // "Anchor as guide": the image still feeds the pipeline (it's the anchor — e.g. an Image
  // Transform derives width/height from it) AND steers as a frame-0 guide; the i2v node is
  // bypassed so the latent stays empty (t2v). The image is required, like any anchor.
  function renderAnchorGuideSource(st, scene, parent) {
    parent = parent || body;
    const ref = scene.source?.media_ref;
    const pick = window.MediaPicker.create({
      value: ref,
      mediaBin: st.mediaBin,
      noneLabel: "— choose anchor / guide image —",
      onChange: (mediaRef) => {
        S.patchScene(scene.id, { source: { ...(scene.source || {}), type: "anchor_guide", media_ref: mediaRef } });
      },
    });
    parent.append(field("Anchor image (guide mode)", pick));
    parent.append(el("div", "insp-hint",
      "The image feeds the pipeline like any anchor (so nodes that need it — e.g. Image Transform "
      + "for width/height — still get it) and steers this scene from a frame-0 guide, but the i2v node "
      + "is bypassed so the latent stays empty (text-to-video). Required, like any anchor."));

    const cur = scene.source?.guide_strength;
    const val = cur == null ? 0.35 : cur;
    const row = el("label", "field");
    row.append(el("span", null, `Guide strength · ${Number(val).toFixed(2)}`));
    const slider = el("input"); slider.type = "range"; slider.min = "0"; slider.max = "1"; slider.step = "0.05";
    slider.value = String(val); slider.dataset.k = "ag-strength";
    slider.oninput = () => { row.firstChild.textContent = `Guide strength · ${Number(slider.value).toFixed(2)}`; };
    slider.onchange = () => {
      S.patchScene(scene.id, { source: { ...(scene.source || {}), type: "anchor_guide", guide_strength: parseFloat(slider.value) } });
    };
    row.append(slider);
    parent.append(row);
    if (!S.getEditorSetting("anchorGuideHasI2v")) {
      parent.append(el("div", "insp-hint warn",
        "No i2v node declared — the latent won't be emptied. Declare your i2v node in "
        + "Editor settings → Anchor as guide so it's bypassed on this pass (otherwise this runs as a "
        + "normal i2v anchor with an added frame-0 guide)."));
    }
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
  const LEN_DEFAULT_MODE = { frames: "project", fps: "project" };

  function modeOf(scene, kind) { return scene[kind + "_mode"] || LEN_DEFAULT_MODE[kind]; }
  function effOf(scene, kind) {
    const m = LEN_META[kind], p = S.get().project;
    if (kind === "frames" && S.sceneEffFrames) return S.sceneEffFrames(scene, p);
    if (kind === "fps" && S.sceneEffFps) return S.sceneEffFps(scene, p);
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
      if (sel.value === "project") patch[kind] = null;
      else if (scene[kind] == null) patch[kind] = effOf(scene, kind);
      S.patchScene(scene.id, patch);
    };
    wrap.append(sel);

    if (mode === "custom") {
      const i = el("input"); i.type = "number"; i.value = effOf(scene, kind); i.dataset.k = "sc-" + kind;
      i.oninput = () => S.patchSceneQuiet(scene.id, { [kind]: m.snap(parseInt(i.value || "0", 10)) });
      // Stored value is snapped (frames → 9/17/25…, fps → ≥1); reflect that back on commit
      // so the field doesn't keep showing an unsnapped number the engine won't actually use.
      i.onchange = () => { i.value = m.snap(parseInt(i.value || "0", 10)); };
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
  const PC = () => window.PipelineCaps;

  function availableSourceOptions(st) {
    if (PC()?.usesChainSampler(st)) return SRC;
    return SRC.filter(([v]) => v === "image");
  }

  function renderSourceField(st, root, scene) {
    const chain = PC()?.usesChainSampler(st);
    // t2v is removed as a mode — surface a legacy/default "empty" scene as the i2v
    // anchor option (no committed change until the user edits or assigns an anchor).
    let stored = root.source?.type || (chain ? "carry" : "image");
    if (stored === "empty") stored = "image";
    const opts = availableSourceOptions(st);
    if (!opts.some(([v]) => v === stored) && PC()?.isChainOnlySource(stored)) {
      const note = el("div", "insp-hint");
      note.textContent = `Stored source “${PC().sourceLabel(stored)}” needs Chain Sampler — runs as text-to-video unless you pick an input image.`;
      body.append(note);
    }
    const src = el("select");
    opts.forEach(([v, label]) => {
      const o = el("option", null, label);
      o.value = v;
      if (v === stored) o.selected = true;
      src.append(o);
    });
    if (stored && !opts.some(([v]) => v === stored)) {
      const o = el("option", null, PC()?.sourceLabel(stored) + " (unavailable)");
      o.value = stored;
      o.selected = true;
      src.append(o);
    }
    src.onchange = () => {
      const prev = root.source?.type || (chain ? "carry" : "empty");
      const next = src.value;
      const patch = { source: { ...(root.source || {}), type: next } };
      if (next === "image" || next === "empty") patch.guides = [];
      else if (chain && (prev === "carry" || next === "carry" || prev === "mixed" || next === "mixed")) patch.guides = [];
      S.patchScene(scene.id, patch);
    };
    body.append(field("Source", src));
    const eff = PC()?.effectiveSourceType(root, st) || stored;
    // image + mixed both pick an anchor image here (renderImageSource branches on mixed);
    // anchor_guide uses its own picker (image + guide strength).
    if (eff === "image" || eff === "mixed") renderImageSource(st, root, body);
    if (eff === "anchor_guide") renderAnchorGuideSource(st, root, body);
  }

  function renderVideoSource(st, scene, parent, label) {
    parent = parent || body;
    if (!window.MediaPicker) return;
    const pick = window.MediaPicker.create({
      mediaBin: st.mediaBin,
      value: scene.source?.media_ref,
      filter: (m) => m.kind === "video",
      noneLabel: "— pick video —",
      onChange: (mediaRef) => {
        S.patchScene(scene.id, {
          source: { ...(scene.source || {}), type: scene.source?.type || "v2v", media_ref: mediaRef },
        });
      },
    });
    parent.append(field(label || "V2V source video", pick));
  }

  function renderVideoClip(st, scene) {
    title.textContent = "Video clip";
    body.append(el("div", "insp-hint", "Imported or converted video — plays as-is. Not included in Generate or the global prompt montage."));
    const ref = scene.source?.media_ref;
    const asset = ref ? (st.mediaBin || []).find((m) => m.id === ref) : null;
    body.append(field("Source", el("span", null, asset?.name || ref || "From last render")));

    const lenRow = el("div", "fields-row");
    lenRow.append(lengthControl(scene, "frames"));
    lenRow.append(lengthControl(scene, "fps"));
    body.append(lenRow);

    foldSection("Scene options", false, (more) => {
      const r = (st.sceneRenders || {})[scene.id];
      if (r?.media || ref) {
        const srcTag = el("div", "insp-tag"); srcTag.textContent = "Source trim (slip)"; more.append(srcTag);
        const inRow = el("div", "fields-row");
        const iIn = el("input"); iIn.type = "number"; iIn.min = "0"; iIn.step = "0.05"; iIn.dataset.k = "vc-src-in";
        iIn.value = scene.source_in || 0;
        iIn.oninput = () => S.setSourceTrim(scene.id, { source_in: parseFloat(iIn.value || 0) });
        inRow.append(field("Source in (s)", iIn));
        more.append(inRow);
      }
    });
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

    if (root.removed_from_plan) {
      const banner = el("div", "insp-hint");
      banner.append(el("span", null, "Removed from plan — its generated clip stays on the timeline. "));
      const restore = el("button", "btn ghost tiny", "Restore to plan");
      restore.onclick = () => S.restoreToPlan(scene.id);
      banner.append(restore);
      body.append(banner);
    }

    const ta = el("textarea"); ta.rows = 4; ta.value = root.text || ""; ta.placeholder = "Describe this scene…"; ta.dataset.k = "sc-text";
    ta.oninput = () => S.patchSceneQuiet(scene.id, { text: ta.value });
    body.append(field("Prompt", ta));
    if (window.ShortcutAutocomplete) window.ShortcutAutocomplete.attach(ta);
    renderSourceField(st, root, scene);
    if (PC()?.usesChainSampler(st)) {
      if ((root.source?.type) === "generated_frame") renderGeneratedFrameSource(st, root);
      if ((root.source?.type) === "v2v") renderVideoSource(st, root, body, "V2V source video");
    }

    const effFrames = effOf(scene, "frames"), effFps = effOf(scene, "fps") || 1;
    const planDur = effFrames / effFps;
    const lenRow = el("div", "fields-row");
    lenRow.append(lengthControl(scene, "frames"));
    lenRow.append(lengthControl(scene, "fps"));
    body.append(lenRow);
    const shownDur = S.sceneDurationSec ? S.sceneDurationSec(scene) : planDur;
    const lenDrift = Math.abs(shownDur - planDur) > 0.05; // render frozen at an older plan length
    body.append(el("div", "insp-hint",
      (lenDrift
        ? `Rendered ≈ ${shownDur.toFixed(2)}s · plan ${planDur.toFixed(2)}s — Regenerate to apply`
        : `Duration ≈ ${planDur.toFixed(2)}s`)
      + ` · trim on timeline · generation splits live in the global prompt (Composer)`));

    const actions = el("div", "insp-block");
    const genBtn = el("button", "btn primary", "Generate this scene");
    genBtn.onclick = () => S.generate(scene.id);
    actions.append(genBtn);
    body.append(actions);

    foldSection("Scene options", false, (more) => {
      // Visual effects (blur / fades / Ken Burns zoom) are applied from the timeline's
      // + Add → Effects menu now; this section keeps source trim, i2v guides, and exclude.
      const _num = (val, k, opts) => {
        const i = el("input"); i.type = "number"; i.value = val; i.dataset.k = k;
        if (opts) Object.assign(i, opts);
        return i;
      };

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

      if (PC()?.usesChainSampler(st)) renderSceneGuides(st, scene, more);

      const row = el("div", "insp-block");
      const chk = el("label", "chk"); const cb = el("input"); cb.type = "checkbox"; cb.checked = !!scene.excluded;
      cb.onchange = () => S.patchScene(scene.id, { excluded: cb.checked });
      chk.append(cb); chk.append(el("span", null, "Exclude from full generation"));
      row.append(chk); more.append(row);
    });
  }

  function slotLabelFor(st, slotId) {
    if (!slotId || slotId === "funpack") return null;
    const slot = (st.models?.slots || []).find((s) => s.id === slotId);
    return slot ? (slot.label || slot.node_class || slot.id) : slotId;
  }

  function renderEngineStrip(st) {
    const p = st.project;
    const strip = el("div", "insp-engine-strip");
    const c = PC()?.caps(st) || {};
    if (c.disable_core || c.imported_workflow) {
      const name = st.models?.workflow_import?.name || "Custom workflow";
      strip.append(el("span", "insp-engine-strip-txt", `Generating with imported workflow (${name})`));
      const btn = el("button", "btn ghost tiny", "Models →");
      btn.onclick = () => window.ModelsModal.open();
      strip.append(btn);
      body.append(strip);
      return;
    }
    const studioOn = PC()?.usesFunpackStudio(st);
    const chainOn = PC()?.usesChainSampler(st);
    const parts = [];
    parts.push(studioOn ? "FunPack Studio" : (slotLabelFor(st, p.conditioning_slot) || p.conditioning_slot));
    parts.push(chainOn ? "Chain Sampler" : (slotLabelFor(st, p.sampler_slot) || p.sampler_slot));
    strip.append(el("span", "insp-engine-strip-txt", "Generating with " + parts.join(" + ")));
    if (studioOn || chainOn) {
      const btn = el("button", "btn ghost tiny", "Engine settings →");
      btn.onclick = () => window.EngineSettingsModal.open();
      strip.append(btn);
    }
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
    row1.append(numberField("FPS", p.frame_rate, (v) => S.patchProjectQuiet({ frame_rate: v }), "pj-fps"));
    body.append(row1);
    const row2 = el("div", "fields-row");
    row2.append(numberField("Width", p.width != null ? p.width : 768, (v) => S.patchProjectQuiet({ width: v }), "pj-w"));
    row2.append(numberField("Height", p.height != null ? p.height : 512, (v) => S.patchProjectQuiet({ height: v }), "pj-h"));
    body.append(row2);

    const promptTag = el("div", "insp-tag"); promptTag.textContent = "Prompt"; body.append(promptTag);
    const anchor = el("textarea"); anchor.rows = 2; anchor.value = p.anchor || ""; anchor.dataset.k = "pj-anchor";
    anchor.placeholder = "World / setting context prepended to every scene";
    anchor.oninput = () => S.patchProjectQuiet({ anchor: anchor.value });
    body.append(field("Anchor", anchor));
    const neg = el("textarea"); neg.rows = 2; neg.value = p.negative_prompt || ""; neg.dataset.k = "pj-neg";
    neg.placeholder = "What to avoid in every scene";
    neg.oninput = () => S.patchProjectQuiet({ negative_prompt: neg.value });
    body.append(field("Negative prompt", neg));

    // Postfix: shared text appended to every scene (mirror of the anchor, which is prepended).
    // Its own project field with an enable toggle — never part of the global prompt.
    const postEnabled = p.postfix_enabled !== false;
    const post = el("textarea"); post.rows = 2; post.value = p.postfix || ""; post.dataset.k = "pj-postfix";
    post.placeholder = "Appended to every scene (e.g. style or quality tags)";
    post.disabled = !postEnabled;
    post.oninput = () => S.patchProjectQuiet({ postfix: post.value });
    const postWrap = el("div", "field");
    const postHead = el("label", "field-toggle");
    const postCb = el("input"); postCb.type = "checkbox"; postCb.checked = postEnabled;
    postCb.onchange = () => { post.disabled = !postCb.checked; S.patchProjectQuiet({ postfix_enabled: postCb.checked }); };
    postHead.append(postCb, el("span", null, "Postfix"));
    postWrap.append(postHead, post);
    body.append(postWrap);

    foldSection("Advanced project settings", false, (adv) => {
      adv.append(numberField("Max scenes", p.max_scenes, (v) => S.patchProjectQuiet({ max_scenes: v }), "pj-max"));
    });
  }

  const STUDIO_DEFAULT_GUIDE = { enabled: true, source: "template", frame_idx: 0, apply_at: 0, strength: 0.35 };

  function normGuideSettings(p) {
    const gs = (p && p.guide_settings) || {};
    return { stack_enabled: !!gs.stack_enabled, accumulate_prior: !!gs.accumulate_prior };
  }

  function renderSceneGuides(st, scene, parent) {
    parent = parent || body;
    if (!PC()?.usesChainSampler(st)) return;
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
      const studioNote = PC()?.usesFunpackStudio(st)
        ? " — next run rebuilds guides and clears stale action/detail repairs; Studio training history is kept."
        : ".";
      w.append(el("span", null, why + studioNote));
      box.append(w);
    }
    if (pv.parse_error) { const w = el("div", "pv-warn"); w.append(el("span", null, "▲")); w.append(el("span", null, "ComfyUI offline — preview paused")); box.append(w); }
    const parsed = pv.parsed || {};
    // Anchor disabled (editor setting): the backend still reports a leading anchor, but
    // generation folds it into Scene 1 — show it that way so the preview matches.
    const foldAnchor = parsed.anchor && !S.getEditorSetting("anchorEnabled");
    if (parsed.anchor && !foldAnchor) { const l = el("div", "pv-line"); l.append(el("span", "pv-badge anchor", "anchor")); l.append(el("span", null, parsed.anchor)); box.append(l); }
    if (foldAnchor) { const l = el("div", "pv-line"); l.append(el("span", "pv-badge", "S1")); l.append(el("span", null, parsed.anchor)); box.append(l); }
    const sBase = foldAnchor ? 2 : 1;
    const rkeys = pv.scene_refinement_keys || [];
    (parsed.scenes || []).forEach((s, i) => {
      const l = el("div", "pv-line");
      l.append(el("span", "pv-badge", "S" + (i + sBase)));
      // Refinement key(s) this scene will steer with (matches generation exactly)
      const rk = rkeys[i];
      if (rk && rk.keys && rk.keys.length) {
        const label = (rk.keys.length > 1 ? "keys: " : "key: ") + rk.keys.join(" + ")
          + (rk.uses_default ? " (default)" : (rk.keys.length > 1 ? " (avg)" : ""));
        const chip = el("span", "pv-badge rkey" + (rk.uses_default ? " default" : ""), label);
        chip.title = rk.uses_default
          ? "No shortcut key fired in this scene — steers with the project default key"
          : "Steers with these refinement keys (averaged when more than one)";
        l.append(chip);
      }
      l.append(el("span", null, s.text || "(empty)"));
      // Show detected transition after this scene
      const t = (parsed.transitions || []).find((tr) => tr.after_scene === i);
      if (t) l.append(el("span", "pv-badge trans", "→ split"));
      box.append(l);
    });
    // Postfix is appended to every scene at generation — show it once after the scenes, mirroring
    // how the anchor (prepended) is shown once above them.
    if (pv.postfix) {
      const l = el("div", "pv-line");
      l.append(el("span", "pv-badge postfix", "postfix"));
      l.append(el("span", null, pv.postfix));
      box.append(l);
    }
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

  function exposedControl(desc, value, on, key) {
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
      ctrl = el("input"); ctrl.type = "number";
      // Real bounds/step from the node spec (captured at expose time) — not step=1 for all.
      if (desc.min != null) ctrl.min = String(desc.min);
      if (desc.max != null) ctrl.max = String(desc.max);
      if (desc.step != null) ctrl.step = String(desc.step);
      else if (desc.kind === "float") ctrl.step = "any";
      ctrl.value = value != null ? value : (desc.default != null ? desc.default : "");
      ctrl.oninput = () => on(desc.kind === "int" ? parseInt(ctrl.value || "0", 10) : parseFloat(ctrl.value || "0"));
    } else {
      ctrl = el("input"); ctrl.type = "text"; ctrl.value = value != null ? value : (desc.default != null ? desc.default : "");
      ctrl.oninput = () => on(ctrl.value);
    }
    if (key) ctrl.dataset.k = key; // protect from autosave rebuilds while editing
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
        const ctrl = exposedControl(
          { kind: l.kind, choices: l.choices, min: l.min, max: l.max, step: l.step, default: l.default },
          l.value, (v) => S.setModelLink(l.id, v), "exp-link-" + l.id);
        wrap.append(field(`🔗 ${l.name} (${(l.members || []).length})`, ctrl));
      }
    });
    slotItems.forEach(([slot, d]) => {
      // Bypass is a slot-level flag (slot.bypassed), not a real node input — route it
      // separately so it never gets sent to ComfyUI as a fake widget value.
      const ctrl = d.isBypass
        ? exposedControl(d, !!slot.bypassed, (v) => S.setModelBypass(slot.id, v), "exp-" + slot.id + "-bypass")
        : exposedControl(d, (slot.inputs || {})[d.name], (v) => S.setModelInput(slot.id, d.name, v),
          "exp-" + slot.id + "-" + d.name);
      wrap.append(field(`${slotLabel(slot)} · ${d.label || d.name}`, ctrl));
    });
    body.append(wrap);
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

  function renderOverlayInspector(st, ov) {
    title.textContent = ov.kind === "text" ? "Text overlay" : "Image overlay";
    const sec = el("div", "insp-block");
    sec.append(el("div", "insp-tag", "Overlay"));

    if (ov.kind === "text") {
      const ta = el("textarea", "insp-global-ta");
      ta.rows = 3;
      ta.value = ov.text || "";
      ta.dataset.k = "ov-text";
      ta.oninput = () => S.updateOverlayTrack(ov.id, { text: ta.value, label: "Text" }, true);
      sec.append(field("Text", ta));

      const size = el("input");
      size.type = "number"; size.min = "8"; size.max = "200"; size.step = "1";
      size.value = ov.font_size != null ? ov.font_size : 42;
      size.dataset.k = "ov-size";
      size.oninput = () => S.updateOverlayTrack(ov.id, { font_size: parseInt(size.value || "42", 10) }, true);
      sec.append(field("Font size (px)", size));

      const fontSel = window.OverlayUI
        ? window.OverlayUI.fontSelect(ov.font_family || "system-ui", (v) => S.updateOverlayTrack(ov.id, { font_family: v }, true))
        : el("select");
      sec.append(field("Font", fontSel));

      const color = el("input");
      color.type = "color";
      color.value = (ov.color || "#ffffff").match(/^#/) ? ov.color : "#ffffff";
      color.dataset.k = "ov-color";
      color.oninput = () => S.updateOverlayTrack(ov.id, { color: color.value }, true);
      sec.append(field("Color", color));

      if (window.OverlayUI?.textStyleControls) {
        sec.append(window.OverlayUI.textStyleControls(
          (k) => ov[k],
          (patch) => S.updateOverlayTrack(ov.id, patch, true),
        ));
      }
    } else {
      const asset = (st.mediaBin || []).find((m) => m.id === ov.media_ref);
      sec.append(el("div", "insp-hint", asset ? `Image: ${asset.name}` : "Missing image in Media Browser"));
      const canvas = S.projectCanvasSize ? S.projectCanvasSize() : { w: st.project?.width || 768, h: st.project?.height || 512 };
      if (window.OverlayUI?.pixelSizeFields) {
        sec.append(window.OverlayUI.pixelSizeFields(ov, canvas.w, canvas.h, (patch) => {
          S.updateOverlayTrack(ov.id, patch, true);
        }));
      }
    }

    const start = el("input");
    start.type = "number"; start.min = "0"; start.step = "0.1";
    start.value = ov.start_sec != null ? ov.start_sec : 0;
    start.dataset.k = "ov-start";
    start.oninput = () => S.updateOverlayTrack(ov.id, { start_sec: Math.max(0, parseFloat(start.value || "0")) }, true);
    sec.append(field("Start (s)", start));

    const dur = el("input");
    dur.type = "number"; dur.min = "0.25"; dur.step = "0.1";
    dur.value = ov.duration_sec != null ? ov.duration_sec : 3;
    dur.dataset.k = "ov-dur";
    dur.oninput = () => S.updateOverlayTrack(ov.id, { duration_sec: Math.max(0.25, parseFloat(dur.value || "3")) }, true);
    sec.append(field("Duration (s)", dur));

    const op = el("input");
    op.type = "range"; op.min = "0"; op.max = "1"; op.step = "0.05";
    op.value = ov.opacity != null ? ov.opacity : 1;
    op.dataset.k = "ov-op";
    op.oninput = () => S.updateOverlayTrack(ov.id, { opacity: parseFloat(op.value) }, true);
    sec.append(field("Opacity", op));

    const flipRow = el("div", "insp-layer-actions");
    const mkFlip = (label, key) => {
      const b = el("button", "btn ghost tiny" + (ov[key] ? " active" : ""), label);
      b.onclick = () => S.updateOverlayTrack(ov.id, { [key]: !ov[key] }, true);
      return b;
    };
    flipRow.append(mkFlip("Flip H", "flip_h"), mkFlip("Flip V", "flip_v"));
    sec.append(flipRow);

    const lane = S.overlayLaneById ? S.overlayLaneById(ov.lane_id) : null;
    const laneIdx = S.overlayLaneIndex ? S.overlayLaneIndex(ov.lane_id) : -1;
    const laneCount = (st.project?.overlay_lanes || []).length;
    sec.append(el("div", "insp-hint", lane
      ? `Track ${laneIdx + 1} of ${laneCount} · higher tracks draw on top.`
      : "Higher overlay tracks draw on top when times overlap."));

    const layerRow = el("div", "insp-layer-actions");
    const mkLayerBtn = (label, fn, title) => {
      const b = el("button", "btn ghost tiny", label);
      b.title = title;
      b.onclick = fn;
      return b;
    };
    layerRow.append(
      mkLayerBtn("To back", () => S.sendOverlayToBack(ov.id), "Move to bottom overlay track"),
      mkLayerBtn("↓", () => S.sendOverlayBackward(ov.id), "One track down"),
      mkLayerBtn("↑", () => S.bringOverlayForward(ov.id), "One track up"),
      mkLayerBtn("To front", () => S.bringOverlayToFront(ov.id), "Move to top overlay track"),
    );
    sec.append(layerRow);

    sec.append(el("div", "insp-hint", "Drag on the preview to reposition. Double-click preview or timeline clip to edit."));
    const rm = el("button", "btn danger tiny", "Remove overlay");
    rm.onclick = () => S.removeOverlayTrack(ov.id, { skipConfirm: true });
    sec.append(rm);
    body.append(sec);
  }

  // While the user is actively editing one of our text/number/range fields, DON'T rebuild
  // the inspector — autosave fires ~1s and a rebuild would yank the field out, drop the
  // caret and make typing append (e.g. "512" + "640" -> "512640"). We defer and re-sync on
  // blur. Checkboxes and selects are DISCRETE commits, not typing: rebuilding right after
  // them is required so their dependent controls (custom length input, Ken Burns knobs,
  // absolute strength, …) appear/disappear immediately instead of only after blur.
  function shouldProtect(a) {
    if (!a || !a.dataset || !a.dataset.k || !body.contains(a)) return false;
    if (a.tagName === "TEXTAREA") return true;
    if (a.tagName !== "INPUT") return false; // SELECT — let it rebuild to reveal dependents
    const t = (a.type || "text").toLowerCase();
    return t !== "checkbox" && t !== "radio"; // protect text/number/range/color; rebuild toggles
  }
  let _editing = false;

  function render(st) {
    if (_editing) {
      // Only skip if a protected (typing) field is actually still focused; else the flag
      // got stuck (focused element removed without focusout, or focus is on a discrete
      // control) — clear it so the inspector resumes updating.
      if (shouldProtect(document.activeElement)) return;
      _editing = false;
    }

    const scrollTop = body.scrollTop;  // preserve scroll across the rebuild
    clear(body);
    if (!st.project) { title.textContent = "Inspector"; body.append(el("div", "pj-meta", "No project open.")); return; }
    const ov = st.selectedOverlayId ? S.overlayTrack(st.selectedOverlayId) : null;
    const scene = !ov && st.selectedSceneId ? S.scene(st.selectedSceneId) : null;
    renderSwitch(st, scene);
    renderEngineStrip(st);
    if (ov) renderOverlayInspector(st, ov);
    else if (scene) {
      if (S.isVideoClip(scene)) renderVideoClip(st, scene);
      else renderScene(st, scene);
    } else renderProject(st);
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
      const k = t.dataset?.k;
      if (k === "global-prompt") {
        S.flushSave();
        render(S.get());
        return;
      }
      if (!S.flushSave()) render(S.get());
    }, 60);
  });

  if (window.ViewBus) window.ViewBus.subscribeInspector(render);
  else S.subscribe(render);
})();
