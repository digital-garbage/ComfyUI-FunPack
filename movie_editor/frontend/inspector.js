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
  ];

  function transitionSelect(value, onChange) {
    const sel = el("select");
    const none = el("option", null, "— none —"); none.value = ""; sel.append(none);
    (S.get().transitions || []).forEach((t) => {
      const name = t.trigger || t.name || t.key; if (!name) return;
      const o = el("option", null, t.visual_effect && t.visual_effect !== "none" ? `${name} (${t.visual_effect})` : name);
      o.value = name; if (name === value) o.selected = true; sel.append(o);
    });
    if (value && ![...sel.options].some((o) => o.value === value)) {
      const o = el("option", null, value); o.value = value; o.selected = true; sel.append(o);
    }
    sel.onchange = () => onChange(sel.value);
    return sel;
  }

  function field(labelText, control) {
    const l = el("label", "field"); l.append(el("span", null, labelText)); l.append(control); return l;
  }

  function numberField(labelText, value, on, key) {
    const i = el("input"); i.type = "number"; i.value = value; if (key) i.dataset.k = key;
    i.oninput = () => on(parseInt(i.value || "0", 10));
    return field(labelText, i);
  }

  function renderImageSource(st, scene) {
    const block = el("div", "insp-block img-src");
    const ref = scene.source?.media_ref;
    const asset = (st.mediaBin || []).find((m) => m.id === ref);
    const prev = el("div", "img-src-prev");
    if (asset && asset.kind === "image") { const img = el("img"); img.src = window.MovieEditorAPI.mediaUrl(asset.id); prev.append(img); }
    else prev.append(el("span", "media-icon", asset ? "▶" : "◇"));
    const info = el("div", "img-src-info");
    info.append(el("div", "img-src-name", asset ? asset.name : "No asset — drag from the Media bin onto the clip, or pick below"));
    block.append(prev); block.append(info);
    body.append(block);

    // pick an asset directly
    const pick = el("select");
    pick.append(new Option("— choose asset —", ""));
    (st.mediaBin || []).forEach((m) => { const o = new Option(m.name, m.id); if (m.id === ref) o.selected = true; pick.append(o); });
    pick.onchange = () => S.patchScene(scene.id, { source: { ...(scene.source || {}), type: "image", media_ref: pick.value || null } });
    body.append(field("Media asset", pick));
    body.append(el("div", "insp-hint", "Feeds the i2v anchor for this scene. Routing is automatic (→ Studio source image); to send it through a node first, set that node's Input source to Timeline in Models."));
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

  function renderScene(st, scene) {
    title.textContent = `Scene · ${st.project.scenes.indexOf(scene) + 1}`;
    const tag = el("div", "insp-tag"); tag.textContent = "Clip properties"; body.append(tag);

    const ta = el("textarea"); ta.rows = 5; ta.value = scene.text || ""; ta.placeholder = "Describe this scene…"; ta.dataset.k = "sc-text";
    ta.oninput = () => S.patchSceneQuiet(scene.id, { text: ta.value });
    body.append(field("Prompt", ta));

    const src = el("select");
    SRC.forEach(([v, label]) => { const o = el("option", null, label); o.value = v; if ((scene.source?.type) === v) o.selected = true; src.append(o); });
    src.onchange = () => S.patchScene(scene.id, { source: { ...(scene.source || {}), type: src.value } });
    body.append(field("Source", src));

    if ((scene.source?.type) === "image") renderImageSource(st, scene);
    if ((scene.source?.type) === "generated_frame") renderGeneratedFrameSource(st, scene);
    if ((scene.source?.type) === "carry") {
      const hint = el("div", "insp-block");
      hint.append(el("div", "insp-hint",
        "No start frame of its own — this scene continues from the previous scene's i2v guide and overlaps with it (chain-sampler carry behaviour). Use this for a continuous shot; use an Image / generated frame to hard-cut to a new anchor."));
      body.append(hint);
    }

    body.append(field("Transition to next scene", transitionSelect(scene.transition_to_next || "",
      (v) => S.patchScene(scene.id, { transition_to_next: v }))));

    // ── Video effects (post-decode pixel ops; applied at render + approximated in preview)
    const fxTag = el("div", "insp-tag"); fxTag.textContent = "Video effects"; body.append(fxTag);
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
    body.append(field(`Gaussian blur (${Math.round((fx.blur || 0) * 100)}%)`, blur));

    const fadeRow = el("div", "fields-row");
    const fi = _num(fx.fade_in || 0, "sc-fx-fi", { min: 0, max: 10, step: 0.1 });
    fi.oninput = () => patchFx("fade_in", parseFloat(fi.value || "0"), true);
    const fo = _num(fx.fade_out || 0, "sc-fx-fo", { min: 0, max: 10, step: 0.1 });
    fo.oninput = () => patchFx("fade_out", parseFloat(fo.value || "0"), true);
    fadeRow.append(field("Fade in (s)", fi)); fadeRow.append(field("Fade out (s)", fo));
    body.append(fadeRow);

    const zoom = el("select"); zoom.dataset.k = "sc-fx-zoom";
    [["none", "None"], ["in", "Zoom in (push)"], ["out", "Zoom out (pull back)"]].forEach(([v, label]) => {
      const o = el("option", null, label); o.value = v; if ((fx.zoom || "none") === v) o.selected = true; zoom.append(o);
    });
    zoom.onchange = () => patchFx("zoom", zoom.value);
    body.append(field("Zoom (Ken Burns)", zoom));

    // ── Seam transition (rendered crossfade / fade — overlaps & shortens total)
    const vt = el("select"); vt.dataset.k = "sc-vt";
    [["", "Hard cut"], ["crossfade", "Crossfade (dissolve)"], ["fadeblack", "Fade through black"],
     ["wipeleft", "Wipe left"], ["wiperight", "Wipe right"]].forEach(([v, label]) => {
      const o = el("option", null, label); o.value = v; if ((scene.video_transition || "") === v) o.selected = true; vt.append(o);
    });
    vt.onchange = () => S.patchScene(scene.id, { video_transition: vt.value });
    body.append(field("Video transition to next", vt));
    if (scene.video_transition) {
      const tf = _num(scene.transition_frames || 16, "sc-tf", { min: 1, max: 120, step: 1 });
      tf.oninput = () => S.patchSceneQuiet(scene.id, { transition_frames: parseInt(tf.value || "0", 10) });
      body.append(field("Transition length (frames)", tf));
      body.append(el("div", "insp-hint", "Crossfades overlap the two clips, so the montage gets a little shorter."));
    }

    // ── Clip audio (original LTXAV audio gain)
    const auTag = el("div", "insp-tag"); auTag.textContent = "Clip audio"; body.append(auTag);
    const vol = el("input"); vol.type = "range"; vol.min = "0"; vol.max = "2"; vol.step = "0.05";
    vol.value = scene.audio_volume != null ? scene.audio_volume : 1; vol.dataset.k = "sc-vol";
    const volPct = Math.round((scene.audio_volume != null ? scene.audio_volume : 1) * 100);
    vol.oninput = () => S.patchSceneQuiet(scene.id, { audio_volume: parseFloat(vol.value) });
    body.append(field(`Original volume (${volPct}%)`, vol));
    body.append(el("div", "insp-hint", "0% mutes this clip. Add background tracks & toggle original audio in Project settings."));

    const p = st.project;
    const lenRow = el("div", "fields-row");
    lenRow.append(lengthControl(scene, "frames"));
    lenRow.append(lengthControl(scene, "fps"));
    body.append(lenRow);
    const effFrames = effOf(scene, "frames"), effFps = effOf(scene, "fps") || 1;
    body.append(el("div", "insp-hint", `Duration ≈ ${(effFrames / effFps).toFixed(2)}s`));

    const row = el("div", "insp-block");
    const chk = el("label", "chk"); const cb = el("input"); cb.type = "checkbox"; cb.checked = !!scene.excluded;
    cb.onchange = () => S.patchScene(scene.id, { excluded: cb.checked });
    chk.append(cb); chk.append(el("span", null, "Exclude from full generation"));
    row.append(chk); body.append(row);

    const actions = el("div", "insp-block");
    const gen = el("button", "btn primary", "Generate this scene"); gen.onclick = () => S.generate(scene.id);
    const del = el("button", "btn danger", "Delete"); del.style.marginLeft = "8px"; del.onclick = () => S.removeScene(scene.id);
    actions.append(gen); actions.append(del); body.append(actions);
  }

  function renderProject(st) {
    const p = st.project;
    title.textContent = "Project";
    const tag = el("div", "insp-tag"); tag.textContent = "Global settings"; body.append(tag);

    const name = el("input"); name.value = p.name || ""; name.dataset.k = "pj-name";
    name.oninput = () => S.patchProjectQuiet({ name: name.value });
    body.append(field("Project name", name));

    const anchor = el("textarea"); anchor.rows = 2; anchor.value = p.anchor || ""; anchor.dataset.k = "pj-anchor";
    anchor.placeholder = "Character / world description prepended to every scene";
    anchor.oninput = () => S.patchProjectQuiet({ anchor: anchor.value });
    body.append(field("Anchor", anchor));

    body.append(field("Opening transition (anchor → scene 1)", transitionSelect(p.intro_transition || "",
      (v) => S.patchProject({ intro_transition: v }))));

    const row1 = el("div", "fields-row");
    row1.append(numberField("Seed", p.seed, (v) => S.patchProjectQuiet({ seed: v }), "pj-seed"));
    row1.append(numberField("Frames / scene", p.num_frames_per_scene, (v) => S.patchProjectQuiet({ num_frames_per_scene: v }), "pj-frames"));
    body.append(row1);
    const row2 = el("div", "fields-row");
    row2.append(numberField("FPS", p.frame_rate, (v) => S.patchProjectQuiet({ frame_rate: v }), "pj-fps"));
    row2.append(numberField("Max scenes", p.max_scenes, (v) => S.patchProjectQuiet({ max_scenes: v }), "pj-max"));
    body.append(row2);
    const row3 = el("div", "fields-row");
    row3.append(numberField("Width", p.width != null ? p.width : 768, (v) => S.patchProjectQuiet({ width: v }), "pj-w"));
    row3.append(numberField("Height", p.height != null ? p.height : 512, (v) => S.patchProjectQuiet({ height: v }), "pj-h"));
    body.append(row3);
    body.append(el("div", "insp-hint", "Link FPS / Frames / Width / Height to node inputs in Models → Linked inputs (set Source = Project …)."));

    const neg = el("textarea"); neg.rows = 2; neg.value = p.negative_prompt || ""; neg.dataset.k = "pj-neg";
    neg.placeholder = "What to avoid in every scene";
    neg.oninput = () => S.patchProjectQuiet({ negative_prompt: neg.value });
    body.append(field("Negative prompt", neg));

    // ── Audio (render-time mix) ───────────────────────────────────────────────────
    const audTag = el("div", "insp-tag"); audTag.textContent = "Audio"; body.append(audTag);
    const keepRow = el("div", "insp-block");
    const keepLbl = el("label", "chk"); const keepCb = el("input"); keepCb.type = "checkbox";
    keepCb.checked = p.keep_original_audio !== false;
    keepCb.onchange = () => S.patchProject({ keep_original_audio: keepCb.checked });
    keepLbl.append(keepCb); keepLbl.append(el("span", null, "Keep original (generated) audio"));
    keepRow.append(keepLbl); body.append(keepRow);
    if (p.keep_original_audio === false)
      body.append(el("div", "insp-hint", "Original audio is OFF — the render uses only inserted tracks below (silent if none)."));

    // inserted tracks
    (p.audio_tracks || []).forEach((t, i) => {
      const asset = (st.mediaBin || []).find((m) => m.id === t.media_ref);
      const card = el("div", "aud-track");
      const head = el("div", "aud-track-head");
      head.append(el("span", "aud-track-name", (asset && asset.name) || t.label || `Track ${i + 1}`));
      const rm = el("button", "ic-btn danger", "✕"); rm.title = "Remove track";
      rm.onclick = () => S.removeAudioTrack(t.id);
      head.append(rm); card.append(head);
      const r = el("div", "fields-row");
      r.append(numberField("Start (s)", t.start_sec || 0, (v) => S.updateAudioTrack(t.id, { start_sec: v }, true), "aud-st-" + t.id));
      const volWrap = el("label", "field");
      volWrap.append(el("span", "field-label", `Volume (${Math.round((t.volume != null ? t.volume : 1) * 100)}%)`));
      const tv = el("input"); tv.type = "range"; tv.min = "0"; tv.max = "2"; tv.step = "0.05";
      tv.value = t.volume != null ? t.volume : 1; tv.dataset.k = "aud-vol-" + t.id;
      tv.oninput = () => S.updateAudioTrack(t.id, { volume: parseFloat(tv.value) }, true);
      volWrap.append(tv); r.append(volWrap);
      card.append(r);
      body.append(card);
    });

    // add a track from the media bin
    const audioAssets = (st.mediaBin || []).filter((m) => m.kind === "audio" || /\.(mp3|wav|m4a|aac|ogg|flac)$/i.test(m.name || ""));
    const addRow = el("div", "insp-block");
    const addSel = el("select"); addSel.dataset.k = "aud-add";
    addSel.append(new Option(audioAssets.length ? "— add audio track… —" : "— upload audio to the Media bin first —", ""));
    audioAssets.forEach((m) => addSel.append(new Option(m.name || m.id, m.id)));
    addSel.onchange = () => { if (addSel.value) { S.addAudioTrack(addSel.value, 0); addSel.value = ""; } };
    addRow.append(field("Add background track", addSel)); body.append(addRow);
    body.append(el("div", "insp-hint", "Inserted tracks are mixed at render only (not in the live preview). Per-clip original volume is on each scene."));

    const slots = (st.models?.slots || []);

    // Conditioning slot selector + its settings panel
    const condTag = el("div", "insp-tag"); condTag.textContent = "Conditioning"; body.append(condTag);
    const condSel = el("select");
    [["funpack", "FunPack Studio (built-in)"], ...slots.map((s) => [s.id, s.label || s.node_class || s.id])]
      .forEach(([v, lbl]) => { const o = new Option(lbl, v); if ((p.conditioning_slot || "funpack") === v) o.selected = true; condSel.append(o); });
    condSel.onchange = () => S.setConditioningSlot(condSel.value);
    body.append(field("Node", condSel));
    if (!p.conditioning_slot || p.conditioning_slot === "funpack") renderStudioSettings(st);

    // Sampler slot selector + its settings panel
    const sampTag = el("div", "insp-tag"); sampTag.textContent = "Sampler"; body.append(sampTag);
    const sampSel = el("select");
    [["funpack", "FunPack Chain Sampler (built-in)"], ...slots.map((s) => [s.id, s.label || s.node_class || s.id])]
      .forEach(([v, lbl]) => { const o = new Option(lbl, v); if ((p.sampler_slot || "funpack") === v) o.selected = true; sampSel.append(o); });
    sampSel.onchange = () => S.setSamplerSlot(sampSel.value);
    body.append(field("Node", sampSel));
    if (!p.sampler_slot || p.sampler_slot === "funpack") renderSamplerSettings(st);
  }

  // Studio refiner flags (live in studio_settings.refiner). Booleans the editor exposes.
  const STUDIO_REFINER_FLAGS = [
    { name: "vision_conditioning", label: "Vision conditioning", default: true },
    { name: "reference_injection", label: "Reference injection", default: false },
    { name: "prompt_repair", label: "Prompt repair", default: true },
  ];

  function renderStudioSettings(st) {
    const p = st.project;
    const si = p.studio_inputs || {};
    let curSettings = {};
    try { curSettings = JSON.parse(si.studio_settings || "{}"); } catch (_) {}
    const samplers = curSettings.samplers || null;

    // Studio refiner toggles
    const rfTag = el("div", "insp-tag sp-sub"); rfTag.textContent = "Studio · refiner"; body.append(rfTag);
    const rf = (curSettings.refiner && typeof curSettings.refiner === "object") ? curSettings.refiner : {};
    STUDIO_REFINER_FLAGS.forEach((f) => {
      const cur = rf[f.name] != null ? rf[f.name] : f.default;
      const ctrl = el("input"); ctrl.type = "checkbox"; ctrl.checked = !!cur; ctrl.style.width = "auto";
      ctrl.dataset.k = "rf-" + f.name;
      ctrl.onchange = () => {
        const next = { ...curSettings, refiner: { ...rf, [f.name]: ctrl.checked } };
        S.setStudioInputNow("studio_settings", JSON.stringify(next));
      };
      body.append(field(f.label, ctrl));
    });

    const tag = el("div", "insp-tag sp-sub"); tag.textContent = "Studio · sampler algorithm"; body.append(tag);

    function persist(updatedSamplers, quiet) {
      const next = JSON.stringify({ ...curSettings, samplers: updatedSamplers });
      if (quiet) S.setStudioInput("studio_settings", next);
      else S.setStudioInputNow("studio_settings", next);
    }

    try {
      window.SamplerPanel.render(body, samplers,
        (s) => persist(s, true),
        (s) => persist(s, false));
    } catch (e) {
      const err = el("div", "insp-hint"); err.style.color = "var(--err,#e55)";
      err.textContent = "Studio sampler panel failed to render: " + e.message;
      body.append(err);
    }
  }

  // Key ChainSampler widget inputs exposed as simple form controls.
  const SAMPLER_KNOBS = [
    { name: "cfg",                   label: "CFG",                   kind: "float", default: 1.0,   min: 0, max: 20,  step: 0.1  },
    { name: "frame_overlap",         label: "Frame overlap",         kind: "int",   default: 16,    min: 0, max: 97,  step: 1    },
    { name: "carry_i2v_guides",      label: "Carry i2v guides",     kind: "bool",  default: false },
    { name: "mid_scene_guide",       label: "Mid-scene guide",      kind: "bool",  default: false },
    { name: "mid_scene_guide_strength", label: "Guide strength",    kind: "float", default: 0.25,  min: 0, max: 1,   step: 0.05, dependsOn: "mid_scene_guide" },
    { name: "embed_guidance",        label: "Embed guidance",       kind: "bool",  default: false },
    { name: "embed_guidance_source", label: "Embed mode",           kind: "combo", choices: ["relative", "absolute"], default: "relative", dependsOn: "embed_guidance" },
    { name: "embed_guidance_strength", label: "Embed strength",     kind: "float", default: 0.02,  min: 0, max: 0.5, step: 0.005, dependsOn: "embed_guidance" },
    { name: "decode_noise_scale",    label: "Decode noise scale",   kind: "float", default: 0.0,   min: 0, max: 1,   step: 0.01 },
    { name: "decode_timestep",       label: "Decode timestep",      kind: "float", default: 0.05,  min: 0, max: 1,   step: 0.01 },
  ];

  function renderSamplerSettings(st) {
    const p = st.project;
    const si = p.sampler_inputs || {};
    const tag = el("div", "insp-tag"); tag.textContent = "Sampler settings"; body.append(tag);

    SAMPLER_KNOBS.forEach((k) => {
      const dep = k.dependsOn;
      if (dep) {
        const depVal = si[dep] != null ? si[dep] : SAMPLER_KNOBS.find((x) => x.name === dep)?.default;
        if (!depVal) return;
      }
      const val = si[k.name] != null ? si[k.name] : k.default;
      let ctrl;
      if (k.kind === "bool") {
        ctrl = el("input"); ctrl.type = "checkbox"; ctrl.checked = !!val; ctrl.style.width = "auto";
        ctrl.dataset.k = "si-" + k.name;
        ctrl.onchange = () => S.setSamplerInputNow(k.name, ctrl.checked);
      } else if (k.kind === "combo") {
        ctrl = el("select"); ctrl.dataset.k = "si-" + k.name;
        (k.choices || []).forEach((c) => { const o = el("option", null, c); o.value = c; if (c === val) o.selected = true; ctrl.append(o); });
        ctrl.onchange = () => S.setSamplerInputNow(k.name, ctrl.value);
      } else {
        ctrl = el("input"); ctrl.type = "number";
        if (k.step != null) ctrl.step = String(k.step);
        if (k.min != null) ctrl.min = String(k.min);
        if (k.max != null) ctrl.max = String(k.max);
        ctrl.value = val; ctrl.dataset.k = "si-" + k.name;
        ctrl.oninput = () => {
          const v = k.kind === "int" ? parseInt(ctrl.value || "0", 10) : parseFloat(ctrl.value || "0");
          S.setSamplerInput(k.name, v);
        };
      }
      body.append(field(k.label, ctrl));
    });
  }

  function renderSplit(st) {
    const pv = st.preview; if (!pv) return;
    const wrap = el("div", "insp-block");
    const tag = el("div", "insp-tag"); tag.textContent = "Split preview"; wrap.append(tag);
    const box = el("div", "split-pv");
    if (pv.warning) { const w = el("div", "pv-warn"); w.append(el("span", null, "▲")); w.append(el("span", null, pv.warning)); box.append(w); }
    if (pv.parse_error) { const w = el("div", "pv-warn"); w.append(el("span", null, "▲")); w.append(el("span", null, "ComfyUI offline — preview paused")); box.append(w); }
    const parsed = pv.parsed || {};
    if (parsed.anchor) { const l = el("div", "pv-line"); l.append(el("span", "pv-badge anchor", "anchor")); l.append(el("span", null, parsed.anchor)); box.append(l); }
    (parsed.scenes || []).forEach((s, i) => {
      const l = el("div", "pv-line");
      l.append(el("span", "pv-badge", "S" + (i + 1)));
      l.append(el("span", null, s.text || "(empty)"));
      // Show detected transition after this scene
      const t = (parsed.transitions || []).find((tr) => tr.after_scene === i);
      if (t?.visual_effect) l.append(el("span", "pv-badge trans", "→ " + t.visual_effect));
      box.append(l);
    });
    const raw = el("details", "pv-raw"); raw.append(el("summary", null, "combined prompt")); raw.append(el("pre", null, pv.combined_prompt || "")); box.append(raw);
    wrap.append(box);

    // Sync button — lets the user push what was parsed back into the scene data
    if ((parsed.scenes || []).length > 0) {
      const syncBtn = el("button", "btn ghost tiny sync-preview-btn", "↺ Sync scenes from preview");
      syncBtn.title = "Distribute the parsed anchor / scene texts / transitions back into the timeline";
      syncBtn.onclick = () => {
        if (confirm("This will overwrite scene texts and transitions with what the parser detected. Continue?"))
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
    const live = (st.preview && st.preview.combined_prompt) || st.project.global_prompt || "";
    const dirty = gpDraft != null && gpDraft !== live;
    const val = gpDraft != null ? gpDraft : live;

    const sec = el("div", "insp-global");
    const head = el("div", "insp-global-head");
    head.append(el("span", "insp-global-title", "Global prompt"));
    const apply = el("button", "btn primary tiny", "Apply →");
    apply.title = "Split this prompt into anchor, scenes and transitions on the timeline (overwrites it)";
    apply.disabled = !val.trim();  // clickable whenever there's a prompt to (re)split — not only after edits
    apply.onclick = async () => {
      apply.disabled = true; apply.textContent = "Applying…";
      await S.applyGlobalPrompt(gpDraft != null ? gpDraft : live);
      gpDraft = null;
      S.set({});  // re-render so the field returns to its live view
    };
    head.append(apply);
    sec.append(head);

    const ta = el("textarea", "insp-global-ta"); ta.rows = 3; ta.value = val; ta.dataset.k = "global-prompt";
    ta.placeholder = "Anchor, then scene texts joined by transition markers — the whole montage as one prompt.";
    ta.oninput = () => { gpDraft = ta.value; apply.disabled = !ta.value.trim(); };
    sec.append(ta);
    sec.append(el("div", "insp-hint", dirty
      ? "Edited — press Apply to (re)split this prompt onto the timeline."
      : "Live view of the assembled prompt. Edit a scene below and it updates here; press Apply to split this prompt into scenes."));
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
    if (scene) renderScene(st, scene); else renderProject(st);
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

  S.subscribe(render);
})();
