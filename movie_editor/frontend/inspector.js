// Right-bottom zone: context inspector (selected scene OR project) + split preview.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const body = document.getElementById("inspector-body");
  const title = document.getElementById("inspector-title");

  const SRC = [["empty", "Empty · text-to-video"], ["image", "Image · i2v anchor"], ["generated_frame", "From generated frame"]];

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

  function numberField(labelText, value, on) {
    const i = el("input"); i.type = "number"; i.value = value;
    i.oninput = () => on(parseInt(i.value || "0", 10));
    return field(labelText, i);
  }

  function renderScene(st, scene) {
    title.textContent = `Scene · ${st.project.scenes.indexOf(scene) + 1}`;
    const tag = el("div", "insp-tag"); tag.textContent = "Clip properties"; body.append(tag);

    const ta = el("textarea"); ta.rows = 5; ta.value = scene.text || ""; ta.placeholder = "Describe this scene…";
    ta.oninput = () => S.patchScene(scene.id, { text: ta.value });
    body.append(field("Prompt", ta));

    const src = el("select");
    SRC.forEach(([v, label]) => { const o = el("option", null, label); o.value = v; if ((scene.source?.type) === v) o.selected = true; src.append(o); });
    src.onchange = () => S.patchScene(scene.id, { source: { ...(scene.source || {}), type: src.value } });
    body.append(field("Source (acts in Phase 2)", src));

    body.append(field("Transition to next scene", transitionSelect(scene.transition_to_next || "",
      (v) => S.patchScene(scene.id, { transition_to_next: v }))));

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

    const name = el("input"); name.value = p.name || "";
    name.oninput = () => S.patchProject({ name: name.value });
    body.append(field("Project name", name));

    const anchor = el("textarea"); anchor.rows = 2; anchor.value = p.anchor || "";
    anchor.placeholder = "Character / world description prepended to every scene";
    anchor.oninput = () => S.patchProject({ anchor: anchor.value });
    body.append(field("Anchor", anchor));

    body.append(field("Opening transition (anchor → scene 1)", transitionSelect(p.intro_transition || "",
      (v) => S.patchProject({ intro_transition: v }))));

    const row1 = el("div", "fields-row");
    row1.append(numberField("Seed", p.seed, (v) => S.patchProject({ seed: v })));
    row1.append(numberField("Frames / scene", p.num_frames_per_scene, (v) => S.patchProject({ num_frames_per_scene: v })));
    body.append(row1);
    const row2 = el("div", "fields-row");
    row2.append(numberField("FPS", p.frame_rate, (v) => S.patchProject({ frame_rate: v })));
    row2.append(numberField("Max scenes", p.max_scenes, (v) => S.patchProject({ max_scenes: v })));
    body.append(row2);
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
    (parsed.scenes || []).forEach((s, i) => { const l = el("div", "pv-line"); l.append(el("span", "pv-badge", "S" + (i + 1))); l.append(el("span", null, s.text || "(empty)")); box.append(l); });
    const raw = el("details", "pv-raw"); raw.append(el("summary", null, "combined prompt")); raw.append(el("pre", null, pv.combined_prompt || "")); box.append(raw);
    wrap.append(box); body.append(wrap);
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
    const items = [];
    ((st.models && st.models.slots) || []).forEach((slot) =>
      (slot.exposed || []).forEach((d) => items.push([slot, d])));
    if (!items.length) return;
    const wrap = el("div", "insp-block");
    const tag = el("div", "insp-tag"); tag.textContent = "Exposed controls"; wrap.append(tag);
    items.forEach(([slot, d]) => {
      const ctrl = exposedControl(d, (slot.inputs || {})[d.name], (v) => S.setModelInput(slot.id, d.name, v));
      wrap.append(field(`${slotLabel(slot)} · ${d.label || d.name}`, ctrl));
    });
    body.append(wrap);
  }

  function render(st) {
    clear(body);
    if (!st.project) { title.textContent = "Inspector"; body.append(el("div", "pj-meta", "No project open.")); return; }
    const scene = st.selectedSceneId ? S.scene(st.selectedSceneId) : null;
    if (scene) renderScene(st, scene); else renderProject(st);
    renderExposed(st);
    renderSplit(st);
  }

  S.subscribe(render);
})();
