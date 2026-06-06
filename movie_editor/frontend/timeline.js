// Movie Editor controller: project state, editable timeline, live preview.
(function () {
  const API = window.MovieEditorAPI;
  let project = null;        // current Project object
  let transitions = [];      // [{trigger/name, visual_effect}]
  let saveTimer = null;

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, txt) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  };

  function transitionOptions(selected) {
    const sel = el("select", "transition-select");
    const none = el("option", null, "— none —");
    none.value = "";
    sel.append(none);
    for (const t of transitions) {
      const name = t.trigger || t.name || t.key;
      if (!name) continue;
      const o = el("option", null, t.visual_effect && t.visual_effect !== "none"
        ? `${name}  (${t.visual_effect})` : name);
      o.value = name;
      if (name === selected) o.selected = true;
      sel.append(o);
    }
    if (selected && !transitions.some((t) => (t.trigger || t.name) === selected)) {
      const o = el("option", null, selected);
      o.value = selected; o.selected = true; sel.append(o);
    }
    return sel;
  }

  function scheduleSave() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveAndPreview, 600);
  }

  async function saveAndPreview() {
    if (!project) return;
    try {
      project = await API.saveProject(project.id, project);
      await refreshPreview();
    } catch (e) {
      $("preview").textContent = "Save failed: " + e.message;
    }
  }

  async function refreshPreview() {
    if (!project) return;
    const box = $("preview");
    box.textContent = "…";
    try {
      const r = await API.preview(project.id, false);
      box.replaceChildren();
      if (r.warning) {
        box.append(el("div", "warn", "⚠ " + r.warning));
      }
      if (r.parse_error) {
        box.append(el("div", "warn", "⚠ " + r.parse_error + " (ComfyUI not reachable — editing still works)"));
      }
      const parsed = r.parsed || {};
      if (parsed.anchor) {
        const a = el("div", "pv-anchor");
        a.append(el("span", "pv-badge", "anchor"));
        a.append(el("span", null, parsed.anchor));
        box.append(a);
      }
      (parsed.scenes || []).forEach((s, i) => {
        const row = el("div", "pv-scene");
        row.append(el("span", "pv-badge", "scene " + (i + 1)));
        row.append(el("span", null, s.text || "(empty)"));
        box.append(row);
      });
      const cp = el("details", "pv-raw");
      cp.append(el("summary", null, "combined prompt"));
      cp.append(el("pre", null, r.combined_prompt || ""));
      box.append(cp);
    } catch (e) {
      box.textContent = "Preview unavailable: " + e.message;
    }
  }

  function sceneCard(scene, index) {
    const card = el("div", "scene-card" + (scene.excluded ? " excluded" : ""));

    const head = el("div", "scene-head");
    head.append(el("span", "scene-num", "Scene " + (index + 1)));
    const srcSel = el("select", "src-select");
    [["empty", "Empty (t2v)"], ["image", "Image (i2v)"], ["generated_frame", "From frame"]].forEach(([v, label]) => {
      const o = el("option", null, label); o.value = v;
      if ((scene.source && scene.source.type) === v) o.selected = true;
      srcSel.append(o);
    });
    srcSel.title = "How this scene's latent is created (acts in Phase 2)";
    srcSel.onchange = () => { scene.source = scene.source || {}; scene.source.type = srcSel.value; scheduleSave(); };
    head.append(srcSel);
    card.append(head);

    const ta = el("textarea", "scene-text");
    ta.rows = 3; ta.value = scene.text || "";
    ta.placeholder = "Describe this scene…";
    ta.oninput = () => { scene.text = ta.value; scheduleSave(); };
    card.append(ta);

    const ctrl = el("div", "scene-ctrl");
    const exc = el("label", "chk");
    const cb = el("input"); cb.type = "checkbox"; cb.checked = !!scene.excluded;
    cb.onchange = () => { scene.excluded = cb.checked; render(); scheduleSave(); };
    exc.append(cb); exc.append(document.createTextNode(" exclude"));
    ctrl.append(exc);

    const solo = el("button", "ghost", "Generate only this");
    solo.onclick = () => window.MoviePlayer.generate(project.id, scene.id);
    ctrl.append(solo);

    const up = el("button", "ghost", "↑"); up.onclick = () => move(index, -1);
    const down = el("button", "ghost", "↓"); down.onclick = () => move(index, 1);
    const del = el("button", "ghost danger", "✕"); del.onclick = () => removeScene(index);
    ctrl.append(up); ctrl.append(down); ctrl.append(del);
    card.append(ctrl);

    // transition to next scene (seam), shown between cards
    if (index < project.scenes.length - 1) {
      const seam = el("div", "seam");
      seam.append(el("span", "seam-label", "→"));
      const sel = transitionOptions(scene.transition_to_next || "");
      sel.onchange = () => { scene.transition_to_next = sel.value; scheduleSave(); };
      seam.append(sel);
      card.append(seam);
    }
    return card;
  }

  function move(index, delta) {
    const j = index + delta;
    if (j < 0 || j >= project.scenes.length) return;
    const s = project.scenes;
    [s[index], s[j]] = [s[j], s[index]];
    render(); scheduleSave();
  }

  function removeScene(index) {
    project.scenes.splice(index, 1);
    render(); scheduleSave();
  }

  function addScene() {
    project.scenes.push({ text: "", transition_to_next: "", source: { type: "empty" }, excluded: false });
    render(); scheduleSave();
  }

  function render() {
    if (!project) return;
    $("project-name").value = project.name || "";
    $("anchor").value = project.anchor || "";
    $("seed").value = project.seed;
    $("frames").value = project.num_frames_per_scene;
    $("fps").value = project.frame_rate;
    $("max-scenes").value = project.max_scenes;

    const intro = $("intro-transition");
    const newIntro = transitionOptions(project.intro_transition || "");
    newIntro.id = "intro-transition";
    newIntro.onchange = () => { project.intro_transition = newIntro.value; scheduleSave(); };
    intro.replaceWith(newIntro);

    const rail = $("rail");
    rail.replaceChildren();
    project.scenes.forEach((s, i) => rail.append(sceneCard(s, i)));
    if (project.scenes.length === 0) {
      rail.append(el("div", "empty-hint", "No scenes yet — add one to start your timeline."));
    }
  }

  async function loadProject(id) {
    project = await API.getProject(id);
    render();
    await refreshPreview();
  }

  async function refreshProjectList(selectId) {
    const { projects } = await API.listProjects();
    const sel = $("project-select");
    sel.replaceChildren();
    for (const p of projects) {
      const o = el("option", null, `${p.name} (${p.scene_count})`);
      o.value = p.id; sel.append(o);
    }
    if (selectId) sel.value = selectId;
    if (sel.value) await loadProject(sel.value);
  }

  function bindGlobals() {
    $("project-select").onchange = (e) => loadProject(e.target.value);
    $("new-project").onclick = async () => {
      const p = await API.createProject("Untitled");
      await refreshProjectList(p.id);
    };
    $("delete-project").onclick = async () => {
      if (!project) return;
      if (!confirm(`Delete project "${project.name}"?`)) return;
      await API.deleteProject(project.id);
      project = null;
      await refreshProjectList();
    };
    $("project-name").oninput = (e) => { project.name = e.target.value; scheduleSave(); };
    $("anchor").oninput = (e) => { project.anchor = e.target.value; scheduleSave(); };
    $("seed").oninput = (e) => { project.seed = parseInt(e.target.value || "0", 10); scheduleSave(); };
    $("frames").oninput = (e) => { project.num_frames_per_scene = parseInt(e.target.value || "1", 10); scheduleSave(); };
    $("fps").oninput = (e) => { project.frame_rate = parseInt(e.target.value || "1", 10); scheduleSave(); };
    $("max-scenes").oninput = (e) => { project.max_scenes = parseInt(e.target.value || "1", 10); scheduleSave(); };
    $("add-scene").onclick = addScene;
    $("refresh-preview").onclick = refreshPreview;
    $("generate-all").onclick = () => project && window.MoviePlayer.generate(project.id, null);
  }

  async function init() {
    bindGlobals();
    try {
      const h = await API.health();
      $("health").textContent = "ComfyUI: " + h.comfy_url;
    } catch (_) { $("health").textContent = "sidecar only"; }
    try {
      const t = await API.transitions();
      transitions = t.transitions || [];
    } catch (_) { transitions = []; }
    try {
      await refreshProjectList();
      const sel = $("project-select");
      if (!sel.value) {
        const p = await API.createProject("My first montage");
        await refreshProjectList(p.id);
      }
    } catch (e) {
      $("preview").textContent = "Backend not reachable: " + e.message;
    }
  }

  window.MovieEditor = { init, getProject: () => project };
})();
