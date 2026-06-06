// Bottom zone: the clip track. Selection drives the inspector; transitions sit
// in the seams between clips. Editing happens in the inspector, not here.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const body = document.getElementById("timeline-body");
  const meta = document.getElementById("timeline-meta");

  const SRC_ICON = { empty: "▦", image: "◐", generated_frame: "⛶" };

  function transitionSelect(value, onChange) {
    const sel = el("select");
    const none = el("option", null, "—"); none.value = ""; sel.append(none);
    (S.get().transitions || []).forEach((t) => {
      const name = t.trigger || t.name || t.key; if (!name) return;
      const o = el("option", null, name); o.value = name; if (name === value) o.selected = true; sel.append(o);
    });
    if (value && ![...sel.options].some((o) => o.value === value)) { const o = el("option", null, value); o.value = value; o.selected = true; sel.append(o); }
    sel.onchange = (e) => { e.stopPropagation(); onChange(sel.value); };
    sel.onclick = (e) => e.stopPropagation();
    return sel;
  }

  function clipEl(st, scene, index) {
    const clip = el("div", "clip" + (scene.id === st.selectedSceneId ? " selected" : "") + (scene.excluded ? " excluded" : ""));
    clip.onclick = () => S.selectScene(scene.id);

    const top = el("div", "clip-top");
    top.append(el("span", "clip-no", String(index + 1).padStart(2, "0")));
    top.append(el("span", "clip-src", SRC_ICON[scene.source?.type] || "▦"));
    const actions = el("div", "clip-actions");
    const mk = (label, cls, fn) => { const b = el("button", "btn ghost tiny" + (cls ? " " + cls : ""), label); b.onclick = (e) => { e.stopPropagation(); fn(); }; return b; };
    actions.append(mk("‹", "", () => S.moveScene(scene.id, -1)));
    actions.append(mk("›", "", () => S.moveScene(scene.id, 1)));
    actions.append(mk("✕", "danger", () => S.removeScene(scene.id)));
    top.append(actions);
    clip.append(top);

    const text = el("div", "clip-text" + (scene.text ? "" : " empty"), scene.text || "empty scene");
    clip.append(text);

    const foot = el("div", "clip-foot");
    foot.append(el("span", null, scene.excluded ? "excluded" : (scene.source?.type || "empty")));
    clip.append(foot);
    return clip;
  }

  function seamEl(scene) {
    const seam = el("div", "seam");
    seam.append(el("div", "seam-line"));
    seam.append(transitionSelect(scene.transition_to_next || "", (v) => S.patchScene(scene.id, { transition_to_next: v })));
    seam.append(el("div", "seam-cap", "transition"));
    return seam;
  }

  function render(st) {
    clear(body); clear(meta);
    if (!st.project) { body.append(el("div", "empty-stage", "Open a project to start cutting.")); return; }

    body.append(el("div", "tl-ruler"));
    const track = el("div", "tl-track");
    const scenes = st.project.scenes || [];
    scenes.forEach((scene, i) => {
      track.append(clipEl(st, scene, i));
      if (i < scenes.length - 1) track.append(seamEl(scene));
    });
    const add = el("button", "tl-add");
    add.append(el("span", "plus", "＋"));
    add.append(el("span", null, "Add scene"));
    add.onclick = () => S.addScene();
    track.append(add);
    body.append(track);

    const active = scenes.filter((s) => !s.excluded).length;
    meta.append(el("span", null, `${scenes.length} clips · ${active} active`));
  }

  S.subscribe(render);
})();
