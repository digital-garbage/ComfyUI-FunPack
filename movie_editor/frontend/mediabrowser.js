// Left zone: projects list + media bin (drag-drop placeholder for Phase 3).
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const body = document.getElementById("media-body");

  function render(st) {
    clear(body);

    // Projects
    const sec = el("div", "mb-section");
    const head = el("div", "mb-section-title");
    head.append(el("span", null, "Projects"));
    const add = el("button", "btn ghost tiny", "＋ New");
    add.onclick = () => S.newProject(prompt("Project name:", "Untitled montage"));
    head.append(add);
    sec.append(head);
    (st.projects || []).forEach((p) => {
      const row = el("div", "project-row" + (st.project && st.project.id === p.id ? " active" : ""));
      row.onclick = () => S.loadProject(p.id);
      row.append(el("span", "pj-name", p.name));
      row.append(el("span", "pj-meta", `${p.scene_count}▦`));
      sec.append(row);
    });
    if (!(st.projects || []).length) sec.append(el("div", "pj-meta", "No projects yet."));
    body.append(sec);

    // Media bin (placeholder)
    const msec = el("div", "mb-section");
    const mhead = el("div", "mb-section-title");
    mhead.append(el("span", null, "Media Bin"));
    mhead.append(el("span", "soon", "soon"));
    msec.append(mhead);
    const bin = el("div", "mediabin");
    bin.append(el("div", "big", "🎞"));
    bin.append(el("div", null, "Drop images & clips here"));
    bin.append(el("div", "pj-meta", "Assign to scenes as i2v anchors"));
    // drag affordance (visual only for now)
    ["dragenter", "dragover"].forEach((ev) => bin.addEventListener(ev, (e) => { e.preventDefault(); bin.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((ev) => bin.addEventListener(ev, (e) => { e.preventDefault(); bin.classList.remove("drag"); }));
    msec.append(bin);
    body.append(msec);
  }

  S.subscribe(render);
})();
