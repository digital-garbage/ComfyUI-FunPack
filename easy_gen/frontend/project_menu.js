// "FunPack Easy Gen" project picker — same /funpack/movie/api/projects* calls
// as the Editor's welcome.js / menubar.js "Open recent" menu, reimplemented as
// a small standalone modal (welcome.js/menubar.js are wired into the full
// Editor chrome and not reusable in isolation).
(function () {
  const { el, clear } = window.dom;
  const API = window.MovieEditorAPI;
  const S = window.Store;
  let overlay = null;

  const loadFileInput = (function () {
    const inp = document.createElement("input");
    inp.type = "file";
    inp.accept = ".json,.funpack_project.json,application/json";
    inp.style.display = "none";
    inp.onchange = async () => {
      const file = inp.files?.[0];
      inp.value = "";
      if (!file) return;
      try {
        const data = JSON.parse(await file.text());
        const p = await API.importProject(data);
        S.set({ project: p });
        close();
      } catch (e) {
        alert("Could not load project file: " + (e.message || e));
      }
    };
    document.body.appendChild(inp);
    return inp;
  })();

  function close() {
    if (overlay) { overlay.remove(); overlay = null; }
  }

  async function open({ dismissable } = {}) {
    close();
    overlay = el("div", "modal-overlay");
    if (!dismissable) overlay.style.pointerEvents = "auto";
    const box = el("div", "modal modal-wide");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", "FunPack Easy Gen"));
    if (dismissable) {
      const x = el("button", "btn ghost tiny", "✕");
      x.onclick = close;
      head.append(el("div", "modal-head-right"), x);
    }
    box.append(head);

    const content = el("div", "modal-content");
    box.append(content);

    const actions = el("div", "welcome-actions");
    const mkBtn = (label, cls, hint, fn) => {
      const b = el("button", "welcome-btn" + (cls ? " " + cls : ""));
      b.type = "button";
      b.append(el("span", "welcome-btn-label", label));
      if (hint) b.append(el("span", "welcome-btn-hint", hint));
      b.onclick = fn;
      return b;
    };
    actions.append(
      mkBtn("New Generation", "primary", "Start with a blank prompt", async () => {
        await S.createProject("Untitled");
        close();
      }),
      mkBtn("Load Project File…", "", "Import a .funpack_project.json", () => loadFileInput.click()),
    );
    content.append(actions);

    // Same three maintenance actions as the Editor's splash — an Easy Gen user on a
    // rental had no way to switch branch or restart without opening the other UI.
    content.append(el("div", "sw-rows-label", "Maintenance"));
    content.append(window.FunPackGit.maintenanceRow());

    content.append(el("div", "sw-rows-label", "Recent"));
    const recentWrap = el("div", "sw-rows");
    content.append(recentWrap);
    try {
      const list = await S.listProjects();
      clear(recentWrap);
      if (!list.length) {
        recentWrap.append(el("div", "sw-hint", "No projects yet."));
      }
      list.slice(0, 12).forEach((p) => {
        const row = el("div", "sw-row");
        const main = el("div", "sw-row-main");
        main.append(el("div", "sw-row-title", p.name));
        main.append(el("div", "sw-row-hint", `${p.scene_count ?? 1} scene(s)`));
        row.append(main);
        const openBtn = el("button", "btn ghost tiny", "Open");
        openBtn.type = "button";
        openBtn.onclick = async () => { await S.loadProject(p.id); close(); };
        row.append(openBtn);
        recentWrap.append(row);
      });
    } catch (e) {
      clear(recentWrap);
      recentWrap.append(el("div", "sw-hint", "Could not load project list: " + (e.message || e)));
    }

    overlay.append(box);
    if (dismissable) overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.body.append(overlay);
  }

  window.ProjectMenu = { open, close };
})();
