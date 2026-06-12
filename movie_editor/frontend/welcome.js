// First-load welcome screen: pick a project before entering the editor.
(function () {
  const { el } = window.dom;
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
      await S.importProject(file);
    };
    document.body.appendChild(inp);
    return inp;
  })();

  function latestProject() {
    const list = S.get().projects || [];
    return list.length ? list[0] : null;
  }

  function promptNewProject() {
    if (window.SlotPicker?.openPrompt) {
      window.SlotPicker.openPrompt({
        title: "New project",
        value: "Untitled montage",
        placeholder: "Project name",
        onPick: (name) => S.newProject(name),
      });
      return;
    }
    const name = prompt("Project name:", "Untitled montage");
    if (name != null) S.newProject(name || "Untitled montage");
  }

  function openLatest() {
    const p = latestProject();
    if (!p) return;
    S.loadProject(p.id);
  }

  function loadProjectFile() {
    loadFileInput.click();
  }

  function welcomeTour() {
    alert("Welcome tour is coming soon.");
  }

  function close() {
    if (overlay) {
      overlay.remove();
      overlay = null;
    }
  }

  function open() {
    close();
    overlay = el("div", "welcome-overlay");
    const card = el("div", "welcome-card");

    const brand = el("div", "welcome-brand");
    brand.append(el("span", "welcome-mark", "◉"));
    brand.append(el("span", "welcome-title", "FunPack"));
    brand.append(el("span", "welcome-sub", "Cutting Room"));
    card.append(brand);

    card.append(el("p", "welcome-lead",
      "Build multi-scene montages with FunPack Studio and preview them on a real timeline."));

    const latest = latestProject();
    const actions = el("div", "welcome-actions");

    const mkBtn = (label, cls, hint, disabled, fn) => {
      const b = el("button", "welcome-btn" + (cls ? " " + cls : "") + (disabled ? " disabled" : ""));
      b.type = "button";
      b.append(el("span", "welcome-btn-label", label));
      if (hint) b.append(el("span", "welcome-btn-hint", hint));
      if (!disabled && fn) b.onclick = fn;
      return b;
    };

    actions.append(
      mkBtn("New Project", "primary", "Start a blank montage", false, promptNewProject),
      mkBtn("Open Latest Project", "",
        latest ? latest.name : "No saved projects yet",
        !latest, openLatest),
      mkBtn("Load Project", "", "Import a .json project file", false, loadProjectFile),
      mkBtn("Welcome Tour", "ghost", "Coming soon", false, welcomeTour),
    );
    card.append(actions);

    overlay.append(card);
    document.body.append(overlay);
  }

  window.WelcomePage = { open, close, isOpen: () => !!overlay };
})();
