// Easy Gen entrypoint: wires the big-preview / prompt / upload / generate bar
// to the shared easy_store.js + preview.js, and opens the "FunPack Easy Gen"
// project picker on boot. No re-attach-on-reload support yet (the Editor's
// /api/active flow) — a reload while a run is in flight loses the progress
// view, though the run itself still completes server-side.
(function () {
  const S = window.Store;
  const API = window.MovieEditorAPI;

  const LAST_PROJECT_KEY = "funpack_easy_last_project";

  const nameEl = document.getElementById("easy-project-name");
  const healthEl = document.getElementById("easy-health-chip");
  const promptEl = document.getElementById("easy-prompt");
  const uploadBtn = document.getElementById("easy-upload-btn");
  const uploadInput = document.getElementById("easy-upload-input");
  const generateBtn = document.getElementById("easy-generate-btn");
  const advancedBtn = document.getElementById("easy-advanced-btn");
  const projectsBtn = document.getElementById("easy-projects-btn");

  const preview = window.EasyPreview.create(document.getElementById("easy-preview"));

  let saveTimer = null;
  function scheduleSave() {
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(() => { S.save().catch(() => {}); }, 800);
  }

  let promptFocused = false;
  promptEl.addEventListener("focus", () => { promptFocused = true; });
  promptEl.addEventListener("blur", () => { promptFocused = false; });
  promptEl.addEventListener("input", () => {
    S.setPromptText(promptEl.value);
    scheduleSave();
  });

  let lastProjectId = null;
  function render(st) {
    const p = st.project;
    nameEl.textContent = p ? p.name : "No project";
    generateBtn.disabled = !p;
    uploadBtn.disabled = !p;
    if (p && !promptFocused) {
      const text = p.scenes[0]?.text || "";
      if (promptEl.value !== text) promptEl.value = text;
    }
    uploadBtn.textContent = p && p.scenes[0]?.source?.type !== "empty" ? "⬆ Uploaded ✕" : "⬆ Upload";
    const pid = p ? p.id : null;
    if (pid !== lastProjectId) {
      lastProjectId = pid;
      preview.stopPolling();
      preview.showEmpty();
      generateBtn.textContent = "Generate";
    }
  }
  S.subscribe(render);

  uploadBtn.onclick = () => {
    const st = S.get();
    const attached = st.project && st.project.scenes[0]?.source?.type !== "empty";
    if (attached) {
      S.setSceneMedia(null, null);
      scheduleSave();
      return;
    }
    uploadInput.click();
  };

  uploadInput.onchange = async () => {
    const file = uploadInput.files?.[0];
    uploadInput.value = "";
    if (!file) return;
    uploadBtn.disabled = true;
    uploadBtn.textContent = "Uploading…";
    try {
      const media = await API.uploadMedia(file);
      const kind = file.type.startsWith("video") ? "video" : "image";
      S.setSceneMedia(media.id, kind);
      await S.save();
    } catch (e) {
      alert("Upload failed: " + (e.message || e));
    } finally {
      render(S.get());
    }
  };

  generateBtn.onclick = async () => {
    const st = S.get();
    if (!st.project) return;
    if (!(st.project.scenes[0]?.text || "").trim()) {
      alert("Type a prompt first.");
      return;
    }
    generateBtn.disabled = true;
    generateBtn.textContent = "Generating…";
    try {
      await S.save();
      const resetSession = S.takeResetSessionFlag();
      const res = await API.generate(st.project.id, null, null, resetSession, null);
      preview.watch(st.project.id, res.prompt_id, {
        onDone: () => { generateBtn.disabled = false; generateBtn.textContent = "Generate"; },
      });
    } catch (e) {
      preview.showError("Generate failed: " + (e.message || e));
      generateBtn.disabled = false;
      generateBtn.textContent = "Generate";
    }
  };

  advancedBtn.onclick = () => window.SettingsWindow.open("generation");
  projectsBtn.onclick = () => window.ProjectMenu.open({ dismissable: !!S.get().project });

  async function boot() {
    S.refreshHealth();
    setInterval(() => S.refreshHealth(), 15000);
    S.subscribe((st) => {
      healthEl.textContent = st.health?.ok ? "● online" : "● offline";
      healthEl.classList.toggle("bad", !(st.health && st.health.ok));
      if (st.project) localStorage.setItem(LAST_PROJECT_KEY, st.project.id);
    });

    const lastId = localStorage.getItem(LAST_PROJECT_KEY);
    if (lastId) {
      try { await S.loadProject(lastId); return; } catch (_) { /* fall through to picker */ }
    }
    window.ProjectMenu.open({ dismissable: false });
  }

  boot();
})();
