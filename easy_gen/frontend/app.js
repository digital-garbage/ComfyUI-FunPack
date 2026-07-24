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
  const galleryBtn = document.getElementById("easy-gallery-btn");
  const generateBtn = document.getElementById("easy-generate-btn");
  const advancedBtn = document.getElementById("easy-advanced-btn");
  const projectsBtn = document.getElementById("easy-projects-btn");
  const saveBtn = document.getElementById("easy-save-btn");

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
    S.scheduleGlobalPromptApply(promptEl.value);
  });
  // Self-contained: reads Store.getEditorSetting("autocomplete")/get().shortcuts
  // and no-ops until a project + shortcut library are loaded.
  window.ShortcutAutocomplete?.attach(promptEl);

  function removeUpload() {
    S.setSceneMedia(null, null);
    scheduleSave();
  }
  preview.setOnClearUpload(removeUpload);

  let lastProjectId = null;
  function render(st) {
    const p = st.project;
    nameEl.textContent = p ? p.name : "No project";
    generateBtn.disabled = !p;
    uploadBtn.disabled = !p;
    galleryBtn.disabled = !p;
    saveBtn.disabled = !p;
    if (p && !promptFocused) {
      const text = p.global_prompt || "";
      if (promptEl.value !== text) promptEl.value = text;
    }
    const source = p?.scenes[0]?.source;
    const attached = source && source.type !== "empty" && source.media_ref;
    uploadBtn.textContent = attached ? "⬆ Uploaded ✕" : "⬆ Upload";
    preview.setUpload(attached ? API.mediaUrl(source.media_ref) : null, source?.type === "v2v" ? "video" : "image");

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
      removeUpload();
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
      const res = await API.uploadMedia(file);
      const media = res.media?.[0]; // POST /media wraps the result: { media: [ {id, ...} ] }
      if (!media) throw new Error("Upload succeeded but returned no media entry.");
      const kind = file.type.startsWith("video") ? "video" : "image";
      S.setSceneMedia(media.id, kind);
      await S.save();
      S.loadMedia();
    } catch (e) {
      alert("Upload failed: " + (e.message || e));
    } finally {
      render(S.get());
    }
  };

  // Easy Gen has no rating UI, so Studio must never apply learned/rated steering here —
  // forced at the graph level (not just hidden in Engine settings) so a project edited in
  // the Cutting Room with these on still generates plainly when run from Easy Gen.
  // "studio"/"sampler" are the built-in pipeline's fixed graph keys (see backend/builder.py).
  const EASY_MODE_OVERRIDES = [
    { node: "studio", input: "mode", value: "Prompt only" },
    { node: "sampler", input: "embed_guidance", value: false },
    { node: "sampler", input: "score_slider", value: false },
    { node: "sampler", input: "output_guidance", value: false },
    { node: "sampler", input: "dynashift", value: false },
    { node: "sampler", input: "taste_nearest_prompt", value: false },
  ];

  generateBtn.onclick = async () => {
    const st = S.get();
    if (!st.project) return;
    if (!(st.project.global_prompt || "").trim()) {
      alert("Type a prompt first.");
      return;
    }
    generateBtn.disabled = true;
    generateBtn.textContent = "Generating…";
    try {
      await S.save();
      const resetSession = S.takeResetSessionFlag();
      const res = await API.generate(st.project.id, null, null, resetSession, EASY_MODE_OVERRIDES);
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
  galleryBtn.onclick = () => window.EasyGallery.open();

  let saveResetTimer = null;
  saveBtn.onclick = async () => {
    if (saveResetTimer) clearTimeout(saveResetTimer);
    saveBtn.disabled = true;
    saveBtn.textContent = "Saving…";
    try {
      await S.save();
      saveBtn.textContent = "Saved ✓";
    } catch (e) {
      saveBtn.textContent = "Save failed";
      alert("Save failed: " + (e.message || e));
    } finally {
      saveResetTimer = setTimeout(() => {
        saveBtn.textContent = "Save";
        saveBtn.disabled = !S.get().project;
      }, 1200);
    }
  };

  async function boot() {
    S.refreshHealth();
    S.loadLibraries();
    S.loadMedia();
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
