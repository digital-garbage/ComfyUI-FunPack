// Easy Gen entrypoint: wires the big-preview / prompt / upload / generate bar
// to the shared easy_store.js + preview.js, and opens the "FunPack Easy Gen"
// project picker on boot. Reloading during a run re-attaches to it via the
// Editor's /api/active flow — see reattachRunning() below.
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
  const exportBtn = document.getElementById("easy-export-btn");

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

  // True whenever a run of ours is in flight — a fresh Generate OR one re-attached after a
  // reload. render() runs on every store notify (health poll, keystroke), so without this
  // it would re-enable Generate underneath a running generation.
  let generating = false;
  function setGenerating(on) {
    generating = on;
    generateBtn.disabled = on || !S.get().project;
    generateBtn.textContent = on ? "Generating…" : "Generate";
  }

  let lastProjectId = null;
  function render(st) {
    const p = st.project;
    nameEl.textContent = p ? p.name : "No project";
    generateBtn.disabled = generating || !p;
    uploadBtn.disabled = !p;
    galleryBtn.disabled = !p;
    saveBtn.disabled = !p;
    exportBtn.disabled = !p;
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
      // Not while a run is in flight: the generation is still going server-side, and
      // tearing down its poll here would orphan it exactly like a reload used to.
      if (!generating) {
        preview.stopPolling();
        preview.showEmpty();
        generateBtn.textContent = "Generate";
      }
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
    setGenerating(true);
    try {
      await S.save();
      const resetSession = S.takeResetSessionFlag();
      const res = await API.generate(st.project.id, null, null, resetSession, EASY_MODE_OVERRIDES);
      preview.watch(st.project.id, res.prompt_id, {
        onDone: () => setGenerating(false),
      });
    } catch (e) {
      preview.showError("Generate failed: " + (e.message || e));
      setGenerating(false);
    }
  };

  advancedBtn.onclick = () => window.SettingsWindow.open("generation");
  projectsBtn.onclick = () => window.ProjectMenu.open({ dismissable: !!S.get().project });
  galleryBtn.onclick = () => window.EasyGallery.open();

  // "Save" persists into FunPack's own project store on THIS machine — the same mechanism
  // as the Editor's autosave. It doesn't help moving a project off a rental GPU box. This
  // downloads the actual .funpack_project.json file (re-importable via "Load Project
  // File…" on another machine), same as the Editor's File ▸ "Save Project File…".
  exportBtn.onclick = async () => {
    const st = S.get();
    if (!st.project) return;
    try { await S.save(); } catch (_) { /* export the last-saved version anyway */ }
    const a = document.createElement("a");
    a.href = API.downloadProjectUrl(st.project.id);
    a.download = "";
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
  };

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

  // Re-attach to a generation that was already running when the page (re)loaded. Easy Gen
  // queues through the Editor's own /generate route, so its prompts are already stamped
  // with the editor client_id + run identity and come back from /api/active — nothing was
  // missing server-side, Easy Gen simply never asked. Same shape as the Cutting Room's
  // store.js resumeRunningGeneration, minus the per-scene recording it has no UI for.
  async function reattachRunning() {
    let act = null;
    try { act = await API.active(); } catch (_) { return; }
    let last = null;
    // Loop: a montage queued from the Cutting Room can have further runs behind the one
    // that happens to be executing now, and Generate should stay blocked until they drain.
    while (act && act.running && act.prompt_id && act.prompt_id !== last) {
      last = act.prompt_id;
      setGenerating(true);
      const st = S.get();
      const queued = act.pending > 0 ? ` · ${act.pending} more queued` : "";
      if (st.project && act.pid && st.project.id === act.pid) {
        // Ours, and the project it targets is loaded — full re-attach, result included.
        await new Promise((resolve) => {
          preview.watch(act.pid, act.prompt_id, {
            onDone: () => resolve(),
            note: `Reconnected after reload${queued}`,
          });
        });
      } else {
        // Started against a different project (typically from the Cutting Room). Do NOT
        // switch projects out from under the user to claim it — just watch the queue so
        // Generate stays blocked and Interrupt is offered while the GPU is busy.
        await preview.monitor(
          act.prompt_id, `A generation is running elsewhere${queued}`);
      }
      try { act = await API.active(); } catch (_) { act = null; }
    }
    setGenerating(false);
  }

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
    let loaded = false;
    if (lastId) {
      try { await S.loadProject(lastId); loaded = true; } catch (_) { /* fall through to picker */ }
    }
    if (!loaded) window.ProjectMenu.open({ dismissable: false });
    // After the project is in place, so a run targeting it re-attaches fully (result and
    // all) instead of falling back to the queue-watch path. Not awaited — it runs for as
    // long as the generation does.
    reattachRunning();
  }

  boot();
})();
