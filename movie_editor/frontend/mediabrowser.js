// Left zone: projects + tabbed library bins — Media, Shortcuts, Transitions.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const API = window.MovieEditorAPI;
  const body = document.getElementById("media-body");

  let tab = "Media";                 // active bin
  let q = { Shortcuts: "", Transitions: "" };

  // ── projects ───────────────────────────────────────────────────────────────────
  function projectsSection(st) {
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
    return sec;
  }

  // ── media bin ──────────────────────────────────────────────────────────────────
  function mediaTab(st) {
    const wrap = el("div", "bin");
    const drop = el("div", "mediabin");
    drop.append(el("div", "big", "🎞"));
    drop.append(el("div", null, "Drop images & clips here"));
    drop.append(el("div", "pj-meta", "or click to browse · drag onto a clip to set its source"));
    const file = el("input"); file.type = "file"; file.accept = "image/*,video/*"; file.multiple = true; file.style.display = "none";
    file.onchange = () => { if (file.files.length) S.uploadMedia([...file.files]); file.value = ""; };
    drop.onclick = () => file.click();
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
    drop.addEventListener("drop", (e) => { const fs = [...(e.dataTransfer?.files || [])]; if (fs.length) S.uploadMedia(fs); });
    wrap.append(drop); wrap.append(file);

    const grid = el("div", "media-grid");
    (st.mediaBin || []).forEach((m) => {
      const card = el("div", "media-card");
      card.draggable = true;
      card.addEventListener("dragstart", (e) => { e.dataTransfer.setData("application/funpack-media", m.id); e.dataTransfer.effectAllowed = "copy"; });
      card.title = `${m.name}\nClick to assign to selected scene · drag onto a clip`;
      const thumb = el("div", "media-thumb");
      if (m.kind === "image") { const img = el("img"); img.src = API.mediaUrl(m.id); img.loading = "lazy"; thumb.append(img); }
      else thumb.append(el("span", "media-icon", m.kind === "video" ? "▶" : "◆"));
      card.append(thumb);
      card.append(el("div", "media-name", m.name));
      const del = el("button", "media-del", "✕"); del.title = "Delete asset";
      del.onclick = (e) => { e.stopPropagation(); if (confirm(`Delete "${m.name}"?`)) S.deleteMedia(m.id); };
      card.append(del);
      card.onclick = () => {
        if (!st.selectedSceneId) { alert("Select a scene first to assign this asset."); return; }
        S.assignMediaToScene(st.selectedSceneId, m.id);
      };
      grid.append(card);
    });
    if (!(st.mediaBin || []).length) grid.append(el("div", "pj-meta", "No media yet."));
    wrap.append(grid);
    return wrap;
  }

  // ── shortcuts bin ────────────────────────────────────────────────────────────────
  function shortcutsTab(st) {
    const wrap = el("div", "bin");
    wrap.append(searchRow("Shortcuts", "Filter shortcuts…", () => render(S.get())));
    const toolbar = el("div", "bin-toolbar");
    const addBtn = el("button", "btn ghost tiny", "＋ Add");
    addBtn.onclick = async () => {
      const name = prompt("Shortcut name / activation phrase:"); if (!name) return;
      const repl = prompt(`Replacement phrase for "${name}":`, ""); if (repl == null) return;
      await S.saveShortcut({ name, triggers: [name], replacements: [repl] });
    };
    const expBtn = el("a", "btn ghost tiny", "↓ Export");
    expBtn.href = API.exportShortcutsUrl(); expBtn.download = "funpack_shortcuts.json"; expBtn.title = "Download shortcuts as JSON";
    const impFile = el("input"); impFile.type = "file"; impFile.accept = ".json"; impFile.style.display = "none";
    impFile.onchange = async () => {
      if (!impFile.files[0]) return;
      const n = await S.importShortcuts(impFile.files[0]); impFile.value = "";
      if (n != null) alert(`Imported ${n} shortcut(s).`);
    };
    const impBtn = el("button", "btn ghost tiny", "↑ Import"); impBtn.title = "Import shortcuts from JSON";
    impBtn.onclick = () => impFile.click();
    toolbar.append(addBtn); toolbar.append(expBtn); toolbar.append(impBtn); toolbar.append(impFile);
    wrap.append(toolbar);

    const list = el("div", "lib-list");
    const items = filtered(st.shortcuts || [], q.Shortcuts, (s) => `${s.name} ${(s.triggers || []).join(" ")} ${(s.replacements || []).join(" ")}`);
    items.forEach((s) => {
      const trig = (s.triggers || [])[0] || s.name;
      const row = el("div", "lib-row");
      const main = el("div", "lib-main");
      main.append(el("div", "lib-name", s.name));
      const rep = (s.replacements || []).join(" / ");
      if (rep) main.append(el("div", "lib-sub", "→ " + rep));
      row.append(main);
      const ins = el("button", "btn ghost tiny", "insert"); ins.title = "Append to selected scene's prompt";
      ins.onclick = () => { if (!S.insertShortcutIntoSelection(trig)) alert("Select a scene first."); };
      const del = el("button", "ic-btn danger", "✕"); del.title = "Delete shortcut";
      del.onclick = () => { if (confirm(`Delete shortcut "${s.name}"?`)) S.deleteShortcut(s.name); };
      row.append(ins); row.append(del);
      list.append(row);
    });
    if (!items.length) list.append(el("div", "pj-meta", (st.shortcuts || []).length ? "No match." : "No shortcuts yet."));
    wrap.append(list);
    return wrap;
  }

  // ── transitions bin ──────────────────────────────────────────────────────────────
  function transitionsTab(st) {
    const wrap = el("div", "bin");
    wrap.append(searchRow("Transitions", "Filter transitions…", () => render(S.get())));
    const toolbar = el("div", "bin-toolbar");
    const addBtn = el("button", "btn ghost tiny", "＋ Add");
    addBtn.onclick = async () => {
      const name = prompt("Transition trigger phrase (e.g. 'hard cut'):"); if (!name) return;
      const fx = prompt("Visual effect (none, crossfade, fade_to_black, …):", "none") || "none";
      await S.saveTransition({ name, trigger: name, visual_effect: fx });
    };
    const expBtn = el("a", "btn ghost tiny", "↓ Export");
    expBtn.href = API.exportTransitionsUrl(); expBtn.download = "funpack_transitions.json"; expBtn.title = "Download transitions as JSON";
    const impFile = el("input"); impFile.type = "file"; impFile.accept = ".json"; impFile.style.display = "none";
    impFile.onchange = async () => {
      if (!impFile.files[0]) return;
      const n = await S.importTransitions(impFile.files[0]); impFile.value = "";
      if (n != null) alert(`Imported ${n} transition(s).`);
    };
    const impBtn = el("button", "btn ghost tiny", "↑ Import"); impBtn.title = "Import transitions from JSON";
    impBtn.onclick = () => impFile.click();
    toolbar.append(addBtn); toolbar.append(expBtn); toolbar.append(impBtn); toolbar.append(impFile);
    wrap.append(toolbar);

    const list = el("div", "lib-list");
    const items = filtered(st.transitions || [], q.Transitions, (t) => `${t.name || ""} ${t.trigger || ""} ${t.visual_effect || ""}`);
    items.forEach((t) => {
      const trig = t.trigger || t.name || t.key;
      const row = el("div", "lib-row");
      const main = el("div", "lib-main");
      main.append(el("div", "lib-name", t.name || trig));
      const sub = [t.trigger && t.trigger !== (t.name || "") ? `"${t.trigger}"` : "", t.visual_effect && t.visual_effect !== "none" ? t.visual_effect : ""].filter(Boolean).join(" · ");
      if (sub) main.append(el("div", "lib-sub", sub));
      row.append(main);
      const apply = el("button", "btn ghost tiny", "apply"); apply.title = "Set as the selected scene's transition";
      apply.onclick = () => { if (!S.applyTransitionToSelection(trig)) alert("Select a scene first."); };
      const del = el("button", "ic-btn danger", "✕"); del.title = "Delete transition";
      del.onclick = () => { if (confirm(`Delete transition "${t.name || trig}"?`)) S.deleteTransition(t.name || trig); };
      row.append(apply); row.append(del);
      list.append(row);
    });
    if (!items.length) list.append(el("div", "pj-meta", (st.transitions || []).length ? "No match." : "No transitions yet."));
    wrap.append(list);
    return wrap;
  }

  // ── helpers ──────────────────────────────────────────────────────────────────────
  function filtered(arr, query, textOf) {
    const s = (query || "").trim().toLowerCase();
    return s ? arr.filter((x) => textOf(x).toLowerCase().includes(s)) : arr;
  }
  function searchRow(key, placeholder, onInput) {
    const inp = el("input", "lib-search"); inp.type = "text"; inp.placeholder = placeholder; inp.value = q[key] || "";
    inp.oninput = () => { q[key] = inp.value; onInput(); };
    return inp;
  }

  function render(st) {
    clear(body);
    body.append(projectsSection(st));

    const sec = el("div", "mb-section");
    const tabs = el("div", "bin-tabs");
    ["Media", "Shortcuts", "Transitions"].forEach((name) => {
      const b = el("button", "bin-tab" + (tab === name ? " active" : ""), name);
      b.onclick = () => { tab = name; render(S.get()); };
      tabs.append(b);
    });
    sec.append(tabs);
    sec.append(tab === "Media" ? mediaTab(st) : tab === "Shortcuts" ? shortcutsTab(st) : transitionsTab(st));
    body.append(sec);
  }

  S.subscribe(render);
})();
