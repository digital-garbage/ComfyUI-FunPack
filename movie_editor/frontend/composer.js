// Composer: the single home for prompt-craft tools — Characters, Shortcuts, Split markers.
// Lives in a draggable FloatingWindow, toggled by the "Composer" button in the menu bar.
// (Moved out of the Media Browser so the left dock is just media; these belong together.)
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const API = window.MovieEditorAPI;

  let win = null;                 // FloatingWindow instance (lazy)
  let tab = "Characters";
  let q = { Characters: "", Shortcuts: "", Splits: "" };
  let editCharacter = null;       // character being edited ({} = new), or null
  let _modal = null;

  function rerender() { if (win && win.isOpen()) render(); }

  // ── generic editor modal (shortcut / split forms) ──────────────────────────────
  function closeModal() { if (_modal) { _modal.remove(); _modal = null; } }
  function openModal(title, build) {
    closeModal();
    const overlay = el("div", "modal-overlay");
    const box = el("div", "modal");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", title));
    const hr = el("div", "modal-head-right");
    const x = el("button", "btn ghost tiny", "✕"); x.onclick = closeModal;
    hr.append(x); head.append(hr); box.append(head);
    const content = el("div", "modal-content");
    build(content, closeModal);
    box.append(content); overlay.append(box);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) closeModal(); });
    document.body.append(overlay);
    _modal = overlay;
  }

  // ── shared field helpers ──────────────────────────────────────────────────────
  // Triggers: one-per-line OR comma-separated. Replacements: one-per-line ONLY (prose
  // phrases often contain commas; comma-splitting tore one phrase into bogus variants).
  const splitTriggers = (v) => String(v || "").split(/[\n,]+/).map((s) => s.trim()).filter(Boolean);
  const splitReplacements = (v) => String(v || "").split(/\n+/).map((s) => s.trim()).filter(Boolean);
  function labeled(label, ctrl) {
    const l = el("label", "lib-field"); l.append(el("span", null, label)); l.append(ctrl); return l;
  }
  function checkRow(label, checked) {
    const row = el("label", "chk"); const cb = el("input"); cb.type = "checkbox"; cb.checked = checked;
    row.append(cb); row.append(el("span", null, label)); row._cb = cb; return row;
  }
  function selectFrom(opts, value) {
    const sel = el("select", "lib-in");
    opts.forEach((o) => { const op = el("option", null, o); op.value = o; if (o === value) op.selected = true; sel.append(op); });
    return sel;
  }
  function filtered(arr, query, textOf) {
    const s = (query || "").trim().toLowerCase();
    return s ? arr.filter((x) => textOf(x).toLowerCase().includes(s)) : arr;
  }
  function searchRow(key, placeholder) {
    const inp = el("input", "lib-search"); inp.type = "text"; inp.placeholder = placeholder; inp.value = q[key] || "";
    inp.oninput = () => { q[key] = inp.value; render(); };
    return inp;
  }
  function mediaRefPicker(st, value, onChange, opts) {
    return window.MediaPicker.create({
      value, mediaBin: st.mediaBin, onChange, compact: true,
      startOpen: !!(opts && opts.startOpen),
    });
  }

  // ── shortcuts ─────────────────────────────────────────────────────────────────
  function openShortcutEditor(item) {
    const isNew = !item.name;
    openModal(isNew ? "New shortcut" : `Edit “${item.name}”`, (content, close) => {
      const box = el("div", "lib-form lib-form-modal");
      const name = el("input", "lib-in"); name.placeholder = "Name"; name.value = item.name || "";
      const trig = el("textarea", "lib-in"); trig.rows = 3; trig.placeholder = "Triggers — one per line or comma-separated"; trig.value = (item.triggers || []).join("\n");
      const reps = el("textarea", "lib-in"); reps.rows = 4; reps.placeholder = "Replacement(s) — one per line; multiple = random pick"; reps.value = (item.replacements || []).join("\n");
      const en = checkRow("Enabled", item.enabled !== false);
      box.append(labeled("Name", name), labeled("Triggers", trig), labeled("Replacements", reps), en);

      // Per-shortcut refinement key: firing this shortcut marks the named key as training.
      const existingKey = String(item.refinement_key || "").trim();
      const useKey = checkRow("Use non-default refinement key", !!existingKey);
      const keyField = labeled("Refinement key", (() => {
        const i = el("input", "lib-in"); i.placeholder = "key name"; i.value = existingKey; return i;
      })());
      const keyInput = keyField.querySelector("input");
      keyField.style.display = existingKey ? "" : "none";
      useKey._cb.onchange = () => { keyField.style.display = useKey._cb.checked ? "" : "none"; if (useKey._cb.checked) keyInput.focus(); };
      box.append(useKey, keyField);

      const actions = el("div", "lib-form-actions");
      const save = el("button", "btn primary tiny", "Save");
      save.onclick = async () => {
        const triggers = splitTriggers(trig.value);
        if (!triggers.length) { alert("At least one trigger is required."); return; }
        const refKey = useKey._cb.checked ? (keyInput.value || "").trim() : "";
        if (useKey._cb.checked && !refKey) { alert("Enter a refinement key name, or uncheck the box."); return; }
        await S.saveShortcut({
          name: name.value.trim() || triggers[0], triggers,
          replacements: splitReplacements(reps.value), enabled: en._cb.checked,
          refinement_key: refKey, original_name: item.name || undefined,
        });
        close(); render();
      };
      const cancel = el("button", "btn ghost tiny", "Cancel"); cancel.onclick = close;
      actions.append(save, cancel); box.append(actions);
      content.append(box);
    });
  }

  function shortcutsTab(st) {
    const wrap = el("div", "bin");
    wrap.append(searchRow("Shortcuts", "Filter shortcuts…"));
    const toolbar = el("div", "bin-toolbar");
    const addBtn = el("button", "btn ghost tiny", "＋ Add"); addBtn.onclick = () => openShortcutEditor({});
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
    toolbar.append(addBtn, expBtn, impBtn, impFile); wrap.append(toolbar);

    const list = el("div", "lib-list");
    const items = filtered(st.shortcuts || [], q.Shortcuts, (s) => `${s.name} ${(s.triggers || []).join(" ")} ${(s.replacements || []).join(" ")}`);
    items.forEach((s) => {
      const trig = (s.triggers || [])[0] || s.name;
      const row = el("div", "lib-row");
      const main = el("div", "lib-main");
      main.append(el("div", "lib-name", s.name + (s.enabled === false ? " (off)" : "")));
      const rep = (s.replacements || []).join(" / ");
      if (rep) main.append(el("div", "lib-sub", "→ " + rep));
      row.append(main);
      const ins = el("button", "btn ghost tiny", "insert"); ins.title = "Append to selected scene's prompt";
      ins.onclick = () => { if (!S.insertShortcutIntoSelection(trig)) alert("Select a scene first."); };
      const edit = el("button", "ic-btn", "✎"); edit.title = "Edit shortcut"; edit.onclick = () => openShortcutEditor(s);
      const del = el("button", "ic-btn danger", "✕"); del.title = "Delete shortcut";
      del.onclick = () => { if (confirm(`Delete shortcut "${s.name}"?`)) S.deleteShortcut(s.name); };
      row.append(ins, edit, del); list.append(row);
    });
    if (!items.length) list.append(el("div", "pj-meta", (st.shortcuts || []).length ? "No match." : "No shortcuts yet."));
    wrap.append(list);
    return wrap;
  }

  // ── split markers (generation prompt splits) ────────────────────────────────────
  const PLACEMENTS = ["global", "start", "end", "silent"];
  function openSplitMarkerEditor(item) {
    const isNew = !item.name && !item.trigger;
    openModal(isNew ? "New split marker" : `Edit “${item.name || item.trigger}”`, (content, close) => {
      const box = el("div", "lib-form lib-form-modal");
      const name = el("input", "lib-in"); name.placeholder = "Name"; name.value = item.name || "";
      const trig = el("input", "lib-in"); trig.placeholder = "Trigger phrase (what appears in the prompt)"; trig.value = item.trigger || "";
      const place = selectFrom(PLACEMENTS, item.placement || "global");
      const en = checkRow("Enabled", item.enabled !== false);
      box.append(labeled("Name", name), labeled("Trigger", trig), labeled("Placement", place), en);
      box.append(el("div", "insp-hint", "Splits the generation prompt only — not a video dissolve on the timeline."));
      const actions = el("div", "lib-form-actions");
      const save = el("button", "btn primary tiny", "Save");
      save.onclick = async () => {
        const trigger = trig.value.trim();
        if (!trigger) { alert("A trigger phrase is required."); return; }
        await S.saveTransition({
          name: name.value.trim() || trigger, trigger,
          placement: place.value, enabled: en._cb.checked,
          original_name: item.name || undefined,
        });
        close(); render();
      };
      const cancel = el("button", "btn ghost tiny", "Cancel"); cancel.onclick = close;
      actions.append(save, cancel); box.append(actions);
      content.append(box);
    });
  }

  function splitMarkersTab(st) {
    const wrap = el("div", "bin");
    wrap.append(searchRow("Splits", "Filter split markers…"));
    const toolbar = el("div", "bin-toolbar");
    const addBtn = el("button", "btn ghost tiny", "＋ Add"); addBtn.onclick = () => openSplitMarkerEditor({});
    const expBtn = el("a", "btn ghost tiny", "↓ Export");
    expBtn.href = API.exportTransitionsUrl(); expBtn.download = "funpack_promptsplit.json"; expBtn.title = "Download split markers as JSON";
    const impFile = el("input"); impFile.type = "file"; impFile.accept = ".json"; impFile.style.display = "none";
    impFile.onchange = async () => {
      if (!impFile.files[0]) return;
      const n = await S.importTransitions(impFile.files[0]); impFile.value = "";
      if (n != null) alert(`Imported ${n} split marker(s).`);
    };
    const impBtn = el("button", "btn ghost tiny", "↑ Import"); impBtn.title = "Import split markers from JSON";
    impBtn.onclick = () => impFile.click();
    toolbar.append(addBtn, expBtn, impBtn, impFile); wrap.append(toolbar);

    const list = el("div", "lib-list");
    const items = filtered(st.transitions || [], q.Splits, (t) => `${t.name || ""} ${t.trigger || ""} ${t.placement || ""}`);
    items.forEach((t) => {
      const trig = t.trigger || t.name || t.key;
      const row = el("div", "lib-row");
      const main = el("div", "lib-main");
      main.append(el("div", "lib-name", (t.name || trig) + (t.enabled === false ? " (off)" : "")));
      const sub = [t.trigger && t.trigger !== (t.name || "") ? `"${t.trigger}"` : "",
                   t.placement && t.placement !== "global" ? t.placement : ""].filter(Boolean).join(" · ");
      if (sub) main.append(el("div", "lib-sub", sub));
      row.append(main);
      const apply = el("button", "btn ghost tiny", "apply"); apply.title = "Set as split marker before the selected scene (generation prompt)";
      apply.onclick = () => { if (!S.applySplitMarkerToSelection(trig)) alert("Select a scene first."); };
      const edit = el("button", "ic-btn", "✎"); edit.title = "Edit split marker"; edit.onclick = () => openSplitMarkerEditor(t);
      const del = el("button", "ic-btn danger", "✕"); del.title = "Delete split marker";
      del.onclick = () => { if (confirm(`Delete split marker "${t.name || trig}"?`)) S.deleteTransition(t.name || trig); };
      row.append(apply, edit, del); list.append(row);
    });
    if (!items.length) list.append(el("div", "pj-meta", (st.transitions || []).length ? "No match." : "No split markers yet."));
    wrap.append(list);
    return wrap;
  }

  // ── characters ──────────────────────────────────────────────────────────────────
  function characterForm(st, item) {
    const isNew = !item.id;
    const box = el("div", "lib-form");
    box.append(el("div", "lib-form-title", isNew ? "New character" : `Edit “${item.name || "character"}”`));
    const name = el("input", "lib-in"); name.placeholder = "Name"; name.value = item.name || "";
    const appearance = el("textarea", "lib-in"); appearance.rows = 2; appearance.placeholder = "Face, hair, eyes, skin…"; appearance.value = item.appearance || "";
    const body = el("textarea", "lib-in"); body.rows = 2; body.placeholder = "Build, height, proportions…"; body.value = item.body || "";
    const wardrobe = el("textarea", "lib-in"); wardrobe.rows = 2; wardrobe.placeholder = "Default outfit or style…"; wardrobe.value = item.wardrobe || "";
    const always = el("textarea", "lib-in"); always.rows = 2; always.placeholder = "Tags always included in this character's prompt"; always.value = item.always_include || "";
    const never = el("textarea", "lib-in"); never.rows = 2; never.placeholder = "Appended to negative when this character is on a scene"; never.value = item.never_include || "";
    let faceRef = item.face_ref || null, bodyRef = item.body_ref || null, detailRef = item.detail_ref || null;
    box.append(
      labeled("Name", name), labeled("Appearance", appearance), labeled("Body", body),
      labeled("Wardrobe", wardrobe), labeled("Always include", always), labeled("Never include", never),
      labeled("Face ref (identity pin)", mediaRefPicker(st, faceRef, (v) => { faceRef = v; }, { startOpen: true })),
      labeled("Body ref", mediaRefPicker(st, bodyRef, (v) => { bodyRef = v; }, { startOpen: true })),
      labeled("Detail ref", mediaRefPicker(st, detailRef, (v) => { detailRef = v; }, { startOpen: true })),
    );
    const actions = el("div", "lib-form-actions");
    const save = el("button", "btn primary tiny", "Save");
    save.onclick = async () => {
      const nm = name.value.trim();
      if (!nm) { alert("Character name is required."); return; }
      await S.saveCharacter({
        id: item.id || undefined, original_id: item.id || undefined, name: nm,
        appearance: appearance.value, body: body.value, wardrobe: wardrobe.value,
        always_include: always.value, never_include: never.value,
        face_ref: faceRef, body_ref: bodyRef, detail_ref: detailRef,
      });
      editCharacter = null; render();
    };
    const cancel = el("button", "btn ghost tiny", "Cancel");
    cancel.onclick = () => { editCharacter = null; render(); };
    actions.append(save, cancel); box.append(actions);
    return box;
  }

  function charactersTab(st) {
    if (editCharacter) return characterForm(st, editCharacter);
    const wrap = el("div", "bin");
    wrap.append(searchRow("Characters", "Filter characters…"));
    const hint = el("div", "pj-meta char-hint");
    hint.textContent = st.selectedSceneId
      ? "Click a character to add/remove it on the selected scene."
      : "Select a scene, then click characters to assign them.";
    wrap.append(hint);
    const head = el("div", "lib-head");
    const add = el("button", "btn ghost tiny", "＋ New"); add.onclick = () => { editCharacter = {}; render(); };
    head.append(add); wrap.append(head);

    const assigned = st.selectedSceneId ? new Set(S.sceneCharacterIds(st.selectedSceneId)) : null;
    const list = el("div", "char-list");
    const items = filtered(st.characters || [], q.Characters, (c) =>
      `${c.name} ${c.appearance || ""} ${c.body || ""} ${c.wardrobe || ""}`);
    items.forEach((c) => {
      const onScene = assigned && assigned.has(c.id);
      const row = el("div", "char-row" + (onScene ? " on-scene" : ""));
      const prev = el("div", "char-thumb");
      if (c.face_ref) { const img = el("img"); img.src = API.mediaUrl(c.face_ref); img.loading = "lazy"; prev.append(img); }
      else prev.append(el("span", null, "◎"));
      row.append(prev);
      const info = el("div", "char-info");
      info.append(el("div", "char-name", c.name || "(unnamed)"));
      const sub = [c.appearance, c.wardrobe].filter(Boolean).join(" · ");
      if (sub) info.append(el("div", "char-sub", sub));
      row.append(info);
      if (onScene) row.append(el("span", "char-badge", "on scene"));
      row.onclick = () => {
        if (st.selectedSceneId) S.toggleSceneCharacter(st.selectedSceneId, c.id);
        else { editCharacter = { ...c }; render(); }
      };
      const edit = el("button", "btn ghost tiny", "✎"); edit.title = "Edit character";
      edit.onclick = (e) => { e.stopPropagation(); editCharacter = { ...c }; render(); };
      const del = el("button", "btn ghost tiny danger", "✕"); del.title = "Delete character";
      del.onclick = async (e) => { e.stopPropagation(); if (confirm(`Delete character “${c.name}”?`)) await S.deleteCharacter(c.id); };
      const actions = el("div", "char-actions"); actions.append(edit, del); row.append(actions);
      list.append(row);
    });
    if (!items.length) list.append(el("div", "pj-meta", (st.characters || []).length ? "No match." : "No characters yet — create one."));
    wrap.append(list);
    return wrap;
  }

  // ── window + tabs ────────────────────────────────────────────────────────────────
  const TABS = ["Characters", "Shortcuts", "Splits"];
  function render() {
    if (!win) return;
    const st = S.get();
    const root = clear(win.body);
    const shell = el("div", "composer-shell");
    const tabs = el("div", "bin-tabs composer-tabs");
    TABS.forEach((name) => {
      const b = el("button", "bin-tab" + (tab === name ? " active" : ""), name);
      b.title = name === "Splits" ? "Split markers (generation prompt)" : name;
      b.onclick = () => { if (tab === name) return; tab = name; editCharacter = null; closeModal(); render(); };
      tabs.append(b);
    });
    shell.append(tabs);
    const scroll = el("div", "composer-scroll");
    scroll.append(
      tab === "Characters" ? charactersTab(st)
        : tab === "Shortcuts" ? shortcutsTab(st)
          : splitMarkersTab(st),
    );
    shell.append(scroll);
    root.append(shell);
  }

  function ensureWin() {
    if (win) return win;
    win = window.FloatingWindow.create({
      id: "composer", title: "Composer", subtitle: "prompt craft",
      width: 520, height: 560, minWidth: 380, minHeight: 320,
      onOpen: render,
      onClose: () => { closeModal(); _syncButton(); },
    });
    return win;
  }

  function toggle() { ensureWin().toggle(); _syncButton(); }
  function isOpen() { return !!(win && win.isOpen()); }

  let _btn = null;
  function registerButton(btn) { _btn = btn; btn.onclick = toggle; _syncButton(); }
  function _syncButton() { if (_btn) _btn.classList.toggle("on", isOpen()); }

  // Re-render on relevant store changes (characters/shortcuts/splits/selection).
  let lastFp = null;
  S.subscribe((st) => {
    if (!isOpen()) return;
    const fp = JSON.stringify({
      c: st.characters?.length, s: st.shortcuts?.length, t: st.transitions?.length,
      sel: st.selectedSceneId, pid: st.project?.id,
    });
    if (fp === lastFp) return;
    lastFp = fp;
    render();
  });

  window.Composer = { toggle, isOpen, registerButton };

  // Self-wire the header button (present in index.html before this script runs).
  const headerBtn = document.getElementById("composer-btn");
  if (headerBtn) registerButton(headerBtn);
})();
