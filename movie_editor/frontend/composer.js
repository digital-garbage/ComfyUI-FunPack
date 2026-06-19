// Composer: the home for prompt-craft tools — Compose (global prompt), Shortcuts, Split markers.
// Lives in a draggable FloatingWindow, toggled by the "Composer" button next to the panel tabs.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const API = window.MovieEditorAPI;

  let win = null;                 // FloatingWindow instance (lazy)
  let tab = "Compose";
  let q = { Shortcuts: "", Splits: "" };
  let gpDraft = null;             // in-progress global-prompt text (Compose tab), or null
  let _modal = null;

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

  // ── compose (global prompt) ─────────────────────────────────────────────────────
  // The whole montage as one editable prompt. Edits debounce into the timeline (which
  // re-splits into scenes); external timeline edits flow back unless we're mid-type.
  function composeTab(st) {
    const wrap = el("div", "bin compose-bin");
    const pv = st.preview;
    const live = st.project.global_prompt
      || (pv && (pv.display_prompt != null ? pv.display_prompt : pv.combined_prompt)) || "";
    const val = gpDraft != null ? gpDraft : live;

    wrap.append(el("div", "lib-form-title", "Global prompt"));
    const ta = el("textarea", "lib-in compose-ta"); ta.rows = 14; ta.value = val;
    ta.placeholder = "Anchor, scene texts, and split markers — one combined montage prompt for generation.";
    ta.oninput = () => { gpDraft = ta.value; S.scheduleGlobalPromptApply(ta.value); };
    wrap.append(ta);
    wrap.append(el("div", "insp-hint",
      "Edits apply automatically and stay in sync with the timeline's per-scene prompts. Shortcuts expand at generation time."));
    composeTextarea = ta;
    return wrap;
  }
  let composeTextarea = null;

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

      // Grouping for the Composer (free-text; sub-category nests under category).
      const cat = el("input", "lib-in"); cat.placeholder = "e.g. Lighting"; cat.value = item.category || "";
      const sub = el("input", "lib-in"); sub.placeholder = "e.g. Golden hour"; sub.value = item.sub_category || "";
      const catRow = el("div", "fields-row");
      catRow.append(labeled("Category", cat), labeled("Sub-category", sub));
      box.append(catRow);

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
          refinement_key: refKey,
          category: cat.value.trim(), sub_category: sub.value.trim(),
          original_name: item.name || undefined,
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
    const expBtn = el("a", "btn ghost tiny", "↑ Export");
    expBtn.href = API.exportShortcutsUrl(); expBtn.download = "funpack_shortcuts.json"; expBtn.title = "Download shortcuts as JSON";
    const impFile = el("input"); impFile.type = "file"; impFile.accept = ".json"; impFile.style.display = "none";
    impFile.onchange = async () => {
      if (!impFile.files[0]) return;
      const n = await S.importShortcuts(impFile.files[0]); impFile.value = "";
      if (n != null) alert(`Imported ${n} shortcut(s).`);
    };
    const impBtn = el("button", "btn ghost tiny", "↓ Import"); impBtn.title = "Import shortcuts from JSON";
    impBtn.onclick = () => impFile.click();
    toolbar.append(addBtn, expBtn, impBtn, impFile); wrap.append(toolbar);

    const list = el("div", "lib-list");
    const items = filtered(st.shortcuts || [], q.Shortcuts, (s) => `${s.name} ${s.category || ""} ${s.sub_category || ""} ${(s.triggers || []).join(" ")} ${(s.replacements || []).join(" ")}`);
    items.forEach((s) => {
      const trig = (s.triggers || [])[0] || s.name;
      const row = el("div", "lib-row");
      const main = el("div", "lib-main");
      const nameLine = el("div", "lib-name", s.name + (s.enabled === false ? " (off)" : ""));
      const catLabel = [s.category, s.sub_category].filter(Boolean).join(" · ");
      if (catLabel) nameLine.append(el("span", "lib-cat-tag", catLabel));
      main.append(nameLine);
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
    const expBtn = el("a", "btn ghost tiny", "↑ Export");
    expBtn.href = API.exportTransitionsUrl(); expBtn.download = "funpack_promptsplit.json"; expBtn.title = "Download split markers as JSON";
    const impFile = el("input"); impFile.type = "file"; impFile.accept = ".json"; impFile.style.display = "none";
    impFile.onchange = async () => {
      if (!impFile.files[0]) return;
      const n = await S.importTransitions(impFile.files[0]); impFile.value = "";
      if (n != null) alert(`Imported ${n} split marker(s).`);
    };
    const impBtn = el("button", "btn ghost tiny", "↓ Import"); impBtn.title = "Import split markers from JSON";
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

  // ── window + tabs ────────────────────────────────────────────────────────────────
  const TABS = ["Compose", "Shortcuts", "Splits"];
  function render() {
    if (!win) return;
    const st = S.get();
    composeTextarea = null;
    const root = clear(win.body);
    const shell = el("div", "composer-shell");
    const tabs = el("div", "bin-tabs composer-tabs");
    TABS.forEach((name) => {
      const b = el("button", "bin-tab" + (tab === name ? " active" : ""), name);
      b.title = name === "Splits" ? "Split markers (generation prompt)"
        : name === "Compose" ? "Global prompt — the whole montage" : name;
      b.onclick = () => { if (tab === name) return; tab = name; closeModal(); render(); };
      tabs.append(b);
    });
    shell.append(tabs);
    const scroll = el("div", "composer-scroll");
    scroll.append(
      tab === "Compose" ? composeTab(st)
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

  // Re-render on relevant store changes (shortcuts / splits / project switch).
  let lastFp = null;
  S.subscribe((st) => {
    if (!isOpen()) return;
    const fp = JSON.stringify({
      s: st.shortcuts?.length, t: st.transitions?.length, pid: st.project?.id,
    });
    if (fp === lastFp) return;
    lastFp = fp;
    render();
  });

  // Keep the Compose textarea in sync with timeline-driven global-prompt changes,
  // but never clobber what the user is actively typing.
  window.addEventListener("funpack-global-prompt-updated", (e) => {
    if (tab !== "Compose" || !composeTextarea) return;
    if (document.activeElement === composeTextarea) return;
    gpDraft = null;
    composeTextarea.value = (e.detail && e.detail.text) || "";
  });

  window.Composer = { toggle, isOpen, registerButton };

  // Self-wire the header button (present in index.html before this script runs).
  const headerBtn = document.getElementById("composer-btn");
  if (headerBtn) registerButton(headerBtn);
})();
