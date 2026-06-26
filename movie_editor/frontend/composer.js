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
    overlay.style.zIndex = _stackZ(0);
    _modal = overlay;
  }

  // z-index helper: keep Composer modals above the floating window (~4000+); `bump`
  // stacks a secondary modal above the primary editor modal beneath it.
  function _stackZ(bump) {
    const winZ = win && win.root ? (parseInt(win.root.style.zIndex || "4000", 10) || 4000) : 4000;
    const base = _modal ? (parseInt(_modal.style.zIndex || winZ, 10) || winZ) : winZ;
    return base + 5 + (bump || 0);
  }

  // A small modal that STACKS above the current editor modal without closing it —
  // used for the quick "Add category / sub-category" prompts.
  function openStackModal(title, build) {
    const overlay = el("div", "modal-overlay");
    const box = el("div", "modal modal-mini");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", title));
    box.append(head);
    const content = el("div", "modal-content");
    const close = () => overlay.remove();
    build(content, close);
    box.append(content); overlay.append(box);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.body.append(overlay);
    overlay.style.zIndex = _stackZ(10);
    setTimeout(() => { const i = box.querySelector("input, select"); if (i) i.focus(); }, 0);
    return close;
  }

  function promptNewCategory(onAdded) {
    openStackModal("Add category", (content, close) => {
      const box = el("div", "lib-form lib-form-modal");
      const name = el("input", "lib-in"); name.placeholder = "Category name";
      box.append(labeled("Name", name));
      const actions = el("div", "lib-form-actions");
      const ok = el("button", "btn primary tiny", "OK");
      ok.onclick = async () => {
        const v = name.value.trim(); if (!v) { name.focus(); return; }
        if (await S.addCategory(v)) { close(); if (onAdded) onAdded(v); }
      };
      const cancel = el("button", "btn ghost tiny", "Cancel"); cancel.onclick = close;
      name.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); ok.click(); } };
      actions.append(ok, cancel); box.append(actions);
      content.append(box);
    });
  }

  function promptNewSubCategory(presetCategory, onAdded) {
    const cats = (S.get().shortcutCategories || []);
    if (!presetCategory && !cats.length) { alert("Add a category first."); return; }
    openStackModal("Add sub-category", (content, close) => {
      const box = el("div", "lib-form lib-form-modal");
      let parentSel = null;
      if (!presetCategory) {
        parentSel = selectFrom(cats.map((c) => c.name), cats[0] && cats[0].name);
        box.append(labeled("Category", parentSel));
      }
      const name = el("input", "lib-in"); name.placeholder = "Sub-category name";
      box.append(labeled("Name", name));
      const actions = el("div", "lib-form-actions");
      const ok = el("button", "btn primary tiny", "OK");
      ok.onclick = async () => {
        const parent = presetCategory || (parentSel && parentSel.value) || "";
        const v = name.value.trim();
        if (!parent) { alert("Pick a category first."); return; }
        if (!v) { name.focus(); return; }
        if (await S.addCategory(parent, v)) { close(); if (onAdded) onAdded(parent, v); }
      };
      const cancel = el("button", "btn ghost tiny", "Cancel"); cancel.onclick = close;
      name.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); ok.click(); } };
      actions.append(ok, cancel); box.append(actions);
      content.append(box);
    });
  }

  function optionEl(value, label, selected) {
    const o = el("option", null, label); o.value = value; if (selected) o.selected = true; return o;
  }

  // Ask whether an import should Merge into the current library or Replace it
  // wholesale. Calls onPick("merge" | "replace") — or never, if cancelled.
  function chooseImportMode(kind, onPick) {
    openStackModal(`Import ${kind}`, (content, close) => {
      const box = el("div", "lib-form lib-form-modal");
      box.append(el("div", "insp-hint",
        `Merge keeps your current ${kind} and adds the imported ones (a matching name overwrites). `
        + `Replace deletes every current ${kind} first, then loads the file.`));
      const actions = el("div", "lib-form-actions");
      const merge = el("button", "btn primary tiny", "Merge");
      merge.onclick = () => { close(); onPick("merge"); };
      const replace = el("button", "btn danger tiny", "Replace all");
      replace.onclick = () => { close(); onPick("replace"); };
      const cancel = el("button", "btn ghost tiny", "Cancel"); cancel.onclick = close;
      actions.append(merge, replace, cancel); box.append(actions);
      content.append(box);
    });
  }

  // A "＋ Add ▾" split button with a small dropdown of [label, action] entries.
  function addDropdown(entries) {
    const wrap = el("div", "composer-add");
    const btn = el("button", "btn ghost tiny", "＋ Add ▾");
    const panel = el("div", "composer-add-menu"); panel.hidden = true;
    entries.forEach(([label, fn]) => {
      const b = el("button", "composer-add-item", label);
      b.onclick = () => { panel.hidden = true; fn(); };
      panel.append(b);
    });
    btn.onclick = (e) => {
      e.stopPropagation();
      const opening = panel.hidden;
      panel.hidden = !opening;
      if (opening) {
        const off = (ev) => { if (!wrap.contains(ev.target)) { panel.hidden = true; document.removeEventListener("mousedown", off, true); } };
        setTimeout(() => document.addEventListener("mousedown", off, true), 0);
      }
    };
    wrap.append(btn, panel);
    return wrap;
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

    // Templates: pick one to apply (prompt + variables), Save to snapshot the current state.
    wrap.append(composeTemplatesBar());

    const titleRow = el("div", "compose-head");
    titleRow.append(el("div", "lib-form-title", "Global prompt"));
    const addSc = el("button", "btn ghost tiny", "＋ Add shortcut");
    addSc.title = "Insert a shortcut trigger — browse by category";
    addSc.onclick = () => openShortcutPicker((trig) => insertIntoCompose(trig));
    titleRow.append(addSc);
    wrap.append(titleRow);

    const ta = el("textarea", "lib-in compose-ta"); ta.rows = 14; ta.value = val;
    ta.placeholder = "Anchor, scene texts, and split markers — one combined montage prompt for generation.";
    ta.oninput = () => { gpDraft = ta.value; S.scheduleGlobalPromptApply(ta.value); updateVarHint(); };
    wrap.append(ta);
    wrap.append(el("div", "insp-hint",
      "Edits apply automatically and stay in sync with the timeline's per-scene prompts. Shortcuts expand at generation time."));
    composeTextarea = ta;
    if (window.ShortcutAutocomplete) window.ShortcutAutocomplete.attach(ta);

    // Variables ($name) — collapsible, same toggle pattern as Settings ▸ Built-in pipeline.
    wrap.append(composeVariables());
    return wrap;
  }
  let composeTextarea = null;
  let varsOpen = false;          // Variables panel expanded?
  let varHintEl = null;          // undeclared / cycle hint line under the prompt

  // ── templates bar (above the global prompt) ─────────────────────────────────────
  function composeTemplatesBar() {
    const bar = el("div", "compose-templates");
    const sel = el("select", "lib-in compose-tpl-select");
    const ph = new Option("Templates…", ""); ph.disabled = true; ph.selected = true; sel.append(ph);
    (S.promptTemplates() || []).forEach((t) => {
      const preview = String(t.prompt || "").replace(/\s+/g, " ").slice(0, 48);
      const o = new Option(`${t.name}${preview ? " — " + preview : ""}`, t.name);
      sel.append(o);
    });
    sel.title = "Apply a saved global prompt + its variables";
    sel.onchange = () => { const n = sel.value; if (n) S.applyPromptTemplate(n).then(render); };
    bar.append(sel);
    const save = el("button", "btn ghost tiny", "Save");
    save.title = "Save the current global prompt + variables as a template";
    save.onclick = () => {
      const name = (window.prompt("Template name:") || "").trim();
      if (!name) return;
      const txt = composeTextarea ? composeTextarea.value : (S.get().project.global_prompt || "");
      S.savePromptTemplate(name, txt);
      render();
    };
    bar.append(save);
    return bar;
  }

  // ── variables panel (below the global prompt) ───────────────────────────────────
  // The list IS the editor: editable name/value rows + Add/Remove. Resolution happens at
  // generation, after shortcuts + split (Studio), so nothing here touches the timeline.
  function composeVariables() {
    const box = el("div", "compose-vars");
    const toggle = el("button", "btn ghost tiny composer-var-toggle",
      (varsOpen ? "▾ " : "▸ ") + "Variables");
    toggle.title = "Project $name variables — substituted into the prompt at generation";
    toggle.onclick = () => { varsOpen = !varsOpen; render(); };
    box.append(toggle);

    varHintEl = el("div", "insp-hint compose-var-hint");
    box.append(varHintEl);

    if (varsOpen) {
      const list = el("div", "compose-var-list");
      // Local working copy; committed (persisted) on every edit, re-rendered only on add/remove.
      const vars = (S.projectVariables() || []).map((v) => ({
        name: String((v && v.name) || ""), value: String((v && v.value != null) ? v.value : ""),
      }));
      const commit = () => S.setProjectVariables(vars);
      vars.forEach((v, i) => {
        const row = el("div", "compose-var-row");
        row.append(el("span", "compose-var-dollar", "$"));
        const nm = el("input", "lib-in compose-var-name");
        nm.value = v.name; nm.placeholder = "name";
        nm.oninput = () => { v.name = nm.value.replace(/^\$+/, ""); commit(); updateVarHint(); };
        row.append(nm);
        row.append(el("span", "compose-var-eq", "="));
        const vv = el("input", "lib-in compose-var-val");
        vv.value = v.value; vv.placeholder = "value (may reference $other)";
        vv.oninput = () => { v.value = vv.value; commit(); updateVarHint(); };
        row.append(vv);
        const rm = el("button", "btn ghost tiny", "✕");
        rm.title = "Remove variable";
        rm.onclick = () => { vars.splice(i, 1); commit(); render(); };
        row.append(rm);
        list.append(row);
      });
      const add = el("button", "btn ghost tiny", "＋ Add variable");
      add.onclick = () => { vars.push({ name: "", value: "" }); commit(); render(); };
      list.append(add);
      box.append(list);
    }
    // Defer so composeTextarea is assigned before the first scan.
    setTimeout(updateVarHint, 0);
    return box;
  }

  // Detect $name cycles among the declared variables (the "yell"); mirrors the safe-degrade
  // guard in templates.resolve_variables but reports the loop instead of silently leaving it.
  function detectVarCycles() {
    const map = {};
    (S.projectVariables() || []).forEach((v) => {
      const k = String((v && v.name) || "").replace(/^\$+/, "").trim();
      if (k) map[k] = String((v && v.value) || "");
    });
    const refs = (n) => (String(map[n] || "").match(/\$([A-Za-z_][A-Za-z0-9_]*)/g) || [])
      .map((m) => m.slice(1)).filter((r) => r in map);
    const color = {}, path = [], cycles = [];
    const dfs = (n) => {
      color[n] = 1; path.push(n);
      for (const r of refs(n)) {
        if (color[r] === 1) cycles.push(path.slice(path.indexOf(r)).concat(r));
        else if (!color[r]) dfs(r);
      }
      path.pop(); color[n] = 2;
    };
    Object.keys(map).forEach((n) => { if (!color[n]) dfs(n); });
    return cycles;
  }

  // Live hint under the prompt: cycle warning takes priority, else list $vars used but not declared.
  function updateVarHint() {
    if (!varHintEl) return;
    const declared = new Set((S.projectVariables() || [])
      .map((v) => String((v && v.name) || "").replace(/^\$+/, "").trim()).filter(Boolean));
    const txt = composeTextarea ? composeTextarea.value : "";
    const used = new Set((txt.match(/\$([A-Za-z_][A-Za-z0-9_]*)/g) || []).map((m) => m.slice(1)));
    const undeclared = [...used].filter((n) => !declared.has(n));
    const cycles = detectVarCycles();
    let msg = "";
    if (cycles.length) {
      msg = "⚠ Variable loop: " + cycles[0].map((n) => "$" + n).join(" → ")
        + " — references itself. Rework it; it won't expand.";
    } else if (undeclared.length) {
      msg = "Referencing " + undeclared.map((n) => "$" + n).join(", ")
        + " — never declared, passed as plain text.";
    }
    varHintEl.textContent = msg;
    varHintEl.classList.toggle("compose-var-warn", !!cycles.length);
    varHintEl.style.display = msg ? "" : "none";
  }

  // Insert text at the compose textarea's caret (or append), then re-sync the timeline.
  function insertIntoCompose(text) {
    const ta = composeTextarea;
    if (!ta) return;
    const v = ta.value;
    let s = ta.selectionStart, e = ta.selectionEnd;
    if (s == null) { s = e = v.length; }
    // Pad with a space when butting up against existing words.
    const before = v.slice(0, s), after = v.slice(e);
    const lead = before && !/\s$/.test(before) ? " " : "";
    const trail = after && !/^\s/.test(after) ? " " : "";
    const ins = lead + text + trail;
    ta.value = before + ins + after;
    const caret = s + ins.length;
    ta.focus(); ta.setSelectionRange(caret, caret);
    gpDraft = ta.value;
    S.scheduleGlobalPromptApply(ta.value);
  }

  // ── shortcut picker (browse by Category → [Sub-category →] Shortcut) ────────────
  // Builds the grouped tree from the shortcut library + managed category list, then
  // drills down. Direct (no sub-category) shortcuts sit beside the sub-category folders.
  function buildShortcutTree() {
    const lc = (s) => String(s || "").toLowerCase();
    const cats = (S.get().shortcutCategories || []).map((c) => ({ name: c.name, subs: (c.sub_categories || []).slice() }));
    const find = (name) => cats.find((c) => lc(c.name) === lc(name));
    const tree = new Map();   // category name → { subs: Map<sub, [sc]>, direct: [sc] }
    const ensure = (name) => {
      if (!tree.has(name)) tree.set(name, { subs: new Map(), direct: [] });
      return tree.get(name);
    };
    cats.forEach((c) => { const n = ensure(c.name); c.subs.forEach((s) => n.subs.set(s, [])); });
    const UNCAT = "Uncategorized";
    (S.get().shortcuts || []).forEach((sc) => {
      if (sc.enabled === false) return;
      const cat = (sc.category || "").trim();
      if (!cat) { ensure(UNCAT).direct.push(sc); return; }
      const node = ensure(cat);
      const sub = (sc.sub_category || "").trim();
      if (!sub) { node.direct.push(sc); return; }
      // Match sub-category case-insensitively to its managed name, else create.
      const known = find(cat) && find(cat).subs.find((s) => lc(s) === lc(sub));
      const key = known || sub;
      if (!node.subs.has(key)) node.subs.set(key, []);
      node.subs.get(key).push(sc);
    });
    // Drop empty categories that have no shortcuts at all (managed-but-unused).
    for (const [name, node] of [...tree]) {
      const hasAny = node.direct.length || [...node.subs.values()].some((a) => a.length);
      if (!hasAny) tree.delete(name);
    }
    return tree;
  }

  function openShortcutPicker(onPick) {
    const tree = buildShortcutTree();
    if (!tree.size) { alert("No shortcuts yet. Add some in the Shortcuts tab."); return; }
    let path = [];   // [] | [category] | [category, sub]

    openModal("Add shortcut", (content, close) => {
      const box = el("div", "sc-picker");
      function pickShortcut(sc) {
        const trig = (sc.triggers || [])[0] || sc.name;
        if (trig) onPick(trig);
        close();
      }
      function shortcutRow(sc) {
        const r = el("button", "sc-pick-item");
        r.append(el("span", "sc-pick-name", (sc.triggers || [])[0] || sc.name));
        const rep = (sc.replacements || [])[0];
        if (rep) r.append(el("span", "sc-pick-prompt", rep));
        r.onclick = () => pickShortcut(sc);
        return r;
      }
      function paint() {
        clear(box);
        const crumb = el("div", "sc-crumb");
        const home = el("button", "sc-crumb-link", "Categories");
        home.onclick = () => { path = []; paint(); };
        crumb.append(home);
        path.forEach((p, i) => {
          crumb.append(el("span", "sc-crumb-sep", "›"));
          const link = el("button", "sc-crumb-link", p);
          link.onclick = () => { path = path.slice(0, i + 1); paint(); };
          crumb.append(link);
        });
        box.append(crumb);

        const list = el("div", "sc-pick-list");
        if (path.length === 0) {
          [...tree.keys()].forEach((name) => {
            const node = tree.get(name);
            const count = node.direct.length + [...node.subs.values()].reduce((n, a) => n + a.length, 0);
            const b = el("button", "sc-pick-folder");
            b.append(el("span", "sc-pick-name", name));
            b.append(el("span", "sc-pick-count", String(count)));
            b.onclick = () => { path = [name]; paint(); };
            list.append(b);
          });
        } else if (path.length === 1) {
          const node = tree.get(path[0]);
          [...node.subs.entries()].forEach(([sub, arr]) => {
            if (!arr.length) return;
            const b = el("button", "sc-pick-folder");
            b.append(el("span", "sc-pick-name", sub));
            b.append(el("span", "sc-pick-count", String(arr.length)));
            b.onclick = () => { path = [path[0], sub]; paint(); };
            list.append(b);
          });
          node.direct.forEach((sc) => list.append(shortcutRow(sc)));
        } else {
          const node = tree.get(path[0]);
          (node.subs.get(path[1]) || []).forEach((sc) => list.append(shortcutRow(sc)));
        }
        box.append(list);
      }
      paint();
      content.append(box);
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

      // Grouping for the Composer: managed category + nested sub-category, chosen from
      // dropdowns. Each dropdown offers the existing entries plus "＋ Add new…".
      const grouping = { category: item.category || "", sub_category: item.sub_category || "" };
      const catRow = el("div", "fields-row");
      box.append(catRow);
      const ADD = " add";
      function paintGrouping() {
        clear(catRow);
        const cats = S.get().shortcutCategories || [];
        const lc = (s) => String(s || "").toLowerCase();
        const catNames = cats.map((c) => c.name);
        if (grouping.category && !catNames.some((n) => lc(n) === lc(grouping.category))) catNames.push(grouping.category);

        const catSel = el("select", "lib-in");
        catSel.append(optionEl("", "— none —", !grouping.category));
        catNames.forEach((n) => catSel.append(optionEl(n, n, lc(n) === lc(grouping.category))));
        catSel.append(optionEl(ADD, "＋ Add new category…"));
        catSel.onchange = () => {
          if (catSel.value === ADD) { catSel.value = grouping.category; promptNewCategory((nm) => { grouping.category = nm; grouping.sub_category = ""; paintGrouping(); }); return; }
          grouping.category = catSel.value; grouping.sub_category = ""; paintGrouping();
        };
        catRow.append(labeled("Category", catSel));

        const entry = cats.find((c) => lc(c.name) === lc(grouping.category));
        const subs = entry ? entry.sub_categories.slice() : [];
        if (grouping.sub_category && !subs.some((s) => lc(s) === lc(grouping.sub_category))) subs.push(grouping.sub_category);
        const subSel = el("select", "lib-in");
        subSel.append(optionEl("", "— none —", !grouping.sub_category));
        subs.forEach((s) => subSel.append(optionEl(s, s, lc(s) === lc(grouping.sub_category))));
        subSel.append(optionEl(ADD, "＋ Add new sub-category…"));
        subSel.disabled = !grouping.category;
        subSel.onchange = () => {
          if (subSel.value === ADD) { subSel.value = grouping.sub_category; promptNewSubCategory(grouping.category, (p, s) => { grouping.category = p; grouping.sub_category = s; paintGrouping(); }); return; }
          grouping.sub_category = subSel.value;
        };
        catRow.append(labeled("Sub-category", subSel));
      }
      paintGrouping();

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
          category: grouping.category, sub_category: grouping.sub_category,
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
    const addBtn = addDropdown([
      ["Shortcut", () => openShortcutEditor({})],
      ["Category", () => promptNewCategory()],
      ["Sub-category", () => promptNewSubCategory("")],
    ]);
    const expBtn = el("a", "btn ghost tiny", "↑ Export");
    expBtn.href = API.exportShortcutsUrl(); expBtn.download = "funpack_shortcuts.json"; expBtn.title = "Download shortcuts as JSON";
    const impFile = el("input"); impFile.type = "file"; impFile.accept = ".json"; impFile.style.display = "none";
    impFile.onchange = () => {
      const file = impFile.files[0]; impFile.value = "";
      if (!file) return;
      chooseImportMode("shortcuts", async (mode) => {
        const n = await S.importShortcuts(file, mode);
        if (n != null) alert(`Imported ${n} shortcut(s)${mode === "replace" ? " (replaced existing)" : ""}.`);
      });
    };
    const impBtn = el("button", "btn ghost tiny", "↓ Import"); impBtn.title = "Import shortcuts from JSON";
    impBtn.onclick = () => impFile.click();
    const delAll = el("button", "btn danger tiny", "✕ Delete all");
    delAll.title = "Delete every shortcut and category";
    delAll.onclick = async () => {
      if (!(st.shortcuts || []).length) { alert("No shortcuts to delete."); return; }
      if (confirm("Delete ALL shortcuts and categories? This cannot be undone.")) await S.clearShortcuts();
    };
    toolbar.append(addBtn, expBtn, impBtn, delAll, impFile); wrap.append(toolbar);

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
    impFile.onchange = () => {
      const file = impFile.files[0]; impFile.value = "";
      if (!file) return;
      chooseImportMode("split markers", async (mode) => {
        const n = await S.importTransitions(file, mode);
        if (n != null) alert(`Imported ${n} split marker(s)${mode === "replace" ? " (replaced existing)" : ""}.`);
      });
    };
    const impBtn = el("button", "btn ghost tiny", "↓ Import"); impBtn.title = "Import split markers from JSON";
    impBtn.onclick = () => impFile.click();
    const delAll = el("button", "btn danger tiny", "✕ Delete all");
    delAll.title = "Delete every split marker";
    delAll.onclick = async () => {
      if (!(st.transitions || []).length) { alert("No split markers to delete."); return; }
      if (confirm("Delete ALL split markers? This cannot be undone.")) await S.clearTransitions();
    };
    toolbar.append(addBtn, expBtn, impBtn, delAll, impFile); wrap.append(toolbar);

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

  // ── files (FunPack on-disk file manager) ────────────────────────────────────────
  // A read/purge view over the files FunPack writes: the prompt library JSONs and the
  // refinement-key store (keys + sidecars + value/latent tensors). Lets the user audit
  // and delete them without leaving the editor. Fetched on demand, not held in the store.
  let filesData = null;
  let filesLoading = false;

  function fmtBytes(n) {
    n = Number(n) || 0;
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
    return (n / (1024 * 1024)).toFixed(1) + " MB";
  }

  async function loadFiles() {
    filesLoading = true;
    try { filesData = await API.listFiles(); }
    catch (e) { filesData = { groups: [], error: e.message }; }
    filesLoading = false;
    if (tab === "Files") render();
  }

  // After a file mutation that could touch the prompt-library JSONs, re-pull the
  // Shortcuts/Splits state so those tabs never show entries whose file is now gone.
  function refreshLibraryAfterFileChange() {
    if (S.loadShortcuts) S.loadShortcuts();
    if (S.loadTransitions) S.loadTransitions();
  }

  function filesTab() {
    const wrap = el("div", "bin");
    const toolbar = el("div", "bin-toolbar");
    const refresh = el("button", "btn ghost tiny", "↻ Refresh"); refresh.onclick = () => loadFiles();
    toolbar.append(refresh);
    wrap.append(toolbar);

    if (filesData == null) { if (!filesLoading) loadFiles(); wrap.append(el("div", "pj-meta", "Loading…")); return wrap; }
    if (filesData.error) wrap.append(el("div", "pj-meta", "Error: " + filesData.error));

    (filesData.groups || []).forEach((g) => {
      const sec = el("div", "files-group");
      const head = el("div", "files-group-head");
      head.append(el("div", "files-group-title", g.label));
      if (g.files.length) {
        const clr = el("button", "btn danger tiny", "✕ Delete all");
        clr.title = "Delete every file in this group";
        clr.onclick = async () => {
          if (!confirm(`Delete ALL ${g.files.length} file(s) under "${g.label}"? This cannot be undone.`)) return;
          try { filesData = await API.clearFiles(g.id); refreshLibraryAfterFileChange(); render(); }
          catch (e) { alert("Delete-all failed: " + e.message); }
        };
        head.append(clr);
      }
      sec.append(head);
      sec.append(el("div", "files-dir", g.dir));

      const list = el("div", "files-list");
      if (!g.files.length) {
        list.append(el("div", "pj-meta", "No files."));
      } else {
        g.files.forEach((f) => {
          const row = el("div", "files-row");
          const main = el("div", "files-main");
          main.append(el("div", "files-name", f.name));
          const meta = [fmtBytes(f.size), f.kind || ""].filter(Boolean).join(" · ");
          main.append(el("div", "files-meta", meta));
          row.append(main);
          const del = el("button", "ic-btn danger", "✕"); del.title = "Delete file";
          del.onclick = async () => {
            if (!confirm(`Delete "${f.name}"? This cannot be undone.`)) return;
            try { filesData = await API.deleteFile(g.id, f.name); refreshLibraryAfterFileChange(); render(); }
            catch (e) { alert("Delete failed: " + e.message); }
          };
          row.append(del); list.append(row);
        });
      }
      sec.append(list);
      wrap.append(sec);
    });
    return wrap;
  }

  // ── window + tabs ────────────────────────────────────────────────────────────────
  const TABS = ["Compose", "Shortcuts", "Splits", "Files"];
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
        : name === "Compose" ? "Global prompt — the whole montage"
          : name === "Files" ? "FunPack files on disk — audit & purge" : name;
      b.onclick = () => {
        if (tab === name) return;
        tab = name; closeModal();
        if (name === "Files") filesData = null;   // re-fetch fresh on every open
        render();
      };
      tabs.append(b);
    });
    shell.append(tabs);
    const scroll = el("div", "composer-scroll");
    scroll.append(
      tab === "Compose" ? composeTab(st)
        : tab === "Shortcuts" ? shortcutsTab(st)
          : tab === "Splits" ? splitMarkersTab(st)
            : filesTab(),
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
      c: (st.shortcutCategories || []).reduce((n, x) => n + 1 + (x.sub_categories?.length || 0), 0),
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
