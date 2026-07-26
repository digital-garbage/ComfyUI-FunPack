// Easy Gen's prompt `$name` variables — quick replacements for full phrases,
// e.g. $vid = "High quality, high fidelity realistic video, motion blur, cinematic fog".
//
// Opens as its own modal off the header's "$ Variables" button (same modal-overlay/.modal
// shell as easy_gallery.js / project_menu.js) — deliberately NOT docked into the main
// screen, which stays prompt + preview only.
//
// Storage is project.variables ([{name, value}]) — the SAME project field the Editor's
// Composer writes and the same one movie_editor/server.py already hands to the builder on
// every generate ("variables": list(target.variables or [])), so nothing backend-side is
// new: templates.resolve_variables substitutes them inside refine_v2 DEAD LAST, after
// shortcut expansion and after the transition split. That order is what makes them work
// INSIDE shortcuts (a shortcut whose replacement contains $vid still resolves) while a
// value containing a comma or a trigger word can never move a scene cut.
//
// Undefined names stay literal ($foo passes through as text) and a self-referencing chain
// is left literal rather than looping — the hint line warns about both before you burn a
// generation on it.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;

  const toggleBtn = document.getElementById("easy-vars-btn");
  const promptEl = document.getElementById("easy-prompt");
  if (!toggleBtn) return;

  let overlay = null;
  let listEl = null;
  let hintEl = null;

  function vars() {
    return (S.projectVariables() || []).map((v) => ({
      name: String((v && v.name) || ""),
      value: String((v && v.value != null) ? v.value : ""),
    }));
  }

  // Cycle detection over the declared variables — mirrors the safe-degrade guard in
  // templates.resolve_variables, but reports the loop instead of silently leaving it literal.
  function detectCycles() {
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

  // A cycle wins over the undeclared list. Only the visible prompt is scanned — a $var used
  // solely inside a shortcut's replacement text is legitimate and stays silent.
  function warning() {
    const cycles = detectCycles();
    if (cycles.length) {
      return {
        bad: true,
        msg: "⚠ Variable loop: " + cycles[0].map((n) => "$" + n).join(" → ")
          + " — references itself. Rework it; it won't expand.",
      };
    }
    const declared = new Set((S.projectVariables() || [])
      .map((v) => String((v && v.name) || "").replace(/^\$+/, "").trim()).filter(Boolean));
    const txt = promptEl ? promptEl.value : "";
    const used = new Set((txt.match(/\$([A-Za-z_][A-Za-z0-9_]*)/g) || []).map((m) => m.slice(1)));
    const undeclared = [...used].filter((n) => !declared.has(n));
    if (undeclared.length) {
      return {
        bad: false,
        msg: "Referencing " + undeclared.map((n) => "$" + n).join(", ")
          + " — never declared, passed through as plain text.",
      };
    }
    return { bad: false, msg: "" };
  }

  // Header button: enabled only with a project, carries the count, and goes amber on a loop
  // so a broken variable is visible without opening the window.
  function updateButton() {
    const st = S.get();
    toggleBtn.disabled = !st.project;
    const n = (S.projectVariables() || []).filter((v) => String((v && v.name) || "").trim()).length;
    toggleBtn.textContent = "$ Variables" + (n ? ` (${n})` : "");
    toggleBtn.classList.toggle("easy-vars-warn", !!warning().bad);
  }

  function updateHint() {
    if (!hintEl) return;
    const w = warning();
    hintEl.textContent = w.msg;
    hintEl.classList.toggle("easy-var-warn", !!w.bad);
    hintEl.style.display = w.msg ? "" : "none";
    updateButton();
  }

  // Grow a value textarea to fit its text (measured after a reset to "auto", or it can only
  // ever get taller). Deferred once on mount because scrollHeight reads 0 before layout.
  function autoGrow(ta) {
    const fit = () => {
      ta.style.height = "auto";
      // scrollHeight covers content + padding; under border-box the borders must be added
      // back or every line is clipped by ~2px and the textarea scrolls anyway.
      const cs = getComputedStyle(ta);
      const border = cs.boxSizing === "border-box"
        ? parseFloat(cs.borderTopWidth) + parseFloat(cs.borderBottomWidth)
        : -(parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom));
      ta.style.height = (ta.scrollHeight + (border || 0)) + "px";
    };
    fit();
    if (!ta.dataset.grown) { ta.dataset.grown = "1"; setTimeout(fit, 0); }
  }

  // Working copy for the open window. It is the source of truth for what's on screen —
  // a half-typed row (blank name AND value) is dropped by the store on persist, so
  // re-reading the project after every edit would delete the row being filled in.
  let rows = [];

  function renderList() {
    if (!listEl) return;
    clear(listEl);
    // Every keystroke commits to the project (debounced save); only add/remove re-renders,
    // so an input never loses focus mid-word.
    const commit = () => { S.setProjectVariables(rows); updateHint(); };

    rows.forEach((v, i) => {
      const row = el("div", "easy-var-row");
      row.append(el("span", "easy-var-dollar", "$"));
      const nm = el("input", "easy-var-in easy-var-name");
      nm.value = v.name; nm.placeholder = "name";
      nm.oninput = () => { v.name = nm.value.replace(/^\$+/, ""); commit(); };
      row.append(nm);
      row.append(el("span", "easy-var-eq", "="));
      // Values are usually long phrases, so this is a textarea that grows with its content —
      // a one-line input makes editing "High quality, high fidelity realistic video, …" a
      // horizontal-scrolling guessing game.
      const vv = el("textarea", "easy-var-in easy-var-val");
      vv.rows = 1;
      vv.value = v.value; vv.placeholder = "value (may reference $other)";
      vv.oninput = () => { v.value = vv.value; autoGrow(vv); commit(); };
      row.append(vv);
      autoGrow(vv);
      const rm = el("button", "btn ghost tiny", "✕");
      rm.type = "button";
      rm.title = "Remove variable";
      rm.onclick = () => { rows.splice(i, 1); S.setProjectVariables(rows); renderList(); updateHint(); };
      row.append(rm);
      listEl.append(row);
    });
    if (!rows.length) listEl.append(el("div", "easy-vars-empty", "No variables yet."));

    const add = el("button", "btn ghost tiny easy-var-add", "＋ Add variable");
    add.type = "button";
    add.onclick = () => { rows.push({ name: "", value: "" }); S.setProjectVariables(rows); renderList(); updateHint(); };
    listEl.append(add);
  }

  function close() {
    if (overlay) { overlay.remove(); overlay = null; }
    listEl = null; hintEl = null;
    updateButton();
  }

  function open() {
    if (!S.get().project) return;
    close();
    overlay = el("div", "modal-overlay");
    const box = el("div", "modal modal-wide");

    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", "Variables"));
    const x = el("button", "btn ghost tiny", "✕");
    x.onclick = close;
    head.append(el("div", "modal-head-right"), x);
    box.append(head);

    const content = el("div", "modal-content easy-vars");
    content.append(el("div", "easy-vars-hint",
      "Shorthand for full phrases. Type $name in the prompt (or inside a shortcut's replacement) "
      + "and it's swapped for the text below at generation — after shortcuts expand and after the "
      + "prompt splits into scenes, so a value can never move a scene cut."));
    hintEl = el("div", "easy-vars-hint easy-var-hint");
    content.append(hintEl);
    listEl = el("div", "easy-var-list");
    content.append(listEl);
    box.append(content);

    overlay.append(box);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.addEventListener("keydown", function onKey(e) {
      if (e.key === "Escape") { close(); document.removeEventListener("keydown", onKey); }
    });
    document.body.append(overlay);

    rows = vars();
    renderList();
    updateHint();
  }

  toggleBtn.onclick = () => (overlay ? close() : open());

  // A project switch replaces the whole variable set — rebuild the open window's rows.
  let renderedProjectId = null;
  S.subscribe((st) => {
    const pid = st.project ? st.project.id : null;
    if (pid !== renderedProjectId) {
      renderedProjectId = pid;
      if (overlay) { rows = vars(); renderList(); updateHint(); return; }
    }
    updateButton();
  });
  if (promptEl) promptEl.addEventListener("input", () => { if (overlay) updateHint(); else updateButton(); });

  updateButton();
  window.EasyVariables = { open, close };
})();
