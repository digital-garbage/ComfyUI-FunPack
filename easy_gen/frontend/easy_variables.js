// Easy Gen's prompt `$name` variables panel — quick replacements for full phrases,
// e.g. $vid = "High quality, high fidelity realistic video, motion blur, cinematic fog".
//
// Storage is project.variables ([{name, value}]) — the SAME project field the Editor
// writes and the same one movie_editor/server.py already hands to the builder on every
// generate ("variables": list(target.variables or [])), so nothing backend-side is new:
// templates.resolve_variables substitutes them inside refine_v2 DEAD LAST, after shortcut
// expansion and after the transition split. That order is what makes them work INSIDE
// shortcuts (a shortcut whose replacement contains $vid still resolves) while a value
// containing a comma or a trigger word can never move a scene cut.
//
// Undefined names stay literal ($foo passes through as text) and a self-referencing chain
// is left literal rather than looping — the hint line below warns about both before you burn
// a generation on it.
(function () {
  const { el } = window.dom;
  const S = window.Store;

  const panel = document.getElementById("easy-vars");
  const toggleBtn = document.getElementById("easy-vars-btn");
  const promptEl = document.getElementById("easy-prompt");
  if (!panel || !toggleBtn) return;

  let open = false;
  let hintEl = null;
  let renderedProjectId = null;

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

  // Live hint: a cycle wins over the undeclared list. Only the visible prompt is scanned —
  // a $var used solely inside a shortcut's replacement text is legitimate and stays silent.
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

  function updateChrome() {
    const st = S.get();
    toggleBtn.disabled = !st.project;
    const n = (S.projectVariables() || []).filter((v) => String((v && v.name) || "").trim()).length;
    toggleBtn.textContent = (open ? "▾ " : "▸ ") + "$ Variables" + (n ? ` (${n})` : "");
    const w = warning();
    toggleBtn.classList.toggle("easy-vars-warn", !!w.bad);
    if (hintEl) {
      hintEl.textContent = w.msg;
      hintEl.classList.toggle("easy-var-warn", !!w.bad);
      hintEl.style.display = w.msg ? "" : "none";
    }
  }

  function render() {
    panel.textContent = "";
    panel.hidden = !open;
    if (!open) { updateChrome(); return; }

    panel.append(el("div", "easy-vars-hint",
      "Shorthand for full phrases. Type $name in the prompt (or inside a shortcut's replacement) "
      + "and it's swapped for the text below at generation — after shortcuts expand and after the "
      + "prompt splits into scenes, so a value can never move a scene cut."));

    hintEl = el("div", "easy-vars-hint easy-var-hint");
    panel.append(hintEl);

    const list = el("div", "easy-var-list");
    // Local working copy: every keystroke commits it to the project (debounced save), and
    // only add/remove re-renders — so an input never loses focus mid-word.
    const rows = vars();
    const commit = () => { S.setProjectVariables(rows); updateChrome(); };

    rows.forEach((v, i) => {
      const row = el("div", "easy-var-row");
      row.append(el("span", "easy-var-dollar", "$"));
      const nm = el("input", "easy-var-in easy-var-name");
      nm.value = v.name; nm.placeholder = "name";
      nm.oninput = () => { v.name = nm.value.replace(/^\$+/, ""); commit(); };
      row.append(nm);
      row.append(el("span", "easy-var-eq", "="));
      const vv = el("input", "easy-var-in easy-var-val");
      vv.value = v.value; vv.placeholder = "value (may reference $other)";
      vv.oninput = () => { v.value = vv.value; commit(); };
      row.append(vv);
      const rm = el("button", "btn ghost tiny", "✕");
      rm.type = "button";
      rm.title = "Remove variable";
      rm.onclick = () => { rows.splice(i, 1); S.setProjectVariables(rows); render(); };
      row.append(rm);
      list.append(row);
    });
    if (!rows.length) list.append(el("div", "easy-vars-empty", "No variables yet."));

    const add = el("button", "btn ghost tiny easy-var-add", "＋ Add variable");
    add.type = "button";
    add.onclick = () => { rows.push({ name: "", value: "" }); S.setProjectVariables(rows); render(); };
    list.append(add);

    panel.append(list);
    updateChrome();
  }

  toggleBtn.onclick = () => {
    if (!S.get().project) return;
    open = !open;
    render();
  };

  // Rebuild only when the project actually changes (a fresh set of rows); every other
  // notify just refreshes the count/warning so in-progress typing is never interrupted.
  S.subscribe((st) => {
    const pid = st.project ? st.project.id : null;
    if (pid !== renderedProjectId) {
      renderedProjectId = pid;
      if (open) { render(); return; }
    }
    updateChrome();
  });
  if (promptEl) promptEl.addEventListener("input", updateChrome);

  updateChrome();
})();
