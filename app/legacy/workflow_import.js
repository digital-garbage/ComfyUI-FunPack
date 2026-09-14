// ComfyUI Workflow Import wizard: parse a workflow, map editor settings to nodes, apply.
(function () {
  const { el, clear } = window.dom;
  const API = window.MovieEditorAPI;

  let overlay = null;
  let step = 0;
  let workflowRaw = null;
  let parsed = null;
  let bindings = {};

  const STEPS = ["Load workflow", "Review nodes", "Link editor inputs", "Apply"];

  function closeModal() {
    if (overlay) overlay.remove();
    overlay = null;
    step = 0;
    workflowRaw = null;
    parsed = null;
    bindings = {};
  }

  function pid() {
    return window.Store?.get().project?.id || null;
  }

  async function loadFile(file) {
    const text = await file.text();
    workflowRaw = JSON.parse(text);
    await parseWorkflow();
  }

  async function parseWorkflow() {
    parsed = await API.parseWorkflow(workflowRaw);
    bindings = {};
    (parsed.bindings || []).forEach((b) => {
      bindings[b.key] = (parsed.suggestions || {})[b.key] || "";
    });
  }

  function stepDots() {
    const row = el("div", "wf-steps");
    STEPS.forEach((label, i) => {
      const dot = el("span", "wf-step" + (i === step ? " active" : i < step ? " done" : ""));
      dot.textContent = `${i + 1}. ${label}`;
      row.append(dot);
    });
    return row;
  }

  function stepLoad() {
    const sec = el("div", "wf-panel");
    sec.append(el("p", "wf-lead", "Import a ComfyUI workflow JSON (UI export or API format). The built-in FunPack pipeline will be disabled; your nodes run instead."));
    const drop = el("div", "wf-drop");
    drop.append(el("div", null, "Drop a .json file here or click to browse"));
    const inp = el("input");
    inp.type = "file";
    inp.accept = ".json,application/json";
    inp.style.display = "none";
    drop.onclick = () => inp.click();
    inp.onchange = async () => {
      if (!inp.files[0]) return;
      try {
        await loadFile(inp.files[0]);
        step = 1;
        render();
      } catch (e) {
        alert("Could not read workflow: " + (e.message || e));
      }
      inp.value = "";
    };
    drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("hover"); };
    drop.ondragleave = () => drop.classList.remove("hover");
    drop.ondrop = async (e) => {
      e.preventDefault();
      drop.classList.remove("hover");
      const f = e.dataTransfer?.files?.[0];
      if (!f) return;
      try {
        await loadFile(f);
        step = 1;
        render();
      } catch (err) {
        alert("Could not read workflow: " + (err.message || err));
      }
    };
    sec.append(drop);
    sec.append(inp);

    const pasteLabel = el("label", "field");
    pasteLabel.append(el("span", null, "Or paste workflow JSON"));
    const ta = el("textarea", "wf-paste");
    ta.rows = 8;
    ta.placeholder = '{ "nodes": [...], "links": [...] }';
    pasteLabel.append(ta);
    sec.append(pasteLabel);

    const parseBtn = el("button", "btn primary", "Parse pasted JSON");
    parseBtn.onclick = async () => {
      try {
        workflowRaw = JSON.parse(ta.value.trim());
        await parseWorkflow();
        step = 1;
        render();
      } catch (e) {
        alert("Invalid JSON: " + (e.message || e));
      }
    };
    sec.append(parseBtn);
    return sec;
  }

  function stepReview() {
    const sec = el("div", "wf-panel");
    sec.append(el("h3", "wf-h3", parsed.name || "Workflow"));
    const meta = el("div", "wf-meta");
    meta.textContent = `${parsed.node_count} nodes · ${parsed.link_count} internal links · ${parsed.format} format`;
    sec.append(meta);
    (parsed.warnings || []).forEach((w) => sec.append(el("div", "wf-warn", w)));

    const list = el("div", "wf-node-list");
    (parsed.slots || []).slice(0, 80).forEach((s) => {
      const row = el("div", "wf-node-row");
      row.append(el("span", "wf-node-cls", s.node_class));
      row.append(el("span", "wf-node-lbl", s.label || s.id));
      list.append(row);
    });
    if ((parsed.slots || []).length > 80) {
      list.append(el("div", "wf-meta", `… and ${parsed.slots.length - 80} more`));
    }
    sec.append(list);

    if ((parsed.links || []).length) {
      sec.append(el("h4", "wf-h4", "Internal links (preserved)"));
      const lbox = el("div", "wf-link-list");
      parsed.links.slice(0, 40).forEach((l) => {
        lbox.append(el("div", "wf-link-row", `${l.from} → ${l.to}`));
      });
      if (parsed.links.length > 40) lbox.append(el("div", "wf-meta", `… ${parsed.links.length - 40} more`));
      sec.append(lbox);
    }

    const nav = el("div", "wf-nav");
    const back1 = el("button", "btn ghost", "Back");
    back1.onclick = () => { step = 0; render(); };
    const next1 = el("button", "btn primary", "Next: link inputs");
    next1.onclick = () => { step = 2; render(); };
    nav.append(back1);
    nav.append(next1);
    sec.append(nav);
    return sec;
  }

  function bindingRow(b) {
    const row = el("div", "wf-bind-row");
    row.append(el("span", "wf-bind-label", b.label));
    const sel = el("select", "wf-bind-sel");
    const none = el("option", null, "— None (set up later) —");
    none.value = "";
    sel.append(none);
    const targets = (parsed.targets || {})[b.key] || [];
    targets.forEach((t) => {
      const o = el("option", null, t.label);
      o.value = t.value;
      if (bindings[b.key] === t.value) o.selected = true;
      sel.append(o);
    });
    if (bindings[b.key] && !targets.some((t) => t.value === bindings[b.key])) {
      const o = el("option", null, bindings[b.key] + " (saved)");
      o.value = bindings[b.key];
      o.selected = true;
      sel.append(o);
    }
    sel.onchange = () => { bindings[b.key] = sel.value; };
    row.append(sel);
    return row;
  }

  function stepBindings() {
    const sec = el("div", "wf-panel");
    sec.append(el("p", "wf-lead", "Connect Movie Editor settings to your workflow. Choose — None — to skip and configure later in Models."));
    const box = el("div", "wf-bind-list");
    (parsed.bindings || []).forEach((b) => box.append(bindingRow(b)));
    sec.append(box);

    sec.append(el("div", "wf-hint", "Built-in FunPack pipeline will be disabled. Wire your final IMAGE output to “Saved video comes from” so results show in the editor."));

    const nav = el("div", "wf-nav");
    const back2 = el("button", "btn ghost", "Back");
    back2.onclick = () => { step = 1; render(); };
    const next2 = el("button", "btn primary", "Next: apply");
    next2.onclick = () => { step = 3; render(); };
    nav.append(back2);
    nav.append(next2);
    sec.append(nav);
    return sec;
  }

  function stepApply() {
    const sec = el("div", "wf-panel");
    sec.append(el("h3", "wf-h3", "Ready to apply"));
    const summary = el("ul", "wf-summary");
    (parsed.bindings || []).forEach((b) => {
      const val = bindings[b.key];
      const li = el("li", null);
      if (!val) {
        li.textContent = `${b.label}: (not linked)`;
      } else {
        const tgt = ((parsed.targets || {})[b.key] || []).find((t) => t.value === val);
        li.textContent = `${b.label}: ${tgt ? tgt.label : val}`;
      }
      summary.append(li);
    });
    sec.append(summary);
    sec.append(el("div", "wf-hint", `${parsed.node_count} nodes will replace your current Models configuration. Built-in pipeline: OFF.`));

    const nav = el("div", "wf-nav");
    const back3 = el("button", "btn ghost", "Back");
    back3.onclick = () => { step = 2; render(); };
    nav.append(back3);
    const applyBtn = el("button", "btn primary", "Apply workflow import");
    applyBtn.onclick = async () => {
      const projectId = pid();
      if (!projectId) { alert("Open a project first."); return; }
      applyBtn.disabled = true;
      applyBtn.textContent = "Applying…";
      try {
        await API.applyWorkflow(projectId, workflowRaw, bindings);
        window.dispatchEvent(new Event("funpack-models-changed"));
        closeModal();
        if (window.ModelsModal?.open) {
          if (confirm("Workflow imported. Open Models to review wiring?")) window.ModelsModal.open();
        }
      } catch (e) {
        alert("Apply failed: " + (e.message || e));
        applyBtn.disabled = false;
        applyBtn.textContent = "Apply workflow import";
      }
    };
    nav.append(applyBtn);
    sec.append(nav);
    return sec;
  }

  function body() {
    const b = el("div", "wf-body");
    b.append(stepDots());
    if (step === 0) b.append(stepLoad());
    else if (step === 1 && parsed) b.append(stepReview());
    else if (step === 2 && parsed) b.append(stepBindings());
    else if (step === 3 && parsed) b.append(stepApply());
    else b.append(stepLoad());
    return b;
  }

  function render() {
    if (!overlay) return;
    const content = overlay.querySelector(".modal-content");
    clear(content);
    content.append(body());
  }

  function open() {
    if (overlay) return;
    if (!pid()) { alert("Open or create a project first."); return; }
    step = 0;
    workflowRaw = null;
    parsed = null;
    bindings = {};

    overlay = el("div", "modal-overlay");
    const modal = el("div", "modal modal-wide");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", "ComfyUI Workflow Import"));
    const close = el("button", "btn ghost", "✕");
    close.onclick = closeModal;
    const heRight = el("div", "modal-head-right");
    heRight.append(close);
    head.append(heRight);
    modal.append(head);
    modal.append(el("div", "modal-content"));
    overlay.append(modal);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) closeModal(); });
    document.body.append(overlay);
    render();
  }

  window.WorkflowImportWizard = { open };
})();
