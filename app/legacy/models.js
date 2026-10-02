// Models & Pipeline: a renderer over the real, live pipeline slots (GET/POST
// /api/pipeline) instead of v4's Studio-shaped "roles" and "candidates per
// role" (`nodeRoles`/`nodeCandidates` -- neither exists in v5: the pipeline is
// a flat, swappable slot list, not a fixed set of named roles). One slot per
// row, grouped the way the pipeline itself groups them (Loaders/Preparation/
// Sampling/Render, or whatever a module's default_pipeline declares); each
// slot's node inputs are read live from GET /api/nodes and edited through the
// same /api/pipeline POST Engine Settings uses (shared state -- see
// pipeline_state.js). Adding/removing/swapping a node or wiring a slot to
// another is NOT built here yet: v4's ~2700 lines of that machinery assumed
// Studio/Chain Sampler's own node set, which no longer exists, and a real
// graph editor is its own piece of work -- this establishes the real,
// working file-picker/settings surface first ("port the looks, leave
// placeholders for features"), per the port plan.
(function () {
  const { el, clear } = window.dom;
  const API = window.MovieEditorAPI;
  const PS = window.PipelineState;

  let _mounted = null;
  let group = null;          // selected sidebar group, chosen once slots load
  let nodesByClass = {};      // class name -> widget spec (core/widgets.py's describe())
  let nodesLoading = false;
  let nodesError = null;
  let openNodeId = null;      // slot id a pinned button asked to jump straight to

  function field(labelText, control, hint) {
    const row = el("div", "sw-row eng-field");
    const main = el("div", "sw-row-main");
    main.append(el("div", "sw-row-title", labelText));
    if (hint) main.append(el("div", "sw-row-hint", hint));
    row.append(main, control);
    return row;
  }
  function toggleField(labelText, checkbox) {
    checkbox.style.width = "auto";
    return field(labelText, checkbox);
  }
  function rowsGroup(parent, label) {
    if (label) parent.append(el("div", "sw-rows-label", label));
    const g = el("div", "sw-rows");
    parent.append(g);
    return g;
  }
  function hintEl(text) { return el("div", "sw-hint", text); }

  function slotDisplayLabel(slot) {
    const spec = nodesByClass[slot.node];
    return (spec && spec.title) || slot.node || slot.id;
  }

  function setInput(slotId, name, value) {
    lastRefusal = [];
    PS.save({ inputs: { [slotId]: { [name]: value } } }).then(render);
  }

  // ── the pipeline's shape: groups, swapping, removing, wiring, adding ────────────
  // Every structural change goes to the server, which refuses what would not build (a type
  // that does not fit, a loop, a node whose output something still reads) and says why; the
  // reason is shown here and nothing changes.
  let lastRefusal = [];

  async function structural(body) {
    const res = await PS.edit(body);
    lastRefusal = res.refused || [];
    await ensureNodesLoaded();
    render();
    return !lastRefusal.length;
  }

  function otherGroups(slot) {
    return groupsWithSlots().filter((g) => g !== (slot.group || "Other"));
  }

  function groupSelect(slot) {
    const sel = el("select", "eng-select");
    const here = slot.group || "Other";
    [here, ...otherGroups(slot)].forEach((g) => { const o = el("option", "", g); o.value = g; sel.append(o); });
    const add = el("option", "", "New group…"); add.value = "\u0000new"; sel.append(add);
    sel.onchange = () => {
      if (sel.value === "\u0000new") {
        const name = (prompt("Name of the new group:", "") || "").trim();
        if (!name) { sel.value = here; return; }
        group = name;
        PS.setGroup(slot.id, name).then(render);
      } else {
        PS.setGroup(slot.id, sel.value).then(render);
      }
    };
    return sel;
  }

  // A search box over every installed node; `onPick(className)` gets the choice.
  function pickNode(title, onPick) {
    const overlay = el("div", "modal-overlay slot-picker-overlay");
    const box = el("div", "modal slot-picker-modal");
    box.append(el("div", "modal-title", title));
    const input = el("input", "slot-picker-input");
    input.type = "text"; input.placeholder = "Search installed nodes by name or category…";
    const list = el("div", "slot-picker-list");
    const note = el("div", "sw-row-hint", "");
    let timer = null, ticket = 0;
    const close = () => overlay.remove();
    async function run() {
      const mine = ++ticket;
      let found;
      try { found = await API.searchNodes(input.value, 40); }
      catch (e) { clear(list); note.textContent = `Could not search: ${e.message || e}`; return; }
      if (mine !== ticket) return;
      clear(list);
      (found.nodes || []).forEach((n) => {
        const row = el("button", "slot-picker-row", n.title || n.node);
        row.append(el("span", "slot-picker-hint", `${n.category || ""}${n.outputs && n.outputs.length ? " → " + n.outputs.join(", ") : ""}`));
        row.onclick = () => { close(); onPick(n.node); };
        list.append(row);
      });
      note.textContent = found.total > (found.nodes || []).length
        ? `${found.nodes.length} of ${found.total} shown: type more to narrow it.` : (found.nodes || []).length ? "" : "Nothing matches.";
    }
    input.oninput = () => { clearTimeout(timer); timer = setTimeout(run, 200); };
    const cancel = el("button", "btn ghost", "Cancel");
    cancel.onclick = close;
    box.append(input, note, list, cancel);
    overlay.append(box);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    overlay.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } });
    document.body.append(overlay);
    input.focus();
    run();
  }

  function structureRow(slot) {
    const wrap = el("div", "eng-structure");
    wrap.append(field("Group", groupSelect(slot), "Cards in the sidebar. Any node can live in any group."));
    const actions = el("div", "eng-structure-actions");
    const swap = el("button", "btn ghost tiny", "Swap node…");
    swap.title = "Use a different node here. It must still produce what the nodes after it read.";
    swap.onclick = () => pickNode(`Swap ${slotDisplayLabel(slot)} for…`, (cls) => {
      if (!confirm(`Swap ${slotDisplayLabel(slot)} for ${cls}?\n\nEverything set on this node (files, values, connections into it) is cleared, and the new node starts empty.`)) return;
      structural({ action: "replace", slot: slot.id, node: cls });
    });
    const rm = el("button", "btn ghost tiny danger", "Remove");
    rm.title = "Take this node out; what it fed is rewired to what fed it, when that is unambiguous.";
    rm.onclick = () => { if (confirm(`Remove ${slotDisplayLabel(slot)} from the pipeline?`)) structural({ action: "remove", slot: slot.id }); };
    actions.append(swap, rm);
    wrap.append(actions);
    return wrap;
  }

  // Outputs of other slots that could feed an input of type `type`, as {value, label}.
  function sourcesFor(slot, type) {
    const out = [];
    (PS.slots() || []).forEach((other) => {
      if (other.id === slot.id) return;
      const spec = nodesByClass[other.node];
      if (!spec || !spec.outputs) return;
      spec.outputs.forEach((kind, i) => {
        if (kind !== type && kind !== "*" && type !== "*") return;
        const name = (spec.output_names && spec.output_names[i]) || kind;
        out.push({ value: `${other.id}\u0000${i}`, label: `${slotDisplayLabel(other)} · ${name}` });
      });
    });
    return out;
  }

  function wiringControl(slot, input, type, current, allowValue) {
    const sel = el("select", "eng-select");
    const wired = Array.isArray(current);
    const value = el("option", "", allowValue ? "A value (typed here)" : "Not connected"); value.value = "";
    sel.append(value);
    if (wired) {
      const known = sourcesFor(slot, type).some((s) => s.value === `${current[0]}\u0000${current[1]}`);
      if (!known) { const o = el("option", "", `${current[0]} · output ${current[1]}`); o.value = `${current[0]}\u0000${current[1]}`; sel.append(o); }
    }
    sourcesFor(slot, type).forEach((s) => { const o = el("option", "", `Fed by ${s.label}`); o.value = s.value; sel.append(o); });
    sel.value = wired ? `${current[0]}\u0000${current[1]}` : "";
    sel.onchange = () => {
      if (!sel.value) { structural({ action: "unwire", slot: slot.id, input }); return; }
      const [from, idx] = sel.value.split("\u0000");
      structural({ action: "wire", slot: slot.id, input, from_slot: from, from_output: Number(idx) });
    };
    return sel;
  }

  function controlForWidget(slot, widget) {
    if (Array.isArray(slot.inputs && slot.inputs[widget.name])) {
      return field(widget.name, wiringControl(slot, widget.name, widget.type, slot.inputs[widget.name], true),
        "Fed by another node: change it there, or pick “A value” to type one here.");
    }
    const row = controlForPlainWidget(slot, widget);
    // An input the app itself writes (the prompt box, the source picture, the length) is the app's:
    // wiring it elsewhere would make every Generate refuse.
    if ((slot.roles || []).some((r) => r.input === widget.name)) return row;
    // Another node can drive this value (one Primitive shared by several inputs is how a "link" is made).
    if (widget.type !== "COMBO" || (widget.choices || []).length) {
      const sources = sourcesFor(slot, widget.type);
      if (sources.length) {
        // Own row beneath the value: squeezed beside a text box it left the box a few pixels wide.
        const both = document.createDocumentFragment();
        both.append(row, field("↳ or take it from", wiringControl(slot, widget.name, widget.type, null, true),
          "Another node's output can drive this value; one Primitive can drive several inputs."));
        return both;
      }
    }
    return row;
  }

  function controlForPlainWidget(slot, widget) {
    const current = slot.inputs && slot.inputs[widget.name] !== undefined
      ? slot.inputs[widget.name] : widget.default;
    const hint = widget.tooltip || "";

    if (widget.type === "BOOLEAN") {
      const cb = el("input", "");
      cb.type = "checkbox";
      cb.checked = !!current;
      cb.onchange = () => setInput(slot.id, widget.name, cb.checked);
      return toggleField(widget.name, cb);
    }
    if (widget.type === "COMBO") {
      const select = el("select", "eng-select");
      const choices = widget.choices || [];
      if (!choices.length) {
        return field(widget.name, hintEl("No choices available — nothing found in the models folder."));
      }
      // A file that was picked before but has since been removed from disk
      // must still show AS the current value, or saving anything else on
      // this slot would silently swap it out from under the user.
      if (current != null && !choices.includes(current)) choices.unshift(current);
      choices.forEach((choice) => {
        const o = el("option", "", String(choice));
        o.value = choice;
        if (choice === current) o.selected = true;
        select.append(o);
      });
      select.onchange = () => setInput(slot.id, widget.name, select.value);
      return field(widget.name, select, hint);
    }
    if (widget.type === "INT" || widget.type === "FLOAT") {
      const input = el("input", "eng-num-input");
      input.type = "number";
      if (widget.min != null) input.min = widget.min;
      if (widget.max != null) input.max = widget.max;
      input.step = widget.step != null ? widget.step : (widget.type === "INT" ? 1 : "any");
      input.value = current != null ? current : "";
      input.onchange = () => {
        const n = widget.type === "INT" ? parseInt(input.value, 10) : parseFloat(input.value);
        if (!Number.isNaN(n)) setInput(slot.id, widget.name, n);
      };
      return field(widget.name, input, hint);
    }
    // STRING
    const input = el(widget.multiline ? "textarea" : "input", "eng-text-input");
    if (!widget.multiline) input.type = "text";
    input.value = current || "";
    input.onblur = () => setInput(slot.id, widget.name, input.value);
    return field(widget.name, input, hint);
  }

  // ── export the pipeline as a picture ──────────────────────────────────────
  // "Which model was that?" outlives the session that could answer it. The
  // card is built on the SERVER (it knows torch/CUDA/GPU; the browser only
  // knows the laptop showing the page) from the pipeline the CLIENT holds --
  // v5's pipeline has no server-side copy of its own (see api.js's own note
  // on /api/pipeline being stateless), so this sends PS.slots() up rather
  // than a project id the way v4's version did.
  function openSettingsCard() {
    // A prior overlay reached only via re-clicking "Export settings…" (never
    // through its own ✕/Close, which is the only path that revokes its blob
    // URL) would otherwise leak that blob every reopen -- found in this
    // feature's own extensive_testing round, carried over unfixed from v4.
    document.querySelectorAll(".sc-overlay").forEach((n) => {
      if (n._scObjectUrl) URL.revokeObjectURL(n._scObjectUrl);
      n.remove();
    });
    const overlay = el("div", "modal-overlay sc-overlay");
    const modal = el("div", "modal sc-modal");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", "Export settings"));
    const hr = el("div", "modal-head-right");
    const x = el("button", "btn ghost tiny", "✕");
    const close = () => { if (url) URL.revokeObjectURL(url); overlay._scObjectUrl = null; overlay.remove(); };
    x.onclick = close;
    hr.append(x); head.append(hr);
    const content = el("div", "modal-content sc-content");
    const foot = el("div", "modal-foot sc-foot");
    modal.append(head, content, foot);
    overlay.append(modal);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.body.append(overlay);

    let url = null, blob = null;
    content.append(el("div", "pj-meta", "Rendering…"));

    const status = el("div", "sc-status");
    const dl = el("button", "btn primary tiny", "⤓ Download");
    const cp = el("button", "btn ghost tiny", "⧉ Copy image");
    const cl = el("button", "btn ghost tiny", "Close");
    cl.onclick = close;
    dl.disabled = true; cp.disabled = true;
    foot.append(status, dl, cp, cl);

    // The card is rendered in whatever theme the app is showing, because it
    // is a document about this install and it should look like this install.
    const theme = window.FunPackTheme?.resolved?.()
      || document.documentElement.getAttribute("data-theme") || "dark";
    const project = window.Store?.get().project;
    API.settingsCard(PS.slots() || [], project?.name, theme)
      .then((b) => {
        if (!overlay.isConnected) return;
        blob = b; url = URL.createObjectURL(b);
        overlay._scObjectUrl = url;
        clear(content);
        const img = el("img", "sc-img");
        img.src = url;
        img.alt = "FunPack settings card";
        content.append(img);
        dl.disabled = false; cp.disabled = false;
      })
      .catch((e) => {
        if (!overlay.isConnected) return;
        clear(content);
        content.append(el("div", "pj-meta", "Could not render the card: " + (e?.message || e)));
      });

    dl.onclick = () => {
      if (!url) return;
      const a = document.createElement("a");
      a.href = url;
      const base = (project?.name || "funpack-settings").replace(/[^\w.-]+/g, "_");
      a.download = `${base}-settings.png`;
      document.body.append(a); a.click(); a.remove();
    };

    cp.onclick = async () => {
      // Clipboard image writes need a secure context, which a rental reached
      // over plain http://<ip> is not. Say that, rather than failing
      // silently -- Download still works.
      status.textContent = "";
      try {
        if (!navigator.clipboard || typeof window.ClipboardItem !== "function") {
          throw new Error("this browser/connection has no image clipboard");
        }
        await navigator.clipboard.write([new window.ClipboardItem({ "image/png": blob })]);
        status.textContent = "Copied.";
      } catch (e) {
        status.textContent = "Could not copy (" + (e?.message || e) + "). Use Download.";
      }
    };
  }

  function groupsWithSlots() {
    const order = [];
    const seen = new Set();
    (PS.slots() || []).forEach((slot) => {
      const g = slot.group || "Other";
      if (!seen.has(g)) { seen.add(g); order.push(g); }
    });
    return order;
  }

  async function ensureNodesLoaded() {
    const classes = Array.from(new Set((PS.slots() || []).map((s) => s.node).filter(Boolean)));
    const missing = classes.filter((c) => !(c in nodesByClass));
    if (!missing.length) return;
    nodesLoading = true; nodesError = null; render();
    try {
      const body = await API.describeNodes(missing);
      nodesByClass = { ...nodesByClass, ...(body.nodes || {}) };
    } catch (e) {
      nodesError = e && e.message ? e.message : String(e);
    }
    nodesLoading = false;
    render();
  }

  function renderSlot(pane, slot) {
    const spec = nodesByClass[slot.node];
    const g = rowsGroup(pane, slotDisplayLabel(slot));
    if (spec === undefined) { g.append(hintEl("Loading…")); return; }
    g.append(structureRow(slot));
    if (spec === null) {
      g.append(hintEl(`${slot.node} is not installed — this slot can't be edited or run. Swap it for another node, or remove it.`));
      return;
    }
    (spec.sockets || []).forEach((socket) => {
      g.append(field(socket.name, wiringControl(slot, socket.name, socket.type, slot.inputs && slot.inputs[socket.name], false),
        socket.required ? `${socket.type} · needed` : socket.type));
    });
    if (!spec.widgets || !spec.widgets.length) {
      if (!(spec.sockets || []).length) g.append(hintEl("Nothing to configure on this node."));
      return;
    }
    spec.widgets.forEach((widget) => g.append(controlForWidget(slot, widget)));
  }

  function renderStatus(pane) {
    const incomplete = PS.incomplete();
    const refused = PS.refused();
    const notes = PS.saveNotes();
    if (lastRefusal.length) pane.append(hintEl(`Not changed: ${lastRefusal.join(" ")}`));
    else if (refused.length) pane.append(hintEl(`Could not save: ${refused.join(" ")}`));
    if (notes.length) pane.append(hintEl(notes.join(" ")));
    if (incomplete.length) {
      const box = el("div", "sw-hint eng-detail");
      box.append(el("div", "sw-rows-label", "Not ready to generate yet"));
      incomplete.forEach((why) => box.append(el("div", "sw-row-hint", why)));
      pane.append(box);
    } else if (PS.queueable()) {
      pane.append(hintEl("Every slot is filled — this pipeline is ready to generate."));
    }
  }

  function renderPane(pane) {
    if (PS.loading()) { pane.append(hintEl("Loading…")); return; }
    if (PS.loadError()) {
      pane.append(hintEl(`Could not load models & pipeline: ${PS.loadError()}`));
      return;
    }
    renderStatus(pane);
    if (nodesError) pane.append(hintEl(`Could not describe some nodes: ${nodesError}`));

    const groups = groupsWithSlots();
    if (!groups.length) { pane.append(hintEl("This pipeline has no slots.")); return; }
    if (!group || !groups.includes(group)) group = groups[0];

    const addRow = el("div", "eng-structure-actions");
    const addBtn = el("button", "btn ghost tiny", `＋ Add a node to ${group}`);
    addBtn.title = "Put any installed node in this group, then wire it to the nodes around it.";
    addBtn.onclick = () => pickNode(`Add a node to ${group}`, (cls) => structural({ action: "add", node: cls, group }));
    addRow.append(addBtn);
    pane.append(addRow);

    (PS.slots() || [])
      .filter((s) => (s.group || "Other") === group)
      .forEach((slot) => renderSlot(pane, slot));
  }

  function renderContent(container) {
    if (PS.loading() || PS.loadError()) {
      const solo = el("div", "models-pane");
      renderPane(solo);
      container.append(solo);
      return;
    }
    const groups = groupsWithSlots();
    if (!groups.length) {
      const solo = el("div", "models-pane");
      renderPane(solo);
      container.append(solo);
      return;
    }
    const cols = el("div", "models-cols");
    const side = el("div", "models-side");
    groups.forEach((g) => {
      side.append(window.SettingsWindow.navItem({
        label: g, active: group === g, onClick: () => { group = g; render(); },
      }));
    });
    const pane = el("div", "models-pane eng-pane");
    renderPane(pane);
    cols.append(side, pane);
    container.append(cols);
  }

  function render() {
    if (!_mounted) return;
    const { content } = _mounted;
    const prevPane = content.querySelector(".models-pane");
    const scrollTop = prevPane ? prevPane.scrollTop : 0;
    clear(content);
    renderContent(content);
    const pane = content.querySelector(".models-pane");
    if (pane) pane.scrollTop = scrollTop;
  }

  function mount(_body, ctx) {
    const content = el("div", "models-mount eng-mount");
    _body.append(content);
    _mounted = { content };
    if (ctx && ctx.sub && ctx.sub.startsWith("node:")) openNodeId = ctx.sub.slice(5);
    if (ctx && ctx.setActions) {
      const card = el("button", "btn ghost tiny", "🖼 Export settings…");
      card.title = "Render this pipeline as a PNG — loaders, typed-in node values, "
                 + "and the host's torch / CUDA / attention";
      card.onclick = openSettingsCard;
      ctx.setActions([card]);
    }
    render();
    PS.ensureLoaded().then(() => {
      if (openNodeId) {
        const slot = (PS.slots() || []).find((s) => s.id === openNodeId);
        if (slot) group = slot.group || "Other";
        openNodeId = null;
      }
      render();
      ensureNodesLoaded();
    });
    return () => { _mounted = null; };
  }

  window.SettingsWindow.register({
    id: "models", group: "Generation", order: 2, title: "Models & Pipeline", flush: true,
    subtitle: "Loaders and nodes wired into the live pipeline.",
    keywords: "models loaders unet vae clip lora nodes pipeline wiring",
    iconBg: "linear-gradient(180deg,#b18cff,#7a4fd0)",
    icon: '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="#fff" stroke-width="1.4" stroke-linejoin="round"><path d="M8 1.8 14 5v6l-6 3.2L2 11V5l6-3.2z"/><path d="M2 5l6 3 6-3M8 8v6.2"/></svg>',
    mount,
    // Group-level pinning only, deliberately -- v4 could pin ONE node (settings_window.js's
    // own comment: "the node page is the thing that takes several clicks to reach"), which
    // needed a concept of "the currently open node" this flat slot list doesn't have yet
    // (nothing here drills into a single slot the way the old per-role node page did).
    // `openNodeId`/`ModelsModal.openNode` below still work if something hands them a real
    // slot id -- pinned_buttons.js still calls them -- but nothing on this screen currently
    // PRODUCES a node-level pin to begin with, so a per-node pin from before this rewrite
    // is the only way this path is reached today. Group-level is the honest current target.
    pinTarget: () => (group
      ? { kind: "section", id: "models", sub: group, label: `Models ▸ ${group}` }
      : null),
  });

  window.ModelsModal = {
    open: () => window.SettingsWindow.open("models"),
    openNode: (slotId) => {
      if (!slotId) return;
      openNodeId = slotId;
      window.SettingsWindow.open("models");
    },
    // Node choices (a checkpoint folder's contents, say) are read live on
    // every describe() call -- nothing here caches them -- so "refresh" only
    // needs to re-ask for the currently visible nodes' specs, never the
    // pipeline's own slot structure.
    refresh: async () => {
      nodesByClass = {};
      await ensureNodesLoaded();
    },
  };
})();
