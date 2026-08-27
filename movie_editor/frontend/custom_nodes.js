// Settings ▸ Custom Nodes — install, update and remove ComfyUI node packs.
//
// Three git operations, not a catalogue: you supply the repository URL. Keeping it that
// plain is the honest shape — a curated list would imply vetting that is not happening.
//
// Every action ends in a ComfyUI restart being NEEDED, not performed: node packs register
// at import, so nothing that happens here takes effect until the server comes back. The
// panel says so rather than restarting under the user.
(function () {
  const { el, clear } = window.dom;
  const API = window.MovieEditorAPI;

  let host = null;
  let nodes = [];
  let rootPath = "";
  let busy = "";        // name (or "__install__") of whatever is running
  let dirty = false;    // something changed → a restart is needed
  let error = "";
  // name -> {checked, behind, ahead, reason}. Empty until the user asks: each entry costs a
  // network round trip, and doing that on open would make the panel feel broken.
  let checked = {};
  let checking = false;

  async function checkUpdates() {
    if (checking || busy) return;
    checking = true;
    render();
    try {
      checked = (await API.customNodesCheck()).checked || {};
    } catch (e) {
      alert("Could not check for updates:\n\n" + (e?.message || e));
    } finally {
      checking = false;
      render();
    }
  }

  async function refresh() {
    try {
      const res = await API.customNodes();
      nodes = res.nodes || [];
      rootPath = res.root || "";
      error = "";
    } catch (e) {
      nodes = [];
      error = e?.message || String(e);
    }
    render();
  }

  // A pack's requirements install can fail while the pack itself is fine. That is worth
  // interrupting for — it is the difference between "restart and use it" and "it will not
  // import until you run this".
  function reportRequirements(res, what) {
    const r = res && res.requirements;
    if (!r || !r.ran || r.ok) return;
    alert(`${what}, but installing its dependencies failed.\n\n${r.detail || ""}`);
  }

  async function run(key, fn, what) {
    if (busy) return;
    busy = key;
    render();
    try {
      const res = await fn();
      reportRequirements(res, what);
      dirty = true;
      // Whatever this was, its behind-count is now stale. Dropping it beats showing a
      // number that was true a moment ago.
      if (key !== "__install__") delete checked[key];
    } catch (e) {
      alert(`${what} failed:\n\n${e?.message || e}`);
    } finally {
      busy = "";
      await refresh();
    }
  }

  function addDialog() {
    const overlay = el("div", "modal-overlay");
    const box = el("div", "modal cn-modal");
    const head = el("div", "modal-head");
    head.append(el("div", "modal-title", "Add a custom node"));
    box.append(head);

    const content = el("div", "modal-content");
    content.append(el("div", "cn-lead", "Paste the repository's git URL."));
    const input = el("input", "lib-in");
    input.type = "text";
    input.placeholder = "https://github.com/owner/repo";
    input.spellcheck = false;
    content.append(input);
    content.append(el("div", "cn-hint",
      "It is cloned into custom_nodes and its requirements.txt is installed, if it has one. "
      + "Nothing about the repository is checked first — install what you trust."));

    const acts = el("div", "lib-form-actions");
    const ok = el("button", "btn primary tiny", "Install");
    ok.type = "button";
    const cancel = el("button", "btn ghost tiny", "Cancel");
    cancel.type = "button";
    const close = () => overlay.remove();
    cancel.onclick = close;
    ok.onclick = () => {
      const url = input.value.trim();
      if (!url) return;
      close();
      run("__install__", () => API.customNodeInstall(url), `Installing ${url}`);
    };
    input.onkeydown = (e) => {
      if (e.key === "Enter") { e.preventDefault(); ok.click(); }
      if (e.key === "Escape") { e.preventDefault(); close(); }
    };
    acts.append(cancel, ok);
    content.append(acts);
    box.append(content);
    overlay.append(box);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.body.append(overlay);
    setTimeout(() => input.focus(), 0);
  }

  function row(n) {
    const r = el("div", "cn-row" + (n.is_funpack ? " self" : ""));
    const left = el("div", "cn-row-text");
    left.append(el("div", "cn-name", n.name));
    const bits = [];
    if (n.is_funpack) bits.push("FunPack itself");
    if (n.git) bits.push(`${n.branch || "?"} · ${n.commit || "?"}`);
    else bits.push("not a git checkout");
    const chk = checked[n.name];
    if (chk && chk.checked) {
      if (chk.behind > 0) bits.push(`${chk.behind} behind`);
      else if (chk.ahead > 0) bits.push(`${chk.ahead} ahead of origin`);
      else bits.push("up to date");
    } else if (chk && chk.reason) {
      bits.push(`not compared — ${chk.reason}`);
    }
    left.append(el("div", "cn-sub", bits.join(" — ")));
    r.append(left);

    const acts = el("div", "cn-row-acts");
    if (chk && chk.checked && chk.behind > 0) r.classList.add("behind");
    if (!n.is_funpack) {
      const label = busy === n.name ? "…"
        : (chk && chk.checked && chk.behind > 0) ? `Update (${chk.behind})` : "Update";
      const up = el("button", "btn ghost tiny" + (chk?.behind > 0 ? " primary" : ""), label);
      up.type = "button";
      up.disabled = !!busy || !n.git;
      up.title = n.git ? `git pull in ${n.name}` : "Not a git checkout — nothing to pull";
      up.onclick = () => run(n.name, () => API.customNodeUpdate(n.name), `Updating ${n.name}`);
      const rm = el("button", "btn ghost tiny danger", "Remove");
      rm.type = "button";
      rm.disabled = !!busy;
      // The path is in the prompt on purpose: this deletes a directory and cannot be undone.
      rm.onclick = () => {
        if (!confirm(`Delete this folder and everything in it?\n\n${rootPath}/${n.name}\n\n`
                     + "This cannot be undone.")) return;
        run(n.name, () => API.customNodeRemove(n.name), `Removing ${n.name}`);
      };
      acts.append(up, rm);
    }
    r.append(acts);
    return r;
  }

  function render() {
    if (!host) return;
    clear(host);

    const bar = el("div", "cn-bar");
    const add = el("button", "btn primary tiny", "＋ Add node");
    add.type = "button";
    add.disabled = !!busy;
    add.onclick = addDialog;
    bar.append(add);
    const chkBtn = el("button", "btn ghost tiny", checking ? "Checking…" : "Check for updates");
    chkBtn.type = "button";
    chkBtn.disabled = checking || !!busy;
    chkBtn.title = "Fetch each pack's origin and report how far behind it is";
    chkBtn.onclick = checkUpdates;
    bar.append(chkBtn);
    if (busy) bar.append(el("span", "cn-busy", busy === "__install__"
      ? "Cloning and installing…" : `Working on ${busy}…`));
    else if (checking) bar.append(el("span", "cn-busy",
      "Fetching each pack — this talks to the network."));
    host.append(bar);

    if (dirty) {
      const note = el("div", "cn-restart");
      note.append(el("span", null,
        "Node packs are registered when ComfyUI starts, so these changes are not live yet."));
      const btn = el("button", "btn tiny", "Restart ComfyUI");
      btn.type = "button";
      // FunPackGit owns the restart: it flushes pending edits and puts up the
      // wait-for-reload overlay, which a bare POST would skip.
      btn.onclick = () => window.FunPackGit?.restartComfy?.();
      note.append(btn);
      host.append(note);
    }

    if (error) {
      host.append(el("div", "pj-meta", "Could not read custom_nodes: " + error));
      return;
    }
    if (!nodes.length) {
      host.append(el("div", "pj-meta", "No custom node packs installed."));
      return;
    }
    const list = el("div", "cn-list");
    nodes.forEach((n) => list.append(row(n)));
    host.append(list);
    host.append(el("div", "cn-hint", rootPath));
  }

  function mount(body) {
    host = el("div", "cn-mount");
    body.append(host);
    dirty = false;
    checked = {};
    render();
    refresh();
    return () => { host = null; };
  }

  window.SettingsWindow.register({
    id: "customnodes", group: "System", order: 3, title: "Custom Nodes",
    subtitle: "Install, update and remove ComfyUI node packs.",
    keywords: "custom nodes packs manager install update remove git clone extension",
    iconBg: "linear-gradient(180deg,#8fb8ff,#4a72c8)",
    icon: '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="#fff" stroke-width="1.4"><rect x="2.2" y="2.2" width="4.6" height="4.6" rx="1"/><rect x="9.2" y="2.2" width="4.6" height="4.6" rx="1"/><rect x="2.2" y="9.2" width="4.6" height="4.6" rx="1"/><path d="M11.5 9.6v4.6M9.2 11.9h4.6"/></svg>',
    mount,
  });
})();
