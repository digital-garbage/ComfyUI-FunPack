// Custom Nodes: install, update and remove ComfyUI node packs. Three git operations, not a catalogue: you supply the URL.
// Packs register when ComfyUI starts, so every change ends with "restart needed", never a restart on its own.
import { composer as c } from "../../composer/composer.js";

export const packs = (app) => function mount() {
  const page = c.region.stack({ gap: "md" });
  let nodes = [], root = "", error = "", busy = "", checked = {}, dirty = false;
  const tell = (text) => c.toast.warn({ text });

  async function refresh() {
    try { const r = await app.api.packs(); nodes = r.nodes || []; root = r.root || ""; error = ""; } catch (err) { nodes = []; error = err.message; }
    draw();
  }
  async function run(key, what, work) {
    if (busy) return;
    busy = key; draw();
    try {
      const res = await work();
      const req = res && res.requirements;
      if (req && req.ran && !req.ok) tell(`${what}, but installing its dependencies failed: ${req.detail || ""}`);
      dirty = true;
      delete checked[key];
    } catch (err) { tell(`${what} failed: ${err.message}`); }
    busy = ""; await refresh();
  }
  const install = async () => {
    const url = await c.modal.prompt({ title: "Add a custom node", label: "Repository git URL", placeholder: "https://github.com/owner/repo", confirmLabel: "Install" }).result;
    if (url && url.trim()) run("__install__", `Installing ${url.trim()}`, () => app.api.pack("install", { url: url.trim() }));
  };
  const check = async () => {
    busy = "__check__"; draw();
    try { checked = (await app.api.pack("check")).checked || {}; } catch (err) { tell(`Could not check for updates: ${err.message}`); }
    busy = ""; draw();
  };
  const remove = async (n) => {
    if (await c.modal.dialogue({ title: "Remove node pack", message: `Delete ${root}/${n.name} and everything in it? This cannot be undone.`, tone: "danger", confirmLabel: "Delete" }).result)
      run(n.name, `Removing ${n.name}`, () => app.api.pack("remove", { name: n.name }));
  };

  function row(n) {
    const chk = checked[n.name], behind = chk && chk.checked && chk.behind > 0;
    const bits = [n.is_funpack ? "FunPack itself" : null, n.git ? `${n.branch || "?"} · ${n.commit || "?"}` : "not a git checkout",
      chk && chk.checked ? (behind ? `${chk.behind} behind` : chk.ahead > 0 ? `${chk.ahead} ahead of origin` : "up to date") : chk && chk.reason ? `not compared — ${chk.reason}` : null].filter(Boolean);
    const acts = n.is_funpack ? c.text.sm({ text: "" }) : c.toolbar.default({ items: [
      c.button.sm({ label: behind ? `Update (${chk.behind})` : "Update", tone: behind ? "primary" : "ghost", disabled: Boolean(busy) || !n.git, onClick: () => run(n.name, `Updating ${n.name}`, () => app.api.pack("update", { name: n.name })) }),
      c.button.sm({ label: "Remove", tone: "danger", disabled: Boolean(busy), onClick: () => remove(n) })] });
    return c.settingsRow.default({ label: n.name, hint: bits.join(" — "), control: acts });
  }
  function draw() {
    page.set([
      c.toolbar.default({ items: [c.button.sm({ label: "＋ Add node", tone: "primary", disabled: Boolean(busy), onClick: install }),
        c.button.sm({ label: busy === "__check__" ? "Checking…" : "Check for updates", tone: "ghost", disabled: Boolean(busy), title: "Fetch each pack's origin and report how far behind it is", onClick: check }),
        busy && busy !== "__check__" ? c.text.sm({ text: busy === "__install__" ? "Cloning and installing…" : `Working on ${busy}…` }) : null].filter(Boolean) }),
      dirty ? c.banner.info({ text: "Node packs are registered when ComfyUI starts, so these changes are not live yet.", action: { label: "Restart ComfyUI", onClick: () => app.maintenance.restart() } }) : null,
      error ? c.banner.warn({ text: `Could not read custom_nodes: ${error}` }) : null,
      !error && !nodes.length ? c.emptyState.default({ icon: "⧉", title: "No node packs", hint: "No custom node packs installed." }) : null,
      ...nodes.map(row),
      root ? c.hint.default({ text: root }) : null,
    ].filter(Boolean));
  }
  draw(); refresh();
  return { node: page.node, destroy: () => page.node.remove() };
};
