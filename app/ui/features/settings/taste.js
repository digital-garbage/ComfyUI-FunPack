// Refinement & Taste: the refinement keys FunPack has learned from your ratings. Keys are files: carry one to another machine
// as a zip, or delete one to start that taste over.
import { composer as c } from "../../composer/composer.js";

export const taste = (app) => function mount() {
  const page = c.region.stack({ gap: "md" });
  const tell = (text) => c.toast.warn({ text });
  let keys = null, error = "";

  async function refresh() {
    try { keys = (await app.api.tasteKeys()).keys || []; error = ""; } catch (err) { keys = []; error = err.message; }
    draw();
  }
  const remove = async (name) => {
    if (!(await c.modal.dialogue({ title: "Delete refinement key", message: `Delete “${name}” and everything it learned? This cannot be undone.`, tone: "danger", confirmLabel: "Delete" }).result)) return;
    try { await app.api.deleteTasteKey(name); } catch (err) { tell(err.message); }
    refresh();
  };
  async function bring([file]) {
    const name = await c.modal.prompt({ title: "Import refinement key", label: "Name", value: file.name.replace(/\.zip$/i, ""), confirmLabel: "Import" }).result;
    if (!name || !name.trim()) return;
    try {
      let r = await app.api.importTasteKey(name.trim(), file, false);
      if (r.exists && await c.modal.dialogue({ title: "Replace key", message: `A key named “${name.trim()}” already exists. Replace it?`, tone: "danger", confirmLabel: "Replace" }).result) r = await app.api.importTasteKey(name.trim(), file, true);
      if (!r.exists) c.toast.good({ text: `Imported “${name.trim()}”.` });
    } catch (err) { tell(err.message); }
    refresh();
  }
  function draw() {
    page.set([
      error ? c.banner.warn({ text: `Could not read the keys: ${error}` }) : null,
      keys && !keys.length && !error ? c.emptyState.default({ icon: "✦", title: "No keys yet", hint: "A key appears once you rate a render with learning on." }) : null,
      ...(keys || []).map((name) => c.settingsRow.default({ label: name, hint: "Refinement key", control: c.toolbar.default({ items: [
        c.button.sm({ label: "⤓ Export", tone: "ghost", onClick: () => { Object.assign(document.createElement("a"), { href: app.api.tasteKeyUrl(name) }).click(); c.toast.good({ text: "Preparing the download…" }); } }),
        c.button.sm({ label: "Delete", tone: "danger", onClick: () => remove(name) })] }) })),
      c.label.section({ text: "Bring a key here" }),
      c.dropzone.default({ label: "Drop or choose an exported key (.zip)", hint: "from another machine", accept: ".zip,application/zip", multiple: false, onFiles: bring }),
    ].filter(Boolean));
  }
  draw(); refresh();
  return { node: page.node, destroy: () => page.node.remove() };
};
