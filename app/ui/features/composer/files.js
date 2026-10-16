// Files: what FunPack keeps on disk for prompts (the shortcut library, cut words) and the taste keys, to see and delete. As v4's tab.
import { composer as c } from "../../composer/composer.js";

const size = (n) => (n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`);

export function files(app) {
  const page = c.region.stack({ gap: "sm" });
  const sure = (title, message) => c.modal.dialogue({ title, message: `${message} This cannot be undone.`, tone: "danger", confirmLabel: "Delete" }).result;
  async function draw() {
    let lib, keys = null;
    try { lib = await app.api.libraryFiles(); } catch (err) { return page.set([c.banner.warn({ text: `Could not list the files: ${err.message}` })]); }
    try { keys = (await app.api.tasteKeys()).keys || []; } catch { /* no taste module: no keys to show */ }
    const row = (label, hint, onDelete) => c.settingsRow.default({ label, hint, control: c.button.sm({ label: "✕ Delete", tone: "danger", onClick: onDelete }) });
    // A key's kinds: one module's learning each. Deleting one kind leaves the key's other kinds as they are.
    const kindRows = async (k) => {
      let kinds = [];
      try { kinds = (await app.api.tasteKindsOf(k)).kinds || []; } catch (err) { return [c.hint.default({ text: `Could not list ${k}: ${err.message}` })]; }
      if (!kinds.length) return [c.hint.default({ text: "Nothing learned under this key yet." })];
      return kinds.map((kd) => c.settingsRow.default({ label: kd.title, hint: `${kd.hint} · ${size(kd.bytes)}`,
        control: c.button.sm({ label: "✕ Delete", tone: "danger", onClick: async () => {
          if (!(await sure("Delete what was learned", `Delete “${kd.title}” under “${k}”? The key's other learning stays.`))) return;
          try { await app.api.clearTasteKind(k, kd.kind); } catch (err) { c.toast.warn({ text: err.message }); }
          draw();
        } }) }));
    };
    const keyBlocks = keys ? await Promise.all(keys.map(async (k) => [
      c.settingsRow.default({ label: k, hint: "what your ratings taught under this name", control: c.button.sm({ label: "✕ Delete key", tone: "danger", onClick: async () => {
        if (!(await sure("Delete taste key", `Delete “${k}” and everything it learned?`))) return;
        try { await app.api.deleteTasteKey(k); } catch (err) { c.toast.warn({ text: err.message }); }
        draw();
      } }) }),
      ...(await kindRows(k)),
    ])) : null;
    page.set([
      c.toolbar.default({ items: [c.button.sm({ label: "↻ Refresh", tone: "ghost", onClick: draw })] }),
      c.label.section({ text: "Prompt library" }),
      c.hint.default({ text: lib.dir }),
      ...(lib.files.length ? lib.files.map((f) => row(f.name, size(f.size), async () => {
        if (!(await sure("Delete file", `Delete “${f.name}”?`))) return;
        try { await app.api.deleteLibraryFile(f.name); } catch (err) { c.toast.warn({ text: err.message }); }
        draw();
      })) : [c.hint.default({ text: "No files." })]),
      ...(keyBlocks ? [c.label.section({ text: "Taste keys" }),
        ...(keys.length ? keyBlocks.flat() : [c.hint.default({ text: "No keys." })])] : []),
    ]);
  }
  draw();
  return page;
}
