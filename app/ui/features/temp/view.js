// Two places a file can be found: the gallery (what was saved, and stays) and the temp folder (what ComfyUI wipes on restart).
// Clicking a file looks at it here; it can be opened in its own tab, or deleted after a question.
import { composer } from "../../composer/composer.js";

const ICON = { image: "▦", video: "▶", audio: "♪" };
const PLACES = [{ value: "output", label: "Saved" }, { value: "temp", label: "Temp files" }];
const EMPTY = { output: "Nothing saved yet", temp: "Nothing in the temp directory" };
const url = (f, where) => `/view?${new URLSearchParams({ filename: f.filename, subfolder: f.subfolder || "", type: where })}`;
const size = (n) => (n < 1024 ? `${n} B` : n < 1048576 ? `${Math.round(n / 1024)} KB` : `${(n / 1048576).toFixed(1)} MB`);
const when = (t) => new Date((Number(t) || 0) * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
const id = (f) => `${f.subfolder}/${f.filename}`;

export function mount({ dom, setFooter, close }) {
  let where = "output", files = [], payload = {}, looking = null, selecting = false, grid = null;
  const picked = new Set();
  const body = composer.region.stack({ gap: "md", fill: true, label: "Files" });
  const content = composer.region.stack({ gap: "sm", fill: true, label: "Files in this folder" });
  const viewer = composer.viewer.cover({ onClose: () => { looking = null; content.node.hidden = false; footer(); } });
  const tabs = composer.tabs.underline({ label: "Where to look", value: where, tabs: PLACES, onChange: (v) => { where = v; selecting = false; picked.clear(); viewer.close(); load(); } });
  body.set([tabs, content, viewer]);

  function look(f) {
    looking = f;
    content.node.hidden = true;                // the grid would show through below the picture
    viewer.show({ src: url(f, where), kind: f.kind });
    footer();
  }

  async function remove(f) {
    const ok = await composer.modal.dialogue({ title: "Delete file", tone: "danger", confirmLabel: "Delete",
      message: `Delete “${f.filename}” from ${where === "output" ? "your saved files" : "the temp folder"} for good? It does not go to the trash.` }).result;
    if (!ok) return;
    setFooter({ note: payload.path || "", actions: [composer.button.md({ label: "Deleting…", disabled: true })] });
    try { await erase(f); } catch (err) { footer(); return composer.toast.warn({ text: `Could not delete it: ${err.message}` }); }
    viewer.close();
    composer.toast.good({ text: `Deleted ${f.filename}.` });
    load();
  }

  async function erase(f) {
    const res = await fetch("/funpack/api/temp/delete", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ where, filename: f.filename, subfolder: f.subfolder || "" }) });
    const out = await res.json().catch(() => ({}));
    if (!res.ok || !out.deleted) throw new Error(out.detail || `HTTP ${res.status}`);
  }

  async function removePicked() {
    const list = files.filter((f) => picked.has(id(f)));
    const ok = await composer.modal.dialogue({ title: `Delete ${list.length} files`, tone: "danger", confirmLabel: `Delete ${list.length}`,
      message: `Delete ${list.length} files from ${where === "output" ? "your saved files" : "the temp folder"} for good? They do not go to the trash.` }).result;
    if (!ok) return;
    let gone = 0, done = 0, why = "";
    const bar = composer.progress.bar({ value: 0, max: list.length, label: "Deleting" });
    const say = composer.text.md({ text: "" });
    const tick = () => { bar.setValue(done); say.setText(`Deleting ${done} of ${list.length}…`); };
    tick();
    content.set([composer.region.stack({ gap: "md", children: [say, bar] })]);
    setFooter({ note: payload.path || "", actions: [composer.button.md({ label: "Deleting…", disabled: true })] });
    const queue = [...list];
    const worker = async () => {                // a few at a time: one request per file, and a network drive answers slowly
      for (let f = queue.shift(); f; f = queue.shift()) {
        try { await erase(f); gone += 1; } catch (err) { why = err.message; }
        done += 1; tick();
      }
    };
    await Promise.all([worker(), worker(), worker(), worker()]);
    picked.clear();
    selecting = false;
    (why ? composer.toast.warn : composer.toast.good)({ text: why ? `Deleted ${gone} of ${list.length}. ${why}` : `Deleted ${gone} files.` });
    load();
  }

  function footer() {
    setFooter({ note: payload.path || "", actions: [
      ...(looking ? [
        composer.button.md({ label: "Open in a tab", onClick: () => dom.openTab(url(looking, where)) }),
        composer.button.md({ label: "Delete", tone: "danger", onClick: () => remove(looking) }),
      ] : selecting ? [
        composer.button.md({ label: picked.size === files.length ? "Select none" : "Select all", onClick: () => { const all = picked.size !== files.length; picked.clear(); if (all) files.forEach((f) => picked.add(id(f))); grid.setValue([...picked]); footer(); } }),
        composer.button.md({ label: picked.size ? `Delete ${picked.size}` : "Delete", tone: "danger", disabled: !picked.size, onClick: removePicked }),
        composer.button.md({ label: "Cancel", onClick: () => { selecting = false; picked.clear(); grid.setValue([]); footer(); } }),
      ] : [composer.button.md({ label: "Select", disabled: !files.length, onClick: () => { selecting = true; footer(); } }), composer.button.md({ label: "Refresh", onClick: load })]),
      composer.button.md({ label: "Close", tone: "primary", onClick: close }),
    ] });
  }

  async function load() {
    looking = null; selecting = false; picked.clear();
    content.set([composer.hint.default({ text: "Looking…" })]);
    try {
      payload = await (await fetch(`/funpack/api/temp?where=${where}`, { cache: "no-store" })).json();
    } catch (err) {
      payload = {};
      content.set([composer.banner.danger({ text: `Could not read the folder: ${err.message}` })]);
      return footer();
    }
    files = payload.files || [];
    content.set([files.length
      ? (grid = composer.gallery.adaptive({
          id: "files", cols: 4,
          items: files.map((f) => ({
            id: id(f), label: f.filename, icon: ICON[f.kind] || "▦", badge: f.kind === "image" ? null : f.kind,
            hint: `${f.kind} · ${size(Number(f.size) || 0)} · ${when(f.mtime)}`,
            thumb: f.kind === "image" ? url(f, where) : null,
          })),
          onActivate: (cell) => {
            if (!selecting) return look(files.find((f) => id(f) === cell.id));
            if (!picked.delete(cell.id)) picked.add(cell.id);
            grid.setValue([...picked]); footer();
          },
          onContext: (cell, e) => {
            const f = files.find((x) => id(x) === cell.id);
            composer.menu.context({ x: e.clientX, y: e.clientY, items: [{ id: "look", label: "Look at it" }, { id: "tab", label: "Open in a tab" }, { separator: true }, { id: "del", label: "Delete…", danger: true }],
              onPick: (pick) => ({ look: () => look(f), tab: () => dom.openTab(url(f, where)), del: () => remove(f) })[pick]?.() });
          },
        }))
      : composer.emptyState.default({ icon: "▤", title: EMPTY[where], hint: payload.detail || "" })]);
    footer();
  }

  load();
  return body;
}
