// Shortcuts: the library of trigger -> replacement phrases, expanded into a prompt when it is generated.
import { composer as c } from "../../composer/composer.js";

const lines = (text) => String(text || "").split("\n").map((l) => l.trim()).filter(Boolean);

export function shortcuts(app, own) {
  const page = c.region.stack({ gap: "sm" });
  let alive = true;
  own(() => { alive = false; });
  const tell = (text) => c.toast.warn({ text });

  function edit(existing) {
    const draft = { name: "", triggers: [], replacements: [], enabled: true, category: "", sub_category: "", ...(existing || {}) };
    const field = (label, control, hint) => c.field.default({ label, hint, control });
    const win = c.modal.generic({ title: existing ? `Edit “${existing.name}”` : "New shortcut", size: "md", body: c.region.stack({ gap: "sm", children: [
      field("Name", c.input.md({ label: "Name", value: draft.name, onInput: (v) => { draft.name = v; } })),
      field("Triggers", c.textarea.md({ label: "Triggers", rows: 3, value: draft.triggers.join("\n"), onInput: (v) => { draft.triggers = lines(v); } }), "One word or phrase per line. Any of them fires the shortcut."),
      field("Replacements", c.textarea.md({ label: "Replacements", rows: 5, value: draft.replacements.join("\n"), onInput: (v) => { draft.replacements = lines(v); } }), "One per line; with several, one is drawn each time."),
      c.field.row({ fields: [field("Category", c.input.md({ label: "Category", value: draft.category, onInput: (v) => { draft.category = v; } })), field("Sub-category", c.input.md({ label: "Sub-category", value: draft.sub_category, onInput: (v) => { draft.sub_category = v; } }))] }),
      c.toggle.default({ label: "Enabled", checked: draft.enabled, onChange: (v) => { draft.enabled = v; } }),
    ] }) });
    const save = async () => {
      try { draw(await app.api.saveShortcut(draft, existing && existing.name)); win.close("done"); } catch (err) { tell(err.message); }
    };
    const remove = async () => {
      if (!(await c.modal.dialogue({ title: "Delete shortcut", message: `Delete “${existing.name}”?`, tone: "danger", confirmLabel: "Delete" }).result)) return;
      try { draw(await app.api.deleteShortcut(existing.name)); win.close("done"); } catch (err) { tell(err.message); }
    };
    win.setFooter({ actions: [existing ? c.button.sm({ label: "Delete", tone: "danger", onClick: remove }) : null, c.button.sm({ label: "Save", tone: "primary", onClick: save })].filter(Boolean) });
  }

  /** Export / import / new category / delete all. */
  function tools(anchor, lib) {
    const did = (p) => p.then(draw).catch((err) => tell(err.message));
    c.menu.dropdown({ anchor: anchor || document.body, items: [{ id: "export", label: "Export all…" }, { id: "import", label: "Import…" }, { id: "category", label: "New category…" }, { separator: true }, { id: "clear", label: "Delete all", danger: true, disabled: !(lib.shortcuts || []).length }],
      onPick: async (id) => {
        if (id === "export") return Object.assign(document.createElement("a"), { href: "/funpack/api/shortcuts/export", download: "funpack_shortcuts.json" }).click();
        if (id === "import") {
          const input = Object.assign(document.createElement("input"), { type: "file", accept: ".json,application/json" });
          input.onchange = async () => {
            const file = input.files && input.files[0];
            if (!file) return;
            let data;
            try { data = JSON.parse(await file.text()); } catch { return tell("That file is not valid JSON."); }
            c.modal.choice({ title: "Import shortcuts", subtitle: "Merge them into your library, or replace the library with them?",
              items: [{ id: "merge", label: "Merge into my library" }, { id: "replace", label: "Replace my library (deletes the shortcuts you have now)" }],
              onPick: (mode) => did(app.api.importShortcuts(data, mode)) });       // closing the window imports nothing
          };
          return input.click();
        }
        if (id === "category") {
          const category = await c.modal.prompt({ title: "New category", label: "Category", confirmLabel: "Next" }).result;
          if (!category || !category.trim()) return;
          const sub = await c.modal.prompt({ title: "Sub-category", label: "Sub-category (optional)", confirmLabel: "Add" }).result;
          if (sub === null || sub === undefined) return;
          return did(app.api.addShortcutCategory(category.trim(), (sub || "").trim()));
        }
        if (await c.modal.dialogue({ title: "Delete all shortcuts", message: "Delete every shortcut in the library? Export first if you might want them back.", tone: "danger", confirmLabel: "Delete all" }).result) did(app.api.clearShortcuts());
      } });
  }

  function draw(lib) {
    if (!alive) return;
    const items = (lib.shortcuts || []).map((s) => ({ id: s.name, label: `${s.enabled ? "" : "⏸ "}${s.name}`, hint: s.triggers.join(", "), keywords: `${s.category} ${s.sub_category} ${s.replacements.join(" ")}`, group: s.category || "" }));
    items.sort((x, y) => x.group.localeCompare(y.group) || x.label.localeCompare(y.label));
    const more = c.button.sm({ label: "⋯", tone: "ghost", title: "Export, import, categories, delete all", onClick: () => tools(more.node, lib) });
    page.set([
      c.toolbar.default({ items: [c.label.section({ text: "Shortcuts" })], trailing: [c.button.sm({ label: "+ New", tone: "primary", onClick: () => edit(null) }), more] }),
      c.filterList.md({ items, placeholder: "Search shortcuts", empty: "No shortcuts yet. Add one with + New.", onChange: (id) => edit((lib.shortcuts || []).find((s) => s.name === id)) }),
      c.hint.default({ text: "Type a trigger in any prompt and it is replaced when you generate." }),
    ]);
  }
  app.api.shortcuts().then(draw).catch((err) => page.set([c.banner.warn({ text: `Could not load shortcuts: ${err.message}` })]));
  return page;
}
