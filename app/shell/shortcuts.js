// The shortcut library: trigger -> replacement text, expanded into a scene's
// prompt at generation time (core/prompt_build.py, core/shortcuts.py). Global,
// not per-project -- a shortcut is reused across every project, so it lives in
// Settings beside Node packs and Updates rather than in the project's own
// properties column.
//
// Same mount({setFooter, close}) / open() shape as packs.js: content built
// once, usable standalone or hosted inside settings_window.js.

import { composer } from "../composer/composer.js";

const BASE = "/funpack/api/shortcuts";

async function ask(method, path = "", body) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((payload.problems && payload.problems[0]) || `HTTP ${res.status}`);
  return payload;
}

//: A short-lived cache so autocomplete.js and the run-queue expansion step
//: are not each holding (and re-fetching) their own copy. Invalidated by
//: every write this file makes; a write from elsewhere (there is none today)
//: would only go stale for the length of one generation.
let cached = null;

export async function fetchAll(force = false) {
  if (cached && !force) return cached;
  try {
    cached = (await ask("GET")).shortcuts || [];
  } catch {
    cached = cached || [];
  }
  return cached;
}

const linesOf = (v) => String(v || "").split("\n").map((s) => s.trim()).filter(Boolean);

/** mount({ setFooter, close }) -> a composer handle. */
export function mount({ setFooter = () => {}, close = () => {} } = {}) {
  let items = null;
  let editing = null;   // the item being edited/added, or null
  let problem = null;

  function startAdd() { editing = { name: "", triggers: [], replacements: [], enabled: true, category: "", sub_category: "" }; draw(); }
  function startEdit(item) { editing = { ...item, original_name: item.name }; draw(); }
  function cancelEdit() { editing = null; draw(); }

  async function save(payload) {
    problem = null;
    try {
      items = (await ask("POST", "", payload)).shortcuts;
      editing = null;
    } catch (err) {
      problem = err.message;
    }
    cached = items;
    draw();
  }

  async function remove(name) {
    problem = null;
    try {
      items = (await ask("DELETE", `/${encodeURIComponent(name)}`)).shortcuts;
    } catch (err) {
      problem = err.message;
    }
    cached = items;
    draw();
  }

  function editForm() {
    const e = editing;
    const nameIn = composer.input.md({ label: "Name", value: e.name, placeholder: "Golden hour" });
    const trigIn = composer.textarea.md({ label: "Triggers", rows: 2,
      value: e.triggers.join("\n"), placeholder: "golden hour\nsunset light" });
    const repIn = composer.textarea.md({ label: "Replacements", rows: 3,
      value: e.replacements.join("\n"),
      placeholder: "warm golden-hour lighting\n(one per line -- more than one picks randomly)" });
    const catIn = composer.input.md({ label: "Category", value: e.category, placeholder: "Lighting" });
    const subIn = composer.input.md({ label: "Sub-category", value: e.sub_category, placeholder: "" });
    const enabledCb = composer.toggle.default({ label: "Enabled", checked: e.enabled !== false });

    const doSave = () => {
      const triggers = linesOf(trigIn.value);
      if (!triggers.length) { problem = "At least one trigger is required."; draw(); return; }
      save({
        name: nameIn.value.trim() || triggers[0],
        triggers,
        replacements: String(repIn.value || "").split("\n").map((s) => s.trim()),
        enabled: enabledCb.value,
        category: catIn.value.trim(),
        sub_category: subIn.value.trim(),
        original_name: e.original_name,
      });
    };

    return composer.region.stack({ gap: "sm", children: [
      composer.label.section({ text: e.original_name ? `Edit "${e.original_name}"` : "New shortcut" }),
      composer.field.default({ label: "Name", control: nameIn }),
      composer.field.default({ label: "Triggers", hint: "One per line, or what's typed to match.", control: trigIn }),
      composer.field.default({ label: "Replacements", hint: "One per line; more than one picks at random each time.", control: repIn }),
      composer.field.default({ label: "Category", control: catIn }),
      composer.field.default({ label: "Sub-category", control: subIn }),
      enabledCb,
      composer.toolbar.default({ label: "Edit shortcut", items: [
        composer.button.md({ label: "Save", tone: "primary", onClick: doSave }),
        composer.button.md({ label: "Cancel", onClick: cancelEdit }),
      ] }),
    ] });
  }

  function rows() {
    const out = [];
    if (problem) out.push(composer.banner.danger({ text: problem }));
    if (editing) out.push(editForm());

    if (items == null) {
      out.push(composer.hint.default({ text: "Loading…" }));
      return out;
    }
    if (!items.length && !editing) {
      out.push(composer.hint.default({ text: "No shortcuts yet. Add one below." }));
    }
    for (const item of items) {
      const tag = [item.category, item.sub_category].filter(Boolean).join(" · ");
      out.push(composer.settingsRow.default({
        label: item.name + (item.enabled === false ? " (off)" : "") + (tag ? ` — ${tag}` : ""),
        hint: `${item.triggers.join(", ")} → ${item.replacements.join(" / ") || "(removes the phrase)"}`,
        control: composer.toolbar.default({ label: item.name, items: [
          composer.button.sm({ label: "Edit", onClick: () => startEdit(item) }),
          composer.button.sm({ label: "Delete", tone: "danger", onClick: () => remove(item.name) }),
        ] }),
      }));
    }
    return out;
  }

  const body = composer.region.stack({ gap: "sm", label: "Shortcuts" });
  function draw() {
    body.set(rows());
    setFooter({
      note: items ? `${items.length} shortcut${items.length === 1 ? "" : "s"}` : "",
      actions: [
        composer.button.md({ label: "+ Add shortcut", disabled: Boolean(editing), onClick: startAdd }),
        composer.button.md({ label: "Close", tone: "primary", onClick: close }),
      ],
    });
  }

  async function load() {
    try { items = await fetchAll(true); }
    catch (err) { items = []; problem = err.message; }
    draw();
  }

  load();
  return body;
}

export function open() {
  let window_ = null;
  const body = mount({
    setFooter: (f) => window_ && window_.setFooter(f),
    close: () => window_ && window_.close("done"),
  });
  window_ = composer.modal.generic({
    title: "Shortcuts", subtitle: "Trigger words that expand into a prompt at generation.",
    size: "lg", body,
    onClose: () => { window_ = null; },
  });
  return window_;
}
