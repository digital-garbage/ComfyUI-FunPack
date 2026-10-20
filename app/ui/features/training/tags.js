// Step 2, Tags: every item with its caption beside its picture, bulk edits, a tag cloud, and automatic tagging.
import { composer as c } from "../../composer/composer.js";
import { LABEL_TONES, LABEL_WORDS, clipWords, srcKey, stillOf } from "./state.js";
import { addTags, bulk, frequency, joinTags, splitTags } from "./captions.js";

const PAGE = 48;
const FILTERS = [
  { value: "all", label: "All" },
  { value: "untagged", label: "No tags yet" },
  { value: "selected", label: "Selected" },
];

export function tagsStep({ app, session, go, ui, peek }) {
  const doc = session.doc;
  if (!doc) return c.emptyState.default({ title: "No set open", hint: "Pick or make a training set in the Data step.", action: c.button.sm({ label: "Back to data", tone: "primary", onClick: () => go("data") }) });
  const all = doc.items;
  const sel = ui.selected;
  for (const id of [...sel]) if (!all.some((i) => i.id === id)) sel.delete(id);
  const cloud = frequency(all);

  // --- what is shown
  const shown = () => all.filter((i) => (ui.filter === "all" ? true : ui.filter === "untagged" ? !i.caption.trim() : sel.has(i.id))
    && (!ui.tag || splitTags(i.caption).some((t) => t.toLowerCase() === ui.tag.toLowerCase())));
  const list = shown();
  const count = c.text.sm({ text: "" });
  const sync = () => count.setText(`${sel.size} selected of ${all.length}${ui.tag ? ` · showing "${ui.tag}"` : ""}`);
  sync();

  const cards = list.slice(0, ui.limit).map((item) => {
    const card = c.captioned.cell({ title: item.name || item.src.filename || item.src.id, src: stillOf(item), kind: item.media_kind, caption: item.caption,
      badge: [LABEL_WORDS[item.label], clipWords(ui.clips[srcKey(item.src)])].filter(Boolean).join(" · "), tone: LABEL_TONES[item.label], selected: sel.has(item.id),
      onChange: (text) => session.edit(() => { item.caption = text; }),
      onSelect: (on) => { if (on) sel.add(item.id); else sel.delete(item.id); sync(); },
      onOpen: () => peek(item) });
    if (item.media_kind === "video") ui.needClips([item.src], () => card.setBadge([LABEL_WORDS[item.label], clipWords(ui.clips[srcKey(item.src)])].filter(Boolean).join(" · ")));
    return card;
  });
  const grid = cards.length ? c.grid.cards({ children: cards }) : c.hint.default({ text: ui.filter === "untagged" ? "Everything has tags." : "Nothing to show." });
  const more = list.length > ui.limit ? c.button.sm({ label: `Show ${Math.min(PAGE, list.length - ui.limit)} more (${list.length - ui.limit} left)`, tone: "ghost", onClick: () => { ui.limit += PAGE; ui.redraw(); } }) : null;

  // --- the bar over the grid
  const filter = c.segmented.sm({ label: "Show", options: FILTERS, value: ui.filter, onChange: (v) => { ui.filter = v; ui.limit = PAGE; ui.redraw(); } });
  const everyone = c.button.sm({ label: "Select all shown", tone: "ghost", onClick: () => { list.forEach((i) => sel.add(i.id)); ui.redraw(); } });
  const none = c.button.sm({ label: "Clear", tone: "ghost", onClick: () => { sel.clear(); ui.redraw(); } });
  const drop = c.button.sm({ label: "Remove selected", tone: "danger", title: "Take the selected items out of this set. The files themselves stay where they are.", onClick: async () => {
    if (!sel.size) { c.toast.info({ text: "Tick the items to remove first." }); return; }
    const n = sel.size;
    if (!(await c.modal.dialogue({ title: "Remove from set", message: `Remove ${n} item${n > 1 ? "s" : ""} from "${session.name}"? The files themselves are not deleted.`, tone: "danger", confirmLabel: "Remove" }).result)) return;
    session.edit((d) => { d.items = d.items.filter((i) => !sel.has(i.id)); });
    sel.clear();
    ui.redraw();
  } });

  // --- bulk edits on the selected (or all, when nothing is selected)
  const tagBox = c.input.md({ label: "Tag", placeholder: "tag" });
  const toBox = c.input.md({ label: "Replace with", placeholder: "replace with" });
  const target = () => (sel.size ? sel : null);
  const apply = (op) => {
    const word = tagBox.value;
    let n = 0;
    session.edit((d) => { n = bulk(d.items, op, word, toBox.value, target()); });
    c.toast.info({ text: n ? `Changed ${n} caption${n > 1 ? "s" : ""}${sel.size ? " (selected)" : " (all)"}.` : "Nothing changed." });
    ui.redraw();
  };
  const bar = c.toolbar.default({ label: "Bulk tag edits", items: [tagBox, toBox, c.button.sm({ label: "Add", tone: "neutral", title: "Add the tag at the end", onClick: () => apply("add") }),
    c.button.sm({ label: "Add first", tone: "neutral", title: "Put the tag at the start", onClick: () => apply("prepend") }),
    c.button.sm({ label: "Remove", tone: "neutral", onClick: () => apply("remove") }),
    c.button.sm({ label: "Replace", tone: "neutral", onClick: () => apply("replace") })] });

  // --- tag cloud: click to show only those, x to take the tag off everything
  const cloudEl = c.tags.cloud({ label: "Tags in this set", items: cloud, selected: ui.tag ? [ui.tag] : [], empty: "No tags yet. Type some, or use automatic tagging below.",
    onPick: (t) => { ui.tag = ui.tag && ui.tag.toLowerCase() === t.toLowerCase() ? "" : t; ui.limit = PAGE; ui.redraw(); },
    onRemove: async (t) => {
      if (!(await c.modal.dialogue({ title: "Remove tag", message: `Remove "${t}" from every caption in this set?`, tone: "danger", confirmLabel: "Remove" }).result)) return;
      session.edit((d) => { bulk(d.items, "remove", t, "", null); });
      ui.redraw();
    } });

  // --- text put before and after every caption when training (the set's own; the project's prefix and postfix are not used)
  const phrase = (key, label, hint) => c.field.default({ label, hint, control: c.input.md({ label, placeholder: "none", value: doc[key] || "",
    onCommit: (v) => session.edit((d) => { d[key] = v.split(/\s+/).filter(Boolean).join(" ").slice(0, 4000); }) }) });

  return c.region.stack({ gap: "md", children: [
    c.field.row({ fields: [
      phrase("trigger", "Prefix", "Put before every caption when training, for example a word to call the LoRA up later. Optional."),
      phrase("postfix", "Postfix", "Put after every caption when training. Optional."),
    ] }),
    c.group.default({ label: "Automatic tagging", rows: [autoTag({ app, session, ui })] }),
    c.group.default({ label: "Tags in this set", hint: "Click one to show only the items that have it. The x removes it from all.", rows: [cloudEl] }),
    c.toolbar.default({ items: [filter, count], trailing: [everyone, none, drop] }),
    bar,
    grid, more,
  ] });
}

// --- automatic tagging -------------------------------------------------------

function autoTag({ app, session, ui }) {
  const t = ui.tagger;
  const row = (t.list || []).find((x) => x.id === t.id) || (t.list || [])[0];
  const options = (t.list || []).map((x) => ({ value: x.id, label: `${x.title}${x.state === "ready" ? "" : " (not downloaded)"}` }));
  const pick = c.select.md({ label: "Tagger", options: options.length ? options : [{ value: "", label: "Loading" }], value: row ? row.id : "", onChange: (v) => { t.id = v; ui.redraw(); } });
  const state = row ? row.state : "absent";
  const fetching = state === "downloading";
  const pct = row && row.total ? Math.round((100 * row.done) / row.total) : 0;
  const download = c.button.sm({ label: state === "ready" ? "Downloaded" : fetching ? "Downloading" : state === "error" ? "Try the download again" : "Download", tone: "primary",
    disabled: !row || state === "ready" || fetching, onClick: async () => { try { await app.api.trainDownloadTagger(row.id); } catch (err) { c.toast.warn({ text: err.message }); } ui.pollTaggers(); } });
  const remove = c.button.sm({ label: "Delete download", tone: "ghost", onClick: async () => { await app.api.trainRemoveTagger(row.id); ui.pollTaggers(); } });
  remove.node.hidden = state !== "ready";
  const custom = c.button.sm({ label: "Use another model", tone: "ghost", title: "Any Hugging Face repo with model.onnx and selected_tags.csv, like owner/name", onClick: async () => {
    const answer = await c.modal.prompt({ title: "Another tagger", label: "Hugging Face repo (owner/name)", placeholder: "SmilingWolf/wd-eva02-large-tagger-v3", confirmLabel: "Add" }).result;
    if (!answer) return;
    t.id = String(answer).trim();
    try { await app.api.trainDownloadTagger(t.id); } catch (err) { c.toast.warn({ text: err.message }); }
    ui.pollTaggers();
  } });
  const info = c.hint.default({ text: row ? (fetching ? `Downloading ${pct}%` : state === "error" ? row.error : row.hint || "") : "" });

  const general = c.number.md({ label: "Confidence", value: t.general, min: 0.05, max: 0.95, step: 0.05, precision: 2, onChange: (v) => { t.general = v; } });
  const limit = c.number.md({ label: "Most tags per item", value: t.limit, min: 0, max: 200, step: 1, onChange: (v) => { t.limit = v; } });
  const characters = c.toggle.default({ label: "Include character names", checked: t.characters, onChange: (v) => { t.characters = v; } });
  const mode = c.segmented.sm({ label: "With existing tags", options: [{ value: "add", label: "Keep and add" }, { value: "replace", label: "Replace" }], value: t.mode, onChange: (v) => { t.mode = v; } });
  const scope = c.segmented.sm({ label: "Which items", options: [{ value: "untagged", label: "No tags yet" }, { value: "selected", label: "Selected" }, { value: "all", label: "All" }], value: t.scope, onChange: (v) => { t.scope = v; } });

  const job = t.job || { state: "idle" };
  const running = job.state === "running";
  const bar = running ? c.progress.bar({ value: job.done || 0, max: job.total || 1, label: "Tagging" }) : null;
  const run = c.button.md({ label: running ? "Stop tagging" : "Tag now", tone: running ? "danger" : "primary", disabled: !running && state !== "ready", onClick: async () => {
    if (running) { await app.api.trainAutotagCancel(); return; }
    await session.flush();
    const items = session.doc.items;
    const ids = items.filter((i) => (t.scope === "all" ? true : t.scope === "selected" ? ui.selected.has(i.id) : !i.caption.trim())).map((i) => i.id);
    if (!ids.length) { c.toast.info({ text: "No items match that choice." }); return; }
    try {
      await app.api.trainAutotag({ set: session.name, tagger: t.id || (row && row.id), ids, general: t.general, characters: t.characters, limit: t.limit });
      ui.pollTaggers(true);
    } catch (err) { c.toast.warn({ text: err.message }); }
  } });
  const why = job.state === "error" ? c.inlineError.default({ text: job.error }) : null;
  const problems = (job.problems || []).length ? c.hint.default({ text: `Skipped: ${job.problems.slice(0, 3).join("; ")}${job.problems.length > 3 ? "…" : ""}` }) : null;
  return c.region.stack({ gap: "sm", children: [
    c.toolbar.default({ items: [pick], trailing: [download, remove, custom] }), info,
    c.field.row({ fields: [c.field.default({ label: "Confidence", hint: "Higher keeps only sure tags.", control: general }), c.field.default({ label: "Most tags per item", hint: "0 means no limit.", control: limit })] }),
    characters, mode, scope, bar, why, problems, c.toolbar.default({ items: [], trailing: [run] }),
  ].filter(Boolean) });
}

/** Merge a finished tagging job into the open set. */
export function applyTagging(session, job, mode) {
  let n = 0;
  session.edit((d) => {
    for (const item of d.items) {
      const found = (job.results || {})[item.id];
      if (!found) continue;
      const text = mode === "replace" ? joinTags(found) : addTags(item.caption, found);
      if (text !== item.caption) { item.caption = text; n += 1; }
    }
  });
  return n;
}
