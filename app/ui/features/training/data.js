// Step 1, Training data: pick or make a set, then put media in it from any mix of sources.
// Skippable: a set that already has what you want goes straight on to tags or to training.
import { composer as c } from "../../composer/composer.js";
import { LABEL_WORDS, clipWords, srcKey, stillOf } from "./state.js";

export const RATED = [
  { value: "liked", label: "Liked results" },
  { value: "disliked", label: "Disliked results" },
  { value: "bad_composition", label: "Bad composition results" },
  { value: "bad_image", label: "Bad image results" },
  { value: "all", label: "All" },
];
const ALL_RATED = ["liked", "disliked", "bad_composition", "bad_image"];

export const binEntry = (m) => ({ src: { kind: "bin", id: m.id }, media_kind: m.kind === "video" ? "video" : "image", label: "own", name: m.name });
export const INCLUDE = [{ value: "prefix", label: "Prefix" }, { value: "main", label: "Main prompt" }, { value: "postfix", label: "Postfix" }, { value: "variables", label: "Variables" }];

export const ratedEntry = (r) => ({ src: { kind: "output", type: r.media.type, subfolder: r.media.subfolder, filename: r.media.filename },
  media_kind: /\.(mp4|webm|mov|mkv|avi)$/i.test(r.media.filename) ? "video" : "image", label: r.label, caption: r.caption || "", name: r.media.filename });

const countsLine = (counts) => Object.keys(LABEL_WORDS).filter((k) => counts[k]).map((k) => `${counts[k]} ${LABEL_WORDS[k].toLowerCase()}`).join(", ");

/** Builds the step. `ui` keeps what is picked between redraws. */
export function dataStep({ app, session, go, ui }) {
  const added = async (entries, what) => {
    if (!entries.length) { c.toast.info({ text: `Nothing to add from ${what}.` }); return; }
    try {
      const n = await session.add(entries);
      c.toast.info({ text: n ? `Added ${n} to "${session.name}".` : "They were all in the set already." });
    } catch (err) { c.toast.warn({ text: err.message }); }
  };

  const names = session.sets.map((x) => ({ value: x.name, label: `${x.name} (${x.count})` }));
  const setSelect = c.select.md({ label: "Training set", options: [{ value: "", label: names.length ? "Pick a set" : "No sets yet" }, ...names], value: session.name, onChange: (v) => session.open(v) });
  const make = c.button.sm({ label: "New set", tone: "neutral", onClick: async () => {
    const answer = await c.modal.prompt({ title: "New training set", label: "Name", placeholder: "my_style", confirmLabel: "Create" }).result;
    if (!answer) return;
    try { await session.create(String(answer).trim()); } catch (err) { c.toast.warn({ text: err.message }); }
  } });
  const drop = c.button.sm({ label: "Delete set", tone: "danger", onClick: async () => {
    if (!(await c.modal.dialogue({ title: "Delete training set", message: `Delete "${session.name}"? Your media and results stay where they are.`, tone: "danger", confirmLabel: "Delete" }).result)) return;
    await session.remove();
  } });
  drop.node.hidden = !session.name;

  const open = !!session.name;
  const counts = session.counts();
  const total = ((session.doc || {}).items || []).length;
  const status = c.hint.default({ text: open ? (total ? `${total} in "${session.name}": ${countsLine(counts)}.` : `"${session.name}" is empty. Add something below, or go on and add later.`) : "Make a set first. It is only a saved list, nothing is copied or changed." });

  const dropzone = c.dropzone.default({ label: "Drop pictures or clips here", hint: "They are kept in your Bin and added to this set", accept: "image/*,video/*", onFiles: async (files) => {
    if (!open) { c.toast.warn({ text: "Make or pick a set first." }); return; }
    const out = await app.api.uploadMedia(files);
    if (out.problems && out.problems.length) c.toast.warn({ text: out.problems.join("; ") });
    if (app.say) app.say("media");
    await added((out.media || []).filter((m) => m.kind === "image" || m.kind === "video").map(binEntry), "the drop");
  } });

  const picked = ui.picked;
  const pickedNote = c.text.sm({ text: `${picked.size} picked` });
  const usable = ui.bin.filter((m) => m.kind === "image" || m.kind === "video");
  const binSrc = (m) => ({ kind: "bin", id: m.id });
  const binItems = () => usable.map((m) => ({ id: m.id, label: m.name, thumb: `/funpack/api/media/${encodeURIComponent(m.id)}/thumb`, badge: m.kind === "video" ? clipWords(ui.clips[srcKey(binSrc(m))]) || m.kind : m.kind }));
  const gallery = c.gallery.adaptive({ id: "training-bin", cols: 3, items: binItems(), selection: [...picked],
    empty: "Your Bin has no pictures or clips yet", onActivate: (item) => {
      if (picked.has(item.id)) picked.delete(item.id); else picked.add(item.id);
      gallery.setValue([...picked]);
      pickedNote.setText(`${picked.size} picked`);
      addBin.node.disabled = !picked.size;
    } });
  ui.needClips(usable.filter((m) => m.kind === "video").map(binSrc), () => { gallery.setItems(binItems()); gallery.setValue([...picked]); });
  const addBin = c.button.sm({ label: "Add picked", tone: "primary", disabled: !picked.size || !open, onClick: async () => {
    const chosen = usable.filter((m) => picked.has(m.id));
    await added(chosen.map(binEntry), "the Bin");
    picked.clear();
  } });

  const want = () => (ui.ratedKind === "all" ? ALL_RATED : [ui.ratedKind]);
  const shownRated = () => (ui.rated.results || []).filter((r) => want().includes(r.label));
  const keyOf = (r) => `${r.media.type}:${r.media.subfolder}/${r.media.filename}`;
  const ratedPicked = ui.ratedPicked;
  const ratedNote = c.text.sm({ text: "" });
  const ratedGallery = c.gallery.adaptive({ id: "training-rated", cols: 3, items: [], selection: [...ratedPicked], empty: "None found yet. Rate results in your projects and they appear here.", onActivate: (item) => {
    if (ratedPicked.has(item.id)) ratedPicked.delete(item.id); else ratedPicked.add(item.id);
    ratedGallery.setValue([...ratedPicked]);
    showRated();
  } });
  const addRated = c.button.sm({ label: "Add these results", tone: "primary", disabled: true, onClick: async () => {
    const all = shownRated(), chosen = all.filter((r) => ratedPicked.has(keyOf(r)));
    await added((chosen.length ? chosen : all).map(ratedEntry), "your results");
    ratedPicked.clear();
    showRated();
  } });
  function showRated() {
    const all = shownRated();
    for (const k of [...ratedPicked]) if (!all.some((r) => keyOf(r) === k)) ratedPicked.delete(k);
    ratedGallery.setItems(all.map((r) => ({ id: keyOf(r), label: r.media.filename, thumb: stillOf({ src: { kind: "output", ...r.media }, media_kind: r.media.filename.match(/\.(mp4|webm|mov|mkv|avi)$/i) ? "video" : "image" }), badge: [r.label, r.doubtful ? "old rating?" : "", clipWords(ui.clips[keyOf(r)])].filter(Boolean).join(" · ") })));
    ui.needClips(all.filter((r) => /\.(mp4|webm|mov|mkv|avi)$/i.test(r.media.filename)).map((r) => ({ kind: "output", ...r.media })), () => showRated());
    ratedGallery.setValue([...ratedPicked]);
    const n = all.length, p = ratedPicked.size;
    const other = (ui.rated.results || []).filter((r) => r.doubtful).length;
    const left = other ? ` ${other} marked "old rating?": the rating may belong to another video that had this name on another ComfyUI instance. Its caption is still the prompt of the file you see.` : "";
    ratedNote.setText((n ? `${n} rated result${n > 1 ? "s" : ""} found in your projects. ${p ? `${p} picked: only those are added.` : "None picked: all are added."}` : "None found yet. Rate results in your projects and they appear here.") + left);
    addRated.node.disabled = !n || !open;
  }
  const include = c.field.default({ label: "Captions include", hint: "What goes into the caption from each result's prompt. Shortcuts are expanded. Variables are filled in only when ticked, otherwise $name stays as written. Nothing ticked adds them with no caption.",
    control: c.checklist.default({ label: "Captions include", items: INCLUDE, values: ui.include, onChange: async (v) => {
      ui.include = v;
      try { ui.rated = await app.api.trainRated(v.length ? v : ["none"]); } catch { /* the old captions stay */ }
      showRated();
    } }) });
  const ratedSel = c.segmented.sm({ label: "Which results", options: RATED, value: ui.ratedKind, onChange: (v) => { ui.ratedKind = v; showRated(); } });
  showRated();

  const next = c.toolbar.default({ items: [c.hint.default({ text: "This step is optional: a set you made earlier is ready as it is." })], trailing: [
    c.button.md({ label: "Skip to training", tone: "ghost", onClick: () => go("train") }),
    c.button.md({ label: "Next: tags", tone: "primary", disabled: !open, onClick: () => go("tags") })] });

  return c.region.stack({ gap: "md", children: [
    c.field.default({ label: "Training set", hint: "A saved list of what to learn from.", control: c.region.stack({ gap: "sm", children: [setSelect, c.toolbar.default({ items: [make, drop] })] }) }),
    status,
    c.group.default({ label: "Drop files", rows: [dropzone] }),
    c.collapsible.default({ label: "Choose in your Bin", open: ui.openBin, onToggle: (o) => { ui.openBin = o; }, body: c.region.stack({ gap: "sm", children: [gallery, c.toolbar.default({ items: [pickedNote], trailing: [addBin] })] }) }),
    c.collapsible.default({ label: "Choose your own results", hint: "Results you rated in your projects. Liked ones teach what you want; disliked ones can teach what to avoid.", open: ui.openRated, onToggle: (o) => { ui.openRated = o; },
      body: c.region.stack({ gap: "sm", children: [ratedSel, ratedGallery, include, c.toolbar.default({ items: [ratedNote], trailing: [addRated] })] }) }),
    next,
  ] });
}
