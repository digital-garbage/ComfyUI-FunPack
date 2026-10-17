// Step 1, Training data: pick or make a set, then put media in it from any mix of sources.
// Skippable: a set that already has what you want goes straight on to tags or to training.
import { composer as c } from "../../composer/composer.js";
import { LABEL_WORDS } from "./state.js";

export const RATED = [
  { value: "liked", label: "Liked results" },
  { value: "disliked", label: "Disliked results" },
  { value: "bad_composition", label: "Bad composition results" },
  { value: "bad_image", label: "Bad image results" },
  { value: "all", label: "All" },
];
const ALL_RATED = ["liked", "disliked", "bad_composition", "bad_image"];

export const binEntry = (m) => ({ src: { kind: "bin", id: m.id }, media_kind: m.kind === "video" ? "video" : "image", label: "own", name: m.name });
export const ratedEntry = (r, withCaption = true) => ({ src: { kind: "output", type: r.media.type, subfolder: r.media.subfolder, filename: r.media.filename },
  media_kind: /\.(mp4|webm|mov|mkv|avi)$/i.test(r.media.filename) ? "video" : "image", label: r.label, caption: withCaption ? r.caption || "" : "", name: r.media.filename });

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
  const gallery = c.gallery.adaptive({ id: "training-bin", items: usable.map((m) => ({ id: m.id, label: m.name, thumb: `/funpack/api/media/${encodeURIComponent(m.id)}/thumb`, badge: m.kind })), selection: [...picked],
    empty: "Your Bin has no pictures or clips yet", onActivate: (item) => {
      if (picked.has(item.id)) picked.delete(item.id); else picked.add(item.id);
      gallery.setValue([...picked]);
      pickedNote.setText(`${picked.size} picked`);
      addBin.node.disabled = !picked.size;
    } });
  const addBin = c.button.sm({ label: "Add picked", tone: "primary", disabled: !picked.size || !open, onClick: async () => {
    const chosen = usable.filter((m) => picked.has(m.id));
    await added(chosen.map(binEntry), "the Bin");
    picked.clear();
  } });

  const want = () => (ui.ratedKind === "all" ? ALL_RATED : [ui.ratedKind]);
  const ratedCount = () => want().reduce((a, k) => a + ((ui.rated.counts || {})[k] || 0), 0);
  const ratedNote = c.text.sm({ text: "" });
  const addRated = c.button.sm({ label: "Add these results", tone: "primary", disabled: true, onClick: async () => {
    await added((ui.rated.results || []).filter((r) => want().includes(r.label)).map((r) => ratedEntry(r, ui.useCaptions)), "your results");
  } });
  const showRated = () => {
    const n = ratedCount();
    ratedNote.setText(n ? `${n} rated result${n > 1 ? "s" : ""} found in your projects.` : "None found yet. Rate results in your projects and they appear here.");
    addRated.node.disabled = !n || !open;
  };
  const useCaptions = c.toggle.default({ label: "Use their prompts as captions", hint: "The prompt each result was made from, with shortcuts expanded. Variables, prefixes and postfixes are left out.", checked: ui.useCaptions, onChange: (v) => { ui.useCaptions = v; } });
  const ratedSel = c.segmented.sm({ label: "Which results", options: RATED, value: ui.ratedKind, onChange: (v) => { ui.ratedKind = v; showRated(); } });
  showRated();

  const next = c.toolbar.default({ items: [c.hint.default({ text: "This step is optional: a set you made earlier is ready as it is." })], trailing: [
    c.button.md({ label: "Skip to training", tone: "ghost", onClick: () => go("train") }),
    c.button.md({ label: "Next: tags", tone: "primary", disabled: !open, onClick: () => go("tags") })] });

  return c.region.stack({ gap: "md", children: [
    c.field.default({ label: "Training set", hint: "A saved list of what to learn from.", control: c.region.stack({ gap: "sm", children: [setSelect, c.toolbar.default({ items: [make, drop] })] }) }),
    status,
    c.group.default({ label: "Drop files", rows: [dropzone] }),
    c.group.default({ label: "Choose in your Bin", rows: [gallery, c.toolbar.default({ items: [pickedNote], trailing: [addBin] })] }),
    c.group.default({ label: "Choose your own results", hint: "Results you rated in your projects. Liked ones teach what you want; disliked ones can teach what to avoid.", rows: [ratedSel, useCaptions, c.toolbar.default({ items: [ratedNote], trailing: [addRated] })] }),
    next,
  ] });
}
