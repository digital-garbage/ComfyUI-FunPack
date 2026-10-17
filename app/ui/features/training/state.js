// What the Training window shares between its steps: which set is open, the set itself, and saving it.
// A caption edited on screen is the truth: the server's copy follows a moment later, and tagger results are merged in by the window, not written behind its back.

const TRAIN = "/funpack/api/m/training_lora";
const SAVE_AFTER = 600;

export const LABEL_WORDS = { own: "Your media", liked: "Liked", disliked: "Disliked", bad_composition: "Bad composition", bad_image: "Bad image" };
export const LABEL_TONES = { own: "neutral", liked: "good", disliked: "bad", bad_composition: "bad", bad_image: "bad" };

/** A picture to show for an item: the bin's thumbnail, or the output file's still. */
export function stillOf(item) {
  const s = item.src;
  if (s.kind === "bin") return `/funpack/api/media/${encodeURIComponent(s.id)}/thumb`;
  const q = `filename=${encodeURIComponent(s.filename)}&subfolder=${encodeURIComponent(s.subfolder || "")}&type=${encodeURIComponent(s.type || "output")}`;
  return item.media_kind === "video" ? `/funpack/api/m/render/poster?${q}` : `/view?${q}`;
}

/** The key a source has on the server (store.src_key). */
export const srcKey = (src) => (src.kind === "bin" ? `bin:${src.id}` : `${src.type || "output"}:${src.subfolder || ""}/${src.filename}`);
/** "124 frames, 24 fps" for a clip whose numbers are known, else "". */
export const clipWords = (info) => (info ? `${info.frames} f · ${info.fps} fps` : "");

export function makeSession(api) {
  const listeners = new Set();
  const s = { name: "", doc: null, sets: [], error: "", saving: false, state: null };
  let timer = null, queued = null;

  const emit = () => listeners.forEach((f) => f());
  s.on = (f) => { listeners.add(f); return () => listeners.delete(f); };
  s.emit = emit;

  async function push() {
    if (!queued) return;
    const { name, doc } = queued;
    queued = null;
    s.saving = true;
    try { await api.trainSaveSet(name, doc); s.error = ""; } catch (err) { s.error = `Could not save the set: ${err.message}`; }
    s.saving = false;
    emit();
  }
  s.flush = async () => { clearTimeout(timer); timer = null; await push(); };
  /** Change the open set in place and save it soon. Does not redraw: the controls on screen already show the edit. */
  s.edit = (change) => {
    if (!s.doc) return;
    change(s.doc);
    queued = { name: s.name, doc: s.doc };
    clearTimeout(timer);
    timer = setTimeout(push, SAVE_AFTER);
  };
  s.refresh = async () => {
    try { s.sets = (await api.trainSets()).sets || []; s.error = ""; } catch (err) { s.error = err.message; }
    emit();
  };
  s.open = async (name) => {
    await s.flush();
    s.name = name || "";
    s.doc = null;
    if (name) {
      try { s.doc = await api.trainSet(name); s.error = ""; } catch (err) { s.error = err.message; s.name = ""; }
    }
    emit();
  };
  /** Take the server's version of the open set (after it changed there). */
  s.reload = async () => { await s.flush(); if (s.name) { try { s.doc = await api.trainSet(s.name); } catch (err) { s.error = err.message; } } await s.refresh(); };
  s.add = async (entries) => {
    if (!s.name) throw new Error("Make or pick a set first.");
    await s.flush();
    const out = await api.trainAdd(s.name, entries);
    s.doc = out.set;
    await s.refresh();
    return out.added;
  };
  s.create = async (name) => { await api.trainCreateSet(name); await s.refresh(); await s.open(name); };
  s.remove = async () => { if (s.name) { await api.trainDeleteSet(s.name); } await s.open(""); await s.refresh(); };
  s.counts = () => {
    const out = {};
    for (const item of (s.doc && s.doc.items) || []) out[item.label] = (out[item.label] || 0) + 1;
    return out;
  };
  return s;
}
