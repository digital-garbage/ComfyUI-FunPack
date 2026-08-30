// The media bin: everything this session has produced, and the one place a
// result is chosen from.
//
// Until this existed a second run replaced the first on screen with no way
// back -- the picture you were comparing against was simply gone, and the only
// record of it was a file on the server nobody in the app could name. So the
// bin accumulates across runs while `run.state.images` deliberately does not:
// the run's state is one generation, and the bin is the session.
//
// Three views, because the same twenty results are read three ways: a grid to
// compare them, a list to read filenames, icons to see many at once. The view
// is the only thing here that is remembered between visits, and it is a
// convenience -- a browser refusing storage still gets a working bin.

import { composer } from "../composer/composer.js";
import { viewUrl } from "./run.js";

const LS_KEY = "funpack.bin.view";

export const VIEWS = [
  { value: "grid", label: "▦", title: "Grid" },
  { value: "list", label: "☰", title: "List" },
  { value: "icons", label: "⣿", title: "Icons" },
];

const NAMES = new Set(VIEWS.map((v) => v.value));

const VIDEO = /\.(mp4|webm|mov|mkv|avi)$/i;

/** Video and image results arrive the same way and cannot be shown the same way. */
export const kindOf = (file) => (VIDEO.test((file && file.filename) || "") ? "video" : "image");

/**
 * A file's identity. ComfyUI answers with the same three fields every time, and
 * the same file can arrive twice -- a reload adopting a finished run replays
 * its outputs -- so an entry is keyed by the file rather than by arrival.
 */
export const keyOf = (file = {}) =>
  `${file.type || "output"}|${file.subfolder || ""}|${file.filename || ""}`;

function remember(view) {
  try { window.localStorage.setItem(LS_KEY, view); } catch { /* private mode */ }
}

export function recallView(fallback = "grid") {
  try {
    const raw = window.localStorage.getItem(LS_KEY);
    return NAMES.has(raw) ? raw : fallback;
  } catch {
    return fallback;
  }
}

/**
 * onOpen is given the whole entry plus the URL to fetch it from, so the caller
 * decides what "open" means -- today it is the viewer, later it will also be
 * dropping one onto a timeline.
 */
export function createBin({ onOpen, view = recallView(), persist = true,
                            empty = "Results appear here as you generate." } = {}) {
  let mode = NAMES.has(view) ? view : "grid";
  let items = [];                                  // newest first
  let selected = null;
  const seen = new Set();

  const host = composer.region.stack({ gap: "sm", label: "Media bin", fill: true });
  const control = composer.buttonGroup.md({
    label: "Bin view", value: mode, items: VIEWS,
    onChange: (next) => setView(next),
  });

  let shown = null;                                // the gallery for the current view

  // No <video> thumbnails, ever. v4 put live <video> elements in the bin and
  // Chrome's six-connections-per-origin pool wedged the whole API behind them
  // -- the app stopped answering while the bin loaded. A glyph and a badge say
  // what it is; the viewer is where a video plays.
  const cellOf = (item) => ({
    id: item.id,
    label: item.label,
    thumb: item.kind === "image" ? item.url : null,
    icon: item.kind === "video" ? "▶" : "▦",
    // Only the odd one out is labelled. A bin of images with "image" written
    // across every thumbnail says nothing and costs a corner of each picture.
    badge: item.kind === "video" ? "video" : null,
    hint: item.kind,
  });

  function build() {
    const props = {
      items: items.map(cellOf),
      selection: selected ? [selected] : [],
      empty,
      onActivate: (cell) => open(cell.id),
    };
    if (mode === "list") return composer.gallery.list(props);
    if (mode === "icons") return composer.gallery.icons(props);
    return composer.gallery.adaptive({ id: "bin", ...props });
  }

  function render() {
    const next = build();
    host.set([next]);
    if (shown && shown !== next && shown.destroy) shown.destroy();
    shown = next;
  }
  render();

  function setView(next) {
    if (!NAMES.has(next) || next === mode) return mode;
    mode = next;
    if (persist) remember(mode);
    control.setValue(mode);
    render();                                      // items and selection survive
    return mode;
  }

  /** Show one entry. Selection is the user's until a run produces something. */
  function open(id) {
    const item = items.find((i) => i.id === id);
    if (!item) return null;
    selected = id;
    if (shown) shown.setValue([id]);
    if (onOpen) onOpen(item);
    return item;
  }

  /**
   * Take in whatever a run has produced so far.
   *
   * Called on every state change, which means it is called far more often than
   * anything new arrives -- so it must be silent when there is nothing new. It
   * was not, once: redrawing on each message walked the selection back to the
   * newest result the moment the user clicked an older one.
   */
  function absorb(files = []) {
    const added = [];
    for (const file of files || []) {
      const id = keyOf(file);
      if (seen.has(id)) continue;
      seen.add(id);
      const item = { id, file, kind: kindOf(file), label: file.filename || "(unnamed)",
                     url: viewUrl(file) };
      items = [item, ...items];
      added.push(item);
    }
    if (!added.length) return added;

    if (shown) shown.setItems(items.map(cellOf));
    // The newest of this batch, which is now at the head. A finished run showing
    // its own result is the whole point; an older selection is not preserved
    // through it, because nobody generates in order to keep looking at the last
    // one.
    open(added[added.length - 1].id);
    return added;
  }

  return {
    host, control,
    absorb, open, setView,
    get view() { return mode; },
    get items() { return [...items]; },
    get selected() { return selected; },
  };
}
