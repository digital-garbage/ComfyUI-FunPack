// The media library: files the user brought IN -- a reference image, an
// imported clip -- kept apart from the bin, which is everything a RUN
// produced. Same reason projects and media are separate stores on the
// server: the two ids look alike and mean different things.

import { composer } from "../composer/composer.js";

export const mediaUrl = (item) => `/funpack/api/media/${item.id}/file`;

const cellOf = (item) => ({
  id: item.id, label: item.name,
  thumb: item.kind === "image" ? mediaUrl(item) : null,
  icon: item.kind === "video" ? "▶" : item.kind === "audio" ? "♪" : "▦",
  badge: item.kind !== "image" ? item.kind : null,
  hint: item.kind,
});

/**
 * onPick is given the whole entry when an item is activated, so the caller
 * decides what that means -- today nothing consumes it, since no pipeline
 * stage takes an image input yet; the hook exists so wiring one up later is
 * not a rewrite of this file.
 */
export function createMediaLibrary({ onPick } = {}) {
  let items = [];

  const gallery = composer.gallery.adaptive({
    id: "media", items: [],
    empty: "Nothing imported yet. Drop a file above, or click it to browse.",
    onActivate: (item) => {
      gallery.setValue([item.id]);
      const found = items.find((i) => i.id === item.id);
      if (found && onPick) onPick(found);
    },
    // Right-click removes -- the bin has no delete at all today, but an
    // import the user reconsiders should not need a trip to the filesystem.
    onContext: (item) => remove(item.id),
  });

  const dropzone = composer.dropzone.default({
    label: "Import media", hint: "Images, clips, audio -- or drop a file here",
    accept: "image/*,video/*,audio/*",
    onFiles: (files) => upload(files),
  });

  const host = composer.region.stack({ gap: "sm", fill: true, label: "Media library",
    children: [dropzone, gallery] });

  function setItems(next) {
    items = next;
    gallery.setItems(items.map(cellOf));
  }

  async function refresh() {
    try {
      const res = await fetch("/funpack/api/media");
      const body = await res.json();
      setItems(body.media || []);
    } catch {
      composer.toast.danger({ text: "Could not load the media library." });
    }
  }

  async function upload(files) {
    const form = new FormData();
    for (const file of files) form.append("file", file, file.name);
    let body;
    try {
      const res = await fetch("/funpack/api/media", { method: "POST", body: form });
      if (!res.ok) {
        // A refusal FunPack's own route made comes back as JSON with a
        // reason. A 413 does not -- aiohttp refuses an oversized body before
        // the route ever runs, as plain text, and res.json() on that would
        // throw a parser error that read as the failure instead of the size
        // limit it actually was.
        if (res.status === 413) throw new Error("that file is too large for this server to accept");
        const problem = await res.json().catch(() => null);
        throw new Error((problem && problem.problems && problem.problems[0]) || `refused (${res.status})`);
      }
      body = await res.json();
    } catch (err) {
      composer.toast.danger({ text: `Could not import that: ${err.message}` });
      return;
    }
    setItems([...(body.media || []), ...items]);
    if (body.problems && body.problems.length) {
      composer.toast.warn({ text: `Imported ${body.media.length}, skipped: ${body.problems[0]}` });
    } else if (body.media.length) {
      composer.toast.good({ text: body.media.length > 1
        ? `Imported ${body.media.length} files.` : `Imported ${body.media[0].name}.` });
    }
  }

  async function remove(id) {
    const found = items.find((i) => i.id === id);
    setItems(items.filter((i) => i.id !== id));
    try {
      const res = await fetch(`/funpack/api/media/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error(`refused (${res.status})`);
      composer.toast.good({ text: found ? `Removed ${found.name}.` : "Removed." });
    } catch {
      composer.toast.danger({ text: "Could not remove that -- it may still be on the server." });
      refresh();
    }
  }

  // No fetch here: this is built once, up front, whether or not the tab
  // hosting it is ever opened -- exactly like every Settings section that
  // waits for its own mount() to be reached. The caller decides when
  // "reach the library" actually happens by calling refresh().
  return { host, refresh, get items() { return [...items]; } };
}
