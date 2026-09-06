// Where a file went when it did not land in the bin.
//
// ComfyUI writes previews and other transient outputs to a temp directory and
// wipes it on restart. Everything in here is therefore findable for exactly as
// long as the server has been up, which is the window in which somebody asks
// "where did that go?".
//
// Read-only on purpose: this is a place to FIND something, not to manage it.
// Deleting temp files is ComfyUI's job and it does it on restart.

import { composer } from "../composer/composer.js";
import { viewUrl } from "./run.js";

const KIND_ICON = { image: "▦", video: "▶", audio: "♪" };

/** 1.4 MB, not 1468006. A size is read to compare two files, not to audit one. */
function size(bytes) {
  const n = Number(bytes) || 0;
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

const when = (mtime) => new Date((Number(mtime) || 0) * 1000).toLocaleTimeString();

export function open({ onOpen } = {}) {
  let window_ = null;
  const body = composer.region.stack({ gap: "sm", fill: true, label: "Temp files" });

  const cells = (files) => files.map((f) => ({
    // The full address, so two files with the same name in different subfolders
    // are two entries rather than one that wins.
    id: `${f.subfolder}/${f.filename}`,
    label: f.filename,
    hint: `${f.kind} · ${size(f.size)} · ${when(f.mtime)}`,
    thumb: f.kind === "image" ? viewUrl({ ...f, type: "temp" }) : null,
    icon: KIND_ICON[f.kind] || "▦",
    badge: f.kind === "image" ? null : f.kind,
  }));

  async function load() {
    body.set([composer.hint.default({ text: "Looking…" })]);
    let payload;
    try {
      const res = await fetch("/funpack/api/temp", { cache: "no-store" });
      payload = await res.json();
    } catch (err) {
      body.set([composer.banner.danger({ text: `Could not read the temp directory: ${err.message}` })]);
      return;
    }

    const files = payload.files || [];
    body.set([files.length
      // Rows, not tiles: this window is for finding a file, and a row shows the
      // whole name, its size and when it was written. The grid draws none of
      // those -- it is for looking at pictures, which is the bin's job.
      ? composer.gallery.list({
          items: cells(files),
          onActivate: (cell) => {
            const file = files.find((f) => `${f.subfolder}/${f.filename}` === cell.id);
            if (!file || !onOpen) return;
            onOpen({ url: viewUrl({ ...file, type: "temp" }), kind: file.kind, file });
            if (window_) window_.close("opened");
          },
        })
      // The reason there is nothing, which is not the same as "nothing was made".
      : composer.emptyState.default({
          icon: "▤", title: "Nothing in the temp directory",
          hint: payload.detail || "",
        })]);

    if (window_) {
      window_.setFooter({
        note: payload.path || "",
        actions: [
          composer.button.md({ label: "Refresh", onClick: () => load() }),
          composer.button.md({ label: "Close", tone: "primary",
                               onClick: () => window_.close("done") }),
        ],
      });
    }
  }

  window_ = composer.modal.generic({
    title: "Temp files",
    subtitle: "Transient ComfyUI outputs. Wiped when the server restarts.",
    size: "xl",
    body,
    onClose: () => { window_ = null; },
  });
  load();
  return window_;
}
