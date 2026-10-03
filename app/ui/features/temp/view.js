// Where a file went when it did not land in the bin. ComfyUI wipes its temp folder on restart, so
// this is a place to FIND something (opens in a new tab), not to manage it.
import { composer } from "../../composer/composer.js";

const ICON = { image: "▦", video: "▶", audio: "♪" };
const url = (f) => `/view?${new URLSearchParams({ filename: f.filename, subfolder: f.subfolder || "", type: "temp" })}`;
const size = (n) => (n < 1024 ? `${n} B` : n < 1048576 ? `${Math.round(n / 1024)} KB` : `${(n / 1048576).toFixed(1)} MB`);

export function mount({ setFooter, close }) {
  const body = composer.region.stack({ gap: "sm", fill: true, label: "Temp files" });

  async function load() {
    body.set([composer.hint.default({ text: "Looking…" })]);
    let payload;
    try {
      payload = await (await fetch("/funpack/api/temp", { cache: "no-store" })).json();
    } catch (err) {
      body.set([composer.banner.danger({ text: `Could not read the temp directory: ${err.message}` })]);
      return;
    }
    const files = payload.files || [];
    const id = (f) => `${f.subfolder}/${f.filename}`;
    body.set([files.length
      ? composer.gallery.list({
          items: files.map((f) => ({
            id: id(f), label: f.filename, icon: ICON[f.kind] || "▦", badge: f.kind === "image" ? null : f.kind,
            hint: `${f.kind} · ${size(Number(f.size) || 0)} · ${new Date((Number(f.mtime) || 0) * 1000).toLocaleTimeString()}`,
            thumb: f.kind === "image" ? url(f) : null,
          })),
          onActivate: (cell) => window.open(url(files.find((f) => id(f) === cell.id)), "_blank", "noopener"),
        })
      : composer.emptyState.default({ icon: "▤", title: "Nothing in the temp directory", hint: payload.detail || "" })]);
    setFooter({ note: payload.path || "", actions: [
      composer.button.md({ label: "Refresh", onClick: load }),
      composer.button.md({ label: "Close", tone: "primary", onClick: close }),
    ] });
  }

  load();
  return body;
}
