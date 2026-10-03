// The media bin as a choice, for fields that need one picture or several: a window listing what is in it.
import { composer as c } from "../../composer/composer.js";

export const names = new Map();          // id -> name, remembered from the last look at the bin
let asked = false;

export async function look(api) {
  const bin = (await api.media()).media || [];
  bin.forEach((m) => names.set(m.id, m.name));
  return bin;
}

/** The names behind some ids; looks the bin up once if any are not known yet, then says so with `again`. */
export function namesOf(ids, api, again) {
  if (!asked && ids.some((id) => !names.has(id))) { asked = true; look(api).then(again, () => {}); }
  return ids.map((id) => names.get(id) || "…");
}

/** -> the chosen ids in the order picked, or null when the window was closed. `multiple` false gives at most one. */
export async function pick(api, { title, kinds, multiple, current = [] }) {
  asked = false;
  const bin = (await look(api)).filter((m) => kinds.includes(m.kind));
  if (!bin.length) { c.toast.warn({ text: `Nothing to pick: the media bin has no ${kinds.join(" or ")} yet.` }); return null; }
  const items = bin.map((m) => ({ value: m.id, label: `${m.name} (${m.kind})` }));
  return new Promise((resolve) => {
    let chosen = multiple ? current.filter((id) => names.has(id)) : [];
    const body = multiple ? c.checklist.default({ label: title, items, values: chosen, onChange: (v) => { chosen = v; } })
      : c.checklist.default({ label: title, items, values: [], onChange: (v) => { chosen = v.filter((id) => !chosen.includes(id)).slice(-1); win.close("done"); } });
    let done = false;
    const finish = (value) => { if (!done) { done = true; resolve(value); } };
    const win = c.modal.generic({ title, size: "sm", body, onClose: (why) => finish(why === "done" ? chosen : null) });
    if (multiple) win.setFooter({ actions: [c.button.sm({ label: "Done", tone: "primary", onClick: () => win.close("done") })] });
  });
}
