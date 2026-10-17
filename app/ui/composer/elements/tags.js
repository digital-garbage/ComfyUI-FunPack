// Tags: a list of short words edited as chips, and a cloud of them with counts.
//
// A tag is text, never markup. The editor owns the list it shows: onChange hears
// a fresh array after a person's edit, and setValue replaces it without a call.

import { define, lookup } from "../internals/register.js";
import { el } from "../internals/el.js";
import { setText } from "../internals/text.js";
import { uid } from "../internals/ids.js";
import { wireDragReorder } from "../internals/dragReorder.js";

const CLOUD_LIMIT = 300;

/** What a tag is allowed to be: one trimmed run of words, no commas, not empty. */
export const clean = (raw) => String(raw ?? "").replace(/[,\n\r\t]+/g, " ").replace(/\s+/g, " ").trim();

function listOf(raw, max) {
  const seen = new Set(), out = [];
  for (const t of Array.isArray(raw) ? raw : []) {
    const tag = clean(t);
    if (tag && !seen.has(tag) && out.length < (max ?? Infinity)) { seen.add(tag); out.push(tag); }
  }
  return out;
}

define("tags", "editor", ({ value = [], suggestions = [], placeholder = "Add a tag", label, onChange, maxTags } = {}) => {
  let tags = listOf(value, maxTags);
  const input = el("input", { cls: ["cx-tags-input", "cx-focusable"], attrs: { type: "text", placeholder, id: uid("tags"), "aria-label": label || placeholder, autocomplete: "off" } });
  const node = el("div", { cls: "cx-tags", attrs: { role: "group", "aria-label": label }, children: [input] });

  const chips = () => [...node.querySelectorAll(".cx-tag")];
  const commit = () => { if (onChange) onChange([...tags]); };
  const full = () => maxTags !== undefined && tags.length >= maxTags;

  function draw(focusAt) {
    chips().forEach((c) => c.remove());
    tags.forEach((tag, i) => {
      const chip = el("span", { cls: "cx-tag cx-tag-edit", attrs: { draggable: "true", tabindex: i === 0 ? "0" : "-1", "data-id": tag, role: "group", "aria-label": tag }, children: [
        el("span", { cls: "cx-tag-text", text: tag }),
        el("button", { cls: "cx-tag-x", text: "×", attrs: { type: "button", tabindex: "-1", "aria-label": `Remove ${tag}` }, on: { click: () => { remove(i); input.focus(); } } }),
      ] });
      node.insertBefore(chip, input);
    });
    input.disabled = full();
    if (focusAt !== undefined) { const list = chips(); (list[Math.min(focusAt, list.length - 1)] || input).focus(); }
  }
  function add(raw) {
    let changed = false;
    for (const piece of String(raw).split(/[,\n]/)) {
      const tag = clean(piece);
      if (tag && !tags.includes(tag) && !full()) { tags.push(tag); changed = true; }
    }
    if (changed) { draw(); commit(); }
    return changed;
  }
  function remove(i, focusAt) { tags.splice(i, 1); draw(focusAt); commit(); }

  wireDragReorder(node, ".cx-tag", (from, to) => {
    const a = tags.indexOf(from), b = tags.indexOf(to);
    if (a < 0 || b < 0) return;
    tags.splice(b, 0, tags.splice(a, 1)[0]);
    draw(); commit();
  });

  const source = (q) => {
    const low = q.trim().toLowerCase();
    return !low ? [] : suggestions.filter((s) => !tags.includes(s))
      .map((s) => [s, s.toLowerCase().indexOf(low)]).filter(([, at]) => at >= 0).sort((a, b) => a[1] - b[1]).slice(0, 8).map(([s]) => ({ label: s }));
  };
  const picker = suggestions.length ? lookup("autocomplete", "default")({ input, source, onPick: (item) => { input.value = ""; add(item.label); input.focus(); } }) : null;

  // After the suggestion list wired its own keys, so an accepted suggestion (already handled, default prevented) is not added twice.
  input.addEventListener("keydown", (e) => {
    if (e.defaultPrevented) return;
    if (e.key === "Enter" || e.key === ",") { e.preventDefault(); if (add(input.value)) input.value = ""; }
    else if (e.key === "Tab" && input.value.trim() && !e.shiftKey) {
      const first = source(input.value)[0];
      if (first) { e.preventDefault(); input.value = ""; add(first.label); picker && picker.close(); }
    } else if (e.key === "Backspace" && !input.value && tags.length) { e.preventDefault(); remove(tags.length - 1); }
    else if (e.key === "ArrowLeft" && !input.selectionStart && chips().length) { e.preventDefault(); chips().pop().focus(); }
  });
  input.addEventListener("paste", (e) => {
    const text = e.clipboardData ? e.clipboardData.getData("text") : "";
    if (/[,\n]/.test(text)) { e.preventDefault(); add(text); }
  });
  node.addEventListener("keydown", (e) => {
    const chip = e.target.closest && e.target.closest(".cx-tag");
    if (!chip || e.target !== chip) return;
    const list = chips(), i = list.indexOf(chip);
    if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); remove(i, i); }
    else if (e.key === "ArrowLeft" && i > 0) { e.preventDefault(); list[i - 1].focus(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); (list[i + 1] || input).focus(); }
  });
  node.addEventListener("focusin", (e) => {                    // one tab stop among the chips: the one last used
    const chip = e.target.closest && e.target.closest(".cx-tag");
    if (chip && e.target === chip) chips().forEach((c) => c.setAttribute("tabindex", c === chip ? "0" : "-1"));
  });
  node.addEventListener("click", (e) => { if (e.target === node) input.focus(); });

  draw();
  return {
    node,
    get: () => [...tags],
    setValue(next) { tags = listOf(next, maxTags); draw(); },
    focus: () => input.focus(),
    destroy() { if (picker) picker.destroy(); node.remove(); },
  };
});

define("tags", "cloud", ({ items = [], onPick, onRemove, selected = [], label, empty = "No tags yet" } = {}) => {
  let shown = [], picked = new Set(selected);
  const list = el("div", { cls: "cx-tagcloud-list", attrs: { role: "list" } });
  const more = el("p", { cls: "cx-tagcloud-more" });
  const note = el("p", { cls: "cx-tagcloud-empty", text: empty });
  const node = el("div", { cls: "cx-tagcloud", attrs: { role: "group", "aria-label": label }, children: [list, more, note] });

  const mark = () => list.querySelectorAll("[data-tag]").forEach((c) => {
    const on = picked.has(c.dataset.tag);
    c.classList.toggle("cx-on", on);
    const pick = c.querySelector(".cx-tag-pick");
    if (pick) pick.setAttribute("aria-pressed", String(on));
  });
  function draw() {
    list.replaceChildren();
    shown.slice(0, CLOUD_LIMIT).forEach(({ tag, count }) => {
      list.append(el("span", { cls: "cx-tag cx-tag-cloud", attrs: { role: "listitem", "data-tag": tag }, children: [
        el("button", { cls: "cx-tag-pick cx-focusable", attrs: { type: "button", "aria-pressed": "false" }, children: [
          el("span", { cls: "cx-tag-text", text: tag }),
          Number.isFinite(count) ? el("span", { cls: "cx-tag-count", text: String(count) }) : null,
        ].filter(Boolean) }),
        onRemove ? el("button", { cls: "cx-tag-x", text: "×", attrs: { type: "button", "aria-label": `Remove ${tag}` } }) : null,
      ].filter(Boolean) }));
    });
    setText(more, shown.length > CLOUD_LIMIT ? `+${shown.length - CLOUD_LIMIT} more` : "");
    node.classList.toggle("cx-tagcloud-none", !shown.length);
    mark();
  }
  node.addEventListener("click", (e) => {
    const chip = e.target.closest && e.target.closest("[data-tag]");
    if (!chip) return;
    if (e.target.closest(".cx-tag-x")) { if (onRemove) onRemove(chip.dataset.tag); }
    else if (e.target.closest(".cx-tag-pick") && onPick) onPick(chip.dataset.tag);
  });
  const set = (next) => { shown = (next || []).filter((i) => i && typeof i.tag === "string" && i.tag).map((i) => ({ tag: i.tag, count: Number(i.count) })); draw(); };
  set(items);
  return {
    node, setItems: set,
    setSelected(next) { picked = new Set(next); mark(); },
    destroy: () => node.remove(),
  };
});
