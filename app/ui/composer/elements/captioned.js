// A picture or clip with its caption beside it: the preview and the tag editor in one card.
//
// The card shows what it is given and reports what a person edits. The caption is a comma separated
// line; the editor inside owns the chips, and onChange hears the whole line after an edit.

import { define, lookup } from "../internals/register.js";
import { el } from "../internals/el.js";
import { setText } from "../internals/text.js";
import { clean } from "./tags.js";

const LAYOUTS = ["cell", "side"];

const tagsOf = (text) => {
  const seen = new Set(), out = [];
  for (const part of String(text ?? "").split(/[,\n]/)) {
    const tag = clean(part);
    if (tag && !seen.has(tag.toLowerCase())) { seen.add(tag.toLowerCase()); out.push(tag); }
  }
  return out;
};

for (const layout of LAYOUTS) {
  define("captioned", layout, ({ src, kind = "image", title = "", caption = "", badge, tone = "neutral", selected = false, suggestions = [], onChange, onSelect, onOpen, label } = {}) => {
    const glyph = () => el("span", { cls: "cx-captioned-glyph", text: kind === "video" ? "▶" : "▨", attrs: { "aria-hidden": "true" } });
    const view = el("button", { cls: ["cx-captioned-view", "cx-focusable"], attrs: { type: "button", "aria-label": title ? `Open ${title}` : "Open" } });
    const show = (next, nextKind) => {
      view.replaceChildren();
      if (!next) { view.append(glyph()); return; }
      const img = el("img", { cls: "cx-captioned-img", attrs: { src: next, alt: "", loading: "lazy" } });
      img.addEventListener("error", () => img.replaceWith(glyph()));
      view.append(img);
      if (nextKind === "video") view.append(el("span", { cls: "cx-captioned-play", text: "▶", attrs: { "aria-hidden": "true" } }));
    };
    show(src, kind);
    if (onOpen) view.addEventListener("click", () => onOpen());
    else view.tabIndex = -1;

    const name = el("span", { cls: "cx-captioned-title", text: title, attrs: { title } });
    const chip = el("span", { cls: ["cx-captioned-badge", `cx-captioned-${tone}`], text: badge ?? "" });
    chip.hidden = !badge;
    const pick = onSelect ? el("input", { cls: ["cx-captioned-pick", "cx-focusable"], attrs: { type: "checkbox", "aria-label": title ? `Select ${title}` : "Select" } }) : null;
    if (pick) { pick.checked = !!selected; pick.addEventListener("change", () => { node.classList.toggle("cx-on", pick.checked); onSelect(pick.checked); }); }

    const editor = lookup("tags", "editor")({ value: tagsOf(caption), suggestions, placeholder: "Add a tag", label: title ? `Tags for ${title}` : "Tags",
      onChange: (list) => { if (onChange) onChange(list.join(", ")); } });

    const head = el("div", { cls: "cx-captioned-head", children: [pick, name, chip].filter(Boolean) });
    const body = el("div", { cls: "cx-captioned-body", children: [head, editor.node] });
    const node = el("div", { cls: ["cx-captioned", `cx-captioned-${layout}`, selected ? "cx-on" : ""].filter(Boolean), attrs: { role: "group", "aria-label": label || title || "Captioned media" }, children: [view, body] });
    return {
      node,
      getCaption: () => editor.get().join(", "),
      setCaption: (text) => editor.setValue(tagsOf(text)),
      setSelected(on) { if (pick) pick.checked = !!on; node.classList.toggle("cx-on", !!on); },
      setBadge(text, nextTone) { setText(chip, text ?? ""); chip.hidden = !text; if (nextTone) chip.className = `cx-captioned-badge cx-captioned-${nextTone}`; },
      setPicture: show,
      focus: () => editor.focus(),
      destroy() { editor.destroy(); node.remove(); },
    };
  });
}
