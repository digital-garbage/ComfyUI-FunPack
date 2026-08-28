// Grids of things.
//
// The adaptive one is v4's media bin generalised: it sizes off its CONTAINER,
// not the viewport, so a 280px panel lays out like a 280px panel whether the
// window is 1200px or 3000px wide.

import { define } from "../internals/register.js";
import { el } from "../internals/el.js";
import { applyDensity, recallCols, rememberCols, normaliseCols } from "../internals/density.js";

define("gallery", "adaptive", ({ id = "gallery", items = [], cols, onActivate, onContext, selection = [], empty = "Nothing here yet" } = {}) => {
  const chosen = new Set(selection);
  const grid = el("div", { cls: "cx-gallery", attrs: { role: "listbox", "aria-multiselectable": "true" } });
  applyDensity(grid, cols === undefined ? recallCols(id) : cols);

  function draw() {
    grid.replaceChildren();
    if (!items.length) {
      grid.append(el("p", { cls: "cx-list-empty", text: empty }));
      return;
    }
    for (const item of items) {
      const cell = el("button", {
        cls: ["cx-cell", chosen.has(item.id) ? "cx-on" : null, "cx-focusable"],
        attrs: { type: "button", role: "option", "aria-selected": String(chosen.has(item.id)) },
      });

      const thumb = el("div", { cls: "cx-cell-thumb" });
      if (item.thumb) {
        // Decorative: the caption below carries the name, so an alt would just
        // make a screen reader say everything twice.
        thumb.append(el("img", { cls: "cx-cell-img", attrs: { src: item.thumb, alt: "", loading: "lazy" } }));
      } else {
        thumb.append(el("span", { cls: "cx-cell-glyph", text: item.icon || "▦", attrs: { "aria-hidden": "true" } }));
      }
      if (item.badge) thumb.append(el("span", { cls: "cx-cell-badge", text: item.badge }));
      if (item.duration) thumb.append(el("span", { cls: "cx-cell-duration", text: item.duration }));

      cell.append(thumb, el("span", { cls: "cx-cell-name", text: item.label }));
      cell.addEventListener("click", () => { if (onActivate) onActivate(item); });
      if (onContext) {
        cell.addEventListener("contextmenu", (e) => { e.preventDefault(); onContext(item, e); });
      }
      grid.append(cell);
    }
  }
  draw();

  const node = el("div", { cls: "cx-gallery-wrap", children: grid });
  return {
    node,
    get value() { return [...chosen]; },
    setValue(next) { chosen.clear(); for (const v of next || []) chosen.add(v); draw(); },
    setCols(next) {
      const n = normaliseCols(next);
      applyDensity(grid, n);
      rememberCols(id, n);
    },
    setItems(next) { items = next; draw(); },
    destroy: () => node.remove(),
  };
});

/** Large clickable cards: the wizard's "what are you making?" choices. */
define("gallery", "cards", ({ items = [], value, onActivate } = {}) => {
  let current = value;
  const node = el("div", { cls: "cx-cards", attrs: { role: "radiogroup" } });
  const cards = items.map((item) => {
    const card = el("button", {
      cls: ["cx-card", "cx-focusable"],
      attrs: { type: "button", role: "radio", "aria-checked": String(item.id === current) },
      children: [
        item.icon ? el("span", { cls: "cx-card-icon", text: item.icon, attrs: { "aria-hidden": "true" } }) : null,
        el("span", { cls: "cx-card-title", text: item.label }),
        item.hint ? el("span", { cls: "cx-hint", text: item.hint }) : null,
      ].filter(Boolean),
    });
    card.classList.toggle("cx-on", item.id === current);
    card.addEventListener("click", () => {
      current = item.id;
      for (const { item: i, card: c } of cards) {
        c.classList.toggle("cx-on", i.id === current);
        c.setAttribute("aria-checked", String(i.id === current));
      }
      if (onActivate) onActivate(item);
    });
    node.append(card);
    return { item, card };
  });
  return { node, get value() { return current; }, destroy: () => node.remove() };
});

/** A single scrolling row: filmstrips, segment strips, recent items. */
define("gallery", "strip", ({ items = [], onActivate, label } = {}) => {
  const node = el("div", { cls: "cx-strip", attrs: { role: "list", "aria-label": label } });
  for (const item of items) {
    const cell = el("button", { cls: ["cx-strip-cell", "cx-focusable"],
      attrs: { type: "button", role: "listitem", title: item.label } });
    if (item.thumb) cell.append(el("img", { cls: "cx-cell-img", attrs: { src: item.thumb, alt: "", loading: "lazy" } }));
    else cell.append(el("span", { cls: "cx-cell-glyph", text: item.icon || "▦", attrs: { "aria-hidden": "true" } }));
    cell.addEventListener("click", () => { if (onActivate) onActivate(item); });
    node.append(cell);
  }
  return { node, destroy: () => node.remove() };
});
