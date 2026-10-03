// Every registered element, in both themes.
//
// It iterates the registry rather than a list, so an element appears here the
// moment it calls define() -- and one with no demo shows up as a gap instead of
// quietly not existing.

import { composer, entries } from "../composer/composer.js";
import { DEMOS, VARIANTS } from "./demos.js";

const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

const resolve = (demo) => (typeof demo === "function" ? demo(composer) : demo);

function renderInto(pane, name, factory) {
  const variants = VARIANTS[name] || [resolve(DEMOS[name])];
  const row = el("div", "cat-specimens");
  for (const props of variants) {
    const cell = el("div", "cat-specimen");
    try {
      // An overlay mounts itself into the portal; appending its node here would
      // drag it out of that root and out of its stacking context. Probe once,
      // close it, and offer a launcher instead -- which is also the only honest
      // way to look at a modal.
      const probe = factory(props);
      if (probe.isOverlay) {
        probe.destroy();
        const open = el("button", null, `Open ${name}`);
        open.className = "cat-launch";
        open.addEventListener("click", () => {
          const live = factory(resolve(DEMOS[name]));
          if (live.result) live.result.then((r) => console.log(`${name} resolved:`, r));
        });
        cell.append(open);
      } else {
        cell.append(probe.node);
      }
    } catch (err) {
      cell.append(el("div", "cat-broke", `threw: ${err.message}`));
    }
    row.append(cell);
  }
  pane.append(row);
}

function render(sideBySide) {
  const page = document.querySelector("#catalogue");
  page.textContent = "";

  const all = [...entries()];
  if (!all.length) {
    const s = el("section", "cat-section");
    s.append(el("h2", null, "No elements defined yet"));
    s.append(el("p", null, "elements/index.js imports nothing, or every element file failed to load."));
    page.append(s);
    return;
  }

  const byGroup = new Map();
  for (const e of all) {
    if (!byGroup.has(e.group)) byGroup.set(e.group, []);
    byGroup.get(e.group).push(e);
  }

  const missing = [];

  for (const [group, items] of byGroup) {
    const section = el("section", "cat-section");
    section.append(el("h2", null, `composer.${group}`));
    section.append(el("p", null, items.map((i) => i.variant).join(" · ")));

    for (const { variant, factory } of items) {
      const name = `${group}.${variant}`;
      if (!DEMOS[name]) { missing.push(name); continue; }

      section.append(el("h3", "cat-variant", `.${variant}`));
      const panes = el("div", "cat-panes" + (sideBySide ? "" : " single"));
      const themes = sideBySide
        ? ["dark", "light"]
        : [document.documentElement.getAttribute("data-theme") || "dark"];
      for (const theme of themes) {
        const pane = el("section", "cat-pane");
        pane.setAttribute("data-theme", theme);
        pane.append(el("h3", null, theme));
        renderInto(pane, name, factory);
        panes.append(pane);
      }
      section.append(panes);
    }
    page.append(section);
  }

  if (missing.length) {
    const s = el("section", "cat-section");
    s.append(el("h2", null, "Registered but not shown"));
    s.append(el("p", null, `${missing.join(", ")} — add a demo to catalogue/demos.js.`));
    page.append(s);
  }
}

let sideBySide = true;

document.querySelector("#toggle-panes").addEventListener("click", (e) => {
  sideBySide = !sideBySide;
  e.currentTarget.setAttribute("aria-pressed", String(sideBySide));
  render(sideBySide);
});

for (const btn of document.querySelectorAll("[data-theme-set]")) {
  btn.addEventListener("click", () => {
    window.ComposerTheme.apply(btn.getAttribute("data-theme-set"));
    if (!sideBySide) render(false);
  });
}

render(sideBySide);
