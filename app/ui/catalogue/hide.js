// Renders the fixture modules through the real shell.

import { mountAll } from "../shell/panels.js";
import { offer, _reset as resetMounts } from "../shell/mounts.js";
import { _reset as resetValues, all as allValues } from "../shell/values.js";
import { GOOD, BROKEN } from "./fixtures.js";

const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

async function render(theme) {
  resetMounts();
  resetValues();

  const page = document.querySelector("#catalogue");
  page.textContent = "";

  const section = el("section", "cat-section");
  section.append(el("h2", null, "What a user sees"));
  section.append(el("p", null,
    "Six modules go in: one valid, five broken in different ways. Everything below " +
    "is the whole of what reaches the screen."));

  const pane = el("section", "cat-pane");
  pane.setAttribute("data-theme", theme);
  pane.append(el("h3", null, theme));
  const host = el("div", "cat-mount");
  pane.append(host);
  section.append(pane);
  page.append(section);

  offer("demo.panel", host);

  // The broken ones are handed to the shell exactly as the good one is. In the
  // real app most never get this far -- core refuses them at import or at
  // validation -- so this is the worst case: everything reaching the browser.
  const manifest = { modules: [GOOD, ...BROKEN.map((b) => b.spec)] };
  const { mounted, hidden } = await mountAll(manifest);

  if (!host.childElementCount) host.append(el("p", "cat-broke", "nothing mounted at all"));

  // Everything from here down is developer commentary, NOT part of what a user
  // would see -- the panel above is.
  const ledger = el("section", "cat-section");
  ledger.append(el("h2", null, "Why each one is absent"));
  ledger.append(el("p", null, "This table exists for you, not for the user. None of it appears in the app."));

  const table = el("div", "cat-ledger");
  for (const { id } of mounted) {
    table.append(el("div", "cat-ledger-row cat-ok"), el("code", "cat-name", id));
    table.append(el("span", null, "mounted"));
  }
  for (const { id, why } of hidden) {
    const source = BROKEN.find((b) => b.spec.id === id);
    table.append(el("code", "cat-name", id));
    table.append(el("span", "cat-why", source ? source.why : "—"));
    table.append(el("span", "cat-caught", why));
  }
  ledger.append(table);
  page.append(ledger);

  const values = el("section", "cat-section");
  values.append(el("h2", null, "Values a headless run would get"));
  values.append(el("p", null,
    "Seeded from the same declaration the panel renders, so a generation with no " +
    "browser open starts from the same numbers."));
  const pre = el("pre", "cx-code-block");
  pre.textContent = JSON.stringify(allValues(), null, 2);
  values.append(pre);
  page.append(values);
}

let theme = "dark";
for (const btn of document.querySelectorAll("[data-theme-set]")) {
  btn.addEventListener("click", () => {
    theme = btn.getAttribute("data-theme-set");
    window.ComposerTheme.apply(theme === "auto" ? "auto" : theme);
    render(theme === "auto" ? window.ComposerTheme.resolved() : theme);
  });
}

render(theme);
