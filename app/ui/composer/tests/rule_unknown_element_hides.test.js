// Rule 3: an element the kit does not have makes its module hide.
//
// The alternative is a panel that renders something approximate, which is worse
// than nothing: it looks like it works.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom } from "./_dom.js";

let composer, UnknownElement, renderPanel, mountAll, mounts;
test.before(async () => {
  setupDom();
  ({ composer, UnknownElement } = await import("../composer.js"));
  ({ renderPanel } = await import("../panel.js"));
  ({ mountAll } = await import("../../shell/panels.js"));
  mounts = await import("../../shell/mounts.js");
});
test.after(() => teardownDom());

test.beforeEach(() => {
  mounts._reset();
  document.body.replaceChildren();
});

const spec = (id, settings) => ({ id, title: id, mount: "test.region", settings });

test("asking the kit for something it does not have throws", () => {
  assert.throws(() => composer.hologram, UnknownElement);
  assert.throws(() => composer.button.hologram, UnknownElement);
});

test("a panel naming an unknown renderer appends nothing", () => {
  const host = document.createElement("div");
  document.body.appendChild(host);
  assert.throws(() => {
    const panel = renderPanel(spec("bad", {
      good: { type: "bool", default: true, label: "Fine" },
      bad: { type: "bool", default: true, label: "Bad", ui: "hologram" },
    }));
    host.appendChild(panel.node);
  });
  assert.equal(host.childElementCount, 0,
    "not even the rows that built successfully reached the document");
});

test("a broken module is absent, and its neighbours still mount", async () => {
  const host = document.createElement("div");
  document.body.appendChild(host);
  mounts.offer("test.region", host);

  const { mounted, hidden } = await mountAll({ modules: [
    spec("first", { on: { type: "bool", default: true, label: "First" } }),
    spec("broken", { on: { type: "bool", default: true, label: "Broken", ui: "hologram" } }),
    spec("third", { on: { type: "bool", default: true, label: "Third" } }),
  ] });

  assert.deepEqual(mounted.map((m) => m.id), ["first", "third"]);
  assert.deepEqual(hidden.map((h) => h.id), ["broken"]);
  assert.match(host.textContent, /First/);
  assert.match(host.textContent, /Third/);
  assert.doesNotMatch(host.textContent, /Broken/);
});

test("nothing about a hidden module reaches the screen", async () => {
  // No placeholder, no greyed row, no warning chip: absent is absent.
  const host = document.createElement("div");
  document.body.appendChild(host);
  mounts.offer("test.region", host);

  await mountAll({ modules: [spec("broken", {
    on: { type: "bool", default: true, label: "Broken", ui: "hologram" },
  })] });

  assert.equal(host.childElementCount, 0);
  assert.equal(host.textContent, "");
});

test("a module whose ui.js fails to load is hidden, not half-mounted", async () => {
  const host = document.createElement("div");
  document.body.appendChild(host);
  mounts.offer("test.region", host);

  const { mounted, hidden } = await mountAll(
    { modules: [{ ...spec("has_ui", { on: { type: "bool", default: true, label: "On" } }), ui: "/missing.js" }] },
    { load: () => { throw new Error("404"); } },
  );

  assert.deepEqual(mounted, []);
  assert.equal(hidden[0].id, "has_ui");
  assert.equal(host.childElementCount, 0, "its panel did not stay behind");
});

test("a module naming a mount point nobody offers is absent", async () => {
  // Hide, don't warn, one level up: a typo'd mount point makes a module vanish
  // rather than landing somewhere arbitrary.
  const { mounted, hidden } = await mountAll({ modules: [
    { ...spec("lost", { on: { type: "bool", default: true, label: "On" } }), mount: "nowhere" },
  ] });
  assert.deepEqual(mounted, []);
  assert.match(hidden[0].why, /no region offers/);
});

test("a module is handed no way to touch the document", async () => {
  const host = document.createElement("div");
  document.body.appendChild(host);
  mounts.offer("test.region", host);

  let context = null;
  await mountAll(
    { modules: [{ ...spec("m", { on: { type: "bool", default: true, label: "On" } }), ui: "/ui.js" }] },
    { load: async () => ({ setup: (ctx) => { context = ctx; } }) },
  );

  assert.deepEqual(Object.keys(context).sort(), ["on", "shell", "values"]);

  // Not "no value IS a node" -- that was the old assertion, and it passed while
  // `composer` sat in this object handing out handles whose .node is a real
  // element, reachable to the whole document through node.ownerDocument. The
  // property that matters is that nothing here PRODUCES one either.
  for (const [name, value] of Object.entries(context)) {
    assert.equal(value instanceof globalThis.Node, false, `${name} is a DOM node`);
  }
  assert.equal("composer" in context, false,
    "an element factory is a way to obtain a DOM node");

  // `on` is callable, and must hand back an unsubscribe rather than anything
  // that leads to the page.
  const off = context.on(() => {});
  assert.equal(typeof off, "function");
  assert.equal(off instanceof globalThis.Node, false);

  // Nothing one level down is a node either -- the services are the other place
  // a handle could have been tucked away.
  for (const [name, value] of Object.entries(context.shell)) {
    for (const [inner, member] of Object.entries(value)) {
      assert.equal(member instanceof globalThis.Node, false, `shell.${name}.${inner}`);
    }
  }

  // The service surface is named one by one, so it can be reviewed.
  assert.deepEqual(Object.keys(context.shell).sort(), ["density", "theme"]);
});
