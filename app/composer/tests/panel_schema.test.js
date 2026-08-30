// The bridge: a declaration in, a panel out.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom, fire } from "./_dom.js";

let renderPanel, defaultsOf, visible, rendererFor, DEFAULT_UI;
test.before(async () => {
  setupDom();
  await import("../composer.js");            // elements register themselves
  ({ renderPanel, defaultsOf, visible, rendererFor, DEFAULT_UI } = await import("../panel.js"));
});
test.after(() => teardownDom());

// Panels are built detached and then mounted; jsdom only runs a control's
// activation behaviour once it is in the document, so anything that clicks has
// to mount first -- which is what the shell does anyway.
const mount = (panel) => { document.body.appendChild(panel.node); return panel; };

const SPEC = {
  id: "audio_clock",
  title: "Audio clock",
  settings: {
    enabled: { type: "bool", default: true, label: "Sync to audio clock", hint: "Locks frame timing." },
    strength: { type: "float", default: 0.65, min: 0, max: 1, step: 0.05,
                label: "Lock strength", unit: "x", ui: "slider", when: { enabled: true } },
    mode: { type: "enum", default: "beat", label: "Alignment", ui: "segmented",
            options: [{ value: "beat", label: "Beat grid" }, { value: "onset", label: "Onset" }],
            when: { enabled: true } },
  },
};

test("defaults come from the declaration, not from the controls", () => {
  assert.deepEqual(defaultsOf(SPEC), { enabled: true, strength: 0.65, mode: "beat" });
});

test("a panel renders one row per visible setting", () => {
  const panel = renderPanel(SPEC);
  assert.equal(panel.node.querySelectorAll(".cx-check-row, .cx-settings-row").length, 3);
  assert.match(panel.node.textContent, /Audio clock/);
  panel.destroy();
});

test("the declared renderer is used, not the default for the type", () => {
  const panel = renderPanel(SPEC);
  assert.ok(panel.node.querySelector(".cx-slider"), "strength asked for a slider");
  assert.ok(panel.node.querySelector(".cx-segmented"), "mode asked for segmented");
  assert.equal(panel.node.querySelector("select"), null, "and not the enum default");
  panel.destroy();
});

test("a type with no declared renderer gets its default", () => {
  const panel = renderPanel({ title: "T", settings: {
    n: { type: "float", default: 1, label: "N" },
    t: { type: "text", default: "x", label: "T" },
  } });
  assert.equal(DEFAULT_UI.float, "number");
  assert.ok(panel.node.querySelector("input[type=number]"));
  assert.ok(panel.node.querySelector("input[type=text]"));
  panel.destroy();
});

test("a self-labelling control is not wrapped in a second label", () => {
  const panel = renderPanel({ title: "T", settings: {
    on: { type: "bool", default: true, label: "Enabled" },
  } });
  const labels = [...panel.node.querySelectorAll("*")]
    .filter((n) => n.textContent.trim() === "Enabled" && !n.children.length);
  assert.equal(labels.length, 1, "the label appears once, not once per wrapper");
  panel.destroy();
});

// --- conditional rows ------------------------------------------------------

test("a row whose condition fails is absent, not disabled", () => {
  // Hide, don't warn, one level down: a greyed control invites clicking on
  // something that cannot do anything.
  const panel = renderPanel(SPEC, { values: { enabled: false } });
  assert.match(panel.node.textContent, /Sync to audio clock/);
  assert.doesNotMatch(panel.node.textContent, /Lock strength/);
  assert.equal(panel.node.querySelector("[disabled]"), null);
  panel.destroy();
});

test("changing the key a row depends on brings it back", () => {
  const panel = mount(renderPanel(SPEC, { values: { enabled: false } }));
  assert.doesNotMatch(panel.node.textContent, /Lock strength/);
  panel.node.querySelector("input[type=checkbox]").click();
  assert.match(panel.node.textContent, /Lock strength/);
  panel.destroy();
});

test("visible() takes a list as any-of", () => {
  const setting = { when: { mode: ["beat", "onset"] } };
  assert.equal(visible(setting, { mode: "onset" }), true);
  assert.equal(visible(setting, { mode: "flat" }), false);
});

test("a setting with no condition is always visible", () => {
  assert.equal(visible({}, {}), true);
});

// --- values ----------------------------------------------------------------

test("edits are reported with the key and the whole value set", () => {
  const seen = [];
  const panel = mount(renderPanel(SPEC, { onChange: (key, value, all) => seen.push([key, value, all.mode]) }));
  panel.node.querySelector("input[type=checkbox]").click();
  assert.deepEqual(seen[0].slice(0, 2), ["enabled", false]);
  assert.equal(seen[0][2], "beat", "the rest of the values come along");
  panel.destroy();
});

test("the panel's values start from the defaults and track edits", () => {
  const panel = mount(renderPanel(SPEC));
  assert.deepEqual(panel.values, { enabled: true, strength: 0.65, mode: "beat" });

  const slider = panel.node.querySelector(".cx-slider input");
  slider.value = "0.3";
  fire(slider, "input");
  assert.equal(panel.values.strength, 0.3);
  panel.destroy();
});

test("supplied values override the defaults", () => {
  const panel = renderPanel(SPEC, { values: { strength: 0.2 } });
  assert.equal(panel.values.strength, 0.2);
  assert.equal(panel.values.mode, "beat", "and the rest still come from the declaration");
  panel.destroy();
});

// --- refusing rather than approximating ------------------------------------

test("a setting with no default is refused by the panel too", () => {
  // Core catches this first; the panel is the layer that cannot proceed without
  // one, since it would render a control holding undefined.
  assert.throws(() => rendererFor({ type: "float", min: 0, max: 1 }), /no default/);
});

test("an unknown renderer throws instead of falling back", () => {
  // A panel that silently renders a different control is worse than an absent
  // one, because nobody notices it is wrong.
  assert.throws(() => rendererFor({ type: "bool", ui: "hologram" }), TypeError);
});

test("a panel that cannot be built appends nothing", () => {
  const host = document.createElement("div");
  document.body.appendChild(host);
  try {
    const panel = renderPanel({ title: "Broken", settings: {
      ok: { type: "bool", default: true, label: "Fine" },
      bad: { type: "bool", default: true, label: "Bad", ui: "hologram" },
    } });
    host.appendChild(panel.node);
    assert.fail("should have thrown");
  } catch (err) {
    assert.ok(err instanceof TypeError);
    assert.equal(host.childElementCount, 0, "nothing half-built reached the document");
  }
});

test("every renderer named by the contract exists", async () => {
  // The contract and the kit have to agree: a hint the schema accepts but the
  // kit cannot render would make every module using it hide itself.
  const HINTS = {
    bool: ["checkboxRow", "toggle"],
    int: ["number", "slider", "stepper", "macroSlider"],
    float: ["number", "slider", "stepper", "macroSlider"],
    enum: ["select", "segmented", "radioGroup", "filterList", "wheel"],
    text: ["input", "search"],
    multiline: ["textarea", "autoTextarea"],
    path: ["filterList"],
  };
  for (const [type, hints] of Object.entries(HINTS)) {
    for (const ui of hints) {
      assert.doesNotThrow(() => rendererFor({ type, ui, default: null }), `${type} + ${ui}`);
    }
    assert.doesNotThrow(() => rendererFor({ type, default: null }), `${type} default`);
  }
});

// --- a redraw is as safe as the first build --------------------------------

test("a row that only appears later cannot empty the panel when it throws", () => {
  // The initial build is protected by a detached fragment, and the redraw was
  // not: it destroyed every control and then built, so a setting revealed by
  // another value -- one the first build never rendered and so never checked --
  // emptied the panel and threw on the way to refilling it. A blank panel with
  // no way back, from a typo in a `ui` name.
  const spec = {
    id: "late", title: "Late",
    settings: {
      mode: { type: "enum", label: "Mode", default: "simple",
              options: [{ value: "simple", label: "Simple" }, { value: "full", label: "Full" }] },
      extra: { type: "int", label: "Extra", default: 1, ui: "noSuchRenderer",
               when: { mode: "full" } },
    },
  };

  const panel = renderPanel(spec, {});
  document.body.replaceChildren(panel.node);
  // The CONTROLS, not the rows. A control's destroy() takes the control out of
  // its settings row and leaves the row behind, so counting rows sees a panel
  // that is intact and is looking at labels with nothing under them -- the
  // first version of this test did exactly that and passed against the bug.
  const controls = () => panel.node.querySelectorAll("input, select, textarea").length;
  const before = controls();
  assert.ok(before >= 1, "nothing was rendered to begin with");

  assert.throws(() => panel.setValue("mode", "full"), /noSuchRenderer/);
  assert.equal(controls(), before,
    "a redraw that could not finish stripped the controls out of the panel");
});

test("a redraw that succeeds destroys the controls it replaced", () => {
  // The fix builds before it tears down, which is exactly where a leak would
  // hide: keeping both maps and forgetting to destroy the old one.
  const destroyed = [];
  const spec = {
    id: "swap", title: "Swap",
    settings: {
      mode: { type: "enum", label: "Mode", default: "a",
              options: [{ value: "a", label: "A" }, { value: "b", label: "B" }] },
      only_a: { type: "int", label: "Only A", default: 1, when: { mode: "a" } },
    },
  };
  const panel = renderPanel(spec, {});
  document.body.replaceChildren(panel.node);
  const was = panel.node.querySelector('[aria-label="Only A"]');
  assert.ok(was, "the gated row was not rendered");

  panel.setValue("mode", "b");
  assert.equal(was.isConnected, false, "the replaced control stayed in the document");
  assert.equal(panel.node.querySelector('[aria-label="Only A"]'), null);
  void destroyed;
});
