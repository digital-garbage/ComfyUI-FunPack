// Behaviour the registry-wide rule tests cannot see: what each element does
// when it is actually used.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom, fire } from "./_dom.js";

let composer;
test.before(async () => {
  setupDom();
  ({ composer } = await import("../composer.js"));
});
test.after(() => teardownDom());

const mount = (handle) => { document.body.appendChild(handle.node); return handle; };

// --- buttons ---------------------------------------------------------------

test("a button reports clicks and can be made busy", () => {
  let clicks = 0;
  const b = mount(composer.button.md({ label: "Go", onClick: () => { clicks += 1; } }));
  b.node.click();
  assert.equal(clicks, 1);

  b.setBusy(true);
  assert.equal(b.node.disabled, true);
  assert.equal(b.node.getAttribute("aria-busy"), "true");
  b.node.click();
  assert.equal(clicks, 1, "a busy button does not fire again");

  b.setBusy(false);
  assert.equal(b.node.disabled, false);
});

test("an unknown tone is refused rather than rendered plain", () => {
  assert.throws(() => composer.button.md({ label: "x", tone: "sparkly" }), RangeError);
});

test("an icon button without a label is refused", () => {
  // Otherwise a toolbar of glyphs is unusable by screen reader and unlabelled
  // on hover, and nobody notices until someone needs it.
  assert.throws(() => composer.iconButton.md({ icon: "x" }), TypeError);
});

test("buttonGroup picks one, or several when multi", () => {
  const seen = [];
  const g = mount(composer.buttonGroup.md({
    value: "a", onChange: (v) => seen.push(v),
    items: [{ value: "a", label: "A" }, { value: "b", label: "B" }],
  }));
  g.node.querySelectorAll("button")[1].click();
  assert.equal(g.value, "b");
  assert.deepEqual(seen, ["b"]);
  assert.equal(g.node.querySelectorAll("button")[1].getAttribute("aria-checked"), "true");

  const m = mount(composer.buttonGroup.md({
    multi: true, value: ["a"],
    items: [{ value: "a", label: "A" }, { value: "b", label: "B" }],
  }));
  m.node.querySelectorAll("button")[1].click();
  assert.deepEqual(m.value.sort(), ["a", "b"]);
  m.node.querySelectorAll("button")[0].click();
  assert.deepEqual(m.value, ["b"], "clicking an active one turns it off");
});

// --- choice ----------------------------------------------------------------

test("checkbox reads and writes its state", () => {
  let last = null;
  const c = mount(composer.checkbox.default({ label: "On", checked: false, onChange: (v) => { last = v; } }));
  const input = c.node.querySelector("input");
  input.click();
  assert.equal(c.value, true);
  assert.equal(last, true);
  c.setValue(false);
  assert.equal(input.checked, false);
});

test("a bare checkbox carries its label as the accessible name", () => {
  const c = composer.checkbox.default({ label: "Enabled" });
  assert.equal(c.node.querySelector("input").getAttribute("aria-label"), "Enabled");
});

test("a checkbox row is labelled by its visible text, not by aria-label", () => {
  // Duplicating it would override what the user can see with a copy that can
  // drift from it.
  const c = composer.checkboxRow.default({ label: "Sync", hint: "Locks timing" });
  const input = c.node.querySelector("input");
  assert.equal(input.getAttribute("aria-label"), null);
  assert.equal(c.node.getAttribute("for"), input.id);
  assert.ok(input.getAttribute("aria-describedby"), "the hint is wired as the description");
});

test("checklist collects several values", () => {
  const c = mount(composer.checklist.default({
    values: ["a"], items: [{ value: "a", label: "A" }, { value: "b", label: "B" }],
  }));
  c.node.querySelectorAll("input")[1].click();
  assert.deepEqual(c.value.sort(), ["a", "b"]);
  c.setValue(["b"]);
  assert.deepEqual(c.value, ["b"]);
});

test("radioGroup keeps exactly one", () => {
  const r = mount(composer.radioGroup.default({
    value: "a", options: [{ value: "a", label: "A" }, { value: "b", label: "B" }],
  }));
  const inputs = r.node.querySelectorAll("input");
  inputs[1].click();
  assert.equal(r.value, "b");
  assert.equal(inputs[0].checked, false);
});

test("segmented selects and reflects aria-checked", () => {
  const s = mount(composer.segmented.md({
    value: "t2v", options: [{ value: "t2v", label: "T" }, { value: "i2v", label: "I" }],
  }));
  const buttons = s.node.querySelectorAll("button");
  buttons[1].click();
  assert.equal(s.value, "i2v");
  assert.equal(buttons[1].getAttribute("aria-checked"), "true");
  assert.equal(buttons[0].getAttribute("aria-checked"), "false");
});

test("select round-trips its value", () => {
  const s = mount(composer.select.md({
    value: "b", options: [{ value: "a", label: "A" }, { value: "b", label: "B" }],
  }));
  assert.equal(s.value, "b");
  s.setValue("a");
  assert.equal(s.value, "a");
});

test("toggle is a switch, not a checkbox, to assistive tech", () => {
  const t = mount(composer.toggle.default({ label: "Second pass", checked: true }));
  assert.equal(t.node.querySelector("input").getAttribute("role"), "switch");
  assert.equal(t.value, true);
});

// --- input -----------------------------------------------------------------

test("onCommit fires on Enter and blur, not on every keystroke", () => {
  // A settings value that updates per keystroke writes nine intermediate values
  // while you type "0.65".
  const commits = [];
  const inputs = [];
  const i = mount(composer.input.md({
    onInput: (v) => inputs.push(v),
    onCommit: (v) => commits.push(v),
  }));
  i.node.value = "roof";
  fire(i.node, "input");
  assert.deepEqual(inputs, ["roof"]);
  assert.deepEqual(commits, []);

  i.node.dispatchEvent(new window.KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true }));
  assert.deepEqual(commits, ["roof"]);

  fire(i.node, "blur");
  assert.deepEqual(commits, ["roof"], "an unchanged value does not commit twice");
});

test("number clamps on commit, not while typing", () => {
  // Clamping mid-keystroke makes a leading "-" or "0." impossible to type.
  let last = null;
  const n = mount(composer.number.md({ value: 3, min: 1, max: 10, onChange: (v) => { last = v; } }));
  const input = n.node.querySelector("input") || n.node;
  input.value = "99";
  fire(input, "input");
  assert.equal(input.value, "99", "not clamped yet");
  fire(input, "blur");
  assert.equal(last, 10);
});

test("number applies its precision", () => {
  let last = null;
  const n = mount(composer.number.md({ value: 1, min: 0, max: 10, precision: 1, onChange: (v) => { last = v; } }));
  const input = n.node.querySelector("input") || n.node;
  input.value = "3.14159";
  fire(input, "blur");
  assert.equal(last, 3.1);
});

test("stepper steps without float drift", () => {
  const s = mount(composer.stepper.md({ value: 0.1, min: 0, max: 1, step: 0.2 }));
  s.node.querySelectorAll("button")[1].click();
  assert.equal(s.value, 0.3, "0.1 + 0.2 must not surface as 0.30000000000000004");
});

test("stepper respects its bounds", () => {
  const s = mount(composer.stepper.md({ value: 1, min: 0, max: 1, step: 1 }));
  s.node.querySelectorAll("button")[1].click();
  assert.equal(s.value, 1);
});

test("filterList filters on label and hint, and reports a pick", () => {
  let picked = null;
  const f = mount(composer.filterList.md({
    onChange: (id) => { picked = id; },
    items: [
      { id: "a", label: "bong_tangent", hint: "validated" },
      { id: "b", label: "karras", hint: "2.3 era" },
    ],
  }));
  const search = f.node.querySelector("input");
  search.value = "valid";
  fire(search, "input");
  assert.equal(f.node.querySelectorAll(".cx-filter-row").length, 1, "matches on hint too");

  f.node.querySelector(".cx-filter-row").click();
  assert.equal(picked, "a");
});

test("filterList says so when nothing matches", () => {
  const f = mount(composer.filterList.md({ items: [{ id: "a", label: "one" }] }));
  const search = f.node.querySelector("input");
  search.value = "zzz";
  fire(search, "input");
  assert.equal(f.node.querySelectorAll(".cx-filter-row").length, 0);
  assert.ok(f.node.querySelector(".cx-filter-empty"), "an empty list must say it is empty");
});

// --- slider ----------------------------------------------------------------

test("slider reports as it moves and commits on release", () => {
  const moves = [];
  const commits = [];
  const s = mount(composer.slider.md({
    value: 0.5, min: 0, max: 1, step: 0.1,
    onChange: (v) => moves.push(v), onCommit: (v) => commits.push(v),
  }));
  const input = s.node.querySelector("input");
  input.value = "0.8";
  fire(input, "input");
  fire(input, "change");
  assert.deepEqual(moves, [0.8]);
  assert.deepEqual(commits, [0.8]);
});

test("the slider paints its fill from the value", () => {
  const s = mount(composer.slider.md({ value: 0.25, min: 0, max: 1, step: 0.05 }));
  assert.equal(s.node.querySelector("input").style.getPropertyValue("--fill"), "25%");
});

test("slider.readout shows the value and its unit", () => {
  const s = mount(composer.slider.readout({ value: 0.65, min: 0, max: 1, step: 0.05, unit: "x", precision: 2 }));
  assert.equal(s.node.querySelector("output").textContent, "0.65x");
  s.setValue(0.3);
  assert.equal(s.node.querySelector("output").textContent, "0.30x");
});

test("slider.macro jumps to a preset", () => {
  const s = mount(composer.slider.macro({
    value: 0.2, min: 0, max: 1, step: 0.05,
    presets: [{ label: "Chill", value: 0.2 }, { label: "Chaos", value: 0.9 }],
  }));
  s.node.querySelectorAll(".cx-macro-presets button")[1].click();
  assert.equal(s.value, 0.9);
});

test("range thumbs cannot pass each other", () => {
  // A range whose start is after its end is a state every consumer would then
  // have to defend against.
  const r = mount(composer.range.md({ from: 0.3, to: 0.7, min: 0, max: 1, step: 0.05 }));
  const [lo, hi] = r.node.querySelectorAll("input");
  lo.value = "0.9";
  fire(lo, "input");
  const { from, to } = r.value;
  assert.ok(from <= to, `from ${from} must not exceed to ${to}`);
});
