// What the chart and the tag elements do when used.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom, fire, key } from "./_dom.js";

let composer;
test.before(async () => {
  setupDom();
  ({ composer } = await import("../composer.js"));
});
test.after(() => teardownDom());

const mount = (handle) => { document.body.appendChild(handle.node); return handle; };
const drawn = (chart) => [...chart.node.querySelectorAll("path.cx-chart-line")];
const points = (path) => (path.getAttribute("d").match(/[ML]/g) || []).length;

// --- chart.line --------------------------------------------------------------

test("a chart with no data shows its empty text instead of a plot", () => {
  const c = mount(composer.chart.line({ series: [], empty: "Nothing yet" }));
  assert.ok(c.node.classList.contains("cx-chart-none"));
  assert.equal(c.node.querySelector(".cx-chart-empty").textContent, "Nothing yet");
  c.setEmpty("Still nothing");
  assert.equal(c.node.querySelector(".cx-chart-empty").textContent, "Still nothing");
  c.setSeries([{ name: "a", values: [1, 2, 3] }]);
  assert.ok(!c.node.classList.contains("cx-chart-none"));
  c.setSeries([]);
  assert.ok(c.node.classList.contains("cx-chart-none"), "going back to no data shows the empty text again");
});

test("a chart skips points that are not numbers and survives one point and constant data", () => {
  const c = mount(composer.chart.line({ series: [{ name: "a", values: [1, NaN, Infinity, -Infinity, 3, undefined, null] }] }));
  assert.equal(points(drawn(c)[0]), 2, "only 1 and 3 are numbers; the rest are left out");
  assert.ok(!/NaN|Infinity|null|undefined/.test(drawn(c)[0].getAttribute("d")));
});

test("a lone point is a dot and a constant series gets a range", () => {
  const one = mount(composer.chart.line({ series: [{ name: "a", values: [5] }] }));
  assert.match(drawn(one)[0].getAttribute("d"), /h0$/);
  const flat = mount(composer.chart.line({ series: [{ name: "a", values: [2, 2, 2, 2] }] }));
  assert.ok(flat.node.querySelectorAll(".cx-chart-yaxis .cx-chart-tick").length >= 2, "a flat line still has ticks");
  assert.ok(!/NaN|Infinity/.test(drawn(flat)[0].getAttribute("d")));
});

test("negative values and a log axis are drawn without NaN", () => {
  const neg = mount(composer.chart.line({ series: [{ name: "a", values: [-3, -1, 2] }] }));
  assert.ok(!/NaN/.test(drawn(neg)[0].getAttribute("d")));
  const log = mount(composer.chart.line({ logY: true, series: [{ name: "a", values: [0.001, 0, -4, 0.1, 10] }] }));
  assert.equal(points(drawn(log)[0]), 3, "a log axis drops what it cannot show");
  assert.ok(!/NaN|Infinity/.test(drawn(log)[0].getAttribute("d")));
});

test("ten thousand points are drawn as a few hundred and the spike survives", () => {
  const values = Array.from({ length: 10000 }, (_, i) => (i === 5001 ? 100 : Math.sin(i / 50)));
  const c = mount(composer.chart.line({ series: [{ name: "a", values }] }));
  const n = points(drawn(c)[0]);
  assert.ok(n <= 600 && n > 100, `drew ${n}`);
  assert.ok(/M[\d.]+ 0\.0|L[\d.]+ 0\.0/.test(drawn(c)[0].getAttribute("d")), "the highest point reaches the top");
});

test("faint, dashed and tone are classes of the line; a bad tone or height is refused", () => {
  const c = mount(composer.chart.line({ series: [{ name: "raw", faint: true, values: [1, 2] }, { name: "smooth", tone: "good", dashed: true, values: [1, 2] }] }));
  const [a, b] = drawn(c);
  assert.ok(a.classList.contains("cx-chart-faint") && a.classList.contains("cx-chart-accent"));
  assert.ok(b.classList.contains("cx-chart-good") && b.classList.contains("cx-chart-dashed"));
  assert.deepEqual([...c.node.querySelectorAll(".cx-chart-key")].map((k) => k.textContent), ["raw", "smooth"]);
  assert.throws(() => composer.chart.line({ series: [{ name: "a", tone: "pink", values: [1] }] }), TypeError);
  assert.throws(() => composer.chart.line({ height: "huge" }), TypeError);
});

test("hovering reads the nearest point, leaving shows the latest", () => {
  const c = mount(composer.chart.line({ xLabel: "step", series: [{ name: "loss", values: [10, 20, 30, 40, 50] }] }));
  assert.match(c.node.querySelector(".cx-chart-readout").textContent, /step.*loss 50|loss 50/);
  const svg = c.node.querySelector(".cx-chart-svg");
  svg.getBoundingClientRect = () => ({ left: 0, width: 400, top: 0, height: 100 });
  const move = (x) => svg.dispatchEvent(Object.assign(new window.Event("pointermove", { bubbles: true }), { clientX: x }));
  move(0);
  assert.match(c.node.querySelector(".cx-chart-readout").textContent, /step 0 · loss 10/);
  move(400);
  assert.match(c.node.querySelector(".cx-chart-readout").textContent, /step 4 · loss 50/);
  svg.dispatchEvent(new window.Event("pointerleave"));
  assert.match(c.node.querySelector(".cx-chart-readout").textContent, /loss 50/);
});

test("series names and axis names are shown as text", () => {
  const evil = '<img src=x onerror="boom()">';
  const c = mount(composer.chart.line({ xLabel: evil, yLabel: evil, label: evil, series: [{ name: evil, values: [1, 2] }] }));
  assert.equal(c.node.querySelector("img"), null);
  assert.ok(c.node.textContent.includes(evil));
});

// --- tags.editor ---------------------------------------------------------------

const chips = (e) => [...e.node.querySelectorAll(".cx-tag-text")].map((n) => n.textContent);
const typeIn = (e, text) => { const i = e.node.querySelector("input"); i.value = text; fire(i, "input"); return i; };

test("Enter and comma add a trimmed tag, empty and repeated ones are refused", () => {
  const heard = [];
  const e = mount(composer.tags.editor({ value: ["a"], onChange: (l) => heard.push(l) }));
  let i = typeIn(e, "  dusk light  "); key(i, "Enter");
  assert.deepEqual(e.get(), ["a", "dusk light"]);
  assert.equal(i.value, "", "the box is emptied");
  i = typeIn(e, "rain"); key(i, ",");
  i = typeIn(e, ""); key(i, "Enter");
  i = typeIn(e, "a"); key(i, "Enter");
  assert.deepEqual(e.get(), ["a", "dusk light", "rain"]);
  assert.deepEqual(heard.at(-1), ["a", "dusk light", "rain"]);
  assert.equal(heard.length, 2, "only real changes are reported");
  assert.deepEqual(chips(e), ["a", "dusk light", "rain"]);
});

test("a comma never ends up inside a tag, and pasted lists split", () => {
  const e = mount(composer.tags.editor({ value: ["x,y", " ", "z"] }));
  assert.deepEqual(e.get(), ["x y", "z"]);
  const i = e.node.querySelector("input");
  const paste = new window.Event("paste", { bubbles: true, cancelable: true });
  paste.clipboardData = { getData: () => "one, two\nthree" };
  i.dispatchEvent(paste);
  assert.deepEqual(e.get(), ["x y", "z", "one", "two", "three"]);
  assert.equal(paste.defaultPrevented, true);
});

test("Backspace in an empty box removes the last tag; a chip's button and Delete remove that chip", () => {
  const e = mount(composer.tags.editor({ value: ["a", "b", "c"] }));
  const i = e.node.querySelector("input");
  key(i, "Backspace");
  assert.deepEqual(e.get(), ["a", "b"]);
  typeIn(e, "q"); key(i, "Backspace");
  assert.deepEqual(e.get(), ["a", "b"], "with text in the box Backspace is for the text");
  e.node.querySelector(".cx-tag-x").click();
  assert.deepEqual(e.get(), ["b"]);
  const chip = e.node.querySelector(".cx-tag");
  chip.focus(); key(chip, "Delete");
  assert.deepEqual(e.get(), []);
});

test("arrow keys move between chips and the box; one chip is a tab stop", () => {
  const e = mount(composer.tags.editor({ value: ["a", "b"] }));
  const [a, b] = e.node.querySelectorAll(".cx-tag");
  assert.deepEqual([a.tabIndex, b.tabIndex], [0, -1]);
  a.focus(); key(a, "ArrowRight");
  assert.equal(document.activeElement, b);
  assert.deepEqual([a.tabIndex, b.tabIndex], [-1, 0], "the focused chip is the tab stop");
  key(b, "ArrowRight");
  assert.equal(document.activeElement, e.node.querySelector("input"));
  key(e.node.querySelector("input"), "ArrowLeft");
  assert.equal(document.activeElement, b);
});

test("dragging a chip onto another moves it", () => {
  const heard = [];
  const e = mount(composer.tags.editor({ value: ["a", "b", "c"], onChange: (l) => heard.push(l) }));
  const store = {};
  const dt = { types: [], effectAllowed: "", setData: (t, v) => { store[t] = v; dt.types.push(t); }, getData: (t) => store[t] };
  const [a, , c] = e.node.querySelectorAll(".cx-tag");
  const ev = (type, target) => { const x = new window.Event(type, { bubbles: true, cancelable: true }); x.dataTransfer = dt; target.dispatchEvent(x); };
  ev("dragstart", c); ev("drop", a);
  assert.deepEqual(e.get(), ["c", "a", "b"]);
  assert.deepEqual(heard.at(-1), ["c", "a", "b"]);
});

test("suggestions are offered, Tab takes the first, and picking one adds it", () => {
  const e = mount(composer.tags.editor({ value: ["rain"], suggestions: ["rain", "rooftop", "dusk", "sunrise"] }));
  const i = typeIn(e, "ro");
  const options = () => [...document.querySelectorAll("[role=option]")].map((o) => o.textContent);
  assert.deepEqual(options(), ["rooftop"], "a tag already in the list is not suggested");
  key(i, "Tab");
  assert.deepEqual(e.get(), ["rain", "rooftop"]);
  typeIn(e, "su");
  document.querySelector("[role=option]").click();
  assert.deepEqual(e.get(), ["rain", "rooftop", "sunrise"]);
  assert.equal(i.value, "");
  e.destroy();
  assert.equal(document.querySelectorAll("[role=option]").length, 0, "destroy closes the list");
});

test("Enter on a highlighted suggestion adds the suggestion once, not the typed text as well", () => {
  const e = mount(composer.tags.editor({ suggestions: ["dusk", "dawn"] }));
  const i = typeIn(e, "d");
  key(i, "ArrowDown"); key(i, "Enter");
  assert.deepEqual(e.get(), ["dusk"]);
});

test("maxTags stops adding, and setValue replaces without calling onChange", () => {
  let calls = 0;
  const e = mount(composer.tags.editor({ value: ["a"], maxTags: 2, onChange: () => { calls += 1; } }));
  const i = typeIn(e, "b"); key(i, "Enter");
  assert.equal(i.disabled, true, "full: the box is off");
  assert.deepEqual(e.get(), ["a", "b"]);
  e.setValue(["x", "y", "z"]);
  assert.deepEqual(e.get(), ["x", "y"]);
  assert.equal(calls, 1);
});

test("a tag is shown as text, never as markup", () => {
  const evil = '<img src=x onerror="boom()">';
  const e = mount(composer.tags.editor({ value: [evil] }));
  assert.equal(e.node.querySelector("img"), null);
  assert.deepEqual(chips(e), [evil]);
});

// --- tags.cloud ----------------------------------------------------------------

test("a cloud shows tags with counts, reports picks and removals, and marks the selected", () => {
  const picks = [], gone = [];
  const t = mount(composer.tags.cloud({ items: [{ tag: "a", count: 3 }, { tag: "b", count: 1 }], selected: ["b"], onPick: (x) => picks.push(x), onRemove: (x) => gone.push(x) }));
  assert.equal(t.node.querySelectorAll(".cx-tag-cloud").length, 2);
  assert.equal(t.node.querySelector(".cx-tag-count").textContent, "3");
  assert.equal(t.node.querySelector("[data-tag=b]").classList.contains("cx-on"), true);
  assert.equal(t.node.querySelector("[data-tag=b] .cx-tag-pick").getAttribute("aria-pressed"), "true");
  t.node.querySelector("[data-tag=a] .cx-tag-pick").click();
  t.node.querySelector("[data-tag=a] .cx-tag-x").click();
  assert.deepEqual([picks, gone], [["a"], ["a"]]);
  t.setSelected(["a"]);
  assert.equal(t.node.querySelector("[data-tag=a]").classList.contains("cx-on"), true);
  assert.equal(t.node.querySelector("[data-tag=b]").classList.contains("cx-on"), false);
  t.setItems([]);
  assert.ok(t.node.classList.contains("cx-tagcloud-none"));
});

test("a cloud with no onRemove has no remove buttons", () => {
  const t = mount(composer.tags.cloud({ items: [{ tag: "a", count: 3 }] }));
  assert.equal(t.node.querySelector(".cx-tag-x"), null);
});

test("five thousand tags draw 300 and say how many more", () => {
  const items = Array.from({ length: 5000 }, (_, i) => ({ tag: `t${i}`, count: 5000 - i }));
  const t = mount(composer.tags.cloud({ items }));
  assert.equal(t.node.querySelectorAll(".cx-tag-cloud").length, 300);
  assert.equal(t.node.querySelector(".cx-tagcloud-more").textContent, "+4700 more");
});

test("a cloud tag is shown as text, never as markup", () => {
  const evil = '<img src=x onerror="boom()">';
  const t = mount(composer.tags.cloud({ items: [{ tag: evil, count: 1 }] }));
  assert.equal(t.node.querySelector("img"), null);
  assert.ok(t.node.textContent.includes(evil));
});


// --- captioned ---------------------------------------------------------------

test("a captioned card shows its tags as chips and hands the whole line back after an edit", () => {
  const heard = [];
  const c = mount(composer.captioned.cell({ title: "a.png", caption: "red, blue, red", onChange: (v) => heard.push(v) }));
  assert.deepEqual([...c.node.querySelectorAll(".cx-tag-text")].map((n) => n.textContent), ["red", "blue"]);
  c.node.querySelector(".cx-tag-x").click();
  assert.deepEqual(heard, ["blue"]);
  c.setCaption("one, two");
  assert.equal(c.getCaption(), "one, two");
  assert.deepEqual(heard, ["blue"], "setCaption does not call onChange");
});

test("a captioned card takes its picture and caption as text, selects, opens and marks a clip", () => {
  const events = [];
  const c = mount(composer.captioned.side({ title: "<b>x</b>", caption: "<i>t</i>", kind: "video", src: "/thumb.png", badge: "Clip", tone: "good",
    onSelect: (on) => events.push(["select", on]), onOpen: () => events.push(["open"]) }));
  assert.equal(c.node.querySelector(".cx-captioned-title").textContent, "<b>x</b>");
  assert.equal(c.node.querySelector(".cx-tag-text").textContent, "<i>t</i>");
  assert.ok(c.node.querySelector(".cx-captioned-play"), "a clip is marked");
  assert.equal(c.node.querySelector(".cx-captioned-badge").textContent, "Clip");
  c.node.querySelector(".cx-captioned-view").click();
  const tick = c.node.querySelector(".cx-captioned-pick");
  tick.checked = true; tick.dispatchEvent(new window.Event("change", { bubbles: true }));
  assert.deepEqual(events, [["open"], ["select", true]]);
  assert.ok(c.node.classList.contains("cx-on"));
  c.setSelected(false);
  assert.ok(!c.node.classList.contains("cx-on"));
});

test("a picture that fails to load is replaced by a glyph, and no picture shows one", () => {
  const c = mount(composer.captioned.cell({ title: "a", src: "/gone.png" }));
  c.node.querySelector("img").dispatchEvent(new window.Event("error"));
  assert.ok(c.node.querySelector(".cx-captioned-glyph") && !c.node.querySelector("img"));
  const none = mount(composer.captioned.cell({ title: "b" }));
  assert.ok(none.node.querySelector(".cx-captioned-glyph"));
});
