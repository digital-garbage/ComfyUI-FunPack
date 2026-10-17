import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";
import { addTags, bulk, frequency, joinTags, splitTags } from "./captions.js";

let feature;
test.before(async () => { setupDom(); feature = (await import("./index.js")).default; });
test.after(() => teardownDom());

const wait = (ms = 20) => new Promise((r) => setTimeout(r, ms));

function fake({ enabled = true } = {}) {
  const calls = [];
  const doc = { name: "cats", trigger: "", settings: {}, items: [
    { id: "aaaaaaaaaaaa", src: { kind: "bin", id: "bbbbbbbbbbbb" }, media_kind: "image", label: "own", caption: "cat, grey", name: "a.png" },
    { id: "cccccccccccc", src: { kind: "output", type: "output", subfolder: "", filename: "v.mp4" }, media_kind: "video", label: "liked", caption: "", name: "v.mp4" },
  ] };
  const run = { run: "", state: "idle", loss: [], loss_from: 0, step: 0, total: 0 };
  const api = {
    trainState: async () => ({ enabled }),
    trainSets: async () => ({ sets: [{ name: "cats", count: 2 }] }),
    trainSet: async () => structuredClone(doc),
    trainSaveSet: async (name, d) => { calls.push(["save", name, d]); return d; },
    trainAdd: async () => ({ set: doc, added: 0 }),
    trainRated: async () => ({ results: [], counts: { liked: 3 } }),
    trainTaggers: async () => ({ taggers: [{ id: "wd_vit_v3", title: "WD ViT v3", state: "absent", hint: "", done: 0, total: 0 }] }),
    trainAutotagState: async () => ({ state: "idle" }),
    trainRun: async () => ({ ...run }),
    trainRuns: async () => [],
    trainPlan: async (set, slots, settings) => { calls.push(["plan", set, settings]); return { prompt: { 1: { class_type: "FunPackTrainLora", inputs: {} } }, problems: [] }; },
    queueGraph: async (g) => { calls.push(["queue", g]); return "pid"; },
    trainStop: async () => { calls.push(["stop"]); return { stopped: true }; },
    media: async () => ({ media: [{ id: "bbbbbbbbbbbb", name: "a.png", kind: "image" }, { id: "dddddddddddd", name: "s.wav", kind: "audio" }] }),
    uploadMedia: async () => ({ media: [], problems: [] }),
  };
  const app = { api, dom: {}, pipeline: { slots: () => [{ id: "model" }] }, say: () => {}, on: () => () => {} };
  return { api, app, calls, doc, run };
}

const open = async (f) => {
  const host = document.createElement("div");
  document.body.append(host);
  const off = feature.setup({ host, app: f.app });
  await wait();
  const button = [...host.querySelectorAll("button")].find((b) => b.textContent.includes("Training"));
  return { host, button, off };
};
const text = () => document.body.textContent;
const click = (label, root = document) => [...root.querySelectorAll("button")].find((b) => b.textContent.trim() === label);

test("captions: tags are trimmed and unique, counts are per caption, bulk edits only the chosen", () => {
  assert.deepEqual(splitTags(" a,B ,, a\nc "), ["a", "B", "c"]);
  assert.equal(joinTags(["x", "X", " y "]), "x, y");
  assert.deepEqual(frequency([{ caption: "a, b" }, { caption: "a, a" }, { caption: "c" }]).map((x) => [x.tag, x.count]), [["a", 2], ["b", 1], ["c", 1]]);
  const items = [{ id: "x", caption: "cat, dog" }, { id: "y", caption: "cat" }];
  assert.equal(bulk(items, "remove", "cat", "", new Set(["x"])), 1);
  assert.deepEqual(items.map((i) => i.caption), ["dog", "cat"]);
  assert.equal(bulk(items, "replace", "cat", "lion"), 1);
  assert.equal(bulk(items, "prepend", "zq"), 2);
  assert.equal(items[1].caption, "zq, lion");
  assert.equal(bulk(items, "add", ""), 0);
  const prose = [{ id: "p", caption: "A fox runs.\nIt jumps, twice,  home" }];
  assert.equal(bulk(prose, "add", "sky"), 1);
  assert.equal(prose[0].caption, "A fox runs.\nIt jumps, twice, home, sky", "only commas are touched: the words stay as written");
  assert.equal(addTags("a fox, running home.", ["A  Fox", "grass", "sky"]), "a fox, running home., grass, sky");
  assert.equal(addTags("", ["x", "y"]), "x, y");
  assert.equal(addTags("a long prompt,", ["x"]), "a long prompt, x");
});

test("the window is hidden while the add-on is off", async () => {
  const f = fake({ enabled: false });
  const { button, off } = await open(f);
  assert.ok(button.hidden);
  off();
});

test("the Data step lists the sets, offers the three sources and can be skipped", async () => {
  const f = fake();
  const { button, off } = await open(f);
  assert.ok(!button.hidden);
  button.click();
  await wait(60);
  const t = text();
  for (const word of ["Drop pictures or clips here", "Choose in your Bin", "Liked results", "Bad composition results", "Bad image results", "Skip to training", "Next: tags"]) assert.ok(t.includes(word), word);
  assert.ok(t.includes("2 in \"cats\""), "the open set is summarised");
  assert.ok(document.querySelectorAll(".cx-cell").length === 1, "only pictures and clips are offered from the Bin, not sound");
  off();
});

test("Tags step: a text box per item under its picture; typing saves the set and keeps the text as typed", async () => {
  const f = fake();
  const { button, off } = await open(f);
  button.click();
  await wait(60);
  click("Next: tags").click();
  await wait(20);
  assert.equal(document.querySelectorAll(".cx-captioned").length, 2);
  const box = document.querySelector(".cx-captioned textarea");
  assert.equal(box.value, "cat, grey");
  box.value = "A cat\nsits, calmly";
  box.dispatchEvent(new window.Event("input", { bubbles: true }));
  await wait(700);
  const saved = f.calls.filter((c) => c[0] === "save").pop();
  assert.equal(saved[2].items[0].caption, "A cat\nsits, calmly");
  off();
});

test("Train step: Start sends the settings and the pipeline to the plan and queues the graph", async () => {
  const f = fake();
  const { button, off } = await open(f);
  button.click();
  await wait(60);
  click("Skip to training").click();
  await wait(20);
  assert.ok(text().includes("Start training"));
  click("Start training").click();
  await wait(40);
  const plan = f.calls.find((c) => c[0] === "plan");
  assert.equal(plan[1], "cats");
  assert.equal(plan[2].steps, 500);
  assert.ok(f.calls.some((c) => c[0] === "queue"));
  off();
});

test("Train step: More settings stays open while an option in it is changed", async () => {
  const f = fake();
  const { button, off } = await open(f);
  button.click();
  await wait(60);
  click("Skip to training").click();
  await wait(20);
  const more = () => document.querySelector("details.cx-collapsible");
  more().open = true;
  more().dispatchEvent(new Event("toggle"));
  const select = [...more().querySelectorAll("select")][0];
  select.value = "LoHa";
  select.dispatchEvent(new Event("change", { bubbles: true }));
  await wait(40);
  assert.equal(more().open, true);
  off();
});

test("Tags step: the prefix and postfix are the set's own, saved with it and sent to the run", async () => {
  const f = fake();
  const { button, off } = await open(f);
  button.click();
  await wait(60);
  click("Next: tags").click();
  await wait(20);
  const field = (label) => [...document.querySelectorAll(".cx-field")].find((x) => x.textContent.startsWith(label)).querySelector("input");
  for (const [label, text] of [["Prefix", "zq"], ["Postfix", "end"]]) {
    const input = field(label);
    input.value = text;
    input.dispatchEvent(new window.Event("change", { bubbles: true }));
  }
  await wait(700);
  const saved = f.calls.filter((c) => c[0] === "save").pop();
  assert.equal(saved[2].trigger, "zq");
  assert.equal(saved[2].postfix, "end");
  click("Next: train").click();
  await wait(20);
  click("Start training").click();
  await wait(40);
  const plan = f.calls.find((c) => c[0] === "plan");
  assert.equal(plan[2].trigger, "zq");
  assert.equal(plan[2].postfix, "end");
  off();
});

test("a refused plan shows the reason and starts nothing", async () => {
  const f = fake();
  f.api.trainPlan = async () => ({ prompt: {}, problems: ["the pipeline has no clip loader"] });
  const { button, off } = await open(f);
  button.click();
  await wait(60);
  click("Skip to training").click();
  await wait(20);
  click("Start training").click();
  await wait(40);
  assert.ok(text().includes("no clip loader"));
  assert.ok(!f.calls.some((c) => c[0] === "queue"));
  off();
});

test("a run in progress draws its loss, then reports the saved file; Stop asks the server", async () => {
  const f = fake();
  let n = 0;
  f.api.trainRun = async (since) => {
    n += 1;
    if (n < 3) return { run: "", state: "idle", loss: [], loss_from: 0 };
    if (n === 3) return { run: "r1", state: "training", step: 2, total: 5, elapsed: 4, message: "", loss: [0.9, 0.7], loss_from: since };
    return { run: "r1", state: "done", step: 5, total: 5, elapsed: 9, message: "", loss: [0.5, 0.4, 0.3].slice(0, 5 - since), loss_from: since, lora: "cats_lora.safetensors" };
  };
  const { button, off } = await open(f);
  button.click();
  await wait(60);
  click("Skip to training").click();
  await wait(20);
  click("Start training").click();
  await wait(1200);
  assert.ok(text().includes("Step 2 of 5"), "progress is shown");
  assert.ok(document.querySelector(".cx-chart path.cx-chart-line"), "the loss is drawn");
  click("Stop").click();
  await wait(20);
  assert.ok(f.calls.some((c) => c[0] === "stop"));
  await wait(1200);
  assert.ok(text().includes("cats_lora.safetensors"));
  off();
});

test("rated results are added with the captions the server built, and changing what to include asks again", async () => {
  const f = fake();
  const asked = [];
  f.api.trainRated = async (include) => { asked.push(include); return { results: [{ label: "liked", media: { type: "output", subfolder: "", filename: "x.mp4" }, caption: include.includes("postfix") ? "a red fox 35mm" : "a red fox" }], counts: { liked: 1 } }; };
  const added = [];
  f.api.trainAdd = async (name, entries) => { added.push(entries); return { set: f.doc, added: entries.length }; };
  const { button, off } = await open(f);
  button.click();
  await wait(60);
  assert.deepEqual(asked[0], ["main"], "the main prompt alone is the default");
  click("Add these results").click();
  await wait(40);
  assert.equal(added[0][0].caption, "a red fox");
  assert.equal(added[0][0].media_kind, "video");
  const box = [...document.querySelectorAll(".cx-checklist .cx-check-row")].find((r) => r.textContent.includes("Postfix")).querySelector("input");
  box.click();
  await wait(40);
  assert.deepEqual(asked[asked.length - 1].sort(), ["main", "postfix"]);
  click("Add these results").click();
  await wait(40);
  assert.equal(added[1][0].caption, "a red fox 35mm");
  off();
});

test("Tags step: Remove selected takes the ticked items out of the set and saves", async () => {
  const f = fake();
  const { button, off } = await open(f);
  button.click();
  await wait(60);
  click("Next: tags").click();
  await wait(20);
  document.querySelector(".cx-captioned-pick").click();
  click("Remove selected").click();
  await wait(40);
  const yes = [...document.querySelectorAll("button")].filter((b) => b.textContent.trim() === "Remove").pop();
  yes.click();
  await wait(700);
  const saved = f.calls.filter((c) => c[0] === "save").pop();
  assert.equal(saved[2].items.length, 1);
  assert.equal(saved[2].items[0].id, "cccccccccccc");
  off();
});

test("rated results are shown as pictures, and only the picked ones are added", async () => {
  const f = fake();
  f.api.trainRated = async () => ({ results: [
    { label: "liked", media: { type: "output", subfolder: "", filename: "x.mp4" }, caption: "a" },
    { label: "liked", media: { type: "output", subfolder: "", filename: "y.png" }, caption: "b" }], counts: { liked: 2 } });
  const added = [];
  f.api.trainAdd = async (name, entries) => { added.push(entries); return { set: f.doc, added: entries.length }; };
  const { button, off } = await open(f);
  button.click();
  await wait(60);
  assert.ok(text().includes("2 rated results found"));
  const cells = [...document.querySelectorAll(".cx-cell")];
  assert.equal(cells.length, 3, "one bin item and the two rated results are drawn as pictures");
  (cells[2].querySelector("button") || cells[2]).click();
  await wait(20);
  assert.ok(text().includes("1 picked: only those are added"));
  click("Add these results").click();
  await wait(40);
  assert.equal(added[0].length, 1);
  assert.equal(added[0][0].src.filename, "y.png");
  off();
});
