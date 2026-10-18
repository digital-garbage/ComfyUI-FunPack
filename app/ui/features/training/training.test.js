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
    trainClips: async () => ({ clips: { "output:/v.mp4": { frames: 124, fps: 24 } } }),
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
const tab = (label) => [...document.querySelectorAll("button, [role=tab]")].find((b) => b.textContent.trim().endsWith(label));
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
  for (const [label, text] of [["Prefix", "zq " + "a long prefix of many words, ".repeat(12) + "[Shot 1]"], ["Postfix", "end"]]) {
    const input = field(label);
    input.value = text;
    input.dispatchEvent(new window.Event("change", { bubbles: true }));
  }
  await wait(700);
  const saved = f.calls.filter((c) => c[0] === "save").pop();
  assert.equal(saved[2].trigger, "zq " + "a long prefix of many words, ".repeat(12) + "[Shot 1]", "a long prefix is kept whole");
  assert.equal(saved[2].postfix, "end");
  click("Next: train").click();
  await wait(20);
  click("Start training").click();
  await wait(40);
  const plan = f.calls.find((c) => c[0] === "plan");
  assert.equal(plan[2].trigger.slice(0, 2), "zq");
  assert.ok(plan[2].trigger.endsWith("[Shot 1]"));
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
  assert.ok(text().includes("Check graphs and previews in the Results tab"), "Train only says where to look");
  assert.ok(!document.querySelector(".cx-chart"), "the graph is not on the Train tab");
  tab("Results").click();
  await wait(20);
  assert.ok(text().includes("Step 2 of 5"), "progress is shown");
  assert.ok(document.querySelector(".cx-chart path.cx-chart-line"), "the loss is drawn");
  tab("Train").click();
  await wait(20);
  click("Stop").click();
  await wait(20);
  assert.ok(f.calls.some((c) => c[0] === "stop"));
  tab("Results").click();
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

test("Tags step: a clip's frame count and rate show beside its label", async () => {
  const f = fake();
  const { button, off } = await open(f);
  button.click();
  await wait(60);
  click("Next: tags").click();
  await wait(100);
  assert.ok(text().includes("124 f · 24 fps"));
  off();
});

test("a preview step pauses the run, the pipeline makes a preview with the LoRA so far, and training carries on", async () => {
  const f = fake();
  f.doc.settings = { preview_strength: -0.8 };
  const previews = [];
  f.app.runner = { preview: async (o) => { previews.push(o); return { image: { filename: "pv.mp4", subfolder: "", type: "output" } }; } };
  let queued = 0, polls = 0;
  f.api.queueGraph = async (g) => { queued += 1; f.calls.push(["queue", g]); return "pid"; };
  f.api.trainRun = async () => {
    polls += 1;
    if (queued === 0) return { run: "", state: "idle", loss: [], loss_from: 0 };
    if (queued === 1) return polls % 2 ? { run: "r1", state: "training", step: 2, total: 6, elapsed: 3, message: "", loss: [0.9, 0.8], loss_from: 0 }
      : { run: "r1", state: "paused", step: 2, total: 6, elapsed: 3, message: "", loss: [], loss_from: 2, paused: { reason: "preview", step: 2, preview: "cats_preview.safetensors", checkpoint: true } };
    if (polls % 3 === 0) return { run: "r1", state: "paused", step: 2, total: 6, message: "", loss: [], loss_from: 2, paused: { reason: "preview", step: 2, preview: "cats_preview.safetensors", checkpoint: true } };   // the record still shows the pause until the new run begins
    return { run: "r1", state: "done", step: 6, total: 6, elapsed: 9, message: "", loss: [0.7, 0.6, 0.5, 0.4], loss_from: 2, lora: "cats_lora.safetensors" };
  };
  const { button, off } = await open(f);
  button.click();
  await wait(60);
  click("Skip to training").click();
  await wait(20);
  click("Start training").click();
  tab("Results").click();
  await wait(3600);
  assert.equal(previews.length, 1);
  assert.equal(previews[0].lora, "cats_preview.safetensors");
  assert.equal(previews[0].strength, -0.8, "the preview uses the strength set for it, here a negative one");
  assert.ok(!("preview_strength" in f.calls.find((c) => c[0] === "plan")[2]), "and it is not sent to the trainer");
  assert.equal(queued, 2, "training was queued again to carry on");
  const plans = f.calls.filter((c) => c[0] === "plan");
  assert.equal(plans[1][2].resume, "r1");
  assert.ok(text().includes("Step 2"), "the preview is shown with its step");
  await wait(3000);
  assert.ok(text().includes("cats_lora.safetensors"));
  off();
});

test("a pause that the window never saw training towards is still shown at the step the server is at", async () => {
  const f = fake();
  f.app.runner = { preview: async () => ({ image: { filename: "pv.png", subfolder: "", type: "output" } }) };
  let queued = 0;
  f.api.queueGraph = async () => { queued += 1; return "pid"; };
  const pause = (step, started) => ({ run: "r1", state: "paused", step, total: 6, started, elapsed: 3, message: "", loss: [0.9, 0.8], loss_from: 0, paused: { reason: "user", step, preview: "p.safetensors", checkpoint: true } });
  f.api.trainRun = async () => (queued === 0 ? { run: "", state: "idle", loss: [], loss_from: 0 } : queued === 1 ? pause(2, 100) : pause(4, 200));
  const { button, off } = await open(f);
  button.click();
  await wait(60);
  click("Skip to training").click();
  await wait(20);
  click("Start training").click();
  await wait(1100);
  assert.ok(text().includes("paused at step 2"));
  click("Resume").click();
  await wait(1400);
  assert.ok(text().includes("paused at step 4"), "the second pause is not mistaken for the old one");
  off();
});

test("Pause asks the server, a paused run offers Resume, Preview now and Stop, and Stop ends it", async () => {
  const f = fake();
  f.app.runner = { preview: async () => ({ error: "x" }) };
  let phase = 0;
  f.api.trainPause = async () => { f.calls.push(["pause"]); phase = 2; return { paused: true }; };
  f.api.queueGraph = async () => { phase = 1; return "pid"; };
  f.api.trainRun = async () => phase === 0 ? { run: "", state: "idle", loss: [], loss_from: 0 }
    : phase === 1 ? { run: "r2", state: "training", step: 1, total: 6, elapsed: 1, message: "", loss: [0.9], loss_from: 0 }
    : { run: "r2", state: "paused", step: 2, total: 6, message: "", loss: [0.8], loss_from: 1, paused: { reason: "user", step: 2, preview: "cats_preview.safetensors", checkpoint: true } };
  const { button, off } = await open(f);
  button.click();
  await wait(60);
  click("Skip to training").click();
  await wait(20);
  click("Start training").click();
  await wait(1400);
  click("Pause").click();
  await wait(20);
  assert.ok(text().includes("Pausing at the end of this step"), "the controls' message is beside the controls");
  tab("Results").click();
  await wait(20);
  assert.ok(!text().includes("Pausing at the end of this step"), "and not with the graph");
  tab("Train").click();
  await wait(1400);
  assert.ok(f.calls.some((c) => c[0] === "pause"));
  assert.ok(text().includes("paused at step 2"));
  assert.ok(click("Resume") && !click("Resume").hidden && !click("Preview now").hidden);
  click("Stop").click();
  await wait(40);
  assert.ok(f.calls.some((c) => c[0] === "stop"));
  assert.ok(text().includes("Training stopped."));
  off();
});

test("Restore and continue is offered for a failed run, goes away while it carries on, and comes back if it fails again", async () => {
  const f = fake();
  let failed = true, phase = 0;
  const rec = () => ({ run: "r9", state: "error", steps: 51, total: 6, set: "cats", loss: [0.5], error: "boom", resumable: failed, settings: {} });
  f.api.trainRuns = async () => (phase === 1 ? [] : [{ run: "r9", name: "cats", set: "cats", state: "error", steps: 51, total: 6 }]);
  f.api.trainRunRecord = async () => rec();
  f.api.queueGraph = async () => { phase = 1; return "pid"; };
  f.api.trainRun = async () => phase === 0 ? { run: "r9", started: 5, state: "error", step: 51, total: 6, loss: [], loss_from: 0 }
    : phase === 1 ? { run: "r9", started: 9, state: "training", step: 52, total: 6, elapsed: 1, loss: [], loss_from: 0 }
    : { run: "r9", started: 9, state: "error", step: 53, total: 6, error: "again", loss: [], loss_from: 0 };
  const { button, off } = await open(f);
  button.click();
  await wait(60);
  tab("Results").click();
  await wait(60);
  assert.ok(click("Restore and continue") && !click("Restore and continue").hidden);
  click("Restore and continue").click();
  await wait(1300);
  assert.equal(f.calls.filter((c) => c[0] === "plan").pop()[2].resume, "r9");
  assert.ok(!click("Restore and continue") || click("Restore and continue").hidden, "hidden while it carries on");
  phase = 2;
  await wait(1500);
  assert.ok(click("Restore and continue") && !click("Restore and continue").hidden, "offered again after the next failure");
  off();
});

test("Train step: the preset picked stays picked when a setting is changed after it, and is not sent to the trainer", async () => {
  const f = fake();
  const { button, off } = await open(f);
  button.click();
  await wait(60);
  click("Skip to training").click();
  await wait(20);
  click("Thorough").click();
  await wait(20);
  const rank = [...document.querySelectorAll(".cx-field")].find((x) => x.textContent.startsWith("Rank")).querySelector("input");
  rank.value = "8";
  rank.dispatchEvent(new window.Event("change", { bubbles: true }));
  await wait(700);
  const picked = [...document.querySelectorAll("[aria-pressed=true], [aria-checked=true], .is-active, .on")].map((e) => e.textContent.trim());
  assert.ok(picked.includes("Thorough") && !picked.includes("Quick check"), `picked: ${picked}`);
  click("Start training").click();
  await wait(60);
  const plan = f.calls.find((c) => c[0] === "plan");
  assert.equal(plan[2].rank, 8);
  assert.equal(plan[2].steps, 2000);
  assert.ok(!("preset" in plan[2]));
  off();
});

test("Results: a finished run has a Download LoRA button that saves the file under its own name", async () => {
  const f = fake();
  const saved = [];
  f.app.dom.saveBlob = (blob, name) => saved.push(name);
  f.api.trainLoraUrl = (id) => `/lora/${id}`;
  f.api.trainRuns = async () => [{ run: "r5", name: "cats", set: "cats", state: "done", steps: 10, total: 10, lora: "cats_lora.safetensors" }];
  f.api.trainRunRecord = async () => ({ run: "r5", state: "done", steps: 10, total: 10, set: "cats", loss: [0.2], lora: "cats_lora.safetensors", settings: {} });
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (url) => (String(url) === "/lora/r5" ? { ok: true, blob: async () => new Blob(["w"]) } : realFetch(url));
  try {
    const { button, off } = await open(f);
    button.click();
    await wait(60);
    tab("Results").click();
    await wait(80);
    assert.ok(click("Download LoRA"));
    click("Download LoRA").click();
    await wait(60);
    assert.deepEqual(saved, ["cats_lora.safetensors"]);
    off();
  } finally { globalThis.fetch = realFetch; }
});

test("Results: steps and the preview can be picked and downloaded, not only the final file", async () => {
  const f = fake();
  const urls = [], saved = [];
  f.app.dom.saveBlob = (blob, name) => saved.push(name);
  f.api.trainLoraUrl = (id, file) => `/lora/${id}?file=${file || ""}`;
  f.api.trainRuns = async () => [{ run: "r6", name: "cats", set: "cats", state: "stopped", steps: 120, total: 500 }];
  f.api.trainRunRecord = async () => ({ run: "r6", state: "stopped", steps: 120, total: 500, set: "cats", loss: [0.2], lora: "", files: ["cats_step50.safetensors", "cats_step100.safetensors", "cats_preview.safetensors"], settings: {} });
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (url) => { urls.push(String(url)); return String(url).startsWith("/lora/") ? { ok: true, blob: async () => new Blob(["w"]) } : realFetch(url); };
  try {
    const { button, off } = await open(f);
    button.click();
    await wait(60);
    tab("Results").click();
    await wait(80);
    const pick = [...document.querySelectorAll("select")].find((s) => [...s.options].some((o) => o.textContent === "Step 100"));
    assert.ok(pick, "a choice of files");
    pick.value = "cats_step100.safetensors";
    pick.dispatchEvent(new window.Event("change", { bubbles: true }));
    await wait(20);
    click("Download LoRA").click();
    await wait(60);
    assert.deepEqual(saved, ["cats_step100.safetensors"]);
    assert.ok(urls.includes("/lora/r6?file=cats_step100.safetensors"));
    off();
  } finally { globalThis.fetch = realFetch; }
});

test("Results: a run shows the time per step and the time spent in all, restarts included", async () => {
  const f = fake();
  let n = 0;
  f.api.trainRun = async (since) => {
    n += 1;
    if (n < 3) return { run: "", state: "idle", loss: [], loss_from: 0 };
    return { run: "r7", state: "training", step: 4, total: 100, elapsed: 125, message: "", loss: [0.9, 0.8, 0.7, 0.6].slice(since), loss_from: since, times: [null, 3.5, 3.3, 3.4].slice(since), step_seconds: 3.4, last_step_seconds: 3.4 };
  };
  const { button, off } = await open(f);
  button.click();
  await wait(60);
  click("Skip to training").click();
  await wait(20);
  click("Start training").click();
  await wait(1800);
  tab("Results").click();
  await wait(40);
  const t = text();
  assert.ok(t.includes("3.4 s per step on average"), "the per-step time");
  assert.ok(t.includes("2 min 5 s spent in all, the restarts included"), "the total");
  assert.ok(t.includes("about 5 min 26 s left"), "the estimate comes from the step time");
  assert.equal(document.querySelectorAll(".cx-chart").length, 2, "a graph of the step times beside the loss");
  off();
});

test("Data step: results whose rating may be from another video are marked, not hidden", async () => {
  const f = fake();
  f.api.trainRated = async () => ({ results: [{ label: "liked", media: { type: "output", subfolder: "", filename: "x.mp4" }, caption: "a red fox", doubtful: true }, { label: "liked", media: { type: "output", subfolder: "", filename: "y.mp4" }, caption: "a cat" }], counts: { liked: 2 }, doubtful: 1 });
  const { button, off } = await open(f);
  button.click();
  await wait(60);
  const rated = [...document.querySelectorAll("button")].find((b) => /rated|results/i.test(b.textContent) && /Add|Use|Open/i.test(b.textContent));
  if (rated) rated.click();
  await wait(60);
  assert.ok(text().includes('1 marked "old rating?"'), "the count is shown, and nothing is hidden");
  off();
});
