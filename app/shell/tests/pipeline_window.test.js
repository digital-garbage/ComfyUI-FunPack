// The models and pipeline window.
//
// Driven through the DOM it actually builds, against a fake server that records
// what it was asked. The two things worth proving are that a value edit does
// not reach the server until Save, and that an edit the server refuses leaves
// the window showing what the server still holds -- the alternative is a screen
// that has quietly disagreed with the pipeline since some earlier click.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom, fire } from "../../composer/tests/_dom.js";

let openWindow, groupsOf, resetWindow;
test.before(async () => {
  setupDom();
  await import("../../composer/composer.js");
  ({ open: openWindow, groupsOf, _reset: resetWindow } = await import("../pipeline_window.js"));
});
// There is one pipeline and so one window onto it; a test that leaves one open
// would otherwise hand it to the next test instead of opening a fresh one.
test.beforeEach(() => resetWindow());
test.after(() => teardownDom());

// --- the fake server -------------------------------------------------------

const SLOTS = () => [
  { id: "model", group: "Loaders", node: "Loader", inputs: { model_name: "a.safetensors" } },
  { id: "sampler", group: "Sampling", node: "Sampler",
    inputs: { model: ["model", 0], steps: 20, sampler_name: "euler" } },
  { id: "save", node: "Save", inputs: { images: ["sampler", 0] } },
];

const DESCRIPTIONS = {
  Loader: {
    node: "Loader", title: "Model loader", widgets: [
      { name: "model_name", type: "COMBO", choices: ["a.safetensors", "b.safetensors"] },
    ], sockets: [], outputs: ["MODEL"], output_names: ["model"],
  },
  Sampler: {
    node: "Sampler", title: "Sampler", widgets: [
      { name: "steps", type: "INT", default: 20, min: 1, max: 100 },
      { name: "sampler_name", type: "COMBO", choices: ["euler", "dpmpp_2m"] },
    ], sockets: [{ name: "model", type: "MODEL" }], outputs: ["LATENT"],
  },
  SamplerV2: {
    node: "SamplerV2", title: "Sampler II", sockets: [], outputs: ["LATENT"],
    widgets: [{ name: "steps", type: "INT", default: 8, min: 1, max: 100 }],
  },
  Toggler: {
    node: "Toggler", title: "Toggler", sockets: [], outputs: [],
    widgets: [{ name: "fp16_accumulation", type: "BOOLEAN", default: false,
                tooltip: "Faster fp16 matmuls." }],
  },
  Save: {
    node: "Save", title: "Save image", widgets: [], outputs: [],
    sockets: [{ name: "images", type: "IMAGE" }],
  },
  PrimitiveInt: {
    node: "PrimitiveInt", title: "Primitive Int", sockets: [], outputs: ["INT"],
    output_names: ["value"], widgets: [{ name: "value", type: "INT", default: 20, min: 1, max: 100 }],
  },
};

/** A server that answers, and remembers every request it was given. */
function server({ refuse = null, slots = SLOTS() } = {}) {
  const calls = [];
  let held = slots;
  return {
    calls,
    get held() { return held; },
    load: async () => ({ slots: held, incomplete: [], queueable: true }),
    describe: async (classes) => {
      calls.push({ describe: [...classes] });
      return Object.fromEntries(classes.map((c) => [c, DESCRIPTIONS[c] ?? null]));
    },
    search: async (query) => ({
      nodes: Object.values(DESCRIPTIONS)
        .filter((d) => d.node.toLowerCase().includes((query || "").toLowerCase()))
        .map((d) => ({ node: d.node, title: d.title, category: "test", outputs: d.outputs })),
      total: 3,
    }),
    check: async (body) => {
      calls.push({ check: body });
      // An empty `slots` on a refusal, because that is what the real one sends
      // when the request itself was malformed -- and a window that trusted it
      // would replace the pipeline on screen with nothing at all.
      if (refuse) return { slots: [], refused: [refuse], incomplete: [], queueable: false };
      if (body.action === "remove") held = held.filter((s) => s.id !== body.slot);
      // Inputs cleared, as the real one does: a different node has different
      // inputs, and keeping the ones whose names happen to match is how a value
      // ends up meaning something else.
      else if (body.action === "replace") {
        held = (body.slots || held).map((s) =>
          (s.id === body.slot ? { ...s, node: body.node, inputs: {} } : s));
      } else if (body.action === "wire") {
        held = held.map((s) => (s.id === body.slot
          ? { ...s, inputs: { ...(s.inputs || {}), [body.input]: [body.from_slot, body.from_output] } }
          : s));
      } else if (body.action === "unwire") {
        held = held.map((s) => {
          if (s.id !== body.slot) return s;
          const inputs = { ...(s.inputs || {}) };
          delete inputs[body.input];
          return { ...s, inputs };
        });
      } else if (body.slots) held = body.slots;
      return { slots: held, refused: [], incomplete: [], queueable: true };
    },
  };
}

async function opened(overrides = {}) {
  const api = server(overrides.server || {});
  const win = openWindow({ ...api, ...(overrides.props || {}) });
  await win.ready;
  return { win, api };
}

// --- reading the DOM the window built --------------------------------------

const rows = (win) => [...win.node.querySelectorAll(".cx-settings-row")];

function rowFor(win, label) {
  const found = rows(win).find(
    (r) => r.querySelector(".cx-settings-label")?.textContent === label);
  assert.ok(found, `no settings row labelled "${label}" (have: ${
    rows(win).map((r) => r.querySelector(".cx-settings-label")?.textContent).join(", ")})`);
  return found;
}

const hintOf = (row) => row.querySelector(".cx-hint")?.textContent ?? "";
const cardLabels = (win) =>
  [...win.node.querySelectorAll(".cx-card-title")].map((n) => n.textContent);
const bannerTexts = (win) =>
  [...win.node.querySelectorAll(".cx-banner-text")].map((n) => n.textContent);

const click = (node) => node.dispatchEvent(
  new window.window.MouseEvent("click", { bubbles: true, cancelable: true }));

function button(win, label) {
  const found = [...win.node.querySelectorAll("button")].find(
    (b) => b.textContent.trim() === label);
  assert.ok(found, `no button labelled "${label}"`);
  return found;
}

// --- grouping --------------------------------------------------------------

test("groups come from the slots, in the order they first appear", () => {
  const { order } = groupsOf(SLOTS());
  assert.deepEqual(order, ["Loaders", "Sampling", "Other"]);
});

test("a group the user made is offered even with nothing in it yet", () => {
  const { order, byGroup } = groupsOf(SLOTS(), ["Upscaling"]);
  assert.deepEqual(order, ["Loaders", "Sampling", "Other", "Upscaling"]);
  assert.deepEqual(byGroup.get("Upscaling"), []);
});

test("the window opens on a card per group, counting what is in each", async () => {
  const { win } = await opened();
  assert.deepEqual(cardLabels(win), ["Loaders", "Sampling", "Other"]);
  const hints = [...win.node.querySelectorAll(".cx-card .cx-hint")].map((n) => n.textContent);
  assert.deepEqual(hints, ["1 node", "1 node", "1 node"]);
  win.close();
});

// --- presets -----------------------------------------------------------------

test("with no presets offered, no Load preset button shows up", async () => {
  const { win } = await opened();
  const buttons = [...win.node.querySelectorAll("button")].map((b) => b.textContent);
  assert.ok(!buttons.includes("Load preset…"));
  win.close();
});

test("picking an offered preset replaces the pipeline with it", async () => {
  const preset = { id: "p1", title: "A Test Preset",
    slots: [{ id: "only", group: "Loaders", node: "Loader", inputs: {} }] };
  const { win, api } = await opened({ props: { presets: async () => [preset] } });

  const load = [...win.node.querySelectorAll("button")].find((b) => b.textContent === "Load preset…");
  assert.ok(load, "no Load preset button, though presets() was offered");
  click(load);
  await new Promise((resolve) => setTimeout(resolve, 0));

  const card = [...document.querySelectorAll(".cx-card-title")].find((n) => n.textContent === "A Test Preset");
  assert.ok(card, "the offered preset was not listed");
  click(card.closest(".cx-card"));
  await Promise.resolve();
  await Promise.resolve();

  assert.deepEqual(api.held, preset.slots);
  assert.deepEqual(cardLabels(win), ["Loaders"]);
  win.close();
});

test("a preset that fails to build is refused, not silently shown as loaded", async () => {
  const preset = { id: "p1", title: "Broken Preset", slots: [{ id: "x", node: "Nope", inputs: {} }] };
  const { win } = await opened({
    server: { refuse: "there is no node called 'Nope' installed" },
    props: { presets: async () => [preset] },
  });

  const load = [...win.node.querySelectorAll("button")].find((b) => b.textContent === "Load preset…");
  click(load);
  await new Promise((resolve) => setTimeout(resolve, 0));
  const card = [...document.querySelectorAll(".cx-card-title")].find((n) => n.textContent === "Broken Preset");
  click(card.closest(".cx-card"));
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.deepEqual(bannerTexts(win), ["there is no node called 'Nope' installed"]);
  // Still the original pipeline -- SLOTS()'s three groups, not the preset's one.
  assert.deepEqual(cardLabels(win), ["Loaders", "Sampling", "Other"]);
  win.close();
});

// --- parameters ------------------------------------------------------------

test("a node's parameters show the value the slot holds, not the node's default", async () => {
  const { win } = await opened();
  win.enter("Sampling");
  // The slot says 20 and the widget's default is also 20, so the test uses the
  // combo: the slot picked euler and the choices start there too -- make the
  // slot disagree with both to see which one is being read.
  win.close();

  const custom = server({ slots: [
    { id: "sampler", group: "Sampling", node: "Sampler",
      inputs: { steps: 35, sampler_name: "dpmpp_2m" } },
  ] });
  const second = openWindow(custom);
  await second.ready;
  second.enter("Sampling");
  assert.equal(rowFor(second, "Steps").querySelector("input").value, "35");
  assert.equal(rowFor(second, "Sampler name").querySelector("select").value, "dpmpp_2m");
  second.close();
});

test("a wired input is reported as wired, not offered as a box to type in", async () => {
  const { win } = await opened();
  win.enter("Sampling");
  const row = rowFor(win, "Model");
  assert.match(hintOf(row), /fed by model/);
  assert.match(hintOf(row), /Model loader/, "the source's own title was not shown");
  // The failure this guards: `["model", 0]` rendered into a text control, where
  // it reads as the string "model,0" and saves as one.
  assert.equal(row.querySelector("input, select, textarea"), null);
  win.close();
});

test("a slot pointing at a node nobody has installed says so where its settings would be", async () => {
  const custom = server({ slots: [{ id: "x", group: "Loaders", node: "FromSomePack", inputs: {} }] });
  const win = openWindow(custom);
  await win.ready;
  win.enter("Loaders");
  assert.match(bannerTexts(win).join(" "), /FromSomePack is not installed here/);
  win.close();
});

// --- drafting --------------------------------------------------------------

test("editing a value changes nothing on the server until Save", async () => {
  const { win, api } = await opened();
  win.enter("Sampling");
  const before = api.calls.filter((c) => c.check).length;

  const steps = rowFor(win, "Steps").querySelector("input");
  steps.value = "35";
  fire(steps, "blur");

  assert.equal(win.pending, 1);
  assert.equal(api.calls.filter((c) => c.check).length, before,
    "a value edit reached the server before Save");
  assert.equal(api.held.find((s) => s.id === "sampler").inputs.steps, 20);

  click(button(win, "Save"));
  await new Promise(setImmediate);

  assert.equal(api.held.find((s) => s.id === "sampler").inputs.steps, 35);
  assert.equal(win.editing, null, "Save did not return to the group index");
  win.close();
});

test("Save carries the whole pipeline, not only the group being edited", async () => {
  const { win, api } = await opened();
  win.enter("Sampling");
  const steps = rowFor(win, "Steps").querySelector("input");
  steps.value = "35";
  fire(steps, "blur");
  click(button(win, "Save"));
  await new Promise(setImmediate);

  const sent = api.calls.filter((c) => c.check).at(-1).check.slots;
  assert.deepEqual(sent.map((s) => s.id), ["model", "sampler", "save"]);
  win.close();
});

test("Cancel puts an edit back", async () => {
  const { win, api } = await opened();
  win.enter("Sampling");
  const steps = rowFor(win, "Steps").querySelector("input");
  steps.value = "35";
  fire(steps, "blur");
  assert.equal(win.pending, 1);

  click(button(win, "Cancel"));
  assert.equal(win.pending, 0);
  assert.equal(api.held.find((s) => s.id === "sampler").inputs.steps, 20);

  win.enter("Sampling");
  assert.equal(rowFor(win, "Steps").querySelector("input").value, "20");
  win.close();
});

// --- structure -------------------------------------------------------------

test("Remove asks the server to take the slot out, so the rewiring is its answer", async () => {
  const { win, api } = await opened();
  win.enter("Sampling");
  click(button(win, "Remove"));
  await new Promise(setImmediate);

  const last = api.calls.filter((c) => c.check).at(-1).check;
  assert.equal(last.action, "remove");
  assert.equal(last.slot, "sampler");
  // The pipeline it was asked to remove FROM travels with it: without the
  // slots, the server applies the removal to its own defaults instead.
  assert.deepEqual(last.slots.map((s) => s.id), ["model", "sampler", "save"]);
  assert.deepEqual(win.slots.map((s) => s.id), ["model", "save"]);
  win.close();
});

test("a refused edit leaves the pipeline as it was, and says why", async () => {
  const { win, api } = await opened({ server: { refuse: "removing 'sampler' is ambiguous" } });
  win.enter("Sampling");
  click(button(win, "Remove"));
  await new Promise(setImmediate);

  assert.deepEqual(win.slots.map((s) => s.id), ["model", "sampler", "save"],
    "the window kept a change the server refused");
  assert.match(bannerTexts(win).join(" "), /ambiguous/);
  win.close();
});

test("a moved node lands under the group it was moved to", async () => {
  const { win } = await opened();
  win.enter("Sampling");
  const select = rowFor(win, "Group").querySelector("select");
  select.value = "Loaders";
  fire(select, "change");
  await new Promise(setImmediate);

  assert.equal(win.slots.find((s) => s.id === "sampler").group, "Loaders");
  win.close();
});

// --- what the rest of the app is told --------------------------------------

test("the pipeline the window settled on is handed out, so a run uses it", async () => {
  let handed = null;
  const { win } = await opened({ props: { onApply: (next) => { handed = next; } } });
  win.enter("Sampling");
  const steps = rowFor(win, "Steps").querySelector("input");
  steps.value = "35";
  fire(steps, "blur");
  click(button(win, "Save"));
  await new Promise(setImmediate);

  assert.ok(handed, "nothing was handed out, so Generate would still run the defaults");
  assert.equal(handed.find((s) => s.id === "sampler").inputs.steps, 35);
  win.close();
});

test("a node is described once, however many slots point at it", async () => {
  const custom = server({ slots: [
    { id: "a", group: "Preparation", node: "Loader", inputs: {} },
    { id: "b", group: "Preparation", node: "Loader", inputs: {} },
  ] });
  const win = openWindow(custom);
  await win.ready;
  const asked = custom.calls.filter((c) => c.describe).flatMap((c) => c.describe);
  assert.deepEqual(asked, ["Loader"]);
  win.close();
});

test("reopening a group re-asks for its node descriptions, so a file added mid-session shows up", async () => {
  // The bug this pins: `nodes` used to cache a class's description for the
  // life of the window, keyed only by class name. A model file dropped onto
  // disk after the window opened was invisible in an already-known node's
  // panel with no way to see it short of reloading the page -- which throws
  // the whole in-progress pipeline edit away with the stale cache.
  let files = ["a.safetensors"];
  const custom = server({ slots: [{ id: "l", group: "Loaders", node: "Loader", inputs: {} }] });
  custom.describe = async (classes) => {
    custom.calls.push({ describe: [...classes] });
    return { Loader: { ...DESCRIPTIONS.Loader, widgets: [
      { name: "model_name", type: "COMBO", choices: [...files] },
    ] } };
  };
  const win = openWindow(custom);
  await win.ready;
  await win.enter("Loaders");
  const optionsOf = () => [...rowFor(win, "Model name").querySelector("select").options]
    .map((o) => o.textContent);
  assert.deepEqual(optionsOf().filter((t) => t !== "— not set —"), ["a.safetensors"]);

  files.push("b.safetensors");
  win.leave();
  await win.enter("Loaders");
  assert.deepEqual(optionsOf().filter((t) => t !== "— not set —"),
    ["a.safetensors", "b.safetensors"],
    "reopening the group did not pick up the file added since it was last open");
  win.close();
});

test("Refresh models re-asks without leaving the group", async () => {
  let files = ["a.safetensors"];
  const custom = server({ slots: [{ id: "l", group: "Loaders", node: "Loader", inputs: {} }] });
  custom.describe = async (classes) => {
    custom.calls.push({ describe: [...classes] });
    return { Loader: { ...DESCRIPTIONS.Loader, widgets: [
      { name: "model_name", type: "COMBO", choices: [...files] },
    ] } };
  };
  const win = openWindow(custom);
  await win.ready;
  await win.enter("Loaders");

  files.push("b.safetensors");
  click(button(win, "Refresh models"));
  await new Promise(setImmediate);

  const optionsOf = [...rowFor(win, "Model name").querySelector("select").options]
    .map((o) => o.textContent);
  assert.deepEqual(optionsOf.filter((t) => t !== "— not set —"), ["a.safetensors", "b.safetensors"]);
  win.close();
});

test("a failed Refresh models says so, instead of looking like there was nothing new", async () => {
  const custom = server({ slots: [{ id: "l", group: "Loaders", node: "Loader", inputs: {} }] });
  // mount() destructures `describe` once, at open time -- reassigning
  // `custom.describe` afterward would not reach it. A stable wrapper that
  // reads a mutable flag is what lets a later part of the test change what
  // happens on the NEXT call.
  let fail = false;
  const base = custom.describe;
  custom.describe = (classes) => {
    if (fail) return Promise.reject(new Error("those nodes could not be described (500)"));
    return base(classes);
  };
  const win = openWindow(custom);
  await win.ready;
  await win.enter("Loaders");

  fail = true;
  click(button(win, "Refresh models"));
  await new Promise(setImmediate);

  assert.match(bannerTexts(win).join(" "), /could not be described/);
  win.close();
});

test("an older learn() answering after a newer one does not put its stale choices back", async () => {
  // Two overlapping force-refreshes (e.g. two quick clicks of Refresh models,
  // or leave+enter fast enough that the first request is still in flight):
  // whichever answer comes back LAST must not be allowed to be the OLDER one.
  const custom = server({ slots: [{ id: "l", group: "Loaders", node: "Loader", inputs: {} }] });
  let deferred = false;
  let resolvers = [];
  const base = custom.describe;
  custom.describe = (classes) => {
    if (!deferred) return base(classes);
    custom.calls.push({ describe: [...classes] });
    return new Promise((resolve) => resolvers.push({ resolve, classes }));
  };
  const win = openWindow(custom);
  await win.ready;   // past the initial, non-force learn() before describe() defers

  deferred = true;
  const first = win.enter("Loaders");   // in flight
  const second = win.enter("Loaders");  // also in flight, started after
  // Answer the newer request first, then the older one.
  resolvers[1].resolve({ Loader: { ...DESCRIPTIONS.Loader, widgets: [
    { name: "model_name", type: "COMBO", choices: ["fresh.safetensors"] },
  ] } });
  await second;
  resolvers[0].resolve({ Loader: { ...DESCRIPTIONS.Loader, widgets: [
    { name: "model_name", type: "COMBO", choices: ["stale.safetensors"] },
  ] } });
  await first;

  const options = [...rowFor(win, "Model name").querySelector("select").options]
    .map((o) => o.textContent);
  assert.ok(options.includes("fresh.safetensors"), "the newer answer should have won");
  assert.ok(!options.includes("stale.safetensors"), "the older, late-arriving answer overwrote it");
  win.close();
});

test("a superseded rescan's late failure does not overwrite a newer rescan's success with a false error", async () => {
  const custom = server({ slots: [{ id: "l", group: "Loaders", node: "Loader", inputs: {} }] });
  let deferred = false;
  let handlers = [];
  const base = custom.describe;
  custom.describe = (classes) => {
    if (!deferred) return base(classes);
    return new Promise((resolve, reject) => handlers.push({ resolve, reject }));
  };
  const win = openWindow(custom);
  await win.ready;

  deferred = true;
  const first = win.enter("Loaders");   // will fail, but only after the second has already won
  const second = win.enter("Loaders");  // succeeds
  handlers[1].resolve({ Loader: DESCRIPTIONS.Loader });
  await second;
  handlers[0].reject(new Error("stale network failure from the OLD request"));
  await first;

  assert.deepEqual(bannerTexts(win), [],
    "a rejection from a superseded request must not overwrite the newer success with an error banner");
  win.close();
});

test("a successful rescan clears a refusal banner an earlier failed one left behind", async () => {
  const custom = server({ slots: [{ id: "l", group: "Loaders", node: "Loader", inputs: {} }] });
  let fail = false;
  const base = custom.describe;
  custom.describe = (classes) => {
    if (fail) return Promise.reject(new Error("those nodes could not be described (500)"));
    return base(classes);
  };
  const win = openWindow(custom);
  await win.ready;
  await win.enter("Loaders");

  fail = true;
  click(button(win, "Refresh models"));
  await new Promise(setImmediate);
  assert.match(bannerTexts(win).join(" "), /could not be described/);

  fail = false;
  click(button(win, "Refresh models"));
  await new Promise(setImmediate);
  assert.deepEqual(bannerTexts(win), [], "the earlier failure's banner should be gone after a success");
  win.close();
});

test("a scan failure does not follow the user to the index or into a different group", async () => {
  const custom = server({ slots: [
    { id: "l", group: "Loaders", node: "Loader", inputs: {} },
    { id: "s", group: "Sampling", node: "Sampler",
      inputs: { model: ["l", 0], steps: 20, sampler_name: "euler" } },
  ] });
  let fail = false;
  const base = custom.describe;
  custom.describe = (classes) => {
    if (fail) return Promise.reject(new Error("could not be described (500)"));
    return base(classes);
  };
  const win = openWindow(custom);
  await win.ready;

  fail = true;
  await win.enter("Loaders");
  assert.match(bannerTexts(win).join(" "), /could not be described/, "sanity: the failure shows in Loaders");

  win.leave();
  assert.deepEqual(bannerTexts(win), [], "Loaders' scan failure bled into the index");

  fail = false;
  await win.enter("Sampling");
  assert.deepEqual(bannerTexts(win), [], "Loaders' scan failure bled into an unrelated group");
  win.close();
});

test("leaving a group before its rescan answers does not let a late failure land on the index", async () => {
  const custom = server({ slots: [{ id: "l", group: "Loaders", node: "Loader", inputs: {} }] });
  let deferred = false;
  let handlers = [];
  const base = custom.describe;
  custom.describe = (classes) => {
    if (!deferred) return base(classes);
    return new Promise((resolve, reject) => handlers.push({ resolve, reject }));
  };
  const win = openWindow(custom);
  await win.ready;

  deferred = true;
  const entering = win.enter("Loaders");   // rescan() now in flight, unresolved
  win.leave();                             // back out before it answers
  assert.deepEqual(bannerTexts(win), [], "sanity: nothing shown right after leaving");

  handlers[0].reject(new Error("stale network failure from the left group"));
  await entering;

  assert.deepEqual(bannerTexts(win), [],
    "a rescan answering after its group was left must not paint the index with its failure");
  win.close();
});

test("leaving a group before its commit() answers does not let a late refusal land on the index", async () => {
  // The commit()-side counterpart of the rescan race above: an edit (Remove,
  // Save, wire...) started in one group can still answer after the user has
  // backed out to the index. Its DATA must still apply -- an edit that
  // happened does not un-happen -- but its refusal banner belongs to the
  // group that made it, not to whatever the user is looking at by the time
  // the answer arrives.
  const custom = server({ refuse: "removing 'l' is ambiguous", slots: [
    { id: "l", group: "Loaders", node: "Loader", inputs: {} },
  ] });
  let deferred = false;
  let handlers = [];
  const base = custom.check;
  custom.check = (body) => {
    if (!deferred) return base(body);
    return new Promise((resolve) => handlers.push({ resolve, body }));
  };
  const win = openWindow(custom);
  await win.ready;
  await win.enter("Loaders");

  deferred = true;
  click(button(win, "Remove"));       // commit() now in flight, unresolved
  win.leave();                        // back out before it answers
  assert.deepEqual(bannerTexts(win), [], "sanity: nothing shown right after leaving");

  handlers[0].resolve({ slots: [], refused: ["removing 'l' is ambiguous"], incomplete: [], queueable: false });
  await new Promise(setImmediate);

  assert.deepEqual(bannerTexts(win), [],
    "a commit() refusal answering after its group was left must not paint the index with it");
  win.close();
});

test("a load() that comes back refused still shows why, not an empty pipeline", async () => {
  // refresh()'s `load` can be backed by a plain stateless read (always
  // refused: []) or, when re-validating an already-saved pipeline on open,
  // by the same check() a structural edit uses -- which CAN refuse (a real
  // network failure/500 from the server mid-request). Silently dropping
  // that field would replace the screen with an unexplained empty pipeline.
  const win = openWindow({
    load: async () => ({ slots: [], refused: ["the pipeline could not be read (500)"], incomplete: [] }),
    describe: async () => ({}), check: async () => ({}), search: async () => ({ nodes: [], total: 0 }),
  });
  await win.ready;
  assert.match(bannerTexts(win).join(" "), /could not be read/);
  win.close();
});

test("a pipeline that cannot be read says so instead of showing an empty window", async () => {
  const win = openWindow({
    load: async () => { throw new Error("the pipeline could not be read (500)"); },
    describe: async () => ({}), check: async () => ({}), search: async () => ({ nodes: [], total: 0 }),
  });
  await win.ready;
  assert.match(win.node.textContent, /could not be read/);
  win.close();
});

test("a control that draws its own label is not given a second one", async () => {
  // A checkbox row carries its label and hint itself. Wrapped in a settings row
  // it printed both twice, one above the other, and the pane read as two
  // settings with the same name.
  const custom = server({ slots: [{ id: "t", group: "Loaders", node: "Toggler", inputs: {} }] });
  const win = openWindow(custom);
  await win.ready;
  win.enter("Loaders");

  const shown = win.node.innerText ?? win.node.textContent;
  const times = shown.split("FP16 accumulation").length - 1;
  assert.equal(times, 1, `the label appeared ${times} times`);
  win.close();
});

// --- a choice nobody has made ---------------------------------------------

test("an unset dropdown shows as unset, not as the first thing in the list", async () => {
  // v4 recorded this one exactly: seeding a combo from its first option "would
  // pre-select an arbitrary model file and make an unconfigured loader look
  // configured". Worse than looking wrong, it is unfixable by the obvious
  // click: picking the entry the box already shows fires no change event, so
  // the one action that looks like the fix does nothing at all.
  const custom = server({ slots: [{ id: "l", group: "Loaders", node: "Loader", inputs: {} }] });
  const win = openWindow(custom);
  await win.ready;
  win.enter("Loaders");

  const select = rowFor(win, "Model name").querySelector("select");
  assert.notEqual(select.value, "a.safetensors",
    "an unset picker was showing a file as though it had been chosen");
  assert.equal(select.selectedOptions[0].textContent, "— not set —");
  win.close();
});

test("choosing a value for an unset dropdown records it", async () => {
  const custom = server({ slots: [{ id: "l", group: "Loaders", node: "Loader", inputs: {} }] });
  const win = openWindow(custom);
  await win.ready;
  win.enter("Loaders");

  const select = rowFor(win, "Model name").querySelector("select");
  select.value = "b.safetensors";
  fire(select, "change");
  assert.equal(win.pending, 1);

  click(button(win, "Save"));
  await new Promise(setImmediate);
  assert.equal(custom.held.find((s) => s.id === "l").inputs.model_name, "b.safetensors");
  win.close();
});

test("a dropdown the pipeline already filled is not offered an unset entry", async () => {
  const { win } = await opened();
  win.enter("Loaders");
  const select = rowFor(win, "Model name").querySelector("select");
  assert.equal(select.value, "a.safetensors");
  assert.equal([...select.options].some((o) => o.textContent === "— not set —"), false);
  win.close();
});

test("the footer says an edit is waiting as soon as it is made", async () => {
  // It reads from the draft, so it went stale the moment editing stopped
  // redrawing the body -- and "No changes to save" over an unsaved change is
  // the one thing that bar must never say.
  const { win } = await opened();
  win.enter("Sampling");
  const foot = win.node.querySelector(".cx-modal-foot");
  assert.match(foot.textContent, /No changes to save/);

  const steps = rowFor(win, "Steps").querySelector("input");
  steps.value = "35";
  fire(steps, "blur");

  assert.match(foot.textContent, /1 node edited/);
  win.close();
});

test("editing does not rebuild the control being typed into", async () => {
  // Rebuilding the pane on every keystroke takes the cursor out of the box.
  const { win } = await opened();
  win.enter("Sampling");
  const before = rowFor(win, "Steps").querySelector("input");
  before.value = "35";
  fire(before, "blur");
  assert.equal(rowFor(win, "Steps").querySelector("input"), before,
    "the control was replaced while it was being edited");
  win.close();
});

test("an unfed socket offers to be wired, not just named", async () => {
  const custom = server({ slots: [{ id: "s", group: "Sampling", node: "Sampler", inputs: {} }] });
  const win = openWindow(custom);
  await win.ready;
  win.enter("Sampling");
  assert.match(hintOf(rowFor(win, "Model")), /nothing feeds it/);
  assert.doesNotMatch(hintOf(rowFor(win, "Model")), /cannot wire/);
  assert.ok(rowFor(win, "Model").querySelector("button")?.textContent.includes("Wire"));
  win.close();
});

// --- wiring ------------------------------------------------------------

test("wiring an input picks from every OTHER slot's compatible outputs", async () => {
  const { win } = await opened();
  const sources = win._sourcesFor("sampler", "model");
  assert.deepEqual(sources.map((s) => s.label), ["model · model"]);
  win.close();
});

test("a slot never offers itself as its own source", async () => {
  // Wiring the loader's own (nonexistent, in this fixture) socket to its own
  // output is always a cycle -- filtered before the server ever has to say so.
  const custom = server({ slots: [
    { id: "a", group: "G", node: "Sampler", inputs: {} },
  ] });
  const win = openWindow(custom);
  await win.ready;
  assert.deepEqual(win._sourcesFor("a", "model"), []);
  win.close();
});

test("only type-compatible outputs are offered", async () => {
  // Save's own "images" socket wants IMAGE; nothing in this fixture produces
  // one, so it must not offer the MODEL or LATENT outputs that do exist.
  const { win } = await opened();
  assert.deepEqual(win._sourcesFor("save", "images"), []);
  win.close();
});

test("wiring sends the chosen source to the server and shows the result", async () => {
  const { win, api } = await opened({ server: {
    slots: [
      { id: "model", group: "Loaders", node: "Loader", inputs: { model_name: "a.safetensors" } },
      { id: "sampler", group: "Sampling", node: "Sampler", inputs: { steps: 20, sampler_name: "euler" } },
    ],
  } });
  await win._wire("sampler", "model", "model", 0);
  const sent = api.calls.at(-1).check;
  assert.equal(sent.action, "wire");
  assert.deepEqual(
    { slot: sent.slot, input: sent.input, from_slot: sent.from_slot, from_output: sent.from_output },
    { slot: "sampler", input: "model", from_slot: "model", from_output: 0 });

  win.enter("Sampling");
  assert.match(hintOf(rowFor(win, "Model")), /fed by model/);
  win.close();
});

test("unwiring asks the server and the row goes back to unfed", async () => {
  const { win, api } = await opened();
  await win._unwire("sampler", "model");
  const sent = api.calls.at(-1).check;
  assert.deepEqual({ action: sent.action, slot: sent.slot, input: sent.input },
                    { action: "unwire", slot: "sampler", input: "model" });

  win.enter("Sampling");
  assert.match(hintOf(rowFor(win, "Model")), /nothing feeds it/);
  win.close();
});

test("the Wire button becomes Change once an input is fed", async () => {
  const { win } = await opened();
  win.enter("Sampling");
  assert.ok(button(win, "Change"), "a wired socket did not offer Change");
  assert.throws(() => button(win, "Wire…"));
  win.close();
});

test("an unfed socket has no Unwire button to press", async () => {
  const custom = server({ slots: [{ id: "s", group: "Sampling", node: "Sampler", inputs: {} }] });
  const win = openWindow(custom);
  await win.ready;
  win.enter("Sampling");
  assert.throws(() => button(win, "Unwire"));
  win.close();
});

// --- a node's own outputs, shown on its own panel ---------------------------
//
// Wiring is always asked for by the CONSUMING input, never from the output's
// own side -- but an output that is never shown anywhere on its own node's
// panel is a real blind spot: open the node that PRODUCES something and
// there was nothing here saying what it produces, or whether anything reads
// it yet. Missed by two full review rounds because every one of them
// reasoned from "does wiring an input work", never "can you see what a node
// makes just by looking at it".

test("a node's own outputs are listed on its panel, named and typed", async () => {
  const { win } = await opened();
  win.enter("Loaders");
  const rows = [...win.node.querySelectorAll(".cx-settings-row")]
    .filter((r) => r.querySelector(".cx-settings-label")?.textContent === "model");
  // "model" is both an INPUT widget on other nodes and this node's OWN
  // output name -- the output row is the one with no input control, only
  // the read-only hint.
  const outputRow = rows.find((r) => !r.querySelector("input, select, textarea, button"));
  assert.ok(outputRow, "the Loader's own 'model' output is not shown at all");
  assert.match(outputRow.querySelector(".cx-hint")?.textContent ?? "", /MODEL/);
  win.close();
});

test("an output already consumed says who reads it", async () => {
  const { win } = await opened();
  win.enter("Loaders");
  const rows = [...win.node.querySelectorAll(".cx-settings-row")]
    .filter((r) => r.querySelector(".cx-settings-label")?.textContent === "model");
  const outputRow = rows.find((r) => !r.querySelector("input, select, textarea, button"));
  assert.match(outputRow.querySelector(".cx-hint")?.textContent ?? "", /feeds sampler\.model/);
  win.close();
});

test("an output nothing reads yet says so", async () => {
  const custom = server({ slots: [{ id: "s", group: "Sampling", node: "Sampler", inputs: {} }] });
  const win = openWindow(custom);
  await win.ready;
  win.enter("Sampling");
  const outputRow = rowFor(win, "LATENT");
  assert.match(outputRow.querySelector(".cx-hint")?.textContent ?? "", /nothing reads this yet/);
  win.close();
});

test("a node with no outputs shows no Outputs section", async () => {
  const { win } = await opened();
  win.enter("Other");                            // "save" has no group of its own
  assert.equal([...win.node.querySelectorAll(".cx-eyebrow")]
    .some((el) => el.textContent === "Outputs"), false);
  win.close();
});

// --- wiring a WIDGET input -- "linked inputs" -------------------------------
//
// A widget-typed input (a number, a string, a combo) can hold a link too --
// that IS what a "linked input" is, several nodes' widgets fed from one
// Primitive node's output. Found by adversarial review: the sockets loop had
// all of this and the widgets loop had none of it, so the one thing this
// feature was built for could not be done through the window at all, and a
// widget that already held a link (from an imported project, or wired by
// hand through the server) rendered as an ordinary editable box holding the
// WRONG value -- the widget's own default -- with nothing saying it was fed
// from elsewhere.

test("an unlinked widget offers a way to wire it, alongside its normal editor", async () => {
  const { win } = await opened();
  win.enter("Sampling");
  const row = rowFor(win, "Steps");
  assert.ok(row.querySelector("input[type=number], input:not([type])"),
    "the normal value editor is gone");
  assert.ok([...row.querySelectorAll("button")].some((b) =>
    b.getAttribute("aria-label") === "Wire Steps from another node"), "no way to wire it");
  win.close();
});

test("a widget input that already holds a link is shown as wired, not as a box with the wrong default", async () => {
  const custom = server({ slots: [
    { id: "shared", group: "Loaders", node: "PrimitiveInt", inputs: { value: 512 } },
    { id: "s", group: "Sampling", node: "Sampler", inputs: { steps: ["shared", 0] } },
  ] });
  const win = openWindow(custom);
  await win.ready;
  win.enter("Sampling");
  const row = rowFor(win, "Steps");
  assert.match(hintOf(row), /fed by shared/);
  assert.match(hintOf(row), /Primitive Int/);
  assert.equal(row.querySelector("input[type=number]"), null,
    "a linked widget still offered a box holding its own default, not the real source");
  assert.ok(button(win, "Change"));
  assert.ok(button(win, "Unwire"));
  win.close();
});

test("wiring a widget from a Primitive node sends the wire and shows the result", async () => {
  const { win, api } = await opened({ server: {
    slots: [
      { id: "shared", group: "Loaders", node: "PrimitiveInt", inputs: { value: 512 } },
      { id: "s", group: "Sampling", node: "Sampler", inputs: { sampler_name: "euler" } },
    ],
  } });
  await win._wire("s", "steps", "shared", 0);
  const sent = api.calls.at(-1).check;
  assert.deepEqual(
    { action: sent.action, slot: sent.slot, input: sent.input, from_slot: sent.from_slot, from_output: sent.from_output },
    { action: "wire", slot: "s", input: "steps", from_slot: "shared", from_output: 0 });

  win.enter("Sampling");
  assert.match(hintOf(rowFor(win, "Steps")), /fed by shared/);
  win.close();
});

test("unwiring a widget puts its normal editor back", async () => {
  const custom = server({ slots: [
    { id: "shared", group: "Loaders", node: "PrimitiveInt", inputs: { value: 512 } },
    { id: "s", group: "Sampling", node: "Sampler", inputs: { steps: ["shared", 0] } },
  ] });
  const win = openWindow(custom);
  await win.ready;
  await win._unwire("s", "steps");

  win.enter("Sampling");
  const row = rowFor(win, "Steps");
  assert.match(hintOf(row), /^$/);
  assert.ok(row.querySelector("input[type=number], input:not([type])"),
    "unwiring did not bring the normal editor back");
  win.close();
});

test("a widget picker offers a real Primitive source, filtered by type", async () => {
  const custom = server({ slots: [
    { id: "shared", group: "Loaders", node: "PrimitiveInt", inputs: { value: 512 } },
    { id: "s", group: "Sampling", node: "Sampler", inputs: {} },
  ] });
  const win = openWindow(custom);
  await win.ready;
  const sources = win._sourcesFor("s", "steps");
  assert.deepEqual(sources.map((s) => s.label), ["shared · value"]);
  win.close();
});

test("wiring an input that already has a drafted, unsaved value does not let the stale value win on Save", async () => {
  // Found by adversarial review, reproduced live: edit steps (drafted, not
  // sent), then wire steps to a real source (a STRUCTURE change, committed
  // immediately) -- then Save. saveDraft() spreads the draft OVER the slot's
  // current inputs, and until this was fixed the stale drafted "35" silently
  // overwrote the link the user had just watched get confirmed on screen.
  const { win, api } = await opened({ server: {
    slots: [
      { id: "shared", group: "Loaders", node: "PrimitiveInt", inputs: { value: 512 } },
      { id: "s", group: "Sampling", node: "Sampler", inputs: { sampler_name: "euler" } },
    ],
  } });
  win.enter("Sampling");
  const steps = rowFor(win, "Steps").querySelector("input");
  steps.value = "35";
  fire(steps, "blur");
  assert.equal(win.pending, 1, "the edit was not drafted");

  await win._wire("s", "steps", "shared", 0);
  assert.equal(win.pending, 0, "the stale draft for the now-wired input was not dropped");

  click(button(win, "Save"));
  await new Promise(setImmediate);

  assert.deepEqual(api.held.find((s) => s.id === "s").inputs.steps, ["shared", 0],
    "the stale drafted value overwrote the wire on Save");
  win.close();
});

// --- a draft outliving the node it was for --------------------------------

test("changing a node drops the edits that were for the old one", async () => {
  // The values are the OLD node's. A replacement that happens to declare an
  // input of the same name -- steps, seed, cfg and denoise are shared across
  // most samplers -- would take them silently: the server refuses an input the
  // new node does not declare, and a same-named one sails straight through, so
  // nothing anywhere says the value was never chosen for this node.
  const { win, api } = await opened();
  win.enter("Sampling");

  const steps = rowFor(win, "Steps").querySelector("input");
  steps.value = "35";
  fire(steps, "blur");
  assert.equal(win.pending, 1);

  await win._replace("sampler", "SamplerV2");

  assert.equal(win.pending, 0, "an edit for the old node was still pending");
  assert.equal(rowFor(win, "Steps").querySelector("input").value, "8",
    "the new node showed the old node's value as though it had been chosen");
});

test("removing a node drops its edits with it", async () => {
  // Otherwise the footer counts an edit to a node that is no longer there, and
  // says one thing is unsaved when nothing is.
  const { win } = await opened();
  win.enter("Sampling");
  const steps = rowFor(win, "Steps").querySelector("input");
  steps.value = "35";
  fire(steps, "blur");

  click(button(win, "Remove"));
  await new Promise(setImmediate);
  assert.equal(win.pending, 0);
  assert.match(win.node.querySelector(".cx-modal-foot").textContent, /No changes to save/);
  win.close();
});

test("a value edit is not mistaken for a structural one", async () => {
  // The rule is "the node this slot points at changed", not "the slots came
  // back from the server" -- Save itself goes through the same path, and a rule
  // that fired on any answer would throw the edit away as it was being saved.
  const { win, api } = await opened();
  win.enter("Sampling");
  const steps = rowFor(win, "Steps").querySelector("input");
  steps.value = "35";
  fire(steps, "blur");
  click(button(win, "Save"));
  await new Promise(setImmediate);
  assert.equal(api.held.find((s) => s.id === "sampler").inputs.steps, 35);
  win.close();
});

// --- one window, and one edit at a time -----------------------------------

test("pressing the button twice does not stack a second window", async () => {
  // Two windows over one pipeline are two drafts of it: whichever is saved
  // last wins and the other was edited against a pipeline that had moved.
  const api = server();
  const first = openWindow(api);
  await first.ready;
  const second = openWindow(api);

  assert.equal(second, first, "a second window was opened over the first");
  assert.equal(document.querySelectorAll(".cx-modal").length, 1);
  first.close();
  assert.equal(document.querySelectorAll(".cx-modal").length, 0);
});

test("closing one lets the next press open a fresh one", async () => {
  const api = server();
  const first = openWindow(api);
  await first.ready;
  first.close();

  const second = openWindow(api);
  await second.ready;
  assert.notEqual(second, first, "the button stopped working after one close");
  second.close();
});

test("a slow edit answering late does not undo the edit made after it", async () => {
  // Moving a node carries the whole slot list; removing one carries an id. The
  // second answers first, and the first's answer describes a pipeline that no
  // longer exists -- taking it put the removed node back with no refusal and
  // nothing said.
  const api = server();
  const held = [];
  const slow = { ...api, check: (body) => new Promise((resolve) => {
    held.push(() => resolve(api.check(body)));
  }) };

  const win = openWindow(slow);
  await win.ready;                     // the load does not go through check()
  win.enter("Sampling");

  const moving = win._moveTo("model", "Sampling");
  const removing = win._remove("sampler");
  assert.equal(held.length, 2, "both edits should be in flight");

  held[1]();                           // the remove answers first
  await removing;
  held[0]();                           // the move answers second, and is stale
  await moving;

  assert.equal(win.slots.some((s) => s.id === "sampler"), false,
    "a removed node came back when an older edit answered late");
  win.close();
});
