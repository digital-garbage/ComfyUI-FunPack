const test = require("node:test");
const assert = require("node:assert");
const { SLOTS, normalizePins, visibleSlots, placePin, pinKey } = require("./pinned_buttons.js");

const A = { kind: "section", id: "engine", label: "Engine" };
const B = { kind: "node", id: "fp_lora", label: "FunPack LoRA Loader" };
const C = { kind: "section", id: "models", label: "Models & Pipeline" };

test("slots are always exactly three, so slot N is index N-1", () => {
  assert.strictEqual(SLOTS, 3);
  assert.deepStrictEqual(normalizePins(null), [null, null, null]);
  assert.deepStrictEqual(normalizePins([A]), [A, null, null]);
  assert.strictEqual(normalizePins([A, B, C, A]).length, 3);
});

test("a malformed entry becomes an empty slot, not a broken button", () => {
  assert.deepStrictEqual(normalizePins([{ kind: "section" }, "junk", { id: "x" }]),
    [null, null, null]);
  assert.deepStrictEqual(normalizePins("not an array"), [null, null, null]);
});

test("holes close up but the buttons keep their own numbers", () => {
  // The user's spec: slots 1 and 3 filled draws slot 1 then slot 3, with no gap and no
  // renumbering — position is the address, so it must not shift when a neighbour goes.
  assert.deepStrictEqual(visibleSlots([A, null, C]).map((e) => e.slotNo), [1, 3]);
  assert.deepStrictEqual(visibleSlots([null, B, null]).map((e) => e.slotNo), [2]);
  assert.deepStrictEqual(visibleSlots([A, B, C]).map((e) => e.slotNo), [1, 2, 3]);
  assert.deepStrictEqual(visibleSlots([null, null, null]), []);
});

test("visible order is left to right by slot number", () => {
  assert.deepStrictEqual(visibleSlots([A, null, C]).map((e) => e.target), [A, C]);
});

test("placing a pin fills the slot asked for", () => {
  assert.deepStrictEqual(placePin([null, null, null], 1, B), [null, B, null]);
  assert.deepStrictEqual(placePin([A, null, null], 0, B), [B, null, null]);
});

test("the same destination never occupies two slots", () => {
  assert.deepStrictEqual(placePin([A, null, null], 2, A), [null, null, A]);
  // Matching is by kind AND id: a node and a section could share an id string.
  const nodeEngine = { kind: "node", id: "engine", label: "a node called engine" };
  assert.deepStrictEqual(placePin([A, null, null], 2, nodeEngine), [A, null, nodeEngine]);
});

test("placing does not mutate the list it was given", () => {
  const before = [A, null, null];
  placePin(before, 1, B);
  assert.deepStrictEqual(before, [A, null, null]);
});


// ── sub-views: a place INSIDE a section is its own destination ─────────────────

const ENG = { kind: "section", id: "engine", label: "Engine" };
const ENG_SAMPLER = { kind: "section", id: "engine", sub: "studio_sampler",
                      label: "Engine ▸ FunPack Studio ▸ Sampler algorithm" };
const ENG_GUIDANCE = { kind: "section", id: "engine", sub: "chain_guidance",
                       label: "Engine ▸ Chain Sampler ▸ Guidance" };

test("two categories of one section are different destinations", () => {
  assert.notStrictEqual(pinKey(ENG_SAMPLER), pinKey(ENG_GUIDANCE));
  // Pinning the second must NOT evict the first — that was the failure the key prevents.
  const after = placePin([ENG_SAMPLER, null, null], 1, ENG_GUIDANCE);
  assert.deepStrictEqual(after, [ENG_SAMPLER, ENG_GUIDANCE, null]);
});

test("a section and one of its categories are different destinations", () => {
  assert.notStrictEqual(pinKey(ENG), pinKey(ENG_SAMPLER));
  assert.deepStrictEqual(placePin([ENG, null, null], 2, ENG_SAMPLER), [ENG, null, ENG_SAMPLER]);
});

test("the SAME category still never occupies two slots", () => {
  assert.deepStrictEqual(placePin([ENG_SAMPLER, null, null], 2, ENG_SAMPLER),
    [null, null, ENG_SAMPLER]);
});

test("a missing sub is the same as an empty one", () => {
  assert.strictEqual(pinKey(ENG), pinKey({ kind: "section", id: "engine", sub: "" }));
  assert.strictEqual(pinKey(null), "");
});
