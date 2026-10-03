import test from "node:test";
import assert from "node:assert/strict";

import { wireReferences } from "../reference_wiring.js";

function refSlot(n) {
  return { id: `ref_source_${n}`, roles: [{ at: `assets.reference_${n}`,
    input: "media_id", wireTo: { slot: "r2v", input: `ref_images.ref_image_${n - 1}` } }] };
}

test("each picked reference is wired to its slot and to r2v", () => {
  const slots = [refSlot(1), refSlot(2)];
  const { overrides, unwired } = wireReferences(["aaa111111111", "bbb222222222"], slots);
  assert.equal(unwired, 0);
  assert.equal(overrides.ref_source_1.media_id, "aaa111111111");
  assert.deepEqual(overrides.r2v["ref_images.ref_image_0"], ["ref_source_1", 0]);
  assert.deepEqual(overrides.r2v["ref_images.ref_image_1"], ["ref_source_2", 0]);
});

test("a reference beyond the pipeline's slots is counted, not silently dropped", () => {
  // This is the bug an adversarial review caught: a scene can carry more
  // references than the CURRENT pipeline offers slots for (e.g. added under
  // a bigger preset, then the preset was swapped for a smaller one). The old
  // code in boot.js just returned from the forEach iteration and moved on --
  // no override, no count, no way for the caller to tell the user anything
  // was left out.
  const slots = [refSlot(1)]; // only one slot offered
  const { overrides, unwired } = wireReferences(
    ["aaa111111111", "bbb222222222", "ccc333333333"], slots);
  assert.equal(unwired, 2);
  assert.equal(overrides.ref_source_1.media_id, "aaa111111111");
  assert.equal(Object.keys(overrides).length, 2); // ref_source_1 + r2v, nothing else
});

test("no references and no slots wires nothing and reports no shortfall", () => {
  const { overrides, unwired } = wireReferences([], []);
  assert.deepEqual(overrides, {});
  assert.equal(unwired, 0);
});

test("slots not yet loaded (null, boot.js's own initial state) is treated as no slots, not a crash", () => {
  const { overrides, unwired } = wireReferences(["aaa111111111"], null);
  assert.deepEqual(overrides, {});
  assert.equal(unwired, 1);
});

test("a slot whose roles claim two reference numbers does not let the second silently overwrite the first", () => {
  // Not producible by the shipped H3 preset (one role per slot), but nothing
  // stops a future preset's roles array from doing this -- the function's own
  // contract ("no reference silently dropped") has to hold for the input
  // shape, not just the one preset that happens to exist today.
  const combo = { id: "combo_slot", roles: [
    { at: "assets.reference_1", wireTo: { slot: "r2v", input: "ref_images.ref_image_0" } },
    { at: "assets.reference_2", wireTo: { slot: "r2v", input: "ref_images.ref_image_1" } },
  ] };
  const { overrides, unwired } = wireReferences(["AAA", "BBB"], [combo]);
  assert.equal(overrides.combo_slot.media_id, "AAA");
  assert.equal(unwired, 1);
});

test("two different slots wired to the same downstream input do not let the second overwrite the first", () => {
  // Same failure as the combo-slot case above, reached a different way: two
  // separate reference slots whose `wireTo` both point at one destination
  // (e.g. a copy-pasted preset). The destination can hold one value.
  const slots = [
    { id: "ref_source_1", roles: [{ at: "assets.reference_1",
      wireTo: { slot: "r2v", input: "ref_images.ref_image_0" } }] },
    { id: "ref_source_2", roles: [{ at: "assets.reference_2",
      wireTo: { slot: "r2v", input: "ref_images.ref_image_0" } }] },
  ];
  const { overrides, unwired } = wireReferences(["AAA", "BBB"], slots);
  assert.deepEqual(overrides.r2v["ref_images.ref_image_0"], ["ref_source_1", 0]);
  assert.equal(overrides.ref_source_1.media_id, "AAA");
  assert.equal(overrides.ref_source_2, undefined);
  assert.equal(unwired, 1);
});

test("a slot the caller already assigned (e.g. the source-image slot) is not silently overwritten by a reference", () => {
  // boot.js sets the source-image slot's media_id BEFORE calling
  // wireReferences() -- if a preset ever reused that same slot id for a
  // reference role too, the reference would clobber the source image with
  // no toast, since wireReferences() has no visibility into that earlier
  // write on its own. The caller passes it in as reserved instead.
  const slots = [refSlot(1)];
  const { overrides, unwired } = wireReferences(
    ["AAA"], slots, ["ref_source_1"]);
  assert.equal(overrides.ref_source_1, undefined);
  assert.equal(unwired, 1);
});
