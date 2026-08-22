// Which sampler settings the loaded model cannot use (pipeline_caps.js).
// Run: node --test movie_editor/frontend/pipeline_caps.test.js
//
// The rule these guard: a feature the model cannot run DISAPPEARS from Engine Settings
// rather than being offered and then explained away once per generation. So the predicate
// has to be exactly right in both directions — hiding a knob that works is worse than
// showing one that doesn't.
const test = require("node:test");
const assert = require("node:assert");

const PC = require("./pipeline_caps.js");

const h3 = { models: { model_family: "minimax_h3" }, project: {} };
const ltx = { models: { model_family: "ltxav" }, project: {} };

test("H3 hides the features that need LTX cross-attention", () => {
  const inert = PC.familyInertInputs(h3);
  assert.ok(inert.has("bounded_attention_enabled"));
  assert.ok(inert.has("identity_transfer_enabled"));
});

test("H3 hides identity transfer's sub-settings with it, not just the toggle", () => {
  const inert = PC.familyInertInputs(h3);
  for (const k of ["source_id", "phase_scale", "id_strength", "arcface_mode"]) {
    assert.ok(inert.has(k), `${k} should travel with its parent feature`);
  }
});

test("H3 hides v2a_grad_scale but keeps the feature it couples to", () => {
  const inert = PC.familyInertInputs(h3);
  assert.ok(inert.has("v2a_grad_scale"));
  // JoyAI audio memory itself works on H3 — only the LTXAV coupling knob is dead.
  assert.ok(!inert.has("joyai_audio_memory"));
});

test("LTX hides the H3-only audio clock", () => {
  const inert = PC.familyInertInputs(ltx);
  assert.ok(inert.has("h3_audio_clock"));
});

test("neither family hides the other's live settings", () => {
  const onH3 = PC.familyInertInputs(h3);
  const onLtx = PC.familyInertInputs(ltx);
  assert.ok(!onH3.has("h3_audio_clock"), "the audio clock is the point of H3");
  assert.ok(!onLtx.has("bounded_attention_enabled"), "bounded attention is an LTX feature");
  assert.ok(!onLtx.has("identity_transfer_enabled"), "Best-FaceID is an LTX feature");
});

test("nothing is hidden when the chain sampler is not in the pipeline", () => {
  // Without the Chain Sampler these knobs are inert for a different reason, and the
  // pipeline warning already covers it — hiding here would attribute it to the model.
  const off = { models: { model_family: "minimax_h3", disable_core: true }, project: {} };
  assert.strictEqual(PC.familyInertInputs(off).size, 0);
});

test("an unknown family is treated as LTX, never as 'hide everything'", () => {
  const inert = PC.familyInertInputs({ models: { model_family: "wan2.2" }, project: {} });
  assert.ok(!inert.has("bounded_attention_enabled"));
  assert.ok(inert.has("h3_audio_clock"));
});

test("the main-window chips no longer name a control that is now hidden", () => {
  // Every family-inert setting is hidden, so "turn it off in Settings → Engine" would
  // point at a row that is not rendered. Only value-shaped issues survive there.
  const st = {
    models: { model_family: "minimax_h3" },
    project: { sampler_inputs: { bounded_attention_enabled: true, identity_transfer_enabled: true } },
  };
  const chips = PC.h3InertSettings(st);
  assert.ok(!chips.some((c) => /can't run/.test(c.short)),
    "a hidden setting should not also be chipped");
});
