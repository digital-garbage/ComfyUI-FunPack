// A node input, turned into something a person can edit.

import test from "node:test";
import assert from "node:assert/strict";

import { settingFor, label, whyNotEditable } from "../widgets.js";

test("an identifier becomes a label a person would read", () => {
  assert.equal(label("sampler_name"), "Sampler name");
  assert.equal(label("filename_prefix"), "Filename prefix");
  // Words nobody spells out. "Cfg" and "Vae" read as typos, not as settings.
  assert.equal(label("cfg"), "CFG");
  assert.equal(label("vae_name"), "VAE name");
  assert.equal(label("clip_name1"), "CLIP name1");
});

test("each ComfyUI primitive maps to a control that can hold it", () => {
  assert.equal(settingFor({ name: "steps", type: "INT", default: 20 }).type, "int");
  assert.equal(settingFor({ name: "cfg", type: "FLOAT", default: 7 }).type, "float");
  assert.equal(settingFor({ name: "on", type: "BOOLEAN", default: true }).type, "bool");
  assert.equal(settingFor({ name: "t", type: "STRING", default: "" }).type, "text");
  assert.equal(settingFor({ name: "t", type: "STRING", multiline: true }).type, "multiline");
});

test("a combo's default is one of its choices", () => {
  // ComfyUI leaves the default off a file picker: the first entry is what its
  // own frontend shows, and a select rendered with `undefined` reports
  // undefined as the value the moment anything reads it.
  const setting = settingFor({ name: "ckpt_name", type: "COMBO", choices: ["a.safetensors", "b.safetensors"] });
  assert.equal(setting.default, "a.safetensors");
  assert.deepEqual(setting.options.map((o) => o.value), ["a.safetensors", "b.safetensors"]);
});

test("a declared default that is not among the choices does not become the value", () => {
  const setting = settingFor({ name: "x", type: "COMBO", choices: ["a", "b"], default: "gone" });
  assert.equal(setting.default, "a");
});

test("an empty combo has no control, and says why rather than showing an empty box", () => {
  const widget = { name: "ckpt_name", type: "COMBO", choices: [] };
  assert.equal(settingFor(widget), null);
  assert.match(whyNotEditable(widget), /no files of this kind/);
});

test("a default of the wrong type does not become NaN in a number box", () => {
  // Nothing checks a node's declared default, and a number control handed a
  // string reports NaN the moment it is read.
  assert.equal(settingFor({ name: "steps", type: "INT", default: "twenty" }).default, 0);
  assert.equal(settingFor({ name: "steps", type: "INT", default: 20 }).default, 20);
});

test("a type this app has no control for gets no control", () => {
  // Not a text box holding "LATENT". An input the app cannot edit has to look
  // different from one it can, or the value behind it is a mystery.
  assert.equal(settingFor({ name: "samples", type: "LATENT" }), null);
  assert.match(whyNotEditable({ name: "samples", type: "LATENT" }), /filled by a wire/);
});

test("bounds and tooltips survive the translation", () => {
  const setting = settingFor({
    name: "cfg", type: "FLOAT", default: 8, min: 0, max: 100, step: 0.5,
    tooltip: "How hard the prompt is enforced.",
  });
  assert.equal(setting.min, 0);
  assert.equal(setting.max, 100);
  assert.equal(setting.step, 0.5);
  assert.equal(setting.hint, "How hard the prompt is enforced.");
});

test("a combo that takes several choices at once gets no single-select", () => {
  // A single-select would save a string where the node wants a list.
  const widget = { name: "tags", type: "COMBO", choices: ["a", "b"], multiselect: true };
  assert.equal(settingFor(widget), null);
  assert.match(whyNotEditable(widget), /several choices at once/);
});

test("a choice that brings settings this window cannot draw says so", () => {
  // A dynamic combo's options carry their own inputs. Silently dropping them
  // offers an incomplete node as a complete one, and the run fails when it is
  // queued over a field nobody was shown.
  const setting = settingFor({
    name: "resize_type", type: "COMBO", choices: ["scale dimensions", "longest side"],
    reveals_more: true, tooltip: "How the image is resized.",
  });
  assert.match(setting.hint, /How the image is resized\./);
  assert.match(setting.hint, /not shown here yet/);
});
