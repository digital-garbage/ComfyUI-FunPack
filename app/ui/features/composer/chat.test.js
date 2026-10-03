import test from "node:test";
import assert from "node:assert/strict";
import { chatHook } from "./chat.js";

const slots = [{ id: "e", node: "FunPackEnhancePrompt" }, { id: "x", node: "Other" }];
const project = { editor_settings: { enhance_chat: [{ comment: "warmer", original: "a cat" }, { comment: "old", original: "a dog" }, { comment: "any" }] } };

test("comments reach the enhancer only when they are about the prompt being run, and the rest are said", () => {
  const r = chatHook({ project, scene: { text: " a cat " }, slots });
  assert.deepEqual(JSON.parse(r.inputs.e.chat).map((x) => x.comment), ["warmer", "any"]);
  assert.equal(r.notes.length, 1);
});

test("no enhancer in the pipeline, or no comments: nothing is sent", () => {
  assert.equal(chatHook({ project, scene: { text: "a cat" }, slots: [slots[1]] }), null);
  assert.equal(chatHook({ project: {}, scene: { text: "a cat" }, slots }), null);
});
