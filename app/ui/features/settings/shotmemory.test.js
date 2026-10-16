import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let shotMemory;
test.before(async () => { setupDom(); shotMemory = (await import("./shotmemory.js")).shotMemory; });
test.after(() => teardownDom());

const wait = () => new Promise((r) => setTimeout(r, 10));
const memory = (extra) => ({ prompts: 4, shots: 6, words: [], views: [], arms: [], details: [], ...extra });

test("the probe lists each subject the camera may infuse with how it was rated and its weight, and can forget it", async () => {
  const forgot = [];
  const api = { shotMemory: async () => memory({ infusion: [{ word: "hand", seen: 3, good: 0.5, bad: 2, weight: 1.2 }] }), forgetShotMemory: async (...a) => { forgot.push(a); } };
  const page = shotMemory({ api })();
  document.body.append(page.node);
  await wait();
  const text = page.node.textContent;
  assert.ok(text.includes("Subjects the camera may infuse") && text.includes("hand"));
  assert.ok(text.includes("in 3 prompts") && text.includes("liked 0.5") && text.includes("disliked 2.0") && text.includes("weight 1.2"), text);
  page.node.remove();
});

test("a memory holding only infusion rows is not shown as empty", async () => {
  const api = { shotMemory: async () => memory({ infusion: [{ word: "head", seen: 2, good: 0, bad: 0, weight: 2 }] }) };
  const page = shotMemory({ api })();
  document.body.append(page.node);
  await wait();
  assert.ok(!page.node.textContent.includes("Nothing learned yet"));
  page.node.remove();
});
