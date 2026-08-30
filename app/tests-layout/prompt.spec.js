// The prompt, where it has a box model and a real server.
//
// jsdom covers what the controls DO. What only a browser can answer is whether
// the pipeline the real server serves actually puts them there.

import { test, expect } from "@playwright/test";

test("the prompt is on the main window, and typing in it is what runs", async ({ page }) => {
  // The prompt is a slot input like any other. Until it was put here the only
  // way to type one was to open the pipeline window and find the node.
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  const boxes = page.locator(".cx-panel textarea");
  await expect(boxes).toHaveCount(2);
  await boxes.first().fill("a cat on a rooftop");

  const sent = await page.evaluate(async () => {
    const body = JSON.stringify({ inputs: window.FunPack.prompts() });
    const res = await fetch("/funpack/api/pipeline",
      { method: "POST", headers: { "Content-Type": "application/json" }, body });
    return res.json();
  });
  const positive = sent.slots.find((s) => s.id === "positive");
  expect(sent.refused).toEqual([]);
  expect(positive.inputs.text).toBe("a cat on a rooftop");
  expect(positive.inputs.clip, "the wiring was replaced by a value").toEqual(["clip", 0]);
});

test("a region that filled does not keep its stand-in", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  const prompt = page.locator(".cx-panel", { hasText: "Prompt" }).first();
  await expect(prompt.locator("textarea")).toHaveCount(2);
  await expect(prompt.getByText("No prompt here")).toHaveCount(0);
});
