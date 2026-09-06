// The prompt, where it has a box model and a real server.
//
// jsdom covers what the controls DO. What only a browser can answer is whether
// the pipeline the real server serves actually puts them there.

import { test, expect } from "@playwright/test";

const openConstructor = (page) =>
  page.locator(".cx-panel-head").getByRole("button", { name: "Constructor" }).click();

test("the prompt is written in the Constructor, and typing in it is what runs", async ({ page }) => {
  // The prompt is a slot input like any other. Until it was wired the only way
  // to type one was to open the pipeline window and find the node; it now has a
  // window of its own, opened from the timeline that a run fills.
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  await expect(page.locator("textarea"), "the prompt is on the main window").toHaveCount(0);
  await openConstructor(page);

  const boxes = page.locator(".cx-modal textarea");
  await expect(boxes).toHaveCount(2);
  await boxes.first().fill("a cat on a rooftop");
  await boxes.first().blur();                   // a control commits on blur

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

test("what was typed is still there the next time the window opens", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  await openConstructor(page);
  await page.locator(".cx-modal textarea").first().fill("a cat on a rooftop");
  await page.locator(".cx-modal textarea").first().blur();
  await page.locator(".cx-modal").getByRole("button", { name: "Done" }).click();
  await expect(page.locator(".cx-modal")).toHaveCount(0);

  await openConstructor(page);
  await expect(page.locator(".cx-modal textarea").first()).toHaveValue("a cat on a rooftop");
});

test("a region that filled does not keep its stand-in", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  await openConstructor(page);
  const modal = page.locator(".cx-modal");
  await expect(modal.locator("textarea")).toHaveCount(2);
  await expect(modal.getByText("Nothing to write yet")).toHaveCount(0);
});
