// Docking, measured.
//
// jsdom can say a class was toggled and an attribute changed; it cannot say the
// preview actually got the space, because getBoundingClientRect is zeros there.
// The v4 fault this shape exists to avoid was invisible for exactly that reason:
// the state was right and the pixels were not.

import { test, expect } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
});

const widths = (page) => page.evaluate(() => ({
  left: document.querySelector(".cx-workspace-left").getBoundingClientRect().width,
  right: document.querySelector(".cx-workspace-right").getBoundingClientRect().width,
  main: document.querySelector(".cx-workspace-main").getBoundingClientRect().width,
}));

test("both panels are docked beside the preview, not over it", async ({ page }) => {
  const before = await widths(page);
  expect(before.left, "the assets panel has no width").toBeGreaterThan(100);
  expect(before.right, "the properties panel has no width").toBeGreaterThan(100);

  // Docked means the three regions share the row. Overlaying would let the sum
  // exceed the workspace, which is the other design and not this one.
  const total = await page.evaluate(() =>
    document.querySelector(".cx-workspace").getBoundingClientRect().width);
  expect(before.left + before.right + before.main).toBeLessThanOrEqual(total + 1);
});

test("collapsing a panel gives its width to the preview", async ({ page }) => {
  const before = await widths(page);
  await page.locator('.cx-workspace-rail-left button').click();
  const after = await widths(page);

  expect(after.left).toBe(0);
  expect(after.main, "the preview did not grow into the freed space")
    .toBeGreaterThan(before.main + before.left - 2);
});

test("the toggle stays reachable once its panel is collapsed", async ({ page }) => {
  // The whole reason the rail is a sibling of the panel. In v4 the control was
  // inside the region it hid, the region became display:none, and the only way
  // back was clearing storage by hand.
  const toggle = page.locator('.cx-workspace-rail-left button');
  await toggle.click();
  await expect(toggle).toBeInViewport();
  await toggle.click();
  expect((await widths(page)).left).toBeGreaterThan(100);
});

test("a collapsed panel is still remembered after a reload", async ({ page }) => {
  await page.locator('.cx-workspace-rail-right button').click();
  expect((await widths(page)).right).toBe(0);

  await page.reload();
  await page.waitForFunction(() => window.FunPack !== undefined);
  expect((await widths(page)).right).toBe(0);

  // And it can still be brought back, which is what makes remembering safe.
  await page.locator('.cx-workspace-rail-right button').click();
  expect((await widths(page)).right).toBeGreaterThan(100);
});

test("the prompt sits under the preview and both have height", async ({ page }) => {
  // A vertical split inside a parent whose height came from its content gave
  // its second pane a basis of zero: the prompt was in the DOM, had no pixels,
  // and nothing reported anything.
  const boxes = await page.evaluate(() => {
    const panels = [...document.querySelectorAll(".cx-workspace-main .cx-panel")];
    return panels.map((p) => {
      const r = p.getBoundingClientRect();
      return { title: p.querySelector(".cx-panel-title")?.textContent, h: r.height, y: r.y };
    });
  });
  const preview = boxes.find((b) => b.title === "Preview");
  const prompt = boxes.find((b) => b.title === "Prompt");
  expect(preview?.h, "the preview has no height").toBeGreaterThan(50);
  expect(prompt?.h, "the prompt has no height").toBeGreaterThan(30);
  expect(prompt.y).toBeGreaterThan(preview.y);
});
