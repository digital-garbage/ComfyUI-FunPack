// The models and pipeline window, where it has a real box model.
//
// jsdom covers what the window DOES; this covers what it looks like, which is
// the half jsdom cannot see -- it has no layout, so a modal collapsed to a
// column of overlapping controls passes there and is unusable here.

import { test, expect } from "@playwright/test";

async function openWindow(page) {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.getByRole("button", { name: "Models and pipeline" }).click();
  await expect(page.locator(".cx-modal")).toBeVisible();
}

test("the window opens on a card per group", async ({ page }) => {
  await openWindow(page);
  const cards = page.locator(".cx-card-title");
  await expect(cards).toHaveText(["Loaders", "Preparation", "Sampling", "Render"]);
});

test("a group is a node list beside its parameters, not one on top of the other", async ({ page }) => {
  await openWindow(page);
  await page.locator(".cx-card", { hasText: "Loaders" }).click();

  const list = page.locator('.cx-split-pane [aria-label="Nodes"]').first();
  const params = page.locator('.cx-split-pane [aria-label="Parameters"]').first();
  const a = await list.boundingBox();
  const b = await params.boundingBox();

  expect(a.width, "the node list has no width").toBeGreaterThan(100);
  expect(b.x, "the parameters are not beside the list").toBeGreaterThan(a.x + a.width - 20);
});

test("every control in a group fits the pane it is in", async ({ page }) => {
  // The failure this catches is the one the user reported about the frame:
  // things that do not scale. A control wider than its pane is clipped, and a
  // select clipped at its right edge hides the value it is showing.
  await openWindow(page);
  await page.locator(".cx-card", { hasText: "Loaders" }).click();

  const overflowing = await page.evaluate(() => {
    const pane = document.querySelector('[aria-label="Parameters"]');
    const box = pane.getBoundingClientRect();
    return [...pane.querySelectorAll("select, input, textarea")]
      .map((c) => ({ label: c.getAttribute("aria-label") || c.type, right: c.getBoundingClientRect().right }))
      .filter((c) => c.right > box.right + 1);
  });
  expect(overflowing, "controls run past the edge of their pane").toEqual([]);
});

test("the window still works when the window is narrow", async ({ page }) => {
  // Mobile is later, but a layout that only holds at one width is not a layout.
  await page.setViewportSize({ width: 420, height: 780 });
  await openWindow(page);
  await page.locator(".cx-card", { hasText: "Loaders" }).click();

  const modal = await page.locator(".cx-modal").boundingBox();
  expect(modal.width).toBeLessThanOrEqual(420);

  // Stacked rather than side by side, and both still readable.
  const list = await page.locator('[aria-label="Nodes"]').first().boundingBox();
  const params = await page.locator('[aria-label="Parameters"]').first().boundingBox();
  expect(list.width).toBeGreaterThan(150);
  expect(params.width).toBeGreaterThan(150);

  const overflow = await page.evaluate(() => {
    const el = document.querySelector(".cx-modal-body");
    return el.scrollWidth - el.clientWidth;
  });
  expect(overflow, "the window scrolls sideways").toBeLessThanOrEqual(1);
});

test("Save and Cancel hold the bottom instead of scrolling with the settings", async ({ page }) => {
  // Being visible is not enough and was not the fault. The bar was sticky
  // INSIDE a padded scroller, so a strip of the settings showed through below
  // it -- half a row of text under the buttons, scrolling as the pane scrolled.
  await openWindow(page);
  await page.locator(".cx-card", { hasText: "Preparation" }).click();

  const save = page.getByRole("button", { name: "Save", exact: true });
  await expect(save).toBeVisible();

  const barIn = page.locator(".cx-modal-foot");
  const bar = await barIn.boundingBox();
  const card = await page.locator(".cx-modal").boundingBox();
  expect(bar.y + bar.height,
    "settings show through below the action bar").toBeGreaterThanOrEqual(
      card.y + card.height - 1);

  // And it stays there once the settings are scrolled.
  await page.evaluate(() => {
    const pane = document.querySelector('[aria-label="Parameters"]').closest(".cx-split-pane");
    pane.scrollTop = pane.scrollHeight;
  });
  const after = await barIn.boundingBox();
  expect(after.y).toBeCloseTo(bar.y, 0);
});
