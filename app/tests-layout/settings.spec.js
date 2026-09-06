// The unified Settings window: one button, a searchable list, About drawn in
// place, everything else a deep link to its own real window.

import { test, expect } from "@playwright/test";

async function app(page) {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
}

test("opening Settings shows About, with the real version and branch", async ({ page }) => {
  await app(page);
  const real = await page.evaluate(async () => (await fetch("/funpack/api/git/status")).json());

  await page.getByRole("button", { name: "Settings" }).click();
  await expect(page.locator(".cx-modal")).toContainText("About FunPack");
  if (real.ok) await expect(page.locator(".cx-modal")).toContainText(real.branch);
});

test("search narrows the section list to what matches", async ({ page }) => {
  await app(page);
  await page.getByRole("button", { name: "Settings" }).click();
  await page.locator(".cx-modal .cx-search").fill("packs");
  const rows = page.locator(".cx-modal .cx-filter-row");
  await expect(rows).toHaveCount(1);
  await expect(rows).toContainText("Node packs");
});

test("picking a deep-link section closes Settings and opens that window instead", async ({ page }) => {
  await app(page);
  await page.getByRole("button", { name: "Settings" }).click();
  await page.locator(".cx-filter-row", { hasText: "Node packs" }).click();

  // Only one modal ever exists at a time -- the deep link replaces this
  // window rather than stacking over it.
  await expect(page.locator(".cx-modal")).toHaveCount(1);
  await expect(page.locator(".cx-modal-title")).toHaveText("Node packs");
});

test("a fresh open always starts on About, even after a deep link was used last", async ({ page }) => {
  await app(page);
  await page.getByRole("button", { name: "Settings" }).click();
  await page.locator(".cx-filter-row", { hasText: "Node packs" }).click();
  await page.locator(".cx-modal .cx-icon-btn").click();       // close the packs window

  await page.getByRole("button", { name: "Settings" }).click();
  await expect(page.locator(".cx-modal")).toContainText("About FunPack");
});
