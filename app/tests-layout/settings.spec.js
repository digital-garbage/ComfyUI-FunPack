// The unified Settings window: one button, a searchable nav list, and every
// section mounted in place beside it -- picking one never leaves the window.

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

test("search also matches a section by its keywords, not only its title", async ({ page }) => {
  await app(page);
  await page.getByRole("button", { name: "Settings" }).click();
  // "loaders" is on the pipeline section's keywords, not its visible title
  // ("Models and pipeline") or subtitle ("What the run is made of.").
  await page.locator(".cx-modal .cx-search").fill("loaders");
  const rows = page.locator(".cx-modal .cx-filter-row");
  await expect(rows).toHaveCount(1);
  await expect(rows).toContainText("Models and pipeline");
});

test("picking a section swaps the content in place -- the window stays open, and there is only ever one", async ({ page }) => {
  await app(page);
  await page.getByRole("button", { name: "Settings" }).click();
  await page.locator(".cx-filter-row", { hasText: "Node packs" }).click();

  await expect(page.locator(".cx-modal")).toHaveCount(1);
  // "Settings" is the window's own title throughout; the section names
  // itself inside the body, beside the still-visible nav list.
  await expect(page.locator(".cx-modal-title")).toHaveText("Settings");
  await expect(page.locator(".cx-modal")).toContainText("Node packs");
  await expect(page.locator(".cx-filter-row")).toHaveCount(6);  // About + the five, still all there
});

test("switching sections tears down the last one -- its own polling stops", async ({ page }) => {
  // The log window polls while mounted. Picking it, then picking something
  // else, must stop that poll rather than leaving it running against a
  // detached view -- proven by leaving the log mounted only briefly and
  // confirming the window still works normally afterwards, requesting
  // nothing further from a section no longer on screen.
  await app(page);
  await page.getByRole("button", { name: "Settings" }).click();
  let logRequests = 0;
  await page.route("**/api/log**", (route) => { logRequests += 1; route.fulfill({ json: { lines: ["hi"] } }); });
  await page.locator(".cx-filter-row", { hasText: "ComfyUI log" }).click();
  await expect.poll(() => logRequests).toBeGreaterThan(0);

  await page.locator(".cx-filter-row", { hasText: "About FunPack" }).click();
  const before = logRequests;
  await page.waitForTimeout(2200);       // longer than the log's own 2s poll
  expect(logRequests, "the log kept polling after it was navigated away from").toBe(before);
});

test("closing and reopening always starts fresh, on About", async ({ page }) => {
  await app(page);
  await page.getByRole("button", { name: "Settings" }).click();
  await page.locator(".cx-filter-row", { hasText: "Node packs" }).click();
  await page.locator(".cx-modal .cx-icon-btn").click();       // the one close button, for the whole window

  await page.getByRole("button", { name: "Settings" }).click();
  await expect(page.locator(".cx-modal")).toContainText("About FunPack");
});
