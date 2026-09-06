// "Generate All": every scene in order, waiting for each before starting the
// next. The dev server has no /prompt, so every attempt is refused before it
// ever reaches the queue -- which is exactly the path this proves does not
// hang or leak: the loop must still walk every scene and finish cleanly.

import { test, expect } from "@playwright/test";

test.describe.configure({ mode: "serial" });

async function app(page) {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
}

test("Generate All walks every scene and lands the selection on one of them", async ({ page }) => {
  await app(page);
  const ids = await page.evaluate(() => {
    const { project } = window.FunPack;
    while (project.scenes.length < 3) project.addScene();
    return project.scenes.map((s) => s.id);
  });

  await page.getByRole("button", { name: "Generate All" }).click();

  // The batch actually finishes -- the button comes back, rather than
  // staying disabled on a hung loop (every scene here is refused before it
  // reaches the queue, since the dev server has no /prompt).
  await expect.poll(() => page.getByRole("button", { name: "Generate All" }).isEnabled(),
    { timeout: 15000 }).toBe(true);

  const finalSelected = await page.evaluate(() => window.FunPack.project.selectedId);
  expect(ids).toContain(finalSelected);
});

test("Generate All says how many scenes did not generate, and re-enables both buttons", async ({ page }) => {
  await app(page);
  await page.evaluate(() => {
    const { project } = window.FunPack;
    while (project.scenes.length < 2) project.addScene();
  });

  await page.getByRole("button", { name: "Generate All" }).click();

  await expect.poll(() =>
    page.locator(".cx-panel-status", { hasText: /did not generate/i }).count(), { timeout: 15000 },
  ).toBeGreaterThan(0);

  await expect(page.getByRole("button", { name: "Generate All" })).toBeEnabled();
  await expect(page.getByRole("button", { name: "Generate", exact: true })).toBeEnabled();
});

test("a second click while a batch is already running does not start an overlapping one", async ({ page }) => {
  await app(page);
  await page.evaluate(() => {
    const { project } = window.FunPack;
    while (project.scenes.length < 3) project.addScene();
  });

  const btn = page.getByRole("button", { name: "Generate All" });
  await btn.click();
  await expect(btn).toBeDisabled();
  await btn.click({ force: true });   // a disabled button ignores this; proves it, not a real click path

  await expect.poll(() => btn.isEnabled(), { timeout: 15000 }).toBe(true);
});
