// Moving a project between machines: Save Project File downloads the real
// server route, Load Project File reads a real file back in. jsdom covers the
// store's own downloadUrl()/importProject() logic against a fake fetch; what
// only a browser can prove is that the menu items actually trigger a real
// download and a real file picker, against the real project routes.

import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";

// Skipped: / now serves the ported v4 frontend (app/legacy/), not composer --
// these specs assert on .cx-workspace*/window.FunPack, which no longer exist
// on the served page. Pending rewrite against the v4-derived DOM as part of
// the v4-UI-onto-v5-backend port's pixel-exactness verification step (see
// the plan file). Not deleted: the assertions are still the right SHAPE of
// check, just against selectors this page no longer has.
test.skip(true, "composer retired in favor of the v4 UI port -- rewrite against app/legacy's DOM");

test.describe.configure({ mode: "serial" });

async function app(page) {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
}

test("Save Project File downloads the current project as real JSON", async ({ page }) => {
  await app(page);
  const name = `Downloadable ${Date.now()}`;
  await page.evaluate((n) => window.FunPack.project.newProject(n), name);

  await page.getByRole("button", { name: "File" }).click();
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("menuitem", { name: "Save Project File…" }).click(),
  ]);

  expect(download.suggestedFilename()).toMatch(/\.funpack_project\.json$/);
  const body = JSON.parse(readFileSync(await download.path(), "utf-8"));

  expect(body.name).toBe(name);
  expect(Array.isArray(body.scenes)).toBe(true);
});

test("Load Project File opens the file as a new project, not the one it came from", async ({ page }) => {
  await app(page);
  const original = await page.evaluate(async (n) => {
    await window.FunPack.project.newProject(n);
    window.FunPack.project.setVideo("width", 777);
    await window.FunPack.project.flush();
    return window.FunPack.project.project.id;
  }, `Original ${Date.now()}`);

  const fileContent = await page.evaluate(async () => {
    const r = await fetch(window.FunPack.project.downloadUrl());
    return r.text();
  });

  await page.getByRole("button", { name: "File" }).click();
  const [chooser] = await Promise.all([
    page.waitForEvent("filechooser"),
    page.getByRole("menuitem", { name: "Load Project File…" }).click(),
  ]);
  await chooser.setFiles({
    name: "project.funpack_project.json",
    mimeType: "application/json",
    buffer: Buffer.from(fileContent),
  });

  await expect.poll(() => page.evaluate(() => window.FunPack.project.project.video.width)).toBe(777);
  const imported = await page.evaluate(() => window.FunPack.project.project.id);
  expect(imported).not.toBe(original);

  // The original is untouched by the import -- a shared id would have
  // overwritten it instead of landing as a project of its own.
  const stillThere = await page.evaluate((id) => fetch(`/funpack/api/projects/${id}`).then((r) => r.status), original);
  expect(stillThere).toBe(200);
});

test("a file that is not a project is refused, not silently opened", async ({ page }) => {
  await app(page);
  const before = await page.evaluate(() => window.FunPack.project.project.id);

  await page.getByRole("button", { name: "File" }).click();
  const [chooser] = await Promise.all([
    page.waitForEvent("filechooser"),
    page.getByRole("menuitem", { name: "Load Project File…" }).click(),
  ]);
  await chooser.setFiles({
    name: "not-a-project.json", mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify({ hello: "world" })),
  });

  await expect(page.locator(".cx-panel-status", { hasText: /could not load/i })).toBeVisible();
  expect(await page.evaluate(() => window.FunPack.project.project.id)).toBe(before);
});
