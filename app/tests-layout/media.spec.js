// The media library against the real server: a real multipart upload, a real
// file on disk, a real Range-capable file response -- none of which jsdom's
// mocked fetch in media.test.js can prove.
//
// This is the real, shared `media/` directory under the repo, not a temp
// fixture -- there is no per-run isolation the way core/tests/test_media.py
// gets from monkeypatching config.MEDIA_DIR. A round of adversarial review
// reproduced exactly what that implies: a stray entry left by ANY other
// writer against the same checkout (a second devserver, an interrupted prior
// run) makes an absolute-count assertion fail for a reason that has nothing
// to do with this test. Every assertion below is therefore scoped to what
// THIS test itself put there -- named cells, or a count taken before/after
// this test's own action -- never a bare "the gallery has N things in it".
// Every upload is also removed in a `finally`, so a failed assertion still
// does not leave litter for the next run to trip over.

import { test, expect } from "@playwright/test";
import path from "node:path";
import fs from "node:fs";
import os from "node:os";

// Skipped: / now serves the ported v4 frontend (app/legacy/), not composer --
// these specs assert on .cx-workspace*/window.FunPack, which no longer exist
// on the served page. Pending rewrite against the v4-derived DOM as part of
// the v4-UI-onto-v5-backend port's pixel-exactness verification step (see
// the plan file). Not deleted: the assertions are still the right SHAPE of
// check, just against selectors this page no longer has.
test.skip(true, "composer retired in favor of the v4 UI port -- rewrite against app/legacy's DOM");

async function app(page) {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.locator(".cx-tabs", { hasText: "Media" }).getByRole("tab", { name: "Media" }).click();
}

function tempFile(name, bytes) {
  const p = path.join(os.tmpdir(), name);
  fs.writeFileSync(p, bytes);
  return p;
}

test("a real upload lands in the library and can be fetched back", async ({ page }) => {
  const name = `playwright-ref-${Date.now()}.png`;
  const file = tempFile(name, Buffer.from([0x89, 0x50, 0x4e, 0x47, 0, 0, 0, 0]));
  let id = null;
  try {
    await app(page);
    await page.locator("input[type=file][accept*=image]").setInputFiles(file);

    const cell = page.locator(".cx-cell", { hasText: name });
    await expect(cell).toHaveCount(1);

    id = await page.evaluate((n) =>
      window.FunPack.media.items.find((i) => i.name === n).id, name);
    const res = await page.request.get(`/funpack/api/media/${id}/file`);
    expect(res.ok()).toBe(true);
  } finally {
    if (id) await page.request.delete(`/funpack/api/media/${id}`).catch(() => {});
    fs.unlinkSync(file);
  }
});

test("a file type the store refuses is rejected, and never appears", async ({ page }) => {
  const file = tempFile(`playwright-bad-${Date.now()}.exe`, Buffer.from("MZ"));
  try {
    await app(page);
    const before = await page.locator(".cx-cell").count();
    await page.locator("input[type=file][accept*=image]").setInputFiles(file);

    await expect(page.locator(".cx-toast-danger")).toBeVisible();
    // Not toHaveCount(0): the real, shared store may already hold entries
    // from something else entirely. What this test owns is that ITS OWN
    // upload added nothing -- the count before and after must match.
    await expect(page.locator(".cx-cell")).toHaveCount(before);
  } finally {
    fs.unlinkSync(file);
  }
});

test("right-click removes an entry from the server, not just the screen", async ({ page }) => {
  const name = `playwright-remove-${Date.now()}.png`;
  const file = tempFile(name, Buffer.from([0x89, 0x50, 0x4e, 0x47]));
  let id = null;
  try {
    await app(page);
    await page.locator("input[type=file][accept*=image]").setInputFiles(file);

    id = await page.evaluate((n) =>
      window.FunPack.media.items.find((i) => i.name === n).id, name);

    await page.locator(".cx-cell", { hasText: name }).click({ button: "right" });
    await expect(page.locator(".cx-cell", { hasText: name })).toHaveCount(0);

    const res = await page.request.get(`/funpack/api/media/${id}/file`);
    expect(res.status()).toBe(404);
  } finally {
    // Already gone via the right-click in the normal case; a backstop for
    // when an earlier assertion in this test throws first.
    if (id) await page.request.delete(`/funpack/api/media/${id}`).catch(() => {});
    fs.unlinkSync(file);
  }
});
