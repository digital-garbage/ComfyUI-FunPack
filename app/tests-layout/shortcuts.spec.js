// Shortcuts (Settings ▸ Shortcuts), autocomplete in the real Constructor
// prompt box, and the Project tab's Anchor/Postfix/Variables fields --
// against a real server, where the layout has a real box model. jsdom
// cannot see either the popover's real position (getBoundingClientRect
// answers zeros there) or a flex-shrink layout bug that only shows up once
// a column's content genuinely outgrows its box.

import { test, expect } from "@playwright/test";

async function app(page) {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
}

async function openShortcuts(page) {
  await page.getByRole("button", { name: "Settings" }).click();
  await page.locator(".cx-filter-row", { hasText: "Shortcuts" }).click();
}

async function saveShortcut(page, { name, triggers, replacements }) {
  await page.locator(".cx-modal").getByRole("button", { name: "+ Add shortcut" }).click();
  const modal = page.locator(".cx-modal");
  await modal.locator(".cx-field", { hasText: "Name" }).locator("input").fill(name);
  await modal.locator(".cx-field", { hasText: "Triggers" }).locator("textarea").fill(triggers);
  await modal.locator(".cx-field", { hasText: "Replacements" }).locator("textarea").fill(replacements);
  await modal.getByRole("button", { name: "Save" }).click();
}

test.afterEach(async ({ page }) => {
  // The library is a real file on disk (core/shortcuts.py), not per-project
  // state -- left behind, it would leak into whichever test runs next.
  await page.evaluate(() => fetch("/funpack/api/shortcuts/clear", { method: "POST" })).catch(() => {});
});

test("a saved shortcut round-trips through Settings", async ({ page }) => {
  await app(page);
  await openShortcuts(page);
  await saveShortcut(page, { name: "Fox", triggers: "fox", replacements: "red fox" });

  await expect(page.locator(".cx-modal")).toContainText("fox → red fox");
  await expect(page.locator(".cx-modal")).toContainText("1 shortcut");
});

test("a shortcut with no trigger is refused, and nothing is sent", async ({ page }) => {
  await app(page);
  await openShortcuts(page);
  await page.locator(".cx-modal").getByRole("button", { name: "+ Add shortcut" }).click();
  await page.locator(".cx-modal").getByRole("button", { name: "Save" }).click();

  await expect(page.locator(".cx-modal")).toContainText("At least one trigger is required.");
  await expect(page.locator(".cx-modal")).toContainText("0 shortcuts");
});

test("typing a saved trigger in the real prompt box suggests it, above the modal", async ({ page }) => {
  await app(page);
  await openShortcuts(page);
  await saveShortcut(page, { name: "Fox", triggers: "fox", replacements: "red fox" });
  // Two "Close"s on this window (the header ✕ and the footer's own button);
  // the footer one is the primary-toned action, the header's is an icon button.
  await page.locator(".cx-modal button.cx-btn-primary", { hasText: "Close" }).click();
  await expect(page.locator(".cx-modal")).toHaveCount(0);

  await page.locator(".cx-panel-head").getByRole("button", { name: "Constructor" }).click();
  const box = page.locator(".cx-modal textarea").first();
  await box.fill("a fo");

  const menu = page.locator(".cx-autocomplete");
  await expect(menu).toContainText("fox");
  await expect(menu).toContainText("red fox");

  await menu.locator(".cx-menu-item").first().click();
  await expect(box).toHaveValue("a fox ");
});

test("the Project tab's Anchor, Postfix and Variables render without overlapping the negative prompt", async ({ page }) => {
  await app(page);
  // A regression test for a real layout bug: `host`'s children (rows,
  // negative) could flex-shrink below their own content once the Project
  // tab grew past a couple of settings rows, spilling their text into each
  // other instead of the panel's own overflow:auto ever getting to scroll.
  await page.getByRole("tab", { name: "Project" }).click();
  await page.locator(".cx-field", { hasText: "Anchor" }).locator("textarea").fill("cinematic");
  await page.locator(".cx-field", { hasText: "Postfix" }).locator("textarea").fill("4k");
  await page.getByRole("button", { name: "+ Add variable" }).click();

  const varsLabel = page.locator(".cx-eyebrow", { hasText: "Variables" });
  const negLabel = page.locator("label.cx-label", { hasText: "Negative prompt" });
  await expect(varsLabel).toBeVisible();
  await expect(negLabel).toBeVisible();

  const [varsBox, negBox] = await Promise.all([varsLabel.boundingBox(), negLabel.boundingBox()]);
  // Not just "different y" -- the whole vertical span of one must sit above
  // the other's, the same assertion that would have caught the original bug.
  expect(varsBox.y + varsBox.height).toBeLessThanOrEqual(negBox.y + 1);
});

test("anchor, postfix and a variable survive a reload", async ({ page }) => {
  await app(page);
  await page.getByRole("tab", { name: "Project" }).click();
  await page.locator(".cx-field", { hasText: "Anchor" }).locator("textarea").fill("cinematic");
  await page.locator(".cx-field", { hasText: "Anchor" }).locator("textarea").blur();
  // A prior test in this file may have left rows of its own (real, persisted
  // project state) -- the row THIS click added is always the last one.
  await page.getByRole("button", { name: "+ Add variable" }).click();
  await page.locator('input[placeholder="name"]').last().fill("subject");
  await page.locator('input[placeholder="name"]').last().blur();
  await page.locator('input[placeholder="value"]').last().fill("a fox");
  await page.locator('input[placeholder="value"]').last().blur();

  await page.reload();
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.getByRole("tab", { name: "Project" }).click();
  await expect(page.locator(".cx-field", { hasText: "Anchor" }).locator("textarea")).toHaveValue("cinematic");
  await expect(page.locator("body")).toContainText("$subject");
});

test("each Generate draws a fresh seed for shortcut expansion, not a fixed hash of the text", async ({ page }) => {
  // core/shortcuts.py's expand() falls back to hashing the literal text when
  // seed is 0/absent -- a FIXED pick, not the "picks at random each time"
  // the Shortcuts editor's own hint promises. Regression test for that: the
  // request boot.js actually sends must carry a real, varying seed.
  await app(page);
  const seeds = [];
  page.on("request", (req) => {
    if (req.url().includes("/api/prompt/expand") && req.method() === "POST") {
      const body = req.postDataJSON();
      if (body && typeof body.seed === "number") seeds.push(body.seed);
    }
  });

  await page.locator(".cx-panel-head").getByRole("button", { name: "Constructor" }).click();
  await page.locator(".cx-modal textarea").first().fill("a fox runs");
  await page.locator(".cx-modal textarea").first().blur();
  await page.locator(".cx-modal").getByRole("button", { name: "Done" }).click();

  const generate = page.getByRole("button", { name: "Generate", exact: true });
  for (let i = 0; i < 2; i += 1) {
    await generate.click();
    await expect.poll(() => seeds.length, { timeout: 10000 }).toBeGreaterThan(i);
    // The dev server has no /prompt route, so this run is refused right
    // after queueing -- the button comes back quickly, same as
    // generate_all.spec.js relies on for its own walk.
    await expect(generate).toBeEnabled({ timeout: 10000 });
  }

  expect(seeds.length).toBeGreaterThanOrEqual(2);
  expect(seeds.every((s) => s > 0)).toBe(true);
  expect(new Set(seeds).size, "the same seed was sent twice in a row").toBeGreaterThan(1);
});

test("removing a variable works even mid-edit on a different row", async ({ page }) => {
  // A real race: row0's blur-commit triggers a full, non-keyed redraw of
  // this whole list (inspector.js's draw() -> node.replaceChildren()) --
  // wired to a plain click, pressing row1's ✕ right after typing in row0
  // (without ever blurring it first) destroys row1's own button mid-
  // gesture, so the click that started on it never arrives and nothing is
  // removed. mousedown+preventDefault on the ✕ (inspector.js's
  // removeButton()) is what this proves still works.
  await app(page);
  await page.getByRole("tab", { name: "Project" }).click();

  const add = page.getByRole("button", { name: "+ Add variable" });
  await add.click();
  await page.locator('input[placeholder="name"]').last().fill("row0");
  await add.click();
  await page.locator('input[placeholder="name"]').last().fill("row1");

  const rowCount = await page.locator('input[placeholder="name"]').count();
  // Type into row0's name field WITHOUT blurring it, then press row1's own
  // ✕ (the last one -- rows are appended, never reordered) in one motion --
  // the exact sequence the race needs.
  const row0Name = page.locator('input[placeholder="name"]').nth(rowCount - 2);
  await row0Name.pressSequentially("-edited", { delay: 20 });
  const removeButtons = page.getByRole("button", { name: "✕" });
  await removeButtons.last().click();

  await expect(page.locator('input[placeholder="name"]')).toHaveCount(rowCount - 1);
  await expect(page.locator("body")).not.toContainText("$row1");
});
