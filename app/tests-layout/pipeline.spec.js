// The models and pipeline window, where it has a real box model.
//
// jsdom covers what the window DOES; this covers what it looks like, which is
// the half jsdom cannot see -- it has no layout, so a modal collapsed to a
// column of overlapping controls passes there and is unusable here.

import { test, expect } from "@playwright/test";
import { openPipelineWindow } from "./_menu.js";

async function openWindow(page) {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await openPipelineWindow(page);
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

test("unwiring and rewiring a socket round-trips against the real server", async ({ page }) => {
  // Caught exactly this way once: jsdom's fake check() forwarded every field
  // by construction, so a real bug -- pipeline.js's check() silently dropping
  // `input`/`from_slot`/`from_output` because they were not in its
  // destructured parameter list -- passed every jsdom test and only showed up
  // as a live refusal ("which input to unwire is named by a string, not a
  // NoneType") on an actual click.
  await openWindow(page);
  await page.locator(".cx-card", { hasText: "Preparation" }).click();
  await page.locator(".cx-filter-row", { hasText: "latent" }).click();

  const modelRow = page.locator(".cx-settings-row")
    .filter({ has: page.locator(".cx-settings-label", { hasText: /^Model$/ }) });
  await expect(modelRow).toContainText("fed by model");
  await modelRow.getByRole("button", { name: "Unwire" }).click();
  await expect(modelRow).toContainText("nothing feeds it");
  await expect(modelRow.getByRole("button", { name: "Wire…" })).toBeVisible();

  await modelRow.getByRole("button", { name: "Wire…" }).click();
  const picker = page.locator(".cx-modal", { hasText: "Wire Model" });
  await expect(picker).toBeVisible();
  // Both real MODEL producers in the default pipeline, not just the one it
  // started wired to -- proves the picker is not just replaying the old link.
  await expect(picker).toContainText("model · model");
  await expect(picker).toContainText("modifiers · model");
  await picker.getByText("model · model", { exact: false }).click();

  await expect(page.locator(".cx-modal", { hasText: "Wire Model" })).toHaveCount(0);
  await expect(modelRow).toContainText("fed by model");
  await expect(modelRow).toContainText("FunPack Diffusion Model Loader");
});

test("a widget input -- a linked input -- can be wired to another node's output, for real", async ({ page }) => {
  // Found by adversarial review: the sockets loop had a Wire button and the
  // widgets loop had none at all, so "linked inputs" -- several widgets
  // sharing one node's output as their value -- could not be built through
  // this window despite the backend fully supporting it. Real nodes, real
  // types: FunPackDiffusionModelLoader's second output is its own STRING
  // status line, and "negative"'s "text" widget is a STRING too.
  await openWindow(page);
  await page.locator(".cx-card", { hasText: "Preparation" }).click();
  await page.locator(".cx-filter-row", { hasText: "negative" }).click();

  const textRow = page.locator(".cx-settings-row")
    .filter({ has: page.locator(".cx-settings-label", { hasText: /^Text$/ }) });
  await expect(textRow.locator("textarea, input")).toBeVisible();

  await textRow.getByRole("button", { name: /^Wire Text/ }).click();
  const picker = page.locator(".cx-modal", { hasText: "Wire Text" });
  await expect(picker).toBeVisible();
  await expect(picker).toContainText("model · status");
  await picker.getByText("model · status", { exact: false }).click();

  await expect(page.locator(".cx-modal", { hasText: "Wire Text" })).toHaveCount(0);
  await expect(textRow).toContainText("fed by model");
  await expect(textRow.locator("textarea, input")).toHaveCount(0);
  await expect(textRow.getByRole("button", { name: "Unwire" })).toBeVisible();

  await textRow.getByRole("button", { name: "Unwire" }).click();
  await expect(textRow).not.toContainText("fed by");
  await expect(textRow.locator("textarea, input")).toBeVisible();
});

test("the Add-node picker has one search box, not two disagreeing ones", async ({ page }) => {
  // filterList draws its own "Filter these results" search box; this picker
  // already has an outer one that re-queries the server on every keystroke.
  // Typing into filterList's own box used to show its correct LOCAL "Nothing
  // matches" right next to a STALE "Showing 40 of 75" hint left over from
  // the outer search's last real answer, before it -- or anything -- had
  // been touched. Found live: two numbers on screen describing two different
  // searches, neither telling you which is which.
  await openWindow(page);
  await page.locator(".cx-card", { hasText: "Sampling" }).click();
  await page.getByRole("button", { name: "Add node" }).click();

  const picker = page.locator(".cx-modal", { hasText: "Add a node to Sampling" });
  await expect(picker.getByPlaceholder("Search installed nodes")).toBeVisible();
  // Hidden via CSS (display: none), not removed from the DOM -- filterList
  // still builds it, this picker just does not show it. toHaveCount(0) would
  // check for removal, which is not what the fix does.
  await expect(picker.getByPlaceholder("Filter these results")).not.toBeVisible();
  await expect(picker).toContainText(/Showing \d+ of \d+/);

  await picker.getByPlaceholder("Search installed nodes").fill("zzz-nothing-real-matches-this");
  await expect(picker).toContainText("Nothing matches");
  await expect(picker).not.toContainText(/Showing \d+ of \d+/);
});

test("the way in is closed while the window is open", async ({ page }) => {
  // Reported from a real session: two of the same modal, one behind the other.
  // Two windows over one pipeline are two drafts of it -- whichever is saved
  // last wins, and the other was edited against a pipeline that had moved.
  //
  // There are two guards and this is the outer one: with the window up, its
  // backdrop covers the menu it was opened from, so a second one cannot be
  // asked for by hand at all. The inner guard -- open() called twice -- is in
  // the jsdom suite, where a second call can actually be made.
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  await openPipelineWindow(page);
  await expect(page.locator(".cx-modal")).toHaveCount(1);

  const settings = page.getByRole("button", { name: "Settings" });
  await expect(settings).toBeVisible();
  const blocked = await settings.evaluate((node) => {
    const b = node.getBoundingClientRect();
    const top = document.elementFromPoint(b.x + b.width / 2, b.y + b.height / 2);
    return !node.contains(top);
  });
  expect(blocked, "the opener is still reachable with the window open").toBe(true);

  await page.locator(".cx-modal .cx-icon-btn").click();
  await expect(page.locator(".cx-modal")).toHaveCount(0);
  await openPipelineWindow(page);
  await expect(page.locator(".cx-modal")).toHaveCount(1);
});
