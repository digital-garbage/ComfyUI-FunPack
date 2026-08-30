// Where things sit.
//
// Every claim here was false at some point today and none of them can be
// checked in jsdom, which has no widths: three columns starting at three
// different heights, panel heads of two different heights depending on what was
// in them, a 7px gap on one side of the centre and 19 on the other, and five
// controls in one list at four different widths.

import { test, expect } from "@playwright/test";

const WIDE = { width: 1440, height: 900 };

async function app(page, size = WIDE) {
  await page.setViewportSize(size);
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
}

const boxes = (page, selector) => page.evaluate((sel) =>
  [...document.querySelectorAll(sel)].map((n) => {
    const b = n.getBoundingClientRect();
    return { x: Math.round(b.x), y: Math.round(b.y), w: Math.round(b.width),
             h: Math.round(b.height), r: Math.round(b.right), b: Math.round(b.bottom) };
  }), selector);

const gutter = (page) => page.evaluate(() => {
  const raw = getComputedStyle(document.documentElement).getPropertyValue("--gutter");
  return Math.round(parseFloat(raw));
});

test("every panel head is the same height, whatever is in it", async ({ page }) => {
  // The Assets head holds the bin's view control and the others hold only a
  // title. Padding plus a min-height made that one 41px and the rest 34, so two
  // panels side by side had their titles on different lines.
  await app(page);
  const heads = await boxes(page, ".cx-panel-head");
  expect(heads.length).toBeGreaterThan(2);
  const heights = new Set(heads.map((h) => h.h));
  expect([...heights], "panel heads have more than one height").toHaveLength(1);
});

test("the three columns start at the same height", async ({ page }) => {
  await app(page);
  const tops = await page.evaluate(() => ["left", "main", "right"].map((which) => {
    const region = document.querySelector(`.cx-workspace-${which}`);
    const panel = region.querySelector(".cx-panel");
    return Math.round(panel.getBoundingClientRect().y);
  }));
  expect(new Set(tops), `columns start at ${tops.join(", ")}`).toHaveProperty("size", 1);
});

test("one gutter, everywhere", async ({ page }) => {
  // Between the columns, and between two panels stacked in one of them. Three
  // different numbers for three kinds of gap is what makes a layout read as
  // accidental.
  await app(page);
  const g = await gutter(page);
  const cols = await page.evaluate(() => ["left", "main", "right"].map((which) => {
    const b = document.querySelector(`.cx-workspace-${which}`).getBoundingClientRect();
    return { x: Math.round(b.x), r: Math.round(b.right) };
  }));

  expect(cols[1].x - cols[0].r, "left column to centre").toBe(g);
  expect(cols[2].x - cols[1].r, "centre to right column").toBe(g);

  const stacked = await boxes(page, ".cx-workspace-main .cx-panel");
  expect(stacked.length, "the centre is not two stacked panels any more").toBe(2);
  expect(stacked[1].y - stacked[0].b, "between two panels in one column").toBe(g);
});

test("every control in a list of settings shares its two edges", async ({ page }) => {
  // A <select> takes the width of its widest OPTION, so the row holding a list
  // of checkpoint filenames was 124px wider than the four around it.
  await app(page);
  await page.getByRole("button", { name: "Models and pipeline" }).click();
  await page.locator(".cx-card", { hasText: "Loaders" }).click();
  await expect(page.locator(".cx-modal .cx-settings-control").first()).toBeVisible();

  const controls = await boxes(page, ".cx-modal .cx-settings-control");
  expect(controls.length).toBeGreaterThan(3);
  expect(new Set(controls.map((c) => c.x)), "controls start at different places")
    .toHaveProperty("size", 1);
  expect(new Set(controls.map((c) => c.r)), "controls end at different places")
    .toHaveProperty("size", 1);
});

test("what the app says is at the start of the transport, what you press is at the end", async ({ page }) => {
  // Everything used to be in one right-aligned group, so "Ready" sat against
  // the Generate button with the whole width of the bar empty to its left.
  await app(page);
  const g = await gutter(page);
  const [bar] = await boxes(page, ".cx-action-bar");
  const [lead] = await boxes(page, ".cx-action-bar-lead");
  const [buttons] = await boxes(page, ".cx-action-bar-buttons");

  expect(lead.x - bar.x, "the status is not at the start of the row").toBe(g);
  expect(bar.r - buttons.r, "the buttons are not at the end of the row").toBe(g);
  expect(buttons.x).toBeGreaterThan(lead.r);
});

test("a closed panel leaves one gutter behind it, not two", async ({ page }) => {
  // A closed panel is still a flex item, so the gap is applied on both sides of
  // nothing -- a double-width hole where a panel used to be.
  await app(page);
  const g = await gutter(page);
  const before = (await boxes(page, ".cx-workspace-main"))[0];

  await page.locator(".cx-workspace-rail-left button").click();
  await expect(page.locator(".cx-workspace-left")).toHaveAttribute("aria-hidden", "true");

  const [rail] = await boxes(page, ".cx-workspace-rail-left");
  const [centre] = await boxes(page, ".cx-workspace-main");
  expect(centre.x - rail.r, "the closed panel left a double gap").toBe(g);
  expect(centre.w, "the centre did not take the room back").toBeGreaterThan(before.w);
});
