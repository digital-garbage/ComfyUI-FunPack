// A floating window must always be reachable.
//
// Position is remembered per window id, so the dangerous case is not "where did
// it open" but "where did it open GIVEN a screen it no longer fits". jsdom
// cannot see this at all: getBoundingClientRect is zeros there, so a window
// entirely off the right edge measures the same as one in the middle.

import { test, expect } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
});

async function openWindow(page, id) {
  return page.evaluate(async (id) => {
    const { composer } = await import("/funpack/app/composer/composer.js");
    const handle = composer.floating.window({
      id, title: "Attack", body: composer.text.md({ text: "content" }),
    });
    const box = handle.node.getBoundingClientRect();
    return { x: box.x, y: box.y, w: box.width, h: box.height,
             vw: innerWidth, vh: innerHeight };
  }, id);
}

test("a window remembered off a wider screen still opens reachable", async ({ page }) => {
  // Remember a position far to the right, as dragging on a wide screen would.
  await page.evaluate(() => localStorage.setItem(
    "cx.win.attack", JSON.stringify({ x: 1400, y: 40, width: 420, height: 320 })));

  await page.setViewportSize({ width: 800, height: 600 });
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  const box = await openWindow(page, "attack");

  // Something grabbable has to be on screen -- a titlebar with no pixels in the
  // viewport cannot be dragged back, and the position is remembered, so the
  // window would be lost until localStorage was cleared by hand.
  expect(box.x, "the whole window is past the right edge").toBeLessThan(box.vw);
  expect(box.x + box.w, "the whole window is past the left edge").toBeGreaterThan(0);
  expect(box.y).toBeGreaterThanOrEqual(0);
  expect(box.y, "the titlebar is below the bottom edge").toBeLessThan(box.vh);

  // And enough of it to actually hit with a pointer.
  const visible = Math.min(box.x + box.w, box.vw) - Math.max(box.x, 0);
  expect(visible, "less than a grabbable strip is on screen").toBeGreaterThanOrEqual(60);
});

test("a window open while the viewport shrinks stays reachable", async ({ page }) => {
  await page.setViewportSize({ width: 1400, height: 900 });
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await openWindow(page, "shrink");

  await page.setViewportSize({ width: 600, height: 500 });
  const box = await page.evaluate(() => {
    const node = document.querySelector(".cx-window");
    const r = node.getBoundingClientRect();
    return { x: r.x, y: r.y, w: r.width, vw: innerWidth };
  });

  expect(box.x).toBeLessThan(box.vw);
  expect(box.x + box.w).toBeGreaterThan(0);
});

test("a window cannot be dragged somewhere it cannot be dragged back from", async ({ page }) => {
  await page.setViewportSize({ width: 900, height: 700 });
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await openWindow(page, "dragged");

  const bar = page.locator(".cx-window-bar").first();
  await bar.hover();
  await page.mouse.down();
  await page.mouse.move(5000, 5000, { steps: 5 });   // far past both edges
  await page.mouse.up();

  const box = await page.evaluate(() => {
    const r = document.querySelector(".cx-window").getBoundingClientRect();
    return { x: r.x, y: r.y, w: r.width, vw: innerWidth, vh: innerHeight };
  });
  expect(box.x).toBeLessThan(box.vw);
  expect(box.y).toBeLessThan(box.vh);
  expect(box.x + box.w).toBeGreaterThan(0);
});
