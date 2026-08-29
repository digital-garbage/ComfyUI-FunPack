// Anchored overlays, at the edges of a real viewport.
//
// This is the reason Playwright exists in this project. anchor.js measures,
// flips and clamps -- and every one of those reads getBoundingClientRect, which
// jsdom answers with zeros. The unit tests feed it faked rects, so they check
// the arithmetic and can say nothing about whether a menu opened off-screen.

import { test, expect } from "@playwright/test";

// The app itself, not the catalogue: it is the page a user actually loads, and
// it finishes booting at a point the test can wait for.
test.beforeEach(async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
});

async function openOverlayNear(page, corner) {
  // Put the launcher in a corner, then open from there: an overlay that fits
  // in the middle of the screen proves nothing about flipping.
  return page.evaluate(async ({ corner }) => {
    const { composer } = await import("/funpack/app/composer/composer.js");
    const anchor = composer.button.md({ label: "anchor" });
    Object.assign(anchor.node.style, {
      position: "fixed", width: "80px", height: "24px",
      left: corner.includes("right") ? "calc(100vw - 90px)" : "10px",
      top: corner.includes("bottom") ? "calc(100vh - 34px)" : "10px",
    });
    document.body.appendChild(anchor.node);

    const body = composer.text.md({ text: "x".repeat(200) });
    const pop = composer.popover.anchored({ anchor: anchor.node, body });
    const box = pop.node.getBoundingClientRect();
    return { x: box.x, y: box.y, w: box.width, h: box.height,
             vw: innerWidth, vh: innerHeight };
  }, { corner });
}

for (const corner of ["top-left", "top-right", "bottom-left", "bottom-right"]) {
  test(`an overlay opened at ${corner} stays on screen`, async ({ page }) => {
    const box = await openOverlayNear(page, corner);

    expect(box.w, "the overlay has no width, so nothing was measured").toBeGreaterThan(0);
    expect(box.h).toBeGreaterThan(0);

    // The whole point: clamped and flipped so every edge is inside the viewport.
    expect(box.x, `left edge off screen at ${corner}`).toBeGreaterThanOrEqual(0);
    expect(box.y, `top edge off screen at ${corner}`).toBeGreaterThanOrEqual(0);
    expect(box.x + box.w, `right edge past the viewport at ${corner}`)
      .toBeLessThanOrEqual(box.vw + 1);
    expect(box.y + box.h, `bottom edge past the viewport at ${corner}`)
      .toBeLessThanOrEqual(box.vh + 1);
  });
}

test("an overlay taller than the viewport is still reachable from its top", async ({ page }) => {
  await page.setViewportSize({ width: 800, height: 240 });
  const box = await openOverlayNear(page, "bottom-left");
  expect(box.y).toBeGreaterThanOrEqual(0);
});
