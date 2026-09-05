// The bin, where it has a box model and a real image loader.
//
// jsdom covers what the bin DOES. Two of its claims can only be checked in a
// browser: that a lazily-loaded thumbnail inside a size container actually
// loads, and that three views of the same results all fit the panel they are
// in.

import { test, expect } from "@playwright/test";

// A 2x2 PNG, so the loader has something real to fetch without a server behind
// it. Every other route on the dev server answers 404, and an image that never
// arrives would make this pass whether the loader ran or not.
const PIXEL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP4z8AARAwQCgAf7gP9i18U1AAAAABJRU5ErkJggg==";

async function app(page) {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
}

const seed = (page, names) => page.evaluate((list) => {
  window.FunPack.bin.absorb(list.map((filename) => ({ filename, subfolder: "", type: "output" })));
}, names);

test("a lazily-loaded thumbnail inside the bin's grid actually loads", async ({ page }) => {
  // `loading="lazy"` inside a `container-type: inline-size` wrapper is the one
  // combination worth proving: a lazy image that never enters the loader leaves
  // an empty box forever, and nothing in the app would report it. Under
  // containment that is a real possibility, and jsdom -- which applies no CSS
  // and loads no images -- cannot tell either way.
  //
  // The image is built here rather than taken from a bin entry because the dev
  // server has no /view: a real entry's thumbnail 404s and the kit replaces it
  // with its glyph before this could look at it. What is under test is the
  // ENVIRONMENT the kit puts an image in, so the grid is the real one.
  await app(page);
  await seed(page, ["pixel.png"]);

  await page.evaluate((src) => {
    const img = document.createElement("img");
    img.className = "cx-cell-img";
    img.loading = "lazy";
    img.alt = "";
    img.src = src;
    document.querySelector(".cx-gallery-wrap .cx-gallery").append(img);
  }, PIXEL);

  const img = page.locator(".cx-gallery > img");
  await expect.poll(() => img.evaluate((i) => i.naturalWidth), { timeout: 5000 }).toBe(2);
});

test("a result whose file cannot be fetched shows the glyph, not a broken picture", async ({ page }) => {
  // The dev server answers 404 for every /view, so this is the real path.
  await app(page);
  await seed(page, ["gone.png"]);

  // Scoped to the bin: the timeline draws cells with the same glyph in them, so
  // an unscoped count here stopped being about the bin the day it arrived.
  const bin = page.locator('[aria-label="Media bin"]');
  await expect(bin.locator(".cx-cell-glyph")).toHaveCount(1);
  await expect(bin.locator("img")).toHaveCount(0);
  await expect(bin.locator(".cx-cell-name")).toHaveText(["gone.png"]);
});

test("every view of the bin fits the panel", async ({ page }) => {
  await app(page);
  await seed(page, ["rooftop_dusk_00001_.png", "clip_take3.mp4",
                    "anchor_frame_reference_that_is_far_too_long_00042_.png"]);

  for (const view of ["grid", "list", "icons"]) {
    await page.evaluate((v) => window.FunPack.bin.setView(v), view);
    const panel = await page.locator('[aria-label="Media bin"]').boundingBox();
    const cells = page.locator(".cx-cell, .cx-media-row");
    await expect(cells).toHaveCount(3);

    for (let i = 0; i < 3; i++) {
      const box = await cells.nth(i).boundingBox();
      expect(box.width, `${view}: an entry is wider than the bin`)
        .toBeLessThanOrEqual(panel.width + 1);
      expect(box.height, `${view}: an entry has no height`).toBeGreaterThan(8);
    }
  }
});

test("a long filename does not widen the bin", async ({ page }) => {
  // The bin is in a docked panel with a fixed width. A name that pushes it wider
  // pushes the preview over, which is v4's media bin exactly.
  await app(page);
  const before = await page.locator('[aria-label="Media bin"]').boundingBox();
  await seed(page, ["a_name_far_longer_than_any_panel_could_ever_be_00000000042_.png"]);

  for (const view of ["grid", "list", "icons"]) {
    await page.evaluate((v) => window.FunPack.bin.setView(v), view);
    const after = await page.locator('[aria-label="Media bin"]').boundingBox();
    expect(after.width, `${view} widened the bin`).toBeLessThanOrEqual(before.width + 1);
  }
});
