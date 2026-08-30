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

test("the regions meet on a hairline, not across a gap", async ({ page }) => {
  // The app is one surface cut into areas, not a tray of cards with the desk
  // showing between them. Before this the columns floated with the page
  // background between them -- and with three different gaps at that.
  await app(page);
  const cols = await page.evaluate(() => ["left", "main", "right"].map((which) => {
    const b = document.querySelector(`.cx-workspace-${which}`).getBoundingClientRect();
    return { x: Math.round(b.x), r: Math.round(b.right) };
  }));

  expect(cols[1].x - cols[0].r, "a gap between the left column and the centre").toBe(0);
  expect(cols[2].x - cols[1].r, "a gap between the centre and the right column").toBe(0);

  const divider = await page.evaluate(() =>
    getComputedStyle(document.querySelector(".cx-workspace-left")).borderRightWidth);
  expect(divider, "the regions are not divided at all").toBe("1px");
});

test("two zones stacked in a column are divided by a grabbable splitter", async ({ page }) => {
  // A splitter is the one place a gap is right: it has to be wide enough to
  // take hold of, and it draws its own hairline down the middle.
  await app(page);
  const g = await gutter(page);
  const stacked = await boxes(page, ".cx-workspace-main .cx-panel");
  expect(stacked.length, "the centre is not two stacked zones any more").toBe(2);
  expect(stacked[1].y - stacked[0].b, "between two zones in one column").toBe(g);
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

test("a closed panel leaves the rail and nothing else", async ({ page }) => {
  await app(page);
  const before = (await boxes(page, ".cx-workspace-main"))[0];

  await page.locator(".cx-workspace-rail-left button").click();
  await expect(page.locator(".cx-workspace-left")).toHaveAttribute("aria-hidden", "true");

  const [rail] = await boxes(page, ".cx-workspace-rail-left");
  const [centre] = await boxes(page, ".cx-workspace-main");
  expect(centre.x - rail.r, "a closed panel left a hole behind it").toBe(0);
  expect(centre.w, "the centre did not take the room back").toBeGreaterThan(before.w);
});

test("the app's own surfaces have depth, and both themes get it", async ({ page }) => {
  // Every band and every filled control was one flat colour with a 1px line
  // round it, which is what made this read as a wireframe of itself. The
  // gradients are derived from each theme's own elevation scale, so this has to
  // hold in both -- a token that resolves in one theme and not the other is the
  // exact fault the derived-token rule exists to prevent.
  for (const theme of ["dark", "light"]) {
    await page.addInitScript((t) => window.localStorage.setItem("funpack_theme", t), theme);
    await app(page);
    await expect.poll(() => page.evaluate(() =>
      document.documentElement.getAttribute("data-theme"))).toBe(theme);

    const painted = await page.evaluate(() => ({
      head: getComputedStyle(document.querySelector(".cx-panel-head")).backgroundImage,
      body: getComputedStyle(document.body).backgroundImage,
      primary: getComputedStyle(document.querySelector(".cx-btn-primary")).backgroundImage,
      glow: getComputedStyle(document.querySelector(".cx-btn-primary")).boxShadow,
      grain: getComputedStyle(document.querySelector(".cx-frame"), "::after").opacity,
    }));

    expect(painted.head, `${theme}: a band is a flat fill`).toContain("gradient");
    expect(painted.body, `${theme}: the page is a flat fill`).toContain("gradient");
    expect(painted.primary, `${theme}: the primary button is a flat fill`).toContain("gradient");
    expect(painted.glow, `${theme}: the primary button throws no light`).not.toBe("none");
    expect(Number(painted.grain), `${theme}: no grain`).toBeGreaterThan(0);
  }
});

test("a zone is square and a card is not", async ({ page }) => {
  await app(page);
  const zone = await page.evaluate(() => {
    const n = document.querySelector(".cx-zone");
    const s = getComputedStyle(n);
    return { radius: s.borderTopLeftRadius, shadow: s.boxShadow };
  });
  expect(zone.radius, "a region of the app has rounded corners").toBe("0px");
  expect(zone.shadow, "a region of the app floats above itself").toBe("none");
});
