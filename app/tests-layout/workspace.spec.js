// Docking, measured.
//
// jsdom can say a class was toggled and an attribute changed; it cannot say the
// preview actually got the space, because getBoundingClientRect is zeros there.
// The v4 fault this shape exists to avoid was invisible for exactly that reason:
// the state was right and the pixels were not.

import { test, expect } from "@playwright/test";
import { regionToggle, menuToggle } from "./_menu.js";

test.beforeEach(async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
});

const widths = (page) => page.evaluate(() => ({
  left: document.querySelector(".cx-workspace-left").getBoundingClientRect().width,
  right: document.querySelector(".cx-workspace-right").getBoundingClientRect().width,
  main: document.querySelector(".cx-workspace-main").getBoundingClientRect().width,
}));

test("both panels are docked beside the preview, not over it", async ({ page }) => {
  const before = await widths(page);
  expect(before.left, "the assets panel has no width").toBeGreaterThan(100);
  expect(before.right, "the properties panel has no width").toBeGreaterThan(100);

  // Docked means the three regions share the row. Overlaying would let the sum
  // exceed the workspace, which is the other design and not this one.
  const total = await page.evaluate(() =>
    document.querySelector(".cx-workspace").getBoundingClientRect().width);
  expect(before.left + before.right + before.main).toBeLessThanOrEqual(total + 1);
});

test("collapsing a panel gives its width to the preview", async ({ page }) => {
  const before = await widths(page);
  await regionToggle(page, "Assets").click();
  const after = await widths(page);

  expect(after.left).toBe(0);
  expect(after.main, "the preview did not grow into the freed space")
    .toBeGreaterThan(before.main + before.left - 2);
});

test("the toggle stays reachable once its panel is collapsed", async ({ page }) => {
  // The whole reason the toggle is outside the region it opens. In v4 the
  // control was inside it, the region became display:none, and the only way
  // back was clearing storage by hand.
  const toggle = regionToggle(page, "Assets");
  await toggle.click();
  await expect(toggle).toBeInViewport();
  await toggle.click();
  expect((await widths(page)).left).toBeGreaterThan(100);
});

test("a collapsed panel is still remembered after a reload", async ({ page }) => {
  await regionToggle(page, "Properties").click();
  expect((await widths(page)).right).toBe(0);

  await page.reload();
  await page.waitForFunction(() => window.FunPack !== undefined);
  expect((await widths(page)).right).toBe(0);

  // And it can still be brought back, which is what makes remembering safe.
  await regionToggle(page, "Properties").click();
  expect((await widths(page)).right).toBeGreaterThan(100);
});

test("the prompt sits under the preview and both have height", async ({ page }) => {
  // A vertical split inside a parent whose height came from its content gave
  // its second pane a basis of zero: the prompt was in the DOM, had no pixels,
  // and nothing reported anything.
  const boxes = await page.evaluate(() => {
    const panels = [...document.querySelectorAll(".cx-workspace-main .cx-panel")];
    return panels.map((p) => {
      const r = p.getBoundingClientRect();
      return { title: p.querySelector(".cx-panel-title")?.textContent, h: r.height, y: r.y };
    });
  });
  const preview = boxes.find((b) => b.title === "Preview");
  const timeline = boxes.find((b) => b.title === "Timeline");
  expect(preview?.h, "the preview has no height").toBeGreaterThan(50);
  expect(timeline?.h, "the timeline has no height").toBeGreaterThan(30);
  expect(timeline.y).toBeGreaterThan(preview.y);
});

// `hidden` has to actually hide.
//
// The UA sheet's `[hidden] { display: none }` is an element-level rule, so any
// class that sets a display beats it -- and nearly every element in this kit
// sets one. The Cancel button was marked hidden and stayed on screen, with
// nothing anywhere reporting a problem. jsdom cannot see this: it reports the
// attribute, which was always correct.
test("an element marked hidden is not on screen, whatever it is", async ({ page }) => {
  const kinds = await page.evaluate(async () => {
    const { composer } = await import("/funpack/app/composer/composer.js");
    const made = {
      button: composer.button.md({ label: "Go" }),
      panel: composer.panel.default({ title: "P", body: composer.hint.default({ text: "x" }) }),
      chip: composer.chip.neutral({ label: "c" }),
      progress: composer.progress.bar({ value: 1, max: 2 }),
    };
    const out = {};
    for (const [name, handle] of Object.entries(made)) {
      handle.node.setAttribute("data-probe", name);
      handle.node.setAttribute("hidden", "");
      document.body.appendChild(handle.node);
      out[name] = getComputedStyle(handle.node).display;
    }
    return out;
  });

  for (const [name, display] of Object.entries(kinds)) {
    expect(display, `a hidden ${name} is still displayed`).toBe("none");
  }
});

test("the transport reports a refusal beside Generate", async ({ page }) => {
  // The dev server has no ComfyUI behind it, so the pipeline genuinely cannot
  // run -- and that is the case worth seeing: the reason appears where the run
  // is started, not in a console nobody has open.
  // The real Generate button specifically: "Generate All" also matches a
  // plain text search for "Generate", and .cx-btn-primary is what actually
  // distinguishes the one this test means.
  await page.locator(".cx-panel-head button.cx-btn-primary").click();
  // Scoped to the Timeline zone specifically: the Preview zone has its own
  // .cx-panel-status now too (a save confirmation), and .first() stopped
  // being safe to assume once there was more than one on the page.
  const bar = page.locator(".cx-zone:has(button.cx-btn-primary) .cx-panel-status");
  await expect(bar).not.toHaveText(/^Ready/);
  // Whatever the reason is, it names the slot it is about. This used to read
  // "there is no node called X installed", because the dev server had no node
  // registry at all and every slot looked like a missing node; now the registry
  // is real and the reason is the genuine one -- nothing has been chosen yet.
  await expect(bar).toContainText(/^[a-z_]+: /);
});

// Narrow windows: the panels stop docking and overlay instead.
//
// Not a phone feature -- a 900px window had a 198px properties column in which
// a two-word label wrapped over three lines, and 375px put three columns side
// by side with clipped headings. The behaviour has to hold at both ends.

test("a narrow window opens with the panels out of the way", async ({ page }) => {
  await page.setViewportSize({ width: 420, height: 780 });
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  const w = await widths(page);
  expect(w.left, "a panel was docked in a window too narrow for it").toBe(0);
  expect(w.right).toBe(0);
  expect(w.main, "the centre did not get the whole window").toBeGreaterThan(300);
});

test("an overlaid panel does not cover the control that closes it", async ({ page }) => {
  // The v4 fault, arriving by a different route: the panel stopped taking its
  // own space and ended up over the control that closes it. The named toggles
  // live in the timeline head, which is exactly what an overlaid panel covers,
  // so at this width they are not offered at all -- the menu bar is, and
  // nothing can cover that.
  await page.setViewportSize({ width: 420, height: 780 });
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  await expect(regionToggle(page, "Properties"),
    "a toggle the panel will cover is still offered").toBeHidden();

  const toggle = menuToggle(page, "Properties");
  await toggle();
  expect((await widths(page)).right).toBeGreaterThan(200);
  await toggle();                             // and the way back is not covered
  expect((await widths(page)).right).toBe(0);
});

test("only one panel overlays at a time", async ({ page }) => {
  await page.setViewportSize({ width: 420, height: 780 });
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  await menuToggle(page, "Assets")();
  await menuToggle(page, "Properties")();
  const w = await widths(page);
  expect(w.left, "two overlays over one narrow centre").toBe(0);
  expect(w.right).toBeGreaterThan(200);
});

test("what was open comes back when there is room again", async ({ page }) => {
  await page.setViewportSize({ width: 1400, height: 900 });
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  expect((await widths(page)).left).toBeGreaterThan(200);

  await page.setViewportSize({ width: 420, height: 780 });
  expect((await widths(page)).left).toBe(0);

  await page.setViewportSize({ width: 1400, height: 900 });
  expect((await widths(page)).left,
    "the window getting small was taken as the user closing the panel").toBeGreaterThan(200);
});

test("a settings label is not squeezed into three lines at a square window", async ({ page }) => {
  // 1:1 is an ordinary window shape and was the worst case: the column was a
  // share of the viewport rather than a width its content could live in.
  await page.setViewportSize({ width: 900, height: 900 });
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  const right = (await widths(page)).right;
  expect(right, "the properties column is too narrow to read").toBeGreaterThanOrEqual(260);
});

test("a page loaded narrow gives both panels back when there is room", async ({ page }) => {
  // The asymmetry nothing caught. Loading narrow with both panels remembered
  // open runs the one-at-a-time rule during construction, which closes the
  // right one -- and the snapshot of "what to restore" was taken AFTER that, so
  // it recorded a panel the rule had just closed as a panel the user wanted
  // closed. Widening then brought back only the left, permanently, with nothing
  // the user did to explain it.
  await page.addInitScript(() => {
    window.localStorage.setItem("cx.ws.main.left", "1");
    window.localStorage.setItem("cx.ws.main.right", "1");
  });
  await page.setViewportSize({ width: 420, height: 780 });
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  const narrow = await widths(page);
  expect(narrow.left, "a panel was docked in a window too narrow for it").toBe(0);
  expect(narrow.right).toBe(0);

  await page.setViewportSize({ width: 1280, height: 900 });
  await page.waitForFunction(() => {
    const el = document.querySelector(".cx-workspace-right");
    return el && el.getBoundingClientRect().width > 0;
  }, null, { timeout: 3000 }).catch(() => {});

  const wide = await widths(page);
  expect(wide.left, "the left panel did not come back").toBeGreaterThan(0);
  expect(wide.right, "the right panel never came back").toBeGreaterThan(0);
});

test("a setting too narrow to sit beside its control stacks instead of shredding", async ({ page }) => {
  // What a docked panel does to a settings row: the control keeps a 140px floor
  // and takes it out of the label, so at panel width the label came out one
  // word per line and its hint four characters wide. It reads as damage rather
  // than as a layout, and no jsdom test can see it -- there are no widths there.
  await page.setViewportSize({ width: 800, height: 700 });
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  // Reveal the settings a switch brings with it: those are the wordy ones. The
  // face, not the input -- the input is visually hidden and the label draws the
  // box, which is how a real click arrives too.
  await page.locator(".cx-panel .cx-check-face").first().click();
  const row = page.locator(".cx-settings-row").first();
  await expect(row).toBeVisible();

  const { rowWidth, textWidth, labelHeight, lineHeight } = await row.evaluate((node) => {
    const text = node.querySelector(".cx-settings-text") || node;
    const label = node.querySelector(".cx-settings-label") || text;
    return {
      rowWidth: node.getBoundingClientRect().width,
      textWidth: text.getBoundingClientRect().width,
      labelHeight: label.getBoundingClientRect().height,
      lineHeight: parseFloat(getComputedStyle(label).lineHeight) || 16,
    };
  });

  expect(textWidth, "the label was squeezed out of the row by its control")
    .toBeGreaterThan(rowWidth * 0.75);
  expect(labelHeight, "a two-word label wrapped onto more than two lines")
    .toBeLessThanOrEqual(lineHeight * 2 + 2);
});
