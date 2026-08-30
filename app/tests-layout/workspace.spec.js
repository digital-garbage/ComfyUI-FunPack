// Docking, measured.
//
// jsdom can say a class was toggled and an attribute changed; it cannot say the
// preview actually got the space, because getBoundingClientRect is zeros there.
// The v4 fault this shape exists to avoid was invisible for exactly that reason:
// the state was right and the pixels were not.

import { test, expect } from "@playwright/test";

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
  await page.locator('.cx-workspace-rail-left button').click();
  const after = await widths(page);

  expect(after.left).toBe(0);
  expect(after.main, "the preview did not grow into the freed space")
    .toBeGreaterThan(before.main + before.left - 2);
});

test("the toggle stays reachable once its panel is collapsed", async ({ page }) => {
  // The whole reason the rail is a sibling of the panel. In v4 the control was
  // inside the region it hid, the region became display:none, and the only way
  // back was clearing storage by hand.
  const toggle = page.locator('.cx-workspace-rail-left button');
  await toggle.click();
  await expect(toggle).toBeInViewport();
  await toggle.click();
  expect((await widths(page)).left).toBeGreaterThan(100);
});

test("a collapsed panel is still remembered after a reload", async ({ page }) => {
  await page.locator('.cx-workspace-rail-right button').click();
  expect((await widths(page)).right).toBe(0);

  await page.reload();
  await page.waitForFunction(() => window.FunPack !== undefined);
  expect((await widths(page)).right).toBe(0);

  // And it can still be brought back, which is what makes remembering safe.
  await page.locator('.cx-workspace-rail-right button').click();
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
  const prompt = boxes.find((b) => b.title === "Prompt");
  expect(preview?.h, "the preview has no height").toBeGreaterThan(50);
  expect(prompt?.h, "the prompt has no height").toBeGreaterThan(30);
  expect(prompt.y).toBeGreaterThan(preview.y);
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
  await page.locator('.cx-action-bar button:has-text("Generate")').click();
  const bar = page.locator(".cx-action-bar");
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
  // own space and the rail ended up underneath it.
  await page.setViewportSize({ width: 420, height: 780 });
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  const toggle = page.locator(".cx-workspace-rail-right button");
  await toggle.click();
  expect((await widths(page)).right).toBeGreaterThan(200);
  await expect(toggle).toBeInViewport();
  await toggle.click();                       // and it still works
  expect((await widths(page)).right).toBe(0);
});

test("only one panel overlays at a time", async ({ page }) => {
  await page.setViewportSize({ width: 420, height: 780 });
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  await page.locator(".cx-workspace-rail-left button").click();
  await page.locator(".cx-workspace-rail-right button").click();
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
