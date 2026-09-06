// The timeline strip: drag to reorder, zoom to change clip width. jsdom
// covers the wiring against a fake store; this is a real drag in a real
// browser against the real store.

import { test, expect } from "@playwright/test";

test.describe.configure({ mode: "serial" });

/** A real HTML5 drag: dragstart on `from`, dragover + drop on `to`. */
async function dragCell(page, from, to) {
  await page.evaluate(([fromSel, toSel]) => {
    const source = document.querySelectorAll(".cx-strip-cell")[fromSel];
    const target = document.querySelectorAll(".cx-strip-cell")[toSel];
    const dt = new DataTransfer();
    source.dispatchEvent(new DragEvent("dragstart", { bubbles: true, dataTransfer: dt }));
    target.dispatchEvent(new DragEvent("dragover", { bubbles: true, cancelable: true, dataTransfer: dt }));
    target.dispatchEvent(new DragEvent("drop", { bubbles: true, cancelable: true, dataTransfer: dt }));
  }, [from, to]);
}

test("dragging a clip to a new spot reorders the project's scenes", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  const ids = await page.evaluate(() => {
    const { project } = window.FunPack;
    while (project.scenes.length < 3) project.addScene();
    return project.scenes.slice(0, 3).map((s) => s.id);
  });

  await expect(page.locator(".cx-strip-cell")).toHaveCount(ids.length);
  await dragCell(page, 0, 2);

  const after = await page.evaluate(() => window.FunPack.project.scenes.map((s) => s.id));
  expect(after.slice(0, 3)).toEqual([ids[1], ids[2], ids[0]]);
});

test("dropping a clip on itself changes nothing", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  const before = await page.evaluate(() => {
    const { project } = window.FunPack;
    while (project.scenes.length < 2) project.addScene();
    return project.scenes.map((s) => s.id);
  });

  await dragCell(page, 0, 0);

  const after = await page.evaluate(() => window.FunPack.project.scenes.map((s) => s.id));
  expect(after).toEqual(before);
});

test("zoom changes clip width and survives a reload", async ({ page }) => {
  // Cells also carry flex-grow (proportional to a scene's length), which can
  // dominate rendered width when there is spare row to fill -- the variable
  // zoom actually sets is the honest thing to check, not the geometry that
  // competes with it.
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.evaluate(() => { while (window.FunPack.project.scenes.length < 1) window.FunPack.project.addScene(); });

  await page.getByRole("radio", { name: "L", exact: true }).click();
  await expect.poll(() => page.locator(".cx-strip").evaluate((el) => el.style.getPropertyValue("--strip-w")))
    .toBe("128px");

  await page.reload();
  await page.waitForFunction(() => window.FunPack !== undefined);
  await expect(page.getByRole("radio", { name: "L", exact: true })).toHaveAttribute("aria-checked", "true");
  await expect.poll(() => page.locator(".cx-strip").evaluate((el) => el.style.getPropertyValue("--strip-w")))
    .toBe("128px");
});
