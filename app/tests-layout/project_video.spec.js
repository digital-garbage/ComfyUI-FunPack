// What the project generates at, end to end against the real server.
//
// jsdom covers the store and the controls. What only a browser and a real
// pipeline can answer is whether the values a person types reach the graph --
// and whether they are still there on the next visit.

import { test, expect } from "@playwright/test";

const openConstructor = (page) =>
  page.locator(".cx-panel-head").getByRole("button", { name: "Constructor" }).click();

const widthBox = (page) => page.locator('.cx-modal input[type="number"]').first();

test("the size the project generates at is typed once and used by the run", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await openConstructor(page);

  // The pipeline says which of its inputs belong here; the app offers the place.
  await expect(page.locator(".cx-modal").getByText("Width")).toBeVisible();
  await expect(page.locator('.cx-modal input[type="number"]')).toHaveCount(3);

  await widthBox(page).fill("832");
  await widthBox(page).blur();

  const sent = await page.evaluate(async () => {
    const body = JSON.stringify({ inputs: window.FunPack.prompts() });
    const res = await fetch("/funpack/api/pipeline",
      { method: "POST", headers: { "Content-Type": "application/json" }, body });
    return res.json();
  });
  const latent = sent.slots.find((s) => s.id === "latent");
  expect(sent.refused).toEqual([]);
  expect(latent.inputs.width).toBe(832);
  expect(latent.inputs.model, "the wiring was replaced by a value").toEqual(["model", 0]);
});

test("it is still what the project generates at on the next visit", async ({ page }) => {
  // The whole point of it being the PROJECT's: a scene regenerated tomorrow
  // comes back at the same size, not at whatever the pipeline defaults to.
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await openConstructor(page);
  await widthBox(page).fill("640");
  await widthBox(page).blur();
  const id = await page.evaluate(async () => {
    await window.FunPack.project.flush();
    return window.FunPack.project.project.id;
  });

  await page.reload();
  await page.waitForFunction(() => window.FunPack !== undefined);
  // Opened by id: another test in this file makes projects, and which one a
  // fresh page picks is "the most recent", not "the one this test used".
  await page.evaluate((pid) => window.FunPack.project.open(pid), id);
  await expect.poll(() => page.evaluate(() => window.FunPack.project.video.width)).toBe(640);

  await openConstructor(page);
  await expect(widthBox(page)).toHaveValue("640");
});

test("what a run produces lands on the scene it was started for", async ({ page }) => {
  // The loop that makes this an app rather than a generate button: the scene is
  // what was generated, so the picture belongs to it and is still there after a
  // reload. The dev server has no /prompt, so the run is driven directly.
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  const scene = await page.evaluate(async () => {
    const { project, run } = window.FunPack;
    const id = project.selectedId || project.addScene().id;
    // What the Generate button does, then what the socket says when it ends.
    document.querySelector('.cx-panel-head button.cx-btn-primary').click();
    // The dev server serves no /prompt, so the run never gets an id of its own.
    // Adopting one is what a reload does, and it is the same path from here on.
    await new Promise((r) => setTimeout(r, 50));
    run.adopt("pretend-prompt-id");
    run.handle({ type: "executed", data: { prompt_id: run.state.promptId,
      output: { images: [{ filename: "scene.png", subfolder: "", type: "output" }] } } });
    run.handle({ type: "execution_success", data: { prompt_id: run.state.promptId } });
    await new Promise((r) => setTimeout(r, 50));
    return { id, result: (project.scenes.find((s) => s.id === id) || {}).result };
  });

  expect(scene.result, "the run went nowhere").toMatch(/scene\.png/);
  // Nothing about the picture ARRIVING is asserted here: the dev server answers
  // 404 for every /view, so the strip shows its glyph -- which is the fallback
  // doing its job, not a missing result.
});

test("a project can be made and switched to from the File menu", async ({ page }) => {
  // The store could do all of this from the day it arrived; there was no way in.
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  // A name of its own: the dev server keeps real project files, so what is
  // already in the store is whatever earlier runs left there.
  const name = `Made ${Date.now()}`;
  await page.getByRole("button", { name: "File" }).click();
  await page.getByRole("menuitem", { name: /New project/ }).click();
  await page.locator(".cx-modal input").fill(name);
  await page.locator(".cx-modal").getByRole("button", { name: "Create" }).click();

  await expect.poll(() => page.evaluate(() => window.FunPack.project.project.name)).toBe(name);
  const made = await page.evaluate(() => window.FunPack.project.project.id);

  // And away again. By position, not by name: every other project in the store
  // may well be called Untitled.
  await page.getByRole("button", { name: "File" }).click();
  const items = page.getByRole("menuitem");
  await expect(items.nth(1)).toHaveText(new RegExp(name));   // the one just made, ticked
  await items.nth(2).click();

  await expect.poll(() => page.evaluate(() => window.FunPack.project.project.id)).not.toBe(made);
});
