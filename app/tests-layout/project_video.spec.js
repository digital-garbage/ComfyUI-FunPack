// What the project generates at, end to end against the real server.
//
// jsdom covers the store and the controls. What only a browser and a real
// pipeline can answer is whether the values a person types reach the graph --
// and whether they are still there on the next visit.

import { test, expect } from "@playwright/test";

// Serial: these share one real project store on the dev server, and the config
// runs tests in a file in parallel. Two of them making and switching projects at
// once is a race over the same JSON directory, not a bug in the app.
test.describe.configure({ mode: "serial" });

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
  // may well be called Untitled. Three fixed items ahead of the project list --
  // New project, Save Project File, Load Project File -- then the newest first.
  await page.getByRole("button", { name: "File" }).click();
  const items = page.getByRole("menuitem");
  await expect(items.nth(3)).toHaveText(new RegExp(name));   // the one just made, ticked
  await items.nth(4).click();

  await expect.poll(() => page.evaluate(() => window.FunPack.project.project.id)).not.toBe(made);
});

test("switching projects switches what the run will use, not just what is stored", async ({ page }) => {
  // The store's own getter reported the new project's numbers while the controls
  // that actually produce them still held the old ones -- so a Generate right
  // after a switch ran at the previous project's size, and the only place that
  // was visible was a window nobody had a reason to reopen.
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  // Two projects, each with a size of its own.
  const make = async (name, width) => {
    await page.evaluate(async (n) => { await window.FunPack.project.newProject(n); }, name);
    await openConstructor(page);
    await widthBox(page).fill(String(width));
    await widthBox(page).blur();
    await page.locator(".cx-modal").getByRole("button", { name: "Done" }).click();
    return page.evaluate(async () => {
      await window.FunPack.project.flush();
      return window.FunPack.project.project.id;
    });
  };
  const a = await make(`A ${Date.now()}`, 900);
  const b = await make(`B ${Date.now()}`, 640);

  await page.evaluate((id) => window.FunPack.project.open(id), a);
  await expect.poll(() => page.evaluate(() => window.FunPack.project.video.width)).toBe(900);
  await expect.poll(() => page.evaluate(() => window.FunPack.prompts().latent.width),
    { message: "the run would use the other project's width" }).toBe(900);

  await page.evaluate((id) => window.FunPack.project.open(id), b);
  await expect.poll(() => page.evaluate(() => window.FunPack.project.video.width)).toBe(640);
  await expect.poll(() => page.evaluate(() => window.FunPack.prompts().latent.width),
    { message: "the run would use the other project's width" }).toBe(640);

  // And the control a person looks at agrees with what the run would send.
  await openConstructor(page);
  await expect(widthBox(page)).toHaveValue("640");
});

test("a project that never touched width is not left showing the last project's", async ({ page }) => {
  // syncVideo() only pushed a project's OWN values into the controls -- a
  // project whose video dict is still {} took neither branch, so the control
  // (and so overrides(), and so the run) kept whatever the PREVIOUS project had
  // set. A brand-new project is exactly that state.
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  await page.evaluate(async (n) => { await window.FunPack.project.newProject(n); }, `A ${Date.now()}`);
  await openConstructor(page);
  await widthBox(page).fill("900");
  await widthBox(page).blur();
  await page.locator(".cx-modal").getByRole("button", { name: "Done" }).click();
  await page.evaluate(() => window.FunPack.project.flush());
  await expect.poll(() => page.evaluate(() => window.FunPack.prompts().latent.width)).toBe(900);

  // A second project that never touches width at all.
  await page.evaluate(async (n) => { await window.FunPack.project.newProject(n); }, `B ${Date.now()}`);
  await expect.poll(() => page.evaluate(() => window.FunPack.project.video.width)).toBe(undefined);

  // The pipeline's OWN declared default (modules/system/pipeline: width 512),
  // not the raw node's widget-schema default (FunPackEmptyLatent's own is
  // 768) -- entry.default silently used the node's schema default once, which
  // is a different, wrong number. Not just "not 900": the actual right value.
  const fallback = await page.evaluate(async () => {
    const r = await fetch("/funpack/api/pipeline");
    const slots = (await r.json()).slots || [];
    return (slots.find((s) => s.id === "latent") || {}).inputs?.width;
  });
  await expect.poll(() => page.evaluate(() => window.FunPack.prompts().latent.width),
    { message: "the run would still use the other project's width, or the wrong default" }).toBe(fallback);
  await openConstructor(page);
  await expect(widthBox(page)).toHaveValue(String(fallback));
});

test("an edit can be taken back from the menu and from the keyboard", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  const scenes = () => page.evaluate(() => window.FunPack.project.scenes.length);
  const before = await scenes();

  await page.getByRole("button", { name: "Edit" }).click();
  await page.getByRole("menuitem", { name: "Add scene" }).click();
  expect(await scenes()).toBe(before + 1);

  await page.getByRole("button", { name: "Edit" }).click();
  await page.getByRole("menuitem", { name: "Undo" }).click();
  expect(await scenes(), "the menu did not undo").toBe(before);

  // And the shortcut anyone reaches for first.
  await page.getByRole("button", { name: "Edit" }).click();
  await page.getByRole("menuitem", { name: "Add scene" }).click();
  expect(await scenes()).toBe(before + 1);
  await page.keyboard.press("ControlOrMeta+z");
  expect(await scenes(), "the keyboard did not undo").toBe(before);
  await page.keyboard.press("ControlOrMeta+Shift+z");
  expect(await scenes(), "redo did not put it back").toBe(before + 1);
});

test("Undo is not offered when there is nothing behind it", async ({ page }) => {
  // A menu that always offers it teaches people it does nothing.
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  await page.getByRole("button", { name: "Edit" }).click();
  await expect(page.getByRole("menuitem", { name: "Undo" })).toBeDisabled();
  await page.keyboard.press("Escape");

  await page.getByRole("button", { name: "Edit" }).click();
  await page.getByRole("menuitem", { name: "Add scene" }).click();
  await page.getByRole("button", { name: "Edit" }).click();
  await expect(page.getByRole("menuitem", { name: "Undo" })).toBeEnabled();
});

test("typing in a box keeps its own undo", async ({ page }) => {
  // Taking the shortcut over inside a field means one keystroke un-typing a
  // paragraph instead of a word.
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  const scenes = await page.evaluate(() => window.FunPack.project.scenes.length);

  await page.locator(".cx-panel-head").getByRole("button", { name: "Constructor" }).click();
  const box = page.locator(".cx-modal textarea").first();
  await box.click();
  await page.keyboard.press("ControlOrMeta+z");

  expect(await page.evaluate(() => window.FunPack.project.scenes.length),
    "the project was undone from inside a text box").toBe(scenes);
});
