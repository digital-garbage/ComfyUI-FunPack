// The log panel, against the real route.

import { test, expect } from "@playwright/test";

const openLog = async (page) => {
  await page.getByRole("button", { name: "Settings" }).click();
  await page.getByRole("menuitem", { name: /ComfyUI log/ }).click();
  await expect(page.locator(".cx-modal")).toBeVisible();
};

test("the log shows real lines and says which file they came from", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  const real = await page.evaluate(async () => (await fetch("/funpack/api/log?limit=5")).json());
  await openLog(page);

  if (real.path) {
    await expect(page.locator(".cx-modal-foot")).toContainText(real.path);
    await expect(page.locator(".cx-code-block")).toContainText(real.lines.at(-1).slice(0, 30));
  } else {
    await expect(page.locator(".cx-code-block")).toContainText(real.detail);
  }
});

test("no log file reads as a reason, not as a quiet log", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/log**", (route) => route.fulfill({
    json: { lines: [], path: null,
            detail: "ComfyUI is not writing a log file here, so there is nothing to show." },
  }));

  await openLog(page);
  await expect(page.locator(".cx-code-block")).toContainText("not writing a log file");
});

test("it keeps the last lines when the server goes away", async ({ page }) => {
  // The most interesting thing a log can say is that the server just died --
  // and replacing the lines from just before it went with an error message
  // throws away the only evidence of why.
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/log**", (route) => route.fulfill({
    json: { lines: ["the line before it went"], path: "/x/comfyui.log", detail: "" },
  }));
  await openLog(page);
  await expect(page.locator(".cx-code-block")).toContainText("the line before it went");

  await page.unroute("**/api/log**");
  await page.route("**/api/log**", (route) => route.abort());
  await expect(page.locator(".cx-modal-foot")).toContainText(/Cannot reach ComfyUI/i);
  await expect(page.locator(".cx-code-block"), "the last lines were thrown away")
    .toContainText("the line before it went");
});

test("it stops redrawing while text is being selected", async ({ page }) => {
  // Copying a stack trace is the one thing anybody opens a log to do, and a
  // panel that redraws under the drag makes it impossible.
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  let served = 0;
  await page.route("**/api/log**", (route) => {
    served += 1;
    return route.fulfill({ json: { lines: [`line ${served}`], path: "/x", detail: "" } });
  });

  await openLog(page);
  await expect(page.locator(".cx-code-block")).toContainText("line 1");

  await page.evaluate(() => {
    const pre = document.querySelector(".cx-code-block");
    const range = document.createRange();
    range.selectNodeContents(pre);
    const sel = getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
  });
  const held = await page.locator(".cx-code-block").textContent();
  await page.waitForTimeout(2600);          // more than one poll
  expect(await page.locator(".cx-code-block").textContent(),
    "the panel redrew under a selection").toBe(held);
});
