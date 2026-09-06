// The temp browser, which is a place to FIND something rather than manage it.

import { test, expect } from "@playwright/test";

const openTemp = async (page) => {
  await page.getByRole("button", { name: "Settings" }).click();
  await page.getByRole("menuitem", { name: /Temp files/ }).click();
  await expect(page.locator(".cx-modal")).toBeVisible();
};

const FILES = {
  path: "/x/temp",
  detail: "",
  files: [
    { filename: "preview_00002_.mp4", subfolder: "runs", kind: "video", size: 2_400_000, mtime: 2000 },
    { filename: "preview_00001_.png", subfolder: "", kind: "image", size: 51_200, mtime: 1000 },
  ],
};

test("it lists what is there, with size and kind, newest first", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/temp**", (route) => route.fulfill({ json: FILES }));
  await openTemp(page);

  const names = page.locator(".cx-media-row-name");
  await expect(names).toHaveText(["preview_00002_.mp4", "preview_00001_.png"]);
  await expect(page.locator(".cx-modal")).toContainText("2.3 MB");
  await expect(page.locator(".cx-modal-foot")).toContainText("/x/temp");
});

test("a video is not handed to an img", async ({ page }) => {
  // Chrome's six-connections-per-origin pool and live <video> thumbnails wedged
  // this app's API once already.
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/temp**", (route) => route.fulfill({ json: FILES }));
  await openTemp(page);

  await expect(page.locator(".cx-modal video")).toHaveCount(0);
  const sources = await page.locator(".cx-modal img").evaluateAll(
    (imgs) => imgs.map((i) => i.getAttribute("src")));
  expect(sources.filter((s) => /\.mp4/.test(s))).toEqual([]);
});

test("opening one shows it in the Preview and closes the window", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/temp**", (route) => route.fulfill({ json: FILES }));
  await openTemp(page);

  // What the viewer FETCHES is the evidence: the dev server answers 404 for
  // /view, so the picture never arrives and the kit replaces it with its
  // fallback -- which is correct, and leaves nothing in the DOM to assert on.
  const asked = [];
  await page.route("**/view**", (route) => {
    asked.push(route.request().url());
    return route.fulfill({ status: 404, body: "" });
  });

  await page.locator(".cx-media-row", { hasText: "preview_00001_.png" }).click();
  await expect(page.locator(".cx-modal")).toHaveCount(0);

  await expect.poll(() => asked.length).toBeGreaterThan(0);
  const url = asked.at(-1);
  expect(url).toContain("preview_00001_.png");
  expect(url, "it was fetched from the output directory, not temp").toContain("type=temp");
});

test("an empty directory says why, and no ComfyUI says something else", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  await page.route("**/api/temp**", (route) => route.fulfill({
    json: { files: [], path: "/x/temp", detail: "Nothing here. Temp files are wiped when ComfyUI restarts." },
  }));
  await openTemp(page);
  await expect(page.locator(".cx-modal")).toContainText("wiped when ComfyUI restarts");
  await page.locator(".cx-modal-foot").getByRole("button", { name: "Close" }).click();

  await page.unroute("**/api/temp**");
  await page.route("**/api/temp**", (route) => route.fulfill({
    json: { files: [], path: null, detail: "ComfyUI is not here to ask where its temp files go." },
  }));
  await openTemp(page);
  await expect(page.locator(".cx-modal")).toContainText("ComfyUI is not here");
});
