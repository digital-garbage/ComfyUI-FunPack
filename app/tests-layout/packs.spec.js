// Node packs, against the real routes.
//
// The dev server puts ComfyUI on the path, so the listing is the real
// custom_nodes directory. Nothing here installs, updates or removes anything:
// those change the machine the tests run on, so each is stubbed at the fetch,
// and the stub is what proves the right request went out.

import { test, expect } from "@playwright/test";

const openPacks = async (page) => {
  await page.getByRole("button", { name: "Settings" }).click();
  await page.getByRole("menuitem", { name: /Node packs/ }).click();
  await expect(page.locator(".cx-modal")).toBeVisible();
};

test("the list is the real custom_nodes directory, and says so", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await openPacks(page);

  const real = await page.evaluate(async () => {
    const r = await fetch("/funpack/api/packs");
    return r.json();
  });
  await expect(page.locator(".cx-modal-foot")).toContainText(real.root);
  await expect(page.locator(".cx-modal")).toContainText(`${real.nodes.length} installed`);
});

test("FunPack itself is listed without a way to delete it", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/packs", (route) => route.fulfill({
    json: { root: "/x/custom_nodes", nodes: [
      { name: "ComfyUI-FunPack", is_funpack: true, git: true, branch: "dev", commit: "abc1234" },
      { name: "ComfyUI-Other", is_funpack: false, git: true, branch: "main", commit: "def5678" },
    ] },
  }));
  await openPacks(page);

  const mine = page.locator(".cx-settings-row", { hasText: "ComfyUI-FunPack" });
  await expect(mine).toContainText("this is FunPack");
  await expect(mine.getByRole("button", { name: "Remove" })).toHaveCount(0);
  await expect(page.locator(".cx-settings-row", { hasText: "ComfyUI-Other" })
    .getByRole("button", { name: "Remove" })).toHaveCount(1);
});

test("a pack that is not a git checkout is not offered an update", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/packs", (route) => route.fulfill({
    json: { root: "/x/custom_nodes", nodes: [{ name: "Hand-Copied", is_funpack: false, git: false }] },
  }));
  await openPacks(page);

  const row = page.locator(".cx-settings-row", { hasText: "Hand-Copied" });
  await expect(row).toContainText("not a git checkout");
  await expect(row.getByRole("button", { name: "Update" })).toHaveCount(0);
});

test("installing sends the URL that was typed, and nothing when it is empty", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/packs", (route) => route.fulfill({
    json: { root: "/x/custom_nodes", nodes: [] },
  }));
  const sent = [];
  await page.route("**/api/packs/install", async (route) => {
    sent.push(route.request().postDataJSON());
    return route.fulfill({ json: { name: "pack" } });
  });
  await openPacks(page);

  // Empty: refused here, not sent to a server to be refused there.
  await page.getByRole("button", { name: "Install" }).click();
  await expect(page.locator(".cx-modal")).toContainText("Paste a git URL first.");
  expect(sent).toEqual([]);

  await page.locator(".cx-modal input[type='text']").fill("https://github.com/o/p");
  await page.getByRole("button", { name: "Install" }).click();
  await expect.poll(() => sent).toEqual([{ url: "https://github.com/o/p" }]);
});

test("a refusal from the server is shown in the window", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/packs", (route) => route.fulfill({
    json: { root: "/x/custom_nodes", nodes: [
      { name: "ComfyUI-Other", is_funpack: false, git: true, branch: "main" }] },
  }));
  await page.route("**/api/packs/remove", (route) => route.fulfill({
    status: 400, json: { detail: "That is FunPack itself." },
  }));
  await openPacks(page);

  await page.locator(".cx-settings-row", { hasText: "ComfyUI-Other" })
    .getByRole("button", { name: "Remove" }).click();
  await expect(page.locator(".cx-modal")).toContainText("That is FunPack itself.");
});

test("checking for updates shows which packs are behind", async ({ page }) => {
  // The client read `nodes` off a response whose key is `checked`: the request
  // succeeded, the JSON was valid, and every pack reported nothing to update
  // however far behind it was. The stub below is the server's real shape.
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/packs", (route) => route.fulfill({
    json: { root: "/x/custom_nodes", nodes: [
      { name: "ComfyUI-Behind", is_funpack: false, git: true, branch: "main", commit: "aaa" },
      { name: "ComfyUI-Current", is_funpack: false, git: true, branch: "main", commit: "bbb" },
    ] },
  }));
  await page.route("**/api/packs/check", (route) => route.fulfill({
    json: { checked: {
      "ComfyUI-Behind": { checked: true, branch: "main", ahead: 0, behind: 3 },
      "ComfyUI-Current": { checked: true, branch: "main", ahead: 0, behind: 0 },
    } },
  }));

  await openPacks(page);
  await page.getByRole("button", { name: "Check for updates" }).click();

  await expect(page.locator(".cx-settings-row", { hasText: "ComfyUI-Behind" }))
    .toContainText("3 updates available");
  await expect(page.locator(".cx-settings-row", { hasText: "ComfyUI-Current" }))
    .toContainText("up to date");
});

test("a pack that could not be checked says why, not 'up to date'", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/packs", (route) => route.fulfill({
    json: { root: "/x/custom_nodes", nodes: [
      { name: "ComfyUI-Offline", is_funpack: false, git: true, branch: "main" }] },
  }));
  await page.route("**/api/packs/check", (route) => route.fulfill({
    json: { checked: { "ComfyUI-Offline": { checked: false, reason: "could not reach origin" } } },
  }));

  await openPacks(page);
  await page.getByRole("button", { name: "Check for updates" }).click();
  await expect(page.locator(".cx-settings-row", { hasText: "ComfyUI-Offline" }))
    .toContainText("could not reach origin");
});
