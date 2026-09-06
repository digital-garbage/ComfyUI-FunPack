// Updating, against the real routes.
//
// The dev server IS a git checkout of this repo, so the status route answers
// truthfully and the window can be read the way a person would read it. What is
// NOT exercised here is a real pull or a real restart: those change the checkout
// the tests are running from. Every path that would do that is stubbed at the
// fetch, and the stub is what proves the request was the right one.

import { test, expect } from "@playwright/test";

const openUpdates = async (page) => {
  await page.getByRole("button", { name: "Settings" }).click();
  await page.getByRole("menuitem", { name: /Updates/ }).click();
  await expect(page.locator(".cx-modal")).toBeVisible();
};

test("the window says which branch you are on and whether there is anything new", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await openUpdates(page);

  const modal = page.locator(".cx-modal");
  // Read from the real checkout, not from a fixture: this repo is on a branch
  // and the route reports it.
  const branch = await page.evaluate(async () => {
    const r = await fetch("/funpack/api/git/status");
    return (await r.json()).branch;
  });
  await expect(modal).toContainText(branch);
  await expect(modal.getByText(/Up to date|update.? available|out of date/i)).toBeVisible();
});

test("a checkout with local changes says so before anything is pressed", async ({ page }) => {
  // The one state that blocks both actions. A refusal after the press would be
  // a surprise; this is the same fact, said first.
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/git/status", (route) => route.fulfill({
    json: { ok: true, version: "5.0", branch: "v5", branches: ["v5", "dev"],
            dirty: true, ahead: 0, behind: 2, fetch_ok: true, repo: "/x" },
  }));
  await openUpdates(page);

  const modal = page.locator(".cx-modal");
  await expect(modal).toContainText(/local changes/i);
  await expect(modal.getByRole("button", { name: /Update/ })).toBeDisabled();
});

test("update asks the server to update, and then waits for it to come back", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/git/status", (route) => route.fulfill({
    json: { ok: true, version: "5.0", branch: "v5", branches: ["v5", "dev"],
            dirty: false, ahead: 0, behind: 3, fetch_ok: true, repo: "/x" },
  }));

  const asked = [];
  await page.route("**/api/git/update", (route) => {
    asked.push(route.request().method());
    return route.fulfill({ json: { restarting: true, updated: true } });
  });
  // The page would reload on a healthy answer; keep the server "down" so the
  // overlay stays up and can be read.
  await page.route("**/api/health", (route) => route.abort());

  await openUpdates(page);
  await page.getByRole("button", { name: /Update \(3\)/ }).click();

  await expect.poll(() => asked).toEqual(["POST"]);
  await expect(page.locator(".cx-blocking")).toBeVisible();
  await expect(page.locator(".cx-blocking")).toContainText(/Restarting/i);
});

test("a refusal from git is shown in the window, not swallowed", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/git/status", (route) => route.fulfill({
    json: { ok: true, version: "5.0", branch: "v5", branches: ["v5", "dev"],
            dirty: false, ahead: 0, behind: 1, fetch_ok: true, repo: "/x" },
  }));
  await page.route("**/api/git/update", (route) => route.fulfill({
    status: 400, json: { detail: "Working tree has local changes." },
  }));

  await openUpdates(page);
  await page.getByRole("button", { name: /Update/ }).click();

  await expect(page.locator(".cx-modal")).toContainText("Working tree has local changes.");
  await expect(page.locator(".cx-blocking"), "it waited for a restart that was refused")
    .toHaveCount(0);
});

test("an install that is not a git checkout says so instead of offering to update", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/git/status", (route) => route.fulfill({
    json: { ok: false, version: "5.0", detail: "FunPack is not a git checkout (no .git directory)." },
  }));

  await openUpdates(page);
  await expect(page.locator(".cx-modal")).toContainText(/not a git checkout/i);
  await expect(page.locator(".cx-modal").getByRole("button", { name: /Update/ })).toBeDisabled();
});
