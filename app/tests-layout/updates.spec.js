// Updating, against the real routes.
//
// The dev server IS a git checkout of this repo, so the status route answers
// truthfully and the window can be read the way a person would read it. What is
// NOT exercised here is a real pull or a real restart: those change the checkout
// the tests are running from. Every path that would do that is stubbed at the
// fetch, and the stub is what proves the request was the right one.

import { test, expect } from "@playwright/test";

// Skipped: / now serves the ported v4 frontend (app/legacy/), not composer --
// these specs assert on .cx-workspace*/window.FunPack, which no longer exist
// on the served page. Pending rewrite against the v4-derived DOM as part of
// the v4-UI-onto-v5-backend port's pixel-exactness verification step (see
// the plan file). Not deleted: the assertions are still the right SHAPE of
// check, just against selectors this page no longer has.
test.skip(true, "composer retired in favor of the v4 UI port -- rewrite against app/legacy's DOM");

const openUpdates = async (page) => {
  await page.getByRole("button", { name: "Settings" }).click();
  await page.locator(".cx-filter-row", { hasText: "Updates" }).click();
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
  // Update alone still blocks on a dirty tree -- pulling new commits onto
  // changes in progress needs asking first. A refusal after the press would
  // be a surprise; this is the same fact, said first.
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

test("a dirty tree does not block switching branch or rolling back -- those auto-stash", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/git/status", (route) => route.fulfill({
    json: { ok: true, version: "5.0", branch: "v5", branches: ["v5", "dev"],
            dirty: true, ahead: 0, behind: 2, fetch_ok: true, repo: "/x",
            rollback_target: { commit: "abc12345", subject: "prior" } },
  }));
  await openUpdates(page);

  const modal = page.locator(".cx-modal");
  await expect(modal).toContainText(/stash.*automatically/i);
  await expect(modal.getByLabel("Branch")).toBeEnabled();
  await expect(modal.getByRole("button", { name: "Roll back" })).toBeEnabled();
  // The one action a dirty tree still stops.
  await expect(modal.getByRole("button", { name: /Update/ })).toBeDisabled();
});

test("switching branch on a dirty tree stashes first, and says so", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/git/status", (route) => route.fulfill({
    json: { ok: true, version: "5.0", branch: "v5", branches: ["v5", "dev"],
            dirty: true, ahead: 0, behind: 0, fetch_ok: true, repo: "/x" },
  }));
  await page.route("**/api/git/checkout", (route) => route.fulfill({
    json: { restarting: true, updated: true, branch: "dev",
            stashed: "FunPack: auto-stashed before switching to dev" },
  }));
  await page.route("**/api/health", (route) => route.abort());

  await openUpdates(page);
  await page.getByLabel("Branch").selectOption("dev");

  await expect(page.locator(".cx-toast-warn")).toContainText(/stashed first/i);
  await expect(page.locator(".cx-blocking")).toContainText(/Restarting/i);
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

test("an update that changed nothing does not wait for a restart", async ({ page }) => {
  // The server only restarts when the checkout moved. Waiting for one that is
  // not coming is three minutes of an overlay.
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/git/status**", (route) => route.fulfill({
    json: { ok: true, version: "5.0", branch: "v5", branches: ["v5"], dirty: false,
            ahead: 0, behind: 0, fetch_ok: true, checked_remote: true, repo: "/x" },
  }));
  await page.route("**/api/git/update", (route) => route.fulfill({
    json: { restarting: false, updated: false },
  }));

  await openUpdates(page);
  await page.getByRole("button", { name: /Update/ }).click();

  await expect(page.locator(".cx-blocking"), "it waited for a restart that never comes")
    .toHaveCount(0);
  await expect(page.locator(".cx-modal")).toContainText("Up to date");
});

test("a restart blocked mid-run is not lost -- the window offers a way to finish it", async ({ page }) => {
  // The client's own "a run is in flight" check is a courtesy, not the guard --
  // it goes stale if the dialog was already open when a run started. The
  // server refuses the restart itself and remembers it owes one; the window
  // has to surface that or the update is permanently applied-but-not-live.
  let pending = false;
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/git/status**", (route) => route.fulfill({
    json: { ok: true, version: "5.0", branch: "v5", branches: ["v5", "dev"], dirty: false,
            ahead: 0, behind: pending ? 0 : 2, fetch_ok: true, checked_remote: true, repo: "/x",
            restart_pending: pending },
  }));
  await page.route("**/api/git/update", (route) => {
    pending = true;
    return route.fulfill({ json: { restarting: false, blocked: "A generation is running.",
                                   updated: true } });
  });
  await page.route("**/api/git/restart", (route) => route.fulfill({ json: { restarting: true } }));
  await page.route("**/api/health", (route) => route.abort());

  await openUpdates(page);
  await page.getByRole("button", { name: /Update \(2\)/ }).click();

  await expect(page.locator(".cx-modal")).toContainText(/waiting to restart/i);
  const restartBtn = page.getByRole("button", { name: "Restart now" });
  await expect(restartBtn).toBeVisible();

  await restartBtn.click();
  await expect(page.locator(".cx-blocking")).toBeVisible();
});

test("a generation in flight is said, and the buttons that would kill it are off", async ({ page }) => {
  // The restart takes the run with it, with the GPU time already spent.
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/git/status**", (route) => route.fulfill({
    json: { ok: true, version: "5.0", branch: "v5", branches: ["v5", "dev"], dirty: false,
            ahead: 0, behind: 4, fetch_ok: true, checked_remote: true, repo: "/x" },
  }));
  // A run this page believes is its own and in flight.
  await page.evaluate(() => window.FunPack.run.adopt("pretend-prompt-id"));

  await openUpdates(page);
  await expect(page.locator(".cx-modal")).toContainText(/generation is running/i);
  await expect(page.getByRole("button", { name: /Update \(4\)/ })).toBeDisabled();
  // The restart it would deliver kills the run just the same.
  await expect(page.locator(".cx-modal").getByRole("button", { name: "Restart" })).toBeDisabled();
});

test("Restart ComfyUI is offered with nothing pending, and works", async ({ page }) => {
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);
  await page.route("**/api/git/status**", (route) => route.fulfill({
    json: { ok: true, version: "5.0", branch: "v5", branches: ["v5"], dirty: false,
            ahead: 0, behind: 0, fetch_ok: true, checked_remote: true, repo: "/x" },
  }));
  const asked = [];
  await page.route("**/api/git/restart", (route) => {
    asked.push(route.request().method());
    return route.fulfill({ json: { restarting: true } });
  });
  await page.route("**/api/health", (route) => route.abort());

  await openUpdates(page);
  await page.locator(".cx-modal").getByRole("button", { name: "Restart" }).click();

  await expect.poll(() => asked).toEqual(["POST"]);
  await expect(page.locator(".cx-blocking")).toContainText(/Restarting/i);
});
