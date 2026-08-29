// Layout is the half jsdom cannot see.
//
// jsdom has no layout engine: getBoundingClientRect returns zeros, container
// queries never fire, focus rings do not paint, and a theme is just an
// attribute. Everything in the unit suite is therefore about behaviour, and
// every claim about POSITION or SIZE is currently unchecked. These run in a
// real browser against the real dev server, so those claims get checked too.
//
// The dev server is ComfyUI's own interpreter serving through core/serve.py, so
// the extension allowlist and the traversal guard are exercised by looking at
// the pages as much as by the Python tests.

import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests-layout",
  fullyParallel: true,
  reporter: process.env.CI ? "line" : "list",
  use: {
    baseURL: "http://127.0.0.1:8231",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "/Users/dex/Documents/ComfyUI/venv/bin/python ../tools/devserver.py 8231",
    url: "http://127.0.0.1:8231/funpack/",
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
