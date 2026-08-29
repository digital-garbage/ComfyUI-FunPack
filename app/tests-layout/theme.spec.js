// Both themes, painted.
//
// jsdom treats a theme as an attribute: it never resolves a custom property,
// so "the light theme is light" is unverifiable there. A real browser computes
// the cascade, which is the only place a token that resolves to the wrong value
// -- or to nothing -- shows up.

import { test, expect } from "@playwright/test";

const rgb = (value) => value.match(/\d+/g).map(Number);
const luminance = ([r, g, b]) => (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;

test.describe("themes", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/funpack/");
    await page.waitForFunction(() => window.FunPack !== undefined);
  });

  for (const theme of ["dark", "light"]) {
    test(`${theme} resolves every token it uses`, async ({ page }) => {
      await page.evaluate((t) => window.ComposerTheme.apply(t), theme);

      const unresolved = await page.evaluate(() => {
        const style = getComputedStyle(document.documentElement);
        const names = [];
        for (const sheet of document.styleSheets) {
          let rules;
          try { rules = sheet.cssRules; } catch { continue; }
          for (const rule of rules) {
            if (!rule.style) continue;
            for (const prop of rule.style) {
              if (prop.startsWith("--")) names.push(prop);
            }
          }
        }
        // A token that resolves to nothing is the failure this catches: the
        // rule still applies and the colour silently comes from elsewhere.
        return [...new Set(names)].filter((n) => style.getPropertyValue(n).trim() === "");
      });

      expect(unresolved, `tokens with no value in ${theme}`).toEqual([]);
    });
  }

  test("light is actually lighter than dark", async ({ page }) => {
    const read = async (theme) => {
      await page.evaluate((t) => window.ComposerTheme.apply(t), theme);
      return page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    };
    const dark = luminance(rgb(await read("dark")));
    const light = luminance(rgb(await read("light")));
    expect(light, "the light theme is not lighter than the dark one")
      .toBeGreaterThan(dark);
  });

  test("the page never scrolls sideways", async ({ page }) => {
    for (const width of [1440, 1024, 768, 480, 360]) {
      await page.setViewportSize({ width, height: 800 });
      const overflows = await page.evaluate(() =>
        document.documentElement.scrollWidth > document.documentElement.clientWidth + 1);
      expect(overflows, `horizontal scrollbar at ${width}px`).toBe(false);
    }
  });
});
