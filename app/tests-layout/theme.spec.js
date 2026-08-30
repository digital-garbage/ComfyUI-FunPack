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

test("the theme you picked survives a reload", async ({ page }) => {
  // The appearance module applied its own declared default over the choice the
  // page had already applied from storage, so an explicit Dark came back as
  // Auto on every load -- silently, and looking like the switch was broken.
  await page.addInitScript(() => window.localStorage.setItem("funpack_theme", "dark"));
  await page.goto("/funpack/");
  await page.waitForFunction(() => window.FunPack !== undefined);

  await expect.poll(() => page.evaluate(() =>
    document.documentElement.getAttribute("data-theme-pref"))).toBe("dark");
  expect(await page.evaluate(() => window.localStorage.getItem("funpack_theme"))).toBe("dark");
});

test("a filled accent button keeps its label, at rest and under the pointer", async ({ page }) => {
  // `filter: brightness()` lightens the LABEL along with the fill under it, so
  // hovering used to wash out the one word that had to stay readable. Only the
  // fill moves now -- and which way it moves is the opposite in each theme,
  // because dark's accent carries near-black text and light's carries white.
  const luminance = (css) => {
    const [r, g, b] = css.match(/[\d.]+/g).slice(0, 3).map(Number).map((v) => {
      const s = v / 255;
      return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };

  for (const [theme, hoverGoes] of [["dark", "lighter"], ["light", "darker"]]) {
    await page.addInitScript((t) => window.localStorage.setItem("funpack_theme", t), theme);
    await page.goto("/funpack/");
    await page.waitForFunction(() => window.FunPack !== undefined);

    const button = page.locator(".cx-btn-primary").first();
    const read = () => page.evaluate(() => {
      const s = getComputedStyle(document.querySelector(".cx-btn-primary"));
      return { label: s.color, fills: s.backgroundImage.match(/(rgba?|color)\([^)]+\)/g) || [] };
    });

    const rest = await read();
    await button.hover();
    await page.waitForTimeout(150);
    const hovered = await read();

    expect(rest.fills.length, `${theme}: the button is a flat fill`).toBeGreaterThan(1);
    for (const [when, state] of [["at rest", rest], ["hovered", hovered]]) {
      for (const fill of state.fills) {
        const [a, b] = [luminance(state.label), luminance(fill)].sort((x, y) => y - x);
        expect((a + 0.05) / (b + 0.05), `${theme} ${when}: the label is lost in its own fill`)
          .toBeGreaterThan(4);
      }
    }

    const moved = luminance(hovered.fills[0]) - luminance(rest.fills[0]);
    if (hoverGoes === "lighter") expect(moved, `${theme}: hover did not brighten`).toBeGreaterThan(0.05);
    else expect(moved, `${theme}: hover did not darken`).toBeLessThan(-0.05);
  }
});
