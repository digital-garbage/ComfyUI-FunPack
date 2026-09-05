/** The window moved into the Settings menu when the app grew a menu bar. */
export async function openPipelineWindow(page) {
  await page.getByRole("button", { name: "Settings" }).click();
  await page.getByRole("menuitem", { name: /Models and pipeline/ }).click();
}

/**
 * The control that shows and hides a docked region.
 *
 * It used to be a glyph in a rail on the outer edge of the window; it is a
 * named button in the timeline head now, which is where v4 keeps it. Written
 * once here because six tests press it and none of them are about where it is.
 */
export const regionToggle = (page, name) =>
  page.locator(".cx-panel-head").getByRole("button", { name, exact: true });

/**
 * The same toggle from the menu bar, which nothing can cover.
 *
 * The way in and out at a width where the panels overlay the centre -- the
 * named buttons live in the timeline head, and that is what gets covered.
 */
export const menuToggle = (page, region) => async () => {
  await page.getByRole("button", { name: "View" }).click();
  await page.getByRole("menuitem", { name: new RegExp(`(Show|Hide) ${region}`) }).click();
};
