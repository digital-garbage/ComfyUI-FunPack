/**
 * Every settings surface (pipeline, updates, node packs, log, temp files)
 * lives behind ONE Settings button now, found by name in its searchable list.
 */
export async function openSettingsSection(page, name) {
  await page.getByRole("button", { name: "Settings" }).click();
  await page.locator(".cx-filter-row", { hasText: name }).click();
  // The sidebar is a hover-expanding rail now (v4's own behaviour): leaving
  // the mouse where the click landed keeps it open, overlaying the content
  // every caller is about to interact with -- exactly what a real person's
  // mouse does NOT do once they move on to the thing they picked. Hovering
  // the section's own head (never under the rail, whatever the viewport) is
  // what actually collapses it, at any width a test runs at.
  await page.locator(".cx-modal-head").hover();
}

/** The window moved into the unified Settings window when the app grew one. */
export async function openPipelineWindow(page) {
  await openSettingsSection(page, "Models and pipeline");
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
