/** The window moved into the Settings menu when the app grew a menu bar. */
export async function openPipelineWindow(page) {
  await page.getByRole("button", { name: "Settings" }).click();
  await page.getByRole("menuitem", { name: /Models and pipeline/ }).click();
}
