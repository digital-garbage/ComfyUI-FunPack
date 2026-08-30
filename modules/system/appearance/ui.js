// Appearance needs a little JavaScript, because its settings change the app
// rather than the generation. Most modules have no ui.js at all.

export function setup({ values, on, shell }) {
  const apply = (key, value) => {
    if (key === "theme") shell.theme.set(value);
    if (key === "density") shell.density.set(value);
  };

  // ADOPT what the app is already set to, rather than applying this module's
  // declared defaults over it.
  //
  // Both of these are remembered by the browser and applied before any module
  // exists -- the theme in the document head, so the first paint is already
  // right. Writing the default back over that threw away an explicit choice on
  // every single page load: pick Dark, reload, and the app came up in Auto with
  // nothing said and nothing to see.
  values.set("theme", shell.theme.get());
  values.set("density", shell.density.get());

  return on(apply);
}
