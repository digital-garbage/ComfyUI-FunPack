// Appearance needs a little JavaScript, because its settings change the app
// rather than the generation. Most modules have no ui.js at all.

export function setup({ values, on, shell }) {
  const apply = (key, value) => {
    if (key === "theme") shell.theme.set(value);
    if (key === "density") shell.density.set(value);
  };

  // Apply what is already stored, then follow every later edit.
  const current = values.get();
  for (const key of Object.keys(current)) apply(key, current[key]);

  return on(apply);
}
