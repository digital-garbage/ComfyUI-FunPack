// The one list of UI features. Adding a feature is a folder under ./features and a line here.
export default [
  "./features/log/index.js",
  "./features/projects/index.js",
  "./features/timeline/index.js",
  "./features/inspector/index.js",
  "./features/generate/index.js",
  "./features/preview/index.js",
  "./features/menus/index.js",
  "./features/placeholders/index.js",
  "./features/temp/index.js",
].map((path) => new URL(path, import.meta.url).href);
