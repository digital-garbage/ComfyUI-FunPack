// store.js is being split into parts. Whatever the file layout, the Store the rest of the app sees must stay the same:
// the same names, each still a function (or the same kind of value). Loads the real store.js with a stubbed browser.
const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

function loadStore() {
  const noop = () => {};
  const el = () => ({ style: {}, classList: { add: noop, remove: noop }, addEventListener: noop, setAttribute: noop });
  const win = {
    MovieEditorAPI: new Proxy({}, { get: () => () => Promise.resolve({}) }),
    addEventListener: noop,
    localStorage: { getItem: () => null, setItem: noop, removeItem: noop },
    document: { addEventListener: noop, querySelector: () => null, querySelectorAll: () => [], getElementById: () => null, createElement: el, body: {} },
  };
  const sandbox = { window: win, ...win, console, setTimeout, clearTimeout, setInterval: noop, clearInterval: noop,
    fetch: () => Promise.reject(new Error("no network")), navigator: {}, requestAnimationFrame: noop, performance: { now: () => 0 } };
  vm.createContext(sandbox);
  const dir = __dirname;
  const files = fs.readdirSync(dir).filter((f) => /^store(_.+)?\.js$/.test(f) && !f.endsWith(".test.js"));
  const order = ["store.js", ...files.filter((f) => f !== "store.js").sort()];
  const hub = fs.existsSync(path.join(dir, "store_parts.js")) ? ["store_parts.js"] : [];
  for (const f of [...hub, ...order.filter((f) => f !== "store_parts.js")]) vm.runInContext(fs.readFileSync(path.join(dir, f), "utf8"), sandbox, { filename: f });
  return win.Store;
}

test("the Store keeps its public surface", () => {
  const store = loadStore();
  const kinds = Object.fromEntries(Object.keys(store).sort().map((k) => [k, typeof store[k]]));
  const expected = path.join(__dirname, "store_surface.json");
  if (!fs.existsSync(expected)) fs.writeFileSync(expected, JSON.stringify(kinds, null, 1));
  assert.deepStrictEqual(kinds, JSON.parse(fs.readFileSync(expected, "utf8")));
});
