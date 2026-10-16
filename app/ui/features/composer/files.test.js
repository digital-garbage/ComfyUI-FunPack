import test from "node:test";
import assert from "node:assert/strict";
import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let files;
test.before(async () => { setupDom(); ({ files } = await import("./files.js")); });
test.after(() => teardownDom());

test("Files lists the library files and the taste keys; with no taste module it shows the library alone", async () => {
  const lib = { dir: "/x", files: [{ name: "shortcuts.json", size: 2048 }] };
  const page = files({ api: { libraryFiles: async () => lib, tasteKeys: async () => ({ keys: ["portraits"] }) } });
  await new Promise((r) => setTimeout(r, 0));
  assert.match(page.node.textContent, /shortcuts\.json.*2\.0 KB/);
  assert.match(page.node.textContent, /Taste keys.*portraits/);
  const bare = files({ api: { libraryFiles: async () => lib, tasteKeys: async () => { throw new Error("404"); } } });
  await new Promise((r) => setTimeout(r, 0));
  assert.match(bare.node.textContent, /shortcuts\.json/);
  assert.doesNotMatch(bare.node.textContent, /Taste keys/);
});

test("a taste key splits into its kinds, each with what it is for and its own delete", async () => {
  const lib = { dir: "/x", files: [] };
  const api = { libraryFiles: async () => lib, tasteKeys: async () => ({ keys: ["portraits"] }),
    tasteKindsOf: async () => ({ kinds: [{ kind: "reins", title: "Taste steering (REINS)", hint: "Per-block steering", bytes: 2048 }] }),
    clearTasteKind: async () => ({ kinds: [] }) };
  const page = files({ api });
  await new Promise((r) => setTimeout(r, 0)); await new Promise((r) => setTimeout(r, 0));
  assert.match(page.node.textContent, /Taste steering \(REINS\).*Per-block steering.*2\.0 KB/);
  assert.ok([...page.node.querySelectorAll("button")].some((b) => /Delete key/.test(b.textContent)));
});
