// The media library: what the user brought in, over a mocked /funpack/api/media.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom, fire } from "../../composer/tests/_dom.js";

let createMediaLibrary, mediaUrl;
test.before(async () => {
  setupDom();
  await import("../../composer/composer.js");
  ({ createMediaLibrary, mediaUrl } = await import("../media.js"));
});
test.after(() => teardownDom());

const entry = (id, over = {}) => ({ id, name: `${id}.png`, filename: `${id}.png`,
  kind: "image", size: 3, added: 1, ...over });

function library(opts = {}) {
  const picked = [];
  const handle = createMediaLibrary({ onPick: (item) => picked.push(item), ...opts });
  document.body.replaceChildren(handle.host.node);
  return {
    picked,
    node: handle.host.node,
    refresh: () => handle.refresh(),
    get items() { return handle.items; },
    cells: () => [...handle.host.node.querySelectorAll(".cx-cell")],
  };
}

test("mediaUrl points at the file route by id", () => {
  assert.equal(mediaUrl({ id: "abc123" }), "/funpack/api/media/abc123/file");
});

test("refresh populates the gallery from the list route", async () => {
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ media: [entry("a"), entry("b")] }) });
  const lib = library();
  await lib.refresh();

  assert.deepEqual(lib.items.map((i) => i.id), ["a", "b"]);
  assert.equal(lib.cells().length, 2);
});

test("nothing is fetched until refresh is called", () => {
  let asked = false;
  globalThis.fetch = async () => { asked = true; return { ok: true, json: async () => ({ media: [] }) }; };
  library();
  assert.equal(asked, false, "the library fetched on construction instead of waiting to be asked");
});

test("clicking an entry calls onPick with the whole item", async () => {
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ media: [entry("x")] }) });
  const lib = library();
  await lib.refresh();

  lib.cells()[0].click();
  assert.equal(lib.picked.length, 1);
  assert.equal(lib.picked[0].id, "x");
});

test("an upload adds to the front of the list without a full refresh", async () => {
  let posted = null;
  globalThis.fetch = async (url, opts) => {
    if (opts && opts.method === "POST") {
      posted = opts.body;
      return { ok: true, json: async () => ({ media: [entry("new")], problems: [] }) };
    }
    return { ok: true, json: async () => ({ media: [entry("old")] }) };
  };
  const lib = library();
  await lib.refresh();
  assert.equal(lib.items.length, 1);

  const file = new File(["data"], "ref.png", { type: "image/png" });
  const dz = lib.node.querySelector("input[type=file]");
  Object.defineProperty(dz, "files", { value: [file], configurable: true });
  dz.dispatchEvent(new Event("change"));
  await new Promise((r) => setTimeout(r, 0));

  assert.ok(posted instanceof FormData, "the upload was not sent as multipart form data");
  assert.deepEqual(lib.items.map((i) => i.id), ["new", "old"]);
});

test("an upload the server refuses does not add anything, and does not throw", async () => {
  globalThis.fetch = async (_url, opts) => {
    if (opts && opts.method === "POST") {
      return { ok: false, status: 400, json: async () => ({ problems: ["that is not something FunPack imports"] }) };
    }
    return { ok: true, json: async () => ({ media: [] }) };
  };
  const lib = library();
  await lib.refresh();

  const file = new File(["MZ"], "payload.exe");
  const dz = lib.node.querySelector("input[type=file]");
  Object.defineProperty(dz, "files", { value: [file], configurable: true });
  dz.dispatchEvent(new Event("change"));
  await new Promise((r) => setTimeout(r, 0));

  assert.equal(lib.items.length, 0);
});

test("a 413 the framework refuses before the route runs shows a clean message, not a JSON-parse error", async () => {
  // aiohttp refuses an oversized body itself, before FunPack's route ever
  // runs, as plain text -- res.json() on that throws a SyntaxError, and an
  // earlier version of upload() let that error's own message reach the
  // toast: "Unexpected token 'M', "Maximum re"... is not valid JSON".
  globalThis.fetch = async (_url, opts) => {
    if (opts && opts.method === "POST") {
      return {
        ok: false, status: 413,
        json: async () => { throw new SyntaxError("Unexpected token 'M', \"Maximum re\"... is not valid JSON"); },
      };
    }
    return { ok: true, json: async () => ({ media: [] }) };
  };
  const lib = library();
  await lib.refresh();

  const file = new File(["x".repeat(10)], "huge.png");
  const dz = lib.node.querySelector("input[type=file]");
  Object.defineProperty(dz, "files", { value: [file], configurable: true });
  dz.dispatchEvent(new Event("change"));
  await new Promise((r) => setTimeout(r, 0));

  const toast = document.querySelector(".cx-toast-danger");
  assert.ok(toast, "no toast was shown at all");
  assert.match(toast.textContent, /too large/i);
  assert.doesNotMatch(toast.textContent, /JSON|SyntaxError|Unexpected token/i);
});

test("removing an entry is optimistic, and reverts if the server refuses", async () => {
  let deleteCalled = false;
  globalThis.fetch = async (url, opts) => {
    if (opts && opts.method === "DELETE") { deleteCalled = true; return { ok: false, status: 500 }; }
    return { ok: true, json: async () => ({ media: [entry("keep")] }) };
  };
  const lib = library();
  await lib.refresh();

  fire(lib.cells()[0], "contextmenu");
  // Removal is optimistic -- gone from the list immediately, before the
  // request resolves -- then the failed DELETE triggers a re-fetch.
  assert.equal(lib.items.length, 0);
  assert.equal(deleteCalled, true);
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(lib.items.length, 1, "a failed delete was not recovered from");
});
