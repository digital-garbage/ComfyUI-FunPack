// The shortcut library UI, over a mocked /funpack/api/shortcuts.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let mount;
test.before(async () => {
  setupDom();
  await import("../../composer/composer.js");
  ({ mount } = await import("../shortcuts.js"));
});
test.after(() => teardownDom());

const sc = (name, over = {}) => ({ name, triggers: [name.toLowerCase()],
  replacements: [`replacement for ${name}`], enabled: true, category: "", sub_category: "", ...over });

function mockFetch(handlers) {
  globalThis.fetch = async (url, opts) => {
    const method = (opts && opts.method) || "GET";
    const key = `${method} ${new URL(url, "http://localhost/").pathname}`;
    const h = handlers[key] || handlers[method + " *"];
    if (!h) throw new Error(`unhandled: ${key}`);
    return h(opts);
  };
}

/** setFooter's actions are handed to whoever hosts this (the modal chrome
 * in production) rather than living inside body.node -- a footer that
 * did nothing with them would make "+ Add shortcut" and "Close" untestable
 * and unclickable, so this stands in for that host: a footer strip actually
 * appended beside the body, exactly like packs.js's own open() does. */
function open(opts = {}) {
  let closed = false;
  const footer = document.createElement("div");
  const body = mount({
    ...opts,
    setFooter: (f) => { footer.replaceChildren(...(f.actions || []).map((a) => a.node)); },
    close: () => { closed = true; if (opts.close) opts.close(); },
  });
  const root = document.createElement("div");
  root.append(body.node, footer);
  document.body.replaceChildren(root);
  return { root, get closed() { return closed; } };
}

const find = (root, text) => [...root.querySelectorAll("button")].find((b) => b.textContent.trim() === text);

test("the list loads on mount", async () => {
  mockFetch({ "GET /funpack/api/shortcuts": async () => ({ ok: true, json: async () => ({ shortcuts: [sc("Fox")] }) }) });
  const { root } = open();
  await new Promise((r) => setTimeout(r, 0));
  assert.match(root.textContent, /Fox/);
});

test("adding a shortcut with no trigger is refused before the request is sent", async () => {
  let posted = false;
  mockFetch({
    "GET /funpack/api/shortcuts": async () => ({ ok: true, json: async () => ({ shortcuts: [] }) }),
    "POST /funpack/api/shortcuts": async () => { posted = true; return { ok: true, json: async () => ({ shortcuts: [] }) }; },
  });
  const { root } = open();
  await new Promise((r) => setTimeout(r, 0));

  find(root, "+ Add shortcut").click();
  find(root, "Save").click();
  assert.equal(posted, false, "an empty trigger list reached the server");
  assert.match(root.textContent, /at least one trigger/i);
});

test("delete calls the route for that name and redraws from the response", async () => {
  const calls = [];
  mockFetch({
    "GET /funpack/api/shortcuts": async () => ({ ok: true, json: async () => ({ shortcuts: [sc("Fox"), sc("Owl")] }) }),
    "DELETE /funpack/api/shortcuts/Fox": async () => { calls.push("Fox"); return { ok: true, json: async () => ({ shortcuts: [sc("Owl")] }) }; },
  });
  const { root } = open();
  await new Promise((r) => setTimeout(r, 0));

  find(root, "Delete").click();
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(calls, ["Fox"]);
  assert.doesNotMatch(root.textContent, /Fox/);
  assert.match(root.textContent, /Owl/);
});

test("a server refusal on save is shown rather than silently dropped", async () => {
  mockFetch({
    "GET /funpack/api/shortcuts": async () => ({ ok: true, json: async () => ({ shortcuts: [] }) }),
    "POST /funpack/api/shortcuts": async () => ({ ok: false, status: 400, json: async () => ({ problems: ["a shortcut needs at least one trigger"] }) }),
  });
  const { root } = open();
  await new Promise((r) => setTimeout(r, 0));

  find(root, "+ Add shortcut").click();
  const triggers = [...root.querySelectorAll("textarea")][0];
  triggers.value = "fox";
  triggers.dispatchEvent(new window.Event("change", { bubbles: true }));
  find(root, "Save").click();
  await new Promise((r) => setTimeout(r, 0));
  assert.match(root.textContent, /a shortcut needs at least one trigger/);
});

test("the footer's Close action calls the close handler it was given", async () => {
  mockFetch({ "GET /funpack/api/shortcuts": async () => ({ ok: true, json: async () => ({ shortcuts: [] }) }) });
  const session = open();
  await new Promise((r) => setTimeout(r, 0));

  find(session.root, "Close").click();
  assert.equal(session.closed, true);
});
