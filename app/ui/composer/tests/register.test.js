import test from "node:test";
import assert from "node:assert/strict";

import { composer, define, entries, has, UnknownElement, _clearRegistry } from "../internals/register.js";

test.beforeEach(() => _clearRegistry());

test("an element exists only because a file defined it", () => {
  assert.equal(has("button", "large"), false);
  define("button", "large", () => "made");
  assert.equal(has("button", "large"), true);
  assert.equal(composer.button.large(), "made");
});

test("asking for an undefined variant throws UnknownElement", () => {
  define("button", "large", () => {});
  assert.throws(() => composer.button.small, UnknownElement);
});

test("asking for an undefined group throws UnknownElement", () => {
  assert.throws(() => composer.wobble, UnknownElement);
});

test("the error names what does exist, so a typo is obvious", () => {
  define("button", "large", () => {});
  define("button", "medium", () => {});
  try {
    composer.button.smal;
    assert.fail("should have thrown");
  } catch (err) {
    assert.match(err.message, /composer\.button\.smal/);
    assert.match(err.message, /large/);
    assert.match(err.message, /medium/);
  }
});

test("defining the same name twice is refused", () => {
  define("button", "large", () => 1);
  // Otherwise one of the two silently wins, decided by import order.
  assert.throws(() => define("button", "large", () => 2), /already defined/);
});

test("define requires a factory", () => {
  assert.throws(() => define("button", "large", "not a function"), TypeError);
});

test("composer cannot be written to", () => {
  assert.throws(() => { composer.button = {}; }, TypeError);
});

test("probing keys answer undefined instead of throwing", () => {
  // An accidental `await composer` must not surface as UnknownElement("then").
  assert.equal(composer.then, undefined);
  assert.equal(composer[Symbol.toStringTag], undefined);
  define("button", "large", () => {});
  assert.equal(composer.button.then, undefined);
});

test("entries() enumerates the registry, so tests need no hand-written list", () => {
  define("button", "large", () => {});
  define("button", "small", () => {});
  define("hint", "default", () => {});
  const found = [...entries()].map((e) => `${e.group}.${e.variant}`).sort();
  assert.deepEqual(found, ["button.large", "button.small", "hint.default"]);
});

test("in/ownKeys reflect what is registered", () => {
  define("button", "large", () => {});
  assert.ok("button" in composer);
  assert.equal("hint" in composer, false);
  assert.deepEqual(Object.keys(composer), ["button"]);
});
