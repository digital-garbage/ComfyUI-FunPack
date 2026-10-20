import test from "node:test";
import assert from "node:assert/strict";
import { startLoading } from "./loading.js";

const fake = () => {
  const log = { closed: 0, text: "", pct: -1 };
  const c = {
    text: { md: ({ text }) => ({ setText: (t) => { log.text = t; log.text0 = text; } }) },
    progress: { bar: () => ({ node: {}, setValue: (v) => { log.pct = v; } }) },
    region: { stack: () => ({}) },
    modal: { generic: (p) => { log.props = p; return { close: () => { log.closed++; } }; } },
  };
  return { c, log };
};

test("the window cannot be dismissed, names the module and its place, and closes once", () => {
  const { c, log } = fake();
  const l = startLoading(c);
  assert.equal(log.props.closeOnEsc, false);
  assert.equal(log.props.closeOnOutside, false);
  l.progress({ n: 2, of: 4, path: "./features/timeline/index.js" });
  assert.equal(log.text, "Loading module 2 out of 4: timeline. Please wait for UI to finish loading.");
  assert.equal(log.pct, 25);
  l.progress({ n: 3, of: 4, path: "./features/render/selected.js" });
  assert.match(log.text, /3 out of 4: render\/selected\./);
  l.done(); l.done();
  assert.equal(log.closed, 1);
});

test("a kit that cannot draw the window does not stop the app", () => {
  const l = startLoading({});
  l.progress({ n: 1, of: 1, path: "./features/a/index.js" });
  l.done();
});
