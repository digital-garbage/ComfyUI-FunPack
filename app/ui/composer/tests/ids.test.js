import test from "node:test";
import assert from "node:assert/strict";

import { uid, _resetIds } from "../internals/ids.js";

test("ids are unique", () => {
  _resetIds();
  const seen = new Set(Array.from({ length: 500 }, () => uid()));
  assert.equal(seen.size, 500);
});

test("the prefix is honoured, so ids say what they belong to", () => {
  assert.match(uid("field"), /^field-\d+$/);
});
