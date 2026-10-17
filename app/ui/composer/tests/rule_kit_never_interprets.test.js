// The kit and the shell never interpret content. Modules ship no JavaScript, so
// there is nothing to scan in modules/.

import test from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { scan, scanSource, code } from "../../../../tools/kit_lint/scan.js";
import { KIT_RULES } from "../../../../tools/kit_lint/rules.js";

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, "..", "..", "..", "..");

test("the kit never interprets content", () => {
  // The kit may style and may build elements; that is its job. What it may not
  // do is parse anything a module gave it.
  const violations = scan(join(repo, "app", "ui", "composer"), KIT_RULES);
  assert.deepEqual(violations, [],
    violations.map((v) => `${v.file}:${v.line} ${v.rule}`).join("\n"));
});

test("the shell never interprets content either", () => {
  assert.deepEqual(scan(join(repo, "app", "ui", "shell"), KIT_RULES), []);
});

// --- the scanner has to actually catch things ------------------------------

test("the scanner catches what it claims to", () => {
  // Without this, "zero violations" could mean "the scanner is broken".
  const caught = new Set(scanSource("el.innerHTML = x; eval(y);", KIT_RULES).map((v) => v.rule));
  for (const rule of ["markup", "eval"]) assert.ok(caught.has(rule), `not caught: "${rule}"`);
});

test("prose is not code", () => {
  // Otherwise every rule becomes unmentionable in the comment explaining it.
  const commented = `
    // never write innerHTML here
    /* and eval( is fine to discuss */
    export const ok = 1;
  `;
  assert.deepEqual(scanSource(commented, KIT_RULES), []);
  assert.ok(!code(commented).includes("innerHTML"));
});
