// Rule 1: a module never styles anything.
//
// The runtime half is stronger than this -- a module is handed { composer,
// values, on } and nothing that could style or build -- but a scanner catches
// the attempt in the file rather than at the moment it would have run.

import test from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { readFileSync } from "node:fs";

import { scan, scanSource, code } from "../../../tools/kit_lint/scan.js";
import { MODULE_RULES, KIT_RULES } from "../../../tools/kit_lint/rules.js";

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, "..", "..", "..");
const FIXTURES = join(here, "lint_fixtures");

test("every module in the tree is clean", () => {
  // Empty today; the value is that it stays empty as modules arrive.
  const violations = scan(join(repo, "modules"), MODULE_RULES);
  assert.deepEqual(violations, [],
    violations.map((v) => `${v.file}:${v.line} ${v.rule} -- ${v.why}`).join("\n"));
});

test("the kit never interprets content", () => {
  // The kit may style and may build elements; that is its job. What it may not
  // do is parse anything a module gave it.
  const violations = scan(join(repo, "app", "composer"), KIT_RULES);
  assert.deepEqual(violations, [],
    violations.map((v) => `${v.file}:${v.line} ${v.rule}`).join("\n"));
});

test("the shell never interprets content either", () => {
  assert.deepEqual(scan(join(repo, "app", "shell"), KIT_RULES), []);
});

// --- the scanner has to actually catch things ------------------------------

test("a clean module passes", () => {
  const source = readFileSync(join(FIXTURES, "clean", "ui.js"), "utf8");
  assert.deepEqual(scanSource(source, MODULE_RULES), []);
});

test("the scanner catches every rule it claims to", () => {
  // Without this, "zero violations" could mean "the scanner is broken" -- which
  // is the failure mode of every lint nobody tests.
  const source = readFileSync(join(FIXTURES, "offenders", "ui.js"), "utf8");
  const caught = new Set(scanSource(source, MODULE_RULES).map((v) => v.rule));
  for (const rule of ["inline-style", "style-attribute", "class-manipulation",
                      "raw-element", "markup", "stylesheet", "hardcoded-colour",
                      "hardcoded-size", "eval"]) {
    assert.ok(caught.has(rule), `the offender fixture was not caught by "${rule}"`);
  }
});

test("prose is not code", () => {
  // Otherwise every rule becomes unmentionable in the comment explaining it.
  const commented = `
    // never write innerHTML here
    /* and #ff0000 and 12px are fine to discuss */
    export const ok = 1;
  `;
  assert.deepEqual(scanSource(commented, MODULE_RULES), []);
  assert.ok(!code(commented).includes("innerHTML"));
});

test("a url containing a hash is not a colour", () => {
  assert.deepEqual(scanSource('export const a = "#abc";', MODULE_RULES).map((v) => v.rule),
    ["hardcoded-colour"], "a real colour is caught");
});
