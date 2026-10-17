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

import { scan, scanSource, code } from "../../../../tools/kit_lint/scan.js";
import { MODULE_RULES, KIT_RULES, FEATURE_RULES } from "../../../../tools/kit_lint/rules.js";

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, "..", "..", "..", "..");
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
  const violations = scan(join(repo, "app", "ui", "composer"), KIT_RULES);
  assert.deepEqual(violations, [],
    violations.map((v) => `${v.file}:${v.line} ${v.rule}`).join("\n"));
});

test("the shell never interprets content either", () => {
  assert.deepEqual(scan(join(repo, "app", "ui", "shell"), KIT_RULES), []);
});

test("every UI feature is built from the kit and touches the page only through app", () => {
  const violations = scan(join(repo, "app", "ui", "features"), FEATURE_RULES);
  assert.deepEqual(violations, [],
    violations.map((v) => `${v.file.replace(repo, "")}:${v.line} ${v.rule}`).join("\n"));
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

test("the feature rules catch the page, the frame's classes and styling, and let prose and imports through", () => {
  const bad = `
    const a = document.createElement("div");
    window.addEventListener("resize", f);
    node.classList.add("x"); node.style.left = "1px";
    q.closest(".cx-panel-head"); el.className = "fp-card";
    box.innerHTML = "x"; const c = "#ff0066";
  `;
  const caught = new Set(scanSource(bad, FEATURE_RULES).map((v) => v.rule));
  for (const rule of ["document", "reaching-for-globals", "class-manipulation", "inline-style", "class-name", "markup", "hardcoded-colour", "hardcoded-size"]) {
    assert.ok(caught.has(rule), `not caught: ${rule}`);
  }
  const fine = `
    import { open } from "./window.js";
    const text = "Check the Log window. Wrong look? Open the document, then close it.";
    localStorage.setItem("k", "1"); app.dom.onPage("focusin", f);
  `;
  assert.deepEqual(scanSource(fine, FEATURE_RULES), []);
});
