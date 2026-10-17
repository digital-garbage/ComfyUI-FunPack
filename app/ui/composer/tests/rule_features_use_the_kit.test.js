// A UI feature builds with kit elements and reaches the page only through app (app.dom).
// The scanner reads text; a feature that slips a rule is a line in this test's failure.

import test from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { scan, scanSource } from "../../../../tools/kit_lint/scan.js";
import { FEATURE_RULES } from "../../../../tools/kit_lint/rules.js";

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, "..", "..", "..", "..");

test("every UI feature is built from the kit and touches the page only through app", () => {
  const violations = scan(join(repo, "app", "ui", "features"), FEATURE_RULES);
  assert.deepEqual(violations, [],
    violations.map((v) => `${v.file.replace(repo, "")}:${v.line} ${v.rule}`).join("\n"));
});


// --- the scanner has to actually catch things ------------------------------

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
