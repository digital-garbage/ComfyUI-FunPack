// A source scanner for the rule tests.
//
// Deliberately dumb: it reads text and matches patterns. Something cleverer
// would need a parser, and a parser is a thing that can be wrong in ways nobody
// notices. A false positive here is a comment away from being fixed; a false
// negative is a rule that quietly stopped being enforced.

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const SKIP = new Set(["node_modules", ".git", "__pycache__", "tests"]);

export function jsFilesIn(root) {
  const out = [];
  let entries;
  try { entries = readdirSync(root); } catch { return out; }
  for (const name of entries) {
    if (SKIP.has(name) || name.startsWith(".")) continue;
    const path = join(root, name);
    if (statSync(path).isDirectory()) out.push(...jsFilesIn(path));
    else if (name.endsWith(".js") && !name.endsWith(".test.js")) out.push(path);
  }
  return out;
}

/** Strip comments so a rule quoted in prose is not a violation. */
export function code(source) {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/(^|[^:])\/\/.*$/gm, "$1");
}

export function scanSource(source, rules) {
  const body = code(source);
  const found = [];
  for (const rule of rules) {
    const line = body.split("\n").findIndex((l) => rule.test.test(l));
    if (line !== -1) found.push({ rule: rule.name, why: rule.why, line: line + 1 });
  }
  return found;
}

export function scan(root, rules) {
  const violations = [];
  for (const file of jsFilesIn(root)) {
    for (const found of scanSource(readFileSync(file, "utf8"), rules)) {
      violations.push({ file, ...found });
    }
  }
  return violations;
}
