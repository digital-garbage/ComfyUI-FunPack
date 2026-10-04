// Every module the page loads must at least parse: a feature with a syntax error is hidden, and a baseline one blanks the screen.
import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readdirSync, statSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const walk = (d) => readdirSync(d).flatMap((n) => { const p = join(d, n); return statSync(p).isDirectory() ? walk(p) : p.endsWith(".js") ? [p] : []; });

test("every .js under app/ui parses", () => {
  const bad = [];
  for (const f of walk(root)) { try { execFileSync(process.execPath, ["--check", f], { stdio: "pipe" }); } catch (e) { bad.push(`${f}: ${String(e.stderr).split("\n").slice(0, 3).join(" ")}`); } }
  assert.deepEqual(bad, []);
});
