// parseBlockSweepConfig: the "blocks;seam|noseam;times" line format for the H3 block-repeat
// sweep panel, exercised rather than syntax-checked (see detail_verdict.test.js for why).
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");

const src = fs.readFileSync(__dirname + "/store.js", "utf8");
const grab = (name) => src.match(new RegExp(`function ${name}[\\s\\S]*?\\n  \\}\\n`))[0];
const { parseBlockSweepConfig } = new Function(
  grab("parseBlockSweepConfig") + "; return { parseBlockSweepConfig };")();

test("parses blocks, seam and times off a semicolon line", () => {
  const [c] = parseBlockSweepConfig("10,11,13;seam;5");
  assert.equal(c.blocks, "10,11,13");
  assert.equal(c.spanLoop, true);
  assert.equal(c.times, 4); // clamped to the widget's max
});

test("noseam and in-range times pass through untouched", () => {
  const [c] = parseBlockSweepConfig("40-41;noseam;1");
  assert.equal(c.blocks, "40-41");
  assert.equal(c.spanLoop, false);
  assert.equal(c.times, 1);
});

test("multiple lines, blank lines and stray whitespace are skipped/trimmed", () => {
  const out = parseBlockSweepConfig("\n 10,11;seam;2 \n\n40-41;noseam;1\n");
  assert.equal(out.length, 2);
  assert.equal(out[0].label, "10,11;seam;2");
  assert.equal(out[1].label, "40-41;noseam;1");
});

test("a line with no blocks field is dropped, not crashed on", () => {
  assert.deepEqual(parseBlockSweepConfig(";seam;2"), []);
});

test("missing/garbage times default to 1 rather than NaN", () => {
  const [c] = parseBlockSweepConfig("35;seam;");
  assert.equal(c.times, 1);
  const [c2] = parseBlockSweepConfig("35;seam;abc");
  assert.equal(c2.times, 1);
});

test("empty input yields no configs", () => {
  assert.deepEqual(parseBlockSweepConfig(""), []);
  assert.deepEqual(parseBlockSweepConfig(null), []);
});
