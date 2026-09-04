// parseBlockSweepConfig: the "blocks;seam|noseam;times" line format for the H3 block-repeat
// sweep panel, exercised rather than syntax-checked (see detail_verdict.test.js for why).
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");

const src = fs.readFileSync(__dirname + "/store.js", "utf8");
const grab = (name) => src.match(new RegExp(`function ${name}[\\s\\S]*?\\n  \\}\\n`))[0];
const { parseBlockSweepConfig } = new Function(
  grab("parseBlockSweepConfig") + "; return { parseBlockSweepConfig };")();
const { _sweepFilenamePrefix } = new Function(
  grab("_sweepFilenamePrefix") + "; return { _sweepFilenamePrefix };")();

test("the render's own filename_prefix carries the sweep config, not a generic uuid", () => {
  const prefix = _sweepFilenamePrefix("10,11,13;seam;5");
  assert.match(prefix, /^funpack_sweep_10-11-13-seam-5_[a-z0-9]{6}$/);
});

test("filename_prefix is unique per call even for the same label", () => {
  const a = _sweepFilenamePrefix("35;seam;1");
  const b = _sweepFilenamePrefix("35;seam;1");
  assert.notEqual(a, b);
});

const bsSrc = fs.readFileSync(__dirname + "/block_sweep.js", "utf8");
const grabBs = (name) => bsSrc.match(new RegExp(`function ${name}[\\s\\S]*?\\n  \\}\\n`))[0];
const { _sweepFilename } = new Function(
  grabBs("_sweepFilename") + "; return { _sweepFilename };")();

test("sweep filenames are filesystem-safe and carry the config", () => {
  const name = _sweepFilename("10,11,13;seam;5");
  assert.match(name, /^sweep_10-11-13-seam-5_\d+\.mp4$/);
});

test("sweep filenames never start or end with a separator dash", () => {
  const name = _sweepFilename(";noseam;1");
  assert.ok(!name.startsWith("sweep_-"));
  assert.match(name, /^sweep_noseam-1_\d+\.mp4$/);
});

test("parses blocks, seam and times off a semicolon line", () => {
  const [c] = parseBlockSweepConfig("10,11,13;seam;5");
  assert.equal(c.blocks, "10,11,13");
  assert.equal(c.spanLoop, true);
  assert.equal(c.times, 4); // clamped to the widget's max
  assert.equal(c.lastSteps, 0); // omitted = every step
});

test("noseam and in-range times pass through untouched", () => {
  const [c] = parseBlockSweepConfig("40-41;noseam;1");
  assert.equal(c.blocks, "40-41");
  assert.equal(c.spanLoop, false);
  assert.equal(c.times, 1);
});

test("optional laststeps field is parsed and clamped to the widget's range", () => {
  const [c] = parseBlockSweepConfig("40-41;noseam;1;2");
  assert.equal(c.lastSteps, 2);
  const [c2] = parseBlockSweepConfig("40-41;noseam;1;999");
  assert.equal(c2.lastSteps, 50);
  const [c3] = parseBlockSweepConfig("40-41;noseam;1;-5");
  assert.equal(c3.lastSteps, 0);
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
