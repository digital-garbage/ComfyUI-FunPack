// parseComboSweepConfig: the "reins:strength;block|sweep:blocks;seam|noseam;times[;laststeps]"
// line format for the H3 combo sweep panel, exercised rather than syntax-checked (see
// detail_verdict.test.js for why).
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");

const src = fs.readFileSync(__dirname + "/store.js", "utf8");
const grab = (name) => src.match(new RegExp(`function ${name}[\\s\\S]*?\\n  \\}\\n`))[0];
const { parseComboSweepConfig } = new Function(
  grab("parseComboSweepConfig") + "; return { parseComboSweepConfig };")();
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

test("a negative laststeps in the label doesn't produce a double dash or leading dash", () => {
  const name = _sweepFilenamePrefix("40-41;noseam;1;-2");
  assert.match(name, /^funpack_sweep_40-41-noseam-1-2_[a-z0-9]{6}$/);
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

// --- parseComboSweepConfig -------------------------------------------------------------

test("reins-only line parses strength and block, sweep stays null", () => {
  const [c] = parseComboSweepConfig("reins:0.1;49");
  assert.equal(c.label, "reins:0.1;49");
  assert.deepEqual(c.reins, { strength: 0.1, block: "49" });
  assert.equal(c.sweep, null);
});

test("sweep-only line parses blocks/seam/times/laststeps, reins stays null", () => {
  const [c] = parseComboSweepConfig("sweep:40-41;noseam;1;0");
  assert.equal(c.reins, null);
  assert.deepEqual(c.sweep, { blocks: "40-41", spanLoop: false, times: 1, lastSteps: 0 });
});

test("combined line via | sets both halves independently", () => {
  const [c] = parseComboSweepConfig("reins:0.15;49|sweep:40-41;noseam;1;0");
  assert.deepEqual(c.reins, { strength: 0.15, block: "49" });
  assert.deepEqual(c.sweep, { blocks: "40-41", spanLoop: false, times: 1, lastSteps: 0 });
});

test("order of the two halves does not matter", () => {
  const [c] = parseComboSweepConfig("sweep:31-40;seam;2;-3|reins:0.2;25");
  assert.deepEqual(c.reins, { strength: 0.2, block: "25" });
  assert.equal(c.sweep.blocks, "31-40");
  assert.equal(c.sweep.spanLoop, true);
});

test("sweep times and laststeps are clamped the same way as before", () => {
  const [c] = parseComboSweepConfig("sweep:40-41;noseam;999;999");
  assert.equal(c.sweep.times, 4);
  assert.equal(c.sweep.lastSteps, 50);
  const [c2] = parseComboSweepConfig("sweep:40-41;noseam;1;-999");
  assert.equal(c2.sweep.lastSteps, -50);
});

test("a line with neither valid segment is dropped", () => {
  assert.deepEqual(parseComboSweepConfig("garbage"), []);
  assert.deepEqual(parseComboSweepConfig("sweep:;noseam;1"), []); // empty blocks
  assert.deepEqual(parseComboSweepConfig("reins:notanumber;49"), []); // bad strength
});

test("multiple lines, blank lines and stray whitespace are skipped/trimmed", () => {
  const out = parseComboSweepConfig("\n reins:0.1;49 \n\nsweep:40-41;noseam;1\n");
  assert.equal(out.length, 2);
  assert.equal(out[0].label, "reins:0.1;49");
  assert.equal(out[1].label, "sweep:40-41;noseam;1");
});

test("empty input yields no configs", () => {
  assert.deepEqual(parseComboSweepConfig(""), []);
  assert.deepEqual(parseComboSweepConfig(null), []);
});
