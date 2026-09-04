// The settings panel's verdict logic, exercised rather than syntax-checked.
//
// It shipped a temporal-dead-zone throw ("can't access lexical declaration 'moved' before
// initialization") that `node --check` passed happily, because a `const` used above its own
// declaration is valid syntax and only fails when the branch actually runs. So these RUN
// every branch.
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");

const src = fs.readFileSync(__dirname + "/settings_window.js", "utf8");
const grab = (name) => src.match(new RegExp(`function ${name}[\\s\\S]*?\\n  \\}\\n`))[0];
const { dpVerdict, dpNoiseFloor } = new Function(
  grab("dpVerdict") + grab("dpNoiseFloor") + "; return { dpVerdict, dpNoiseFloor };")();

const row = (o) => Object.assign(
  { changed: {}, same_seed: true, structure: 1, detail: 1, edge_aligned: 0, cond_shift: 0 }, o);

test("every branch runs without throwing", () => {
  const cases = [
    row({}),
    row({ structure: 0.42, changed: { h3_repr_steering: [true, false] } }),
    row({ changed: { conditioning: [null, 1] }, cond_shift: 1.0, structure: 0.5 }),
    row({ changed: { h3_block_repeat: ["", "40"] }, structure: 0.95, detail: 1.3, edge_aligned: 0.7 }),
    row({ same_seed: false, changed: { seed: [1, 2] }, structure: 0.3 }),
    row({ structure: 0.95, detail: 0.8, changed: { cfg: [1, 2] } }),
  ];
  cases.forEach((r) => [null, 0.1].forEach((f) => assert.equal(typeof dpVerdict(r, f), "string")));
});

test("a moved shot still gets a detail verdict — the old 0.85 cliff hid it", () => {
  // The block-repeat sweep's best row was structure 0.831, so a >=0.85 gate returned "the
  // shot moved" for all 19 comparisons and never once evaluated sharpness. Both halves get
  // reported now: how much the shot moved AND what happened to the detail.
  const v = dpVerdict(row({ structure: 0.831, detail: 0.99,
                            changed: { h3_block_repeat: ["", "40"] } }), 0);
  assert.match(v, /shot moved 17%/);
  assert.match(v, /detail unchanged/);
});

test("sharper is reachable when all three numbers agree", () => {
  const v = dpVerdict(row({ structure: 0.95, detail: 1.3, edge_aligned: 0.7,
                            changed: { h3_block_repeat: ["", "40"] } }), 0);
  assert.match(v, /SHARPER/);
});

test("more detail spread evenly is grain, not sharpening", () => {
  const v = dpVerdict(row({ structure: 0.95, detail: 1.3, edge_aligned: 0.05,
                            changed: { h3_block_repeat: ["", "40"] } }), 0);
  assert.match(v, /grain, not sharpening/);
});

test("two nearly unrelated pictures refuse a detail verdict", () => {
  const v = dpVerdict(row({ structure: 0.09, detail: 1.25,
                            changed: { h3_block_repeat: ["", "40"] } }), 0);
  assert.match(v, /detail cannot be compared/);
});

test("a pair with nothing changed reports the noise floor", () => {
  const v = dpVerdict(row({ structure: 0.9 }), 0.1);
  assert.match(v, /noise floor: two identical runs differ by 10%/);
});

test("movement is expressed against the measured floor", () => {
  const v = dpVerdict(row({ structure: 0.42, changed: { h3_repr_steering: [true, false] } }), 0.1);
  assert.match(v, /5\.8x your noise floor/);
  assert.match(v, /detail unchanged/);   // and the detail half survives the prefix
});

test("a real prompt change voids the comparison, a small drift does not", () => {
  assert.match(dpVerdict(row({ changed: { conditioning: [null, 1] }, cond_shift: 1.0, structure: 0.5 }), 0.1),
    /prompt or scenes changed/);
  const drift = dpVerdict(row({ changed: { conditioning: [null, 0.02] }, cond_shift: 0.02,
                                structure: 0.95, detail: 1.3, edge_aligned: 0.7 }), 0.1);
  assert.match(drift, /SHARPER/i);
});

test("the floor is the median of the quiet pairs, so one outlier cannot set it", () => {
  const rows = [row({ structure: 0.9 }), row({ structure: 0.91 }), row({ structure: 0.2 }),
                row({ structure: 0.5, changed: { cfg: [1, 2] } })];
  assert.ok(Math.abs(dpNoiseFloor(rows) - 0.1) < 0.02);
});

test("no quiet pairs means no floor rather than a made-up one", () => {
  assert.equal(dpNoiseFloor([row({ changed: { cfg: [1, 2] } })]), null);
});
