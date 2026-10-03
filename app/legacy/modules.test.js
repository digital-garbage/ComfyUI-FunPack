const test = require("node:test");
const assert = require("node:assert");
const fs = require("fs");
const vm = require("vm");

function load() {
  const added = [];
  const root = { console: { warn() {} }, document: {},
    MutationObserver: class { observe() {} disconnect() {} takeRecords() { return [{ addedNodes: added }]; } } };
  vm.runInNewContext(fs.readFileSync(__dirname + "/modules.js", "utf8"), root);
  return { root, added };
}

test("a module that mounts is ok", () => {
  const { root } = load();
  let ran = false;
  assert.equal(root.Modules.define("a", {}, () => { ran = true; }).ok, true);
  assert.ok(ran);
});

test("a module that throws is rejected with its reason, its page additions removed, and the next one still loads", () => {
  const { root, added } = load();
  let removed = false;
  added.push({ remove() { removed = true; } });
  const bad = root.Modules.define("bad", {}, () => { throw new Error("boom"); });
  assert.equal([bad.ok, bad.why, removed].join(), "false,boom,true");
  assert.equal(root.Modules.define("good", {}, () => {}).ok, true);
  assert.equal(root.Modules.status().map((m) => m.ok).join(), "false,true");
});

test("a module missing what it stands on is rejected without running", () => {
  const { root } = load();
  let ran = false;
  const m = root.Modules.define("x", { needs: ["Nope"] }, () => { ran = true; });
  assert.equal([m.ok, ran, m.why].join(), "false,false,needs Nope");
});
