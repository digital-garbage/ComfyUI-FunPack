// What the project generates at.
//
// The scene text has its own path (boot wires the prompt box to it); this is
// the other half -- settings that belong to the project rather than to any one
// scene, which is what makes "regenerate" mean "again, at the same size".

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let createProject;
test.before(async () => {
  setupDom();
  ({ createProject } = await import("../projects.js"));
});
test.after(() => teardownDom());

/** A server that remembers the last body it was PUT. */
function server({ video = {} } = {}) {
  const sent = [];
  const project = { id: "abcdef012345", name: "Untitled", scenes: [], video, updated_at: 1 };
  globalThis.fetch = async (path, opts = {}) => {
    const body = opts.body ? JSON.parse(opts.body) : null;
    if (opts.method === "PUT") sent.push(body);
    return {
      ok: true, status: 200,
      json: async () => (path.endsWith("/projects") && !opts.method
        ? { projects: [{ id: project.id }] }
        : (opts.method === "PUT" ? { ...body, updated_at: 2 } : project)),
    };
  };
  return { sent, project };
}

test("a setting is kept on the project and saved with it", async () => {
  const { sent } = server();
  const p = createProject({});
  await p.start();

  p.setVideo("width", 832);
  p.setVideo("length", 97);
  await p.flush();

  assert.deepEqual(p.video, { width: 832, length: 97 });
  assert.deepEqual(sent.at(-1).video, { width: 832, length: 97 });
});

test("setting a value to what it already is saves nothing", async () => {
  // Every commit of a number box calls this, including the ones that changed
  // nothing -- a blur is a commit. Each would be a PUT of the whole project.
  const { sent } = server({ video: { width: 512 } });
  const p = createProject({});
  await p.start();

  p.setVideo("width", 512);
  await p.flush();
  assert.equal(sent.length, 0, "an unchanged value was saved anyway");

  p.setVideo("width", 832);
  await p.flush();
  assert.equal(sent.length, 1);
});

test("the settings a project was opened with are what it reports", async () => {
  server({ video: { width: 832, height: 480 } });
  const p = createProject({});
  await p.start();
  assert.deepEqual(p.video, { width: 832, height: 480 });
});

test("with no project open there is nothing to set and nothing breaks", () => {
  const p = createProject({});
  p.setVideo("width", 832);
  assert.deepEqual(p.video, {});
});

test("a run's result lands on the scene it was started for, not the current one", async () => {
  // A run takes minutes and the user goes on clicking. Reading "the selected
  // scene" when it finishes attaches the picture to whatever they wandered to.
  const { sent } = server();
  const p = createProject({});
  await p.start();
  p.addScene();
  p.addScene();
  const [first, second] = p.scenes.map((s) => s.id);

  const startedFor = first;                    // what boot captures at Generate
  p.select(second);                            // the user moves on
  p.setResult(startedFor, "/view?filename=a.png");

  assert.equal(p.scenes[0].result, "/view?filename=a.png");
  assert.equal(p.scenes[1].result, null);
  await p.flush();
  assert.equal(sent.at(-1).scenes[0].result, "/view?filename=a.png");
});

test("renaming a project moves the name in the File menu too", async () => {
  // The listing and the open project are the same name seen twice; a menu that
  // kept the old one until a reload is a menu that lies about what is open.
  const { sent } = server();
  const p = createProject({});
  await p.start();

  p.rename("Rooftops");
  assert.equal(p.project.name, "Rooftops");
  assert.deepEqual(p.recent.map((r) => r.name), ["Rooftops"]);

  await p.flush();
  assert.equal(sent.at(-1).name, "Rooftops");
});

test("a rename that says nothing changes nothing", async () => {
  const { sent } = server();
  const p = createProject({});
  await p.start();

  p.rename("   ");
  p.rename("Untitled");
  await p.flush();
  assert.equal(sent.length, 0);
});

test("changing a project setting tells whoever draws from it", async () => {
  // The timeline's clip widths and its ruler are computed from the project's
  // length. A setting that changes without saying so leaves them showing the
  // proportions of the value before it.
  server();
  const drew = [];
  const p = createProject({ onChange: () => drew.push(1) });
  await p.start();
  const before = drew.length;

  p.setVideo("length", 97);
  assert.ok(drew.length > before, "nothing was told the project changed");
});
