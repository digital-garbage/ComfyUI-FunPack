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

test("a run's result still lands on its scene after the user opened a DIFFERENT project", async () => {
  // A run takes minutes; nothing stops the user opening another project while
  // one is in flight. `setResult` alone looks the scene id up in whatever is
  // open NOW -- finds nothing in the other project, and drops the result with
  // no trace. `setResultFor` is what boot.js calls instead once it knows which
  // project the run belongs to.
  const store = {
    a: { id: "aaaaaaaaaaaa", name: "A", scenes: [{ id: "scene-a", text: "", result: null }], updated_at: 1 },
    b: { id: "bbbbbbbbbbbb", name: "B", scenes: [{ id: "scene-b", text: "", result: null }], updated_at: 1 },
  };
  const puts = [];
  globalThis.fetch = async (path, opts = {}) => {
    if (path === "/funpack/api/projects") {
      return { ok: true, status: 200, json: async () => ({ projects: [{ id: store.a.id }, { id: store.b.id }] }) };
    }
    const id = decodeURIComponent(String(path).split("/").pop());
    const key = id === store.a.id ? "a" : "b";
    if (opts.method === "PUT") {
      const body = JSON.parse(opts.body);
      store[key] = body;
      puts.push({ key, body });
      return { ok: true, status: 200, json: async () => ({ ...body, updated_at: (store[key].updated_at || 1) + 1 }) };
    }
    return { ok: true, status: 200, json: async () => store[key] };
  };

  const p = createProject({});
  await p.start();               // opens whatever `list()` names first -- project A
  const startedInProject = p.project.id;
  const startedForScene = p.selected.id;
  assert.equal(startedInProject, store.a.id);

  await p.open(store.b.id);      // the user wanders off to a different project
  assert.equal(p.project.id, store.b.id);

  await p.setResultFor(startedInProject, startedForScene, "/view?filename=a.png");

  // The project on screen (B) must be untouched.
  assert.equal(p.scenes[0].result, null);
  // Project A, not open, got the result written straight to its saved copy.
  const saved = puts.find((x) => x.key === "a");
  assert.ok(saved, "project A was never saved");
  assert.equal(saved.body.scenes[0].result, "/view?filename=a.png");
});

test("a result for a project that is open again goes through the normal live path", async () => {
  // The user switched away and back before the run finished -- the project is
  // open again by the time the result lands, so it should update on screen
  // immediately rather than only reaching the server.
  server();
  const p = createProject({});
  await p.start();
  p.addScene();
  const id = p.project.id;
  const sceneId = p.selectedId;

  await p.setResultFor(id, sceneId, "/view?filename=b.png");
  assert.equal(p.scenes[0].result, "/view?filename=b.png");
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

// --- undo ---------------------------------------------------------------

test("an edit can be taken back, and put back again", async () => {
  server();
  const p = createProject({});
  await p.start();
  p.addScene();
  const id = p.selectedId;

  p.setText(id, "a cat");
  p.setText(id, "a cat on a roof");
  assert.equal(p.selected.text, "a cat on a roof");

  assert.equal(p.undo(), true);
  assert.equal(p.selected.text, "a cat");
  assert.equal(p.undo(), true);
  assert.equal(p.selected.text, "");

  assert.equal(p.redo(), true);
  assert.equal(p.selected.text, "a cat");
});

test("undo puts back a scene that was removed, and the selection with it", async () => {
  server();
  const p = createProject({});
  await p.start();
  p.addScene();
  p.addScene();
  const [first, second] = p.scenes.map((s) => s.id);
  p.select(second);

  p.removeScene(second);
  assert.equal(p.scenes.length, 1);

  p.undo();
  assert.equal(p.scenes.length, 2);
  assert.ok(p.scenes.some((s) => s.id === second), "the scene came back as a different one");
  assert.equal(p.selectedId, second, "the selection did not follow the scene back");
  void first;
});

test("a selection that no longer names anything falls back rather than emptying", async () => {
  // Undo can remove the scene that is selected. A selection naming nothing
  // reads as "no scene" in every panel that draws from it.
  server();
  const p = createProject({});
  await p.start();
  p.addScene();
  p.addScene();
  const added = p.selectedId;

  p.undo();                               // takes the second scene away again
  assert.ok(p.selectedId, "the selection was left naming nothing");
  assert.notEqual(p.selectedId, added);
  assert.ok(p.scenes.some((s) => s.id === p.selectedId));
});

test("a new edit ends the redo line", async () => {
  // Otherwise a redo walks into a version that never followed from what is on
  // screen.
  server();
  const p = createProject({});
  await p.start();
  p.addScene();
  const id = p.selectedId;

  p.setText(id, "one");
  p.undo();
  assert.equal(p.canRedo, true);

  p.setText(id, "two");
  assert.equal(p.canRedo, false);
  assert.equal(p.redo(), false);
});

test("undo with nothing behind it does nothing, and says so", async () => {
  server();
  const p = createProject({});
  await p.start();
  assert.equal(p.canUndo, false);
  assert.equal(p.undo(), false);
  assert.equal(p.redo(), false);
});

test("a project just opened has no past to step into", async () => {
  // Undoing across a project switch would replace the open project with a
  // version of a DIFFERENT one.
  server();
  const p = createProject({});
  await p.start();
  p.addScene();
  assert.equal(p.canUndo, true);

  await p.newProject("Another");
  assert.equal(p.canUndo, false, "undo reached back past the project switch");
});

test("taking an edit back saves it, and tells whoever draws it", async () => {
  const { sent } = server();
  const drew = [];
  const opens = [];
  const p = createProject({ onChange: () => drew.push(1), onOpen: () => opens.push(1) });
  await p.start();
  p.addScene();
  p.setText(p.selectedId, "a cat");
  await p.flush();

  const before = drew.length;
  p.undo();
  assert.ok(drew.length > before, "nothing was told the project moved");
  assert.ok(opens.length > 1, "the controls that follow a project were not put back");

  await p.flush();
  assert.equal(sent.at(-1).scenes.at(-1).text, "", "the version saved was not the one on screen");
});
