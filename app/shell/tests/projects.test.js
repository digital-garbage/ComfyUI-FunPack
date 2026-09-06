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
  await p.flush();   // setResultFor's same-project branch schedules a save like
                      // any other edit -- left pending, its real debounce timer
                      // outlives this test and can fire mid a LATER one.
});

test("downloadUrl names the current project, and nothing when there is none", async () => {
  const p = createProject({});
  assert.equal(p.downloadUrl(), null, "a download link before anything is open");

  server();                             // default project id abcdef012345
  await p.start();
  assert.equal(p.downloadUrl(), "/funpack/api/projects/abcdef012345/download");
});

test("importing a project file opens whatever the server hands back, as its own project", async () => {
  // The server strips the id (a file imported twice, or on the machine it
  // came from, must not overwrite anything already here) -- this only has to
  // send what the file held and open what comes back.
  const posted = [];
  globalThis.fetch = async (path, opts = {}) => {
    if (String(path).endsWith("/projects/import")) {
      posted.push(JSON.parse(opts.body));
      return { ok: true, status: 200, json: async () => ({
        id: "newid000001", name: "Imported", scenes: [{ id: "s1", text: "", result: null }],
        video: {}, updated_at: 5,
      }) };
    }
    return { ok: true, status: 200, json: async () => ({ projects: [] }) };
  };

  const p = createProject({});
  const result = await p.importProject({ id: "someoldid0001", name: "Imported", scenes: [] });

  assert.equal(p.project.id, "newid000001");
  assert.equal(p.project.name, "Imported");
  assert.equal(p.selectedId, "s1", "an imported project with scenes has nothing selected");
  assert.deepEqual(p.recent.map((r) => r.id), ["newid000001"]);
  assert.equal(posted[0].id, "someoldid0001", "the client's job is to send the file, not clean it");
  assert.equal(result.id, "newid000001");
  assert.equal(p.canUndo, false, "an import just opened has no past to step into");
});

test("a project open when an import happens is not what flush() saves afterward", async () => {
  // flush() is called before the import switches `project` -- if it ran
  // AFTER, whatever debounced edit was pending on the OLD project would be
  // saved against the NEW one's id instead.
  const { sent } = server();
  const p = createProject({});
  await p.start();
  p.rename("About to leave");           // scheduled, not yet sent

  globalThis.fetch = async (path, opts = {}) => {
    if (String(path).endsWith("/projects/import")) {
      return { ok: true, status: 200, json: async () => ({
        id: "newid000002", name: "Imported", scenes: [], video: {}, updated_at: 5 }) };
    }
    const body = opts.body ? JSON.parse(opts.body) : null;
    if (opts.method === "PUT") sent.push(body);
    return { ok: true, status: 200, json: async () => (body ? { ...body, updated_at: 2 } : {}) };
  };

  await p.importProject({ name: "Imported", scenes: [] });

  assert.equal(sent.length, 1, "the pending rename was never saved, or was saved against the wrong project");
  assert.equal(sent[0].id, "abcdef012345", "flushed against the OLD project, which is what flush() is for");
  assert.equal(p.project.id, "newid000002");
});

test("an edit made WHILE an import is in flight is still saved, against the old project", async () => {
  // The gap this closes: importProject()'s OWN initial flush() finds nothing
  // dirty (the edit hasn't happened yet), then the edit lands DURING the
  // import's network round trip, on the project still open at that moment.
  // scheduleSave() has to remember which object that was -- the debounce
  // timer fires only after the import has already switched `project` out
  // from under it.
  const { sent } = server();
  const p = createProject({});
  await p.start();                    // project abcdef012345

  let resolveImport;
  globalThis.fetch = async (path, opts = {}) => {
    if (String(path).endsWith("/projects/import")) {
      return new Promise((resolve) => {
        resolveImport = () => resolve({ ok: true, status: 200, json: async () => (
          { id: "newid000003", name: "Imported", scenes: [], video: {}, updated_at: 5 }) });
      });
    }
    const body = opts.body ? JSON.parse(opts.body) : null;
    if (opts.method === "PUT") sent.push(body);
    return { ok: true, status: 200, json: async () => (body ? { ...body, updated_at: 2 } : {}) };
  };

  const importing = p.importProject({ name: "Imported", scenes: [] });
  await Promise.resolve(); await Promise.resolve(); await Promise.resolve();
  assert.equal(p.project.id, "abcdef012345", "the import switched before the edit -- this test proves nothing");

  p.rename("Typed while importing");  // scheduled against the still-current OLD project

  resolveImport();
  await importing;
  assert.equal(p.project.id, "newid000003", "the import did not complete");

  await new Promise((r) => setTimeout(r, 700));    // past the real debounce window

  assert.equal(sent.length, 1, "the edit made during the import never reached the server");
  assert.equal(sent[0].id, "abcdef012345", "saved against the OLD project, not the one now open");
  assert.equal(sent[0].name, "Typed while importing");
});

test("two imports racing land on whichever was clicked last, not whichever answered last", async () => {
  const p = createProject({});
  await p.start();

  const pending = {};
  globalThis.fetch = async (path, opts = {}) => {
    if (String(path).endsWith("/projects/import")) {
      const body = JSON.parse(opts.body);
      return new Promise((resolve) => {
        pending[body.name] = () => resolve({ ok: true, status: 200, json: async () => (
          { id: `id-${body.name}`, name: body.name, scenes: [], video: {}, updated_at: 5 }) });
      });
    }
    return { ok: true, status: 200, json: async () => ({ projects: [] }) };
  };

  const first = p.importProject({ name: "A", scenes: [] });    // clicked first
  await Promise.resolve();
  const second = p.importProject({ name: "B", scenes: [] });   // clicked second, while A is still in flight
  await Promise.resolve();

  // A answers LAST, after B -- a real, ordinary network race, not a
  // contrived one.
  pending.B();
  await second;
  assert.equal(p.project.name, "B");

  pending.A();
  await first;

  assert.equal(p.project.name, "B", "the earlier, superseded import overwrote the one clicked last");
  assert.equal(p.project.id, "id-B");
});

test("a superseded import deletes its own orphaned project instead of leaving it behind", async () => {
  const deleted = [];
  const pending = {};
  globalThis.fetch = async (path, opts = {}) => {
    if (String(path).endsWith("/projects/import")) {
      const body = JSON.parse(opts.body);
      return new Promise((resolve) => {
        pending[body.name] = () => resolve({ ok: true, status: 200, json: async () => (
          { id: `id-${body.name}`, name: body.name, scenes: [], video: {}, updated_at: 5 }) });
      });
    }
    if (opts.method === "DELETE") { deleted.push(path); return { ok: true, status: 200, json: async () => ({}) }; }
    return { ok: true, status: 200, json: async () => ({ projects: [] }) };
  };

  const p = createProject({});
  await p.start();
  const first = p.importProject({ name: "A", scenes: [] });
  await Promise.resolve();
  const second = p.importProject({ name: "B", scenes: [] });
  await Promise.resolve();

  pending.B(); await second;
  pending.A(); await first;

  assert.deepEqual(deleted, ["/funpack/api/projects/id-A"], "the superseded project was never cleaned up");
});

test("start() losing a race to a switch the user already made does not revert it", async () => {
  // build() wires the File menu (and so onProject -> newProject/open/import)
  // before start() is ever awaited -- start()'s own network calls are real
  // gaps a user's click can finish inside of.
  let resolveList;
  globalThis.fetch = async (path, opts = {}) => {
    if (path === "/funpack/api/projects" && opts.method === "GET") {
      return new Promise((resolve) => { resolveList = () => resolve(
        { ok: true, status: 200, json: async () => ({ projects: [] }) }); });
    }
    if (opts.method === "POST" && path.endsWith("/projects")) {
      const body = JSON.parse(opts.body);
      return { ok: true, status: 200, json: async () => (
        { id: "usercreated0", name: body.name, scenes: [{ id: "s1" }], video: {}, updated_at: 2 }) };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };

  const p = createProject({});
  const starting = p.start();          // in flight -- list() has not answered yet
  await Promise.resolve();

  await p.newProject("MyNewThing");    // the user's own action completes first
  assert.equal(p.project.name, "MyNewThing");

  resolveList();                       // start()'s list() finally answers
  await starting;

  assert.equal(p.project.name, "MyNewThing", "start() caught up and silently reverted the switch");
});

test("an undo during an in-flight open supersedes it too, not just another switch", async () => {
  const store = {
    a: { id: "aaaaaaaaaaaa", name: "A", scenes: [{ id: "s1", text: "", result: null }], updated_at: 1 },
    b: { id: "bbbbbbbbbbbb", name: "B", scenes: [{ id: "s1", text: "", result: null }], updated_at: 1 },
  };
  let resolveOpenB;
  globalThis.fetch = async (path, opts = {}) => {
    if (path === "/funpack/api/projects" && opts.method === "GET") {
      return { ok: true, status: 200, json: async () => ({ projects: [{ id: store.a.id }] }) };
    }
    if (String(path).endsWith(store.b.id) && opts.method === "GET") {
      return new Promise((resolve) => { resolveOpenB = () => resolve(
        { ok: true, status: 200, json: async () => store.b }); });
    }
    if (opts.method === "PUT") return { ok: true, status: 200, json: async () => JSON.parse(opts.body) };
    return { ok: true, status: 200, json: async () => store.a };
  };

  const p = createProject({});
  await p.start();                    // project A
  p.setText(p.selectedId, "edited");  // undo history now has the original

  const opening = p.open(store.b.id); // in flight
  await new Promise((r) => setTimeout(r, 0));   // past open()'s own flush() first

  p.undo();                           // restores A's original text, DURING the open
  assert.equal(p.project.id, store.a.id);
  assert.equal(p.scenes[0].text, "");

  resolveOpenB();
  await opening;

  // The open lost the race it was already in when the undo happened -- it
  // must not land on top of the undo a moment later.
  assert.equal(p.project.id, store.a.id, "the in-flight open overwrote the undo");
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
