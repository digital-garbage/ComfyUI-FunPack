// Starting a run, and taking one back over.
//
// This wiring used to live in boot.js, which runs on import and needs a
// document, so none of it could be driven -- and both faults found here were
// ordering faults in exactly that untested wiring.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let createGenerator, reattach, wire, waitForTerminal, createTransport;
test.before(async () => {
  setupDom();
  await import("../../composer/composer.js");
  ({ createGenerator, reattach, wire, waitForTerminal } = await import("../session.js"));
  ({ createTransport } = await import("../transport.js"));
});
test.after(() => teardownDom());

// The REAL transport, not a stand-in.
//
// A hand-rolled one recorded setText and setDisabled into arrays and had a
// draw() that did nothing -- so it could not show the fault it was standing in
// for: the real draw() writes the status from the run state, and a refusal set
// just before it was overwritten with "Ready" a moment after it appeared. The
// test passed and the button did nothing on screen.
function transport() {
  // The real transport, mounted the way the shell mounts it: its controls go in
  // a zone head and what it is saying goes at the far end of the same head.
  // There is no bar -- an editor's controls belong to the region they act on.
  const t = createTransport({});
  const head = document.createElement("header");
  for (const handle of [...t.actions, ...t.status, t.warning]) head.appendChild(handle.node);
  document.body.replaceChildren(head);
  return {
    ...t,
    head,
    get text() { return t.statusText.node.textContent; },
    get disabled() { return t.generate.node.disabled; },
  };
}

/** A run whose start() can be held open, like a real POST.
 *
 *  Open by DEFAULT, and held only when a test asks. A gate that starts shut
 *  means any regression letting an extra start() through hangs the suite
 *  forever instead of failing it, and a hang says far less than an assertion. */
function fakeRun(phase = "idle") {
  const started = [];
  let release = () => {};
  let gate = Promise.resolve();
  return {
    hold() {
      gate = new Promise((r) => { release = () => r(); });
    },
    started,
    release: () => release(),
    state: { phase, promptId: null, progress: null, images: [], audio: [], error: null, node: null },
    listened: 0,
    listen() { this.listened += 1; },
    // The real run delivers the CURRENT state on subscribe, at once. Leaving
    // that off is how a fake stays tidier than the thing it stands in for: the
    // synchronous first delivery is exactly what handed a deliberately disabled
    // button straight back, and a fake without it could not show that.
    subscribers: new Set(),
    subscribe(fn) {
      this.subscribers.add(fn);
      try { fn(this.state); } catch { /* as the real one does */ }
      return () => this.subscribers.delete(fn);
    },
    announce() { for (const fn of this.subscribers) fn(this.state); },
    seen: () => [],
    adopt(id) {
      this.state = { ...this.state, phase: "running", promptId: id, images: [] };
      this.announce();
      return true;
    },
    async start(prompt) {
      started.push(prompt);
      this.state = { ...this.state, phase: "queued" };
      await gate;
      return "p-1";
    },
  };
}

const plan = (over = {}) => async () => ({
  slots: [], refused: [], incomplete: [], queueable: true, prompt: { "1": {} }, ...over,
});

test("two clicks during the round trip queue one run, not two", async () => {
  // The button only goes dead when the RUN is queued, and getting there takes a
  // round trip to ask what to queue. That round trip is the whole window.
  const run = fakeRun();
  run.hold();
  const t = transport();
  let asked = 0;
  const check = async () => { asked += 1; return (await plan()()); };
  const onGenerate = createGenerator({ run, transport: t, check });

  const first = onGenerate();
  const second = onGenerate();
  run.release();
  const [a, b] = await Promise.all([first, second]);

  assert.equal(run.started.length, 1, "two /prompt posts went out");
  assert.equal(asked, 1, "the pipeline was assembled twice for one click");
  assert.equal(a, true);
  assert.equal(b, false, "the second click reported that it had started a run");
});

test("the button goes dead at the click, not at the queue", async () => {
  const run = fakeRun();
  run.hold();
  const t = transport();
  const onGenerate = createGenerator({ run, transport: t, check: plan() });
  const running = onGenerate();
  assert.equal(t.disabled, true, "the button stayed live during the round trip");
  run.release();
  await running;
});

test("a run already in flight is not started again", async () => {
  for (const phase of ["queued", "running"]) {
    const run = fakeRun(phase);
    const onGenerate = createGenerator({ run, transport: transport(), check: plan() });
    assert.equal(await onGenerate(), false);
    assert.equal(run.started.length, 0, `a ${phase} run was started a second time`);
  }
});

test("a pipeline that is not ready says why, beside Generate", async () => {
  const run = fakeRun();
  const t = transport();
  const onGenerate = createGenerator({
    run, transport: t,
    check: plan({ queueable: false, prompt: null, incomplete: ["model: nothing is chosen"] }),
  });
  assert.equal(await onGenerate(), false);
  // Read off the element, so a later redraw wiping it is visible here.
  assert.equal(t.text, "model: nothing is chosen");
  assert.equal(run.started.length, 0);
  assert.equal(t.disabled, false, "a refused attempt left Generate dead");
});

test("a refusal is shown even when nothing is incomplete", async () => {
  const t = transport();
  const onGenerate = createGenerator({
    run: fakeRun(), transport: t,
    check: plan({ queueable: false, prompt: null, refused: ["a slot cannot feed itself"] }),
  });
  await onGenerate();
  assert.equal(t.text, "a slot cannot feed itself");
});

test("a pipeline that cannot be read at all is reported rather than swallowed", async () => {
  const t = transport();
  const onGenerate = createGenerator({
    run: fakeRun(), transport: t,
    check: async () => { throw new TypeError("Failed to fetch"); },
  });
  assert.equal(await onGenerate(), false);
  assert.match(t.text, /could not be read/);
});

test("a failed attempt leaves Generate usable again", async () => {
  const run = fakeRun();
  const t = transport();
  const onGenerate = createGenerator({
    run, transport: t, check: plan({ queueable: false, prompt: null }),
  });
  await onGenerate();
  assert.equal(t.disabled, false, "nothing gave the button back after the attempt ended");
  // And a second click is allowed, which it would not be if the guard stuck.
  assert.equal(await onGenerate(), false);
  assert.equal(t.disabled, false);
});

// --- reattach ---------------------------------------------------------------

test("a run still in the queue is taken back over", async () => {
  const run = fakeRun();
  const got = await reattach(run, "me", {
    queuedFor: async () => ({ promptId: "still-going", running: true, sceneId: null, projectId: null }),
    finishedFor: async () => { throw new Error("history should not have been asked"); },
  });
  assert.equal(got.promptId, "still-going");
  assert.equal(run.state.phase, "running");
});

test("a run that finished during the load is found in history", async () => {
  const run = fakeRun();
  run.seen = () => ["ended-just-now"];
  const got = await reattach(run, "me", {
    queuedFor: async () => null,
    finishedFor: async (_id, seen) => (seen.includes("ended-just-now") ? "ended-just-now" : null),
  });
  assert.equal(got.promptId, "ended-just-now");
});

test("which scene/project a run belongs to travels with it, for both halves", async () => {
  const run = fakeRun();
  const queued = await reattach(run, "me", {
    queuedFor: async () => ({ promptId: "q1", running: true, sceneId: "s1", projectId: "p1" }),
    finishedFor: async () => { throw new Error("should not be asked"); },
  });
  assert.deepEqual(queued, { promptId: "q1", sceneId: "s1", projectId: "p1" });

  const run2 = fakeRun();
  run2.seen = () => ["f1"];
  const finished = await reattach(run2, "me", {
    queuedFor: async () => null,
    finishedFor: async (_id, _seen, { onFound } = {}) => {
      if (onFound) onFound({ client_id: "me", funpack_scene_id: "s2", funpack_project_id: "p2" });
      return "f1";
    },
  });
  assert.deepEqual(finished, { promptId: "f1", sceneId: "s2", projectId: "p2" });
});

test("onAdopt fires BEFORE the run is adopted, not after reattach resolves", async () => {
  // An already-finished run can go straight to DONE inside adopt() itself
  // (whatever the socket buffered gets replayed there) -- a listener that
  // learns which scene this is for only once reattach's own promise settles
  // would learn it one tick after that DONE already reached run.subscribe with
  // no scene to attach it to. This is the ordering the fix actually depends on.
  const order = [];
  const run = fakeRun();
  const realAdopt = run.adopt.bind(run);
  run.adopt = (id, opts) => { order.push("adopt"); return realAdopt(id, opts); };
  run.subscribe((state) => { if (state.phase === "running") order.push("subscriber-saw-it"); });

  await reattach(run, "me", {
    queuedFor: async () => ({ promptId: "q1", running: true, sceneId: "s1", projectId: "p1" }),
    finishedFor: async () => { throw new Error("should not be asked"); },
    onAdopt: (sceneId, projectId) => { order.push(`onAdopt:${sceneId}:${projectId}`); },
  });

  assert.deepEqual(order, ["onAdopt:s1:p1", "adopt", "subscriber-saw-it"]);
});

test("history is not asked about a run this page never saw", async () => {
  const run = fakeRun();
  let asked = false;
  const got = await reattach(run, "me", {
    queuedFor: async () => null,
    finishedFor: async () => { asked = true; return "something-old"; },
  });
  assert.equal(got, null);
  assert.equal(asked, false, "a result from before this page load could be resurrected");
});

test("a page already running its own generation is not reattached to another", async () => {
  const run = fakeRun("running");
  const got = await reattach(run, "me", {
    queuedFor: async () => ({ promptId: "some-other", running: true }),
    finishedFor: async () => null,
  });
  assert.equal(got, null);
});

test("no queue and no history is silence, not an error", async () => {
  const run = fakeRun();
  const got = await reattach(run, "me", {
    queuedFor: async () => { throw new TypeError("Failed to fetch"); },
    finishedFor: async () => null,
  });
  assert.equal(got, null);
  assert.equal(run.state.phase, "idle");
});

test("a reason for not starting is not wiped by the next redraw", () => {
  // The redraw happens on the way out of every attempt, and an attempt that was
  // refused leaves the run exactly where it was: idle. Drawing an idle run says
  // "Ready", which is how the reason appeared and vanished in the same frame.
  const t = transport();
  t.say("model: nothing is chosen");
  t.draw({ phase: "idle", progress: null, images: [], error: null, node: null });
  assert.equal(t.text, "model: nothing is chosen");
});

test("a run that actually starts outranks the last refusal", () => {
  const t = transport();
  t.say("model: nothing is chosen");
  t.draw({ phase: "queued", progress: null, images: [], error: null, node: null });
  assert.equal(t.text, "Queued");
  t.draw({ phase: "idle", progress: null, images: [], error: null, node: null });
  assert.equal(t.text, "Ready", "the stale reason came back after a real run");
});

// --- the load ordering, which is the thing that keeps breaking ---------------
//
// Every fault on this path has been an ordering fault, and every one lived in
// boot.js, where nothing could reach it. These drive the real order.

function wired({ running = null, finished = null, phase = "idle" } = {}) {
  const t = transport();
  const run = fakeRun(phase);
  let releaseQueue;
  const queueAnswered = new Promise((r) => { releaseQueue = r; });

  const session = wire({
    run,
    page: { transport: t },
    check: plan(),
    id: "me",
    queuedFor: async () => { await queueAnswered; return running ? { promptId: running, running: true } : null; },
    finishedFor: async () => finished,
  });
  return { t, run, session, releaseQueue };
}

test("Generate waits for the page to work out whether a run is already going", async () => {
  // The reload-during-a-generation case. The button is on screen and the phase
  // still reads idle for the whole length of the queue lookup, so a click in
  // that window used to queue a second job and orphan the first.
  const { t, run, session, releaseQueue } = wired({ running: "already-going" });

  assert.equal(t.disabled, true, "Generate was live before anything was known");
  const clicked = session.generate();

  releaseQueue();
  await session.ready;
  assert.equal(await clicked, false, "a second run was queued over the one already going");
  assert.equal(run.started.length, 0);
  assert.equal(run.state.promptId, "already-going", "the run in progress was orphaned");
});

test("with nothing already running, Generate comes back and works", async () => {
  const { t, run, session, releaseQueue } = wired({ running: null });
  releaseQueue();
  await session.ready;

  assert.equal(t.disabled, false, "Generate never came back");
  assert.equal(t.text, "Ready", `the loading note stuck: ${t.text}`);

  const started = session.generate();
  run.release();
  assert.equal(await started, true);
  assert.equal(run.started.length, 1);
});

test("the page says what it is doing while it works it out", async () => {
  const { t, releaseQueue, session } = wired({ running: null });
  assert.match(t.text, /Looking for a run/);
  releaseQueue();
  await session.ready;
});

test("a run found in history is adopted before Generate is offered again", async () => {
  const { t, run, session, releaseQueue } = wired({ running: null, finished: "ended-just-now" });
  run.seen = () => ["ended-just-now"];
  releaseQueue();
  await session.ready;
  assert.equal(run.state.promptId, "ended-just-now");
  // The button now follows the RUN, not the lookup: the loading note is gone
  // and what is on screen is whatever that run is doing.
  assert.doesNotMatch(t.text, /Looking for a run/);
});

test("a queue that cannot be reached still gives the button back", async () => {
  // The dev server, and any ComfyUI that is down. Waiting forever on a lookup
  // that will never answer is the same as no button at all.
  const t = transport();
  const run = fakeRun();
  const session = wire({
    run, page: { transport: t }, check: plan(), id: "me",
    queuedFor: async () => { throw new TypeError("Failed to fetch"); },
    finishedFor: async () => null,
  });
  await session.ready;
  assert.equal(t.disabled, false);
  assert.equal(t.text, "Ready");
});

test("a state delivered while the page is still looking does not hand Generate back", () => {
  // subscribe() delivers the current state at once, and every later change
  // redraws. Each of those draws decides the button from the run's phase alone,
  // so an idle run -- which is what a page has while it is still finding out
  // whether it has a run at all -- re-enabled a button that had just been
  // deliberately disabled. Live-looking, and doing nothing when pressed.
  const t = transport();
  const run = fakeRun();
  wire({
    run, page: { transport: t }, check: plan(), id: "me",
    queuedFor: () => new Promise(() => {}),      // never answers
    finishedFor: async () => null,
  });

  assert.equal(t.disabled, true);
  run.announce();                                 // any redraw at all
  assert.equal(t.disabled, true, "a redraw handed the button back mid-lookup");
  t.draw(run.state);
  assert.equal(t.disabled, true, "drawing the idle run handed the button back");
});

test("results of a run go to the bin, and only to the bin", async () => {
  // The subscription draws the transport AND hands results on. It hands them to
  // the BIN, which owns what is on screen: pointing the viewer at the newest
  // image from here as well put two things in charge of one view, and every
  // progress message then walked a user's choice of an older result back to the
  // newest one.
  const taken = [];
  const shown = [];
  const t = transport();
  const run = fakeRun();
  wire({
    run,
    page: {
      transport: t,
      bin: { absorb: (images) => { taken.push(images); return []; } },
      viewer: { setSource: (src, kind) => shown.push([src, kind]) },
    },
    check: plan(), id: "me",
    queuedFor: async () => null, finishedFor: async () => null,
  });

  const images = [{ filename: "a.png", subfolder: "", type: "output" }];
  run.state = { ...run.state, phase: "done", images };
  run.announce();

  assert.deepEqual(taken[taken.length - 1], images);
  assert.deepEqual(shown, [], "the viewer was driven from here as well as from the bin");
});

test("a run found waiting in the queue is adopted as waiting", async () => {
  const t = transport();
  const run = fakeRun();
  const adopted = [];
  run.adopt = (id, opts) => { adopted.push([id, opts]); return true; };

  const session = wire({
    run, page: { transport: t }, check: plan(), id: "me",
    queuedFor: async () => ({ promptId: "waiting", running: false }),
    finishedFor: async () => null,
  });
  await session.ready;
  assert.deepEqual(adopted, [["waiting", { running: false }]]);
});

test("what the panels hold is sent with every run", async () => {
  // Until this was wired, every module panel in the app was decoration: the
  // values lived in the browser and nothing ever sent them, so a modifier
  // switched on was a modifier the run never heard about.
  const run = fakeRun();
  const asked = [];
  const check = async (body) => { asked.push(body); return plan()(); };
  const onGenerate = createGenerator({
    run, transport: transport(), check,
    values: () => ({ sampling_alg: { enabled: true } }),
  });

  await onGenerate();
  assert.deepEqual(asked[0].values, { sampling_alg: { enabled: true } });
});

test("with no panels holding anything, an empty set is still sent", async () => {
  // "Nothing is set" is an answer. Sending nothing at all is a client that
  // cannot say the difference, and the server would have to guess.
  const run = fakeRun();
  const asked = [];
  const check = async (body) => { asked.push(body); return plan()(); };
  await createGenerator({ run, transport: transport(), check })();
  assert.deepEqual(asked[0].values, {});
});

test("a setting that will not be applied is said, and does not stop the run", async () => {
  const run = fakeRun();
  const t = transport();
  const note = "nothing in this pipeline accepts module settings";
  const started = await createGenerator({
    run, transport: t, check: plan({ notes: [note] }),
  })();

  assert.equal(started, true, "a note stopped a run that was otherwise fine");
  assert.match(t.warning.node.textContent, /accepts module settings/);
  assert.equal(t.warning.node.hasAttribute("hidden"), false);
});

test("a note stays up while the run it is about is going", async () => {
  // The status line is redrawn by every message a run sends, so a warning put
  // there was gone the moment the run said "Queued" -- which is roughly when
  // the user looks at it.
  const run = fakeRun();
  const t = transport();
  await createGenerator({ run, transport: t, check: plan({ notes: ["will not be applied"] }) })();

  t.draw({ phase: "running", images: [], progress: { value: 1, max: 20 } });
  t.draw({ phase: "done", images: [] });
  assert.match(t.warning.node.textContent, /will not be applied/);
  assert.equal(t.warning.node.hasAttribute("hidden"), false);
});

test("a pipeline with nothing to say takes the note back down", async () => {
  const run = fakeRun();
  const t = transport();
  await createGenerator({ run, transport: t, check: plan({ notes: ["will not be applied"] }) })();
  assert.equal(t.warning.node.hasAttribute("hidden"), false);

  run.state = { ...run.state, phase: "idle" };
  await createGenerator({ run, transport: t, check: plan() })();
  assert.equal(t.warning.node.hasAttribute("hidden"), true, "a stale warning stayed up");
  assert.equal(t.warning.node.textContent.trim(), "");
});

test("what is typed on the main window is sent with the run", async () => {
  const run = fakeRun();
  const asked = [];
  const check = async (body) => { asked.push(body); return plan()(); };
  await createGenerator({
    run, transport: transport(), check,
    inputs: () => ({ positive: { text: "a cat on a roof" } }),
  })();

  assert.deepEqual(asked[0].inputs, { positive: { text: "a cat on a roof" } });
});

// --- waitForTerminal ---------------------------------------------------------

test("waitForTerminal resolves with the phase a run actually ends at", async () => {
  const run = fakeRun();
  const waiting = waitForTerminal(run);
  run.state = { ...run.state, phase: "running" };
  run.announce();
  run.state = { ...run.state, phase: "done" };
  run.announce();
  assert.equal(await waiting, "done");
});

test("the snapshot delivered at subscribe time is not mistaken for a transition", async () => {
  // A page reload adopting a run that already finished delivers ITS finished
  // state the moment subscribe() is called -- if that counted, this would
  // resolve before the caller's own "start" step had even run.
  const run = fakeRun("done");
  let resolved = false;
  waitForTerminal(run).then(() => { resolved = true; });
  await Promise.resolve(); await Promise.resolve();
  assert.equal(resolved, false, "resolved from the snapshot, not a real transition");

  run.announce();                     // the same "done" state, delivered again -- now a real one
  await Promise.resolve(); await Promise.resolve();
  assert.equal(resolved, true);
});

test("cancel() unsubscribes a waiter whose run never actually started", () => {
  // A generation refused before it ever reaches run.start() (an incomplete
  // pipeline, a queue that says no) never transitions the run -- nothing
  // would ever resolve this promise, and the subscription would sit on the
  // run forever without a way out.
  const run = fakeRun();
  assert.equal(run.subscribers.size, 0);
  const waiting = waitForTerminal(run);
  assert.equal(run.subscribers.size, 1);
  waiting.cancel();
  assert.equal(run.subscribers.size, 0, "cancel() did not unsubscribe");
});

test("cancel() after the promise already resolved is harmless", async () => {
  const run = fakeRun();
  const waiting = waitForTerminal(run);
  run.state = { ...run.state, phase: "done" };
  run.announce();
  assert.equal(await waiting, "done");
  assert.doesNotThrow(() => waiting.cancel());
});
