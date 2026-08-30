// Starting a run, and taking one back over.
//
// This wiring used to live in boot.js, which runs on import and needs a
// document, so none of it could be driven -- and both faults found here were
// ordering faults in exactly that untested wiring.

import test from "node:test";
import assert from "node:assert/strict";

import { setupDom, teardownDom } from "../../composer/tests/_dom.js";

let createGenerator, reattach, wire, createTransport;
test.before(async () => {
  setupDom();
  await import("../../composer/composer.js");
  ({ createGenerator, reattach, wire } = await import("../session.js"));
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
  const t = createTransport({});
  document.body.appendChild(t.node);
  return {
    ...t,
    get text() { return t.status.node.textContent; },
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
    runningFor: async () => "still-going",
    finishedFor: async () => { throw new Error("history should not have been asked"); },
  });
  assert.equal(got, "still-going");
  assert.equal(run.state.phase, "running");
});

test("a run that finished during the load is found in history", async () => {
  const run = fakeRun();
  run.seen = () => ["ended-just-now"];
  const got = await reattach(run, "me", {
    runningFor: async () => null,
    finishedFor: async (_id, seen) => (seen.includes("ended-just-now") ? "ended-just-now" : null),
  });
  assert.equal(got, "ended-just-now");
});

test("history is not asked about a run this page never saw", async () => {
  const run = fakeRun();
  let asked = false;
  const got = await reattach(run, "me", {
    runningFor: async () => null,
    finishedFor: async () => { asked = true; return "something-old"; },
  });
  assert.equal(got, null);
  assert.equal(asked, false, "a result from before this page load could be resurrected");
});

test("a page already running its own generation is not reattached to another", async () => {
  const run = fakeRun("running");
  const got = await reattach(run, "me", {
    runningFor: async () => "some-other",
    finishedFor: async () => null,
  });
  assert.equal(got, null);
});

test("no queue and no history is silence, not an error", async () => {
  const run = fakeRun();
  const got = await reattach(run, "me", {
    runningFor: async () => { throw new TypeError("Failed to fetch"); },
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
    runningFor: async () => { await queueAnswered; return running; },
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
    runningFor: async () => { throw new TypeError("Failed to fetch"); },
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
    runningFor: () => new Promise(() => {}),      // never answers
    finishedFor: async () => null,
  });

  assert.equal(t.disabled, true);
  run.announce();                                 // any redraw at all
  assert.equal(t.disabled, true, "a redraw handed the button back mid-lookup");
  t.draw(run.state);
  assert.equal(t.disabled, true, "drawing the idle run handed the button back");
});

test("the result of a run is shown as it arrives", async () => {
  // The subscription draws the transport AND the preview. Moving it had to
  // carry both, and the viewer is the half with nothing else watching it.
  const shown = [];
  const t = transport();
  const run = fakeRun();
  wire({
    run,
    page: { transport: t, viewer: { setSource: (src, kind) => shown.push([src, kind]) } },
    check: plan(), id: "me",
    runningFor: async () => null, finishedFor: async () => null,
  });

  run.state = { ...run.state, phase: "done", images: [{ filename: "a.png", subfolder: "", type: "output" }] };
  run.announce();
  assert.deepEqual(shown, [["/view?filename=a.png&subfolder=&type=output", "image"]]);

  run.state = { ...run.state, images: [{ filename: "clip.mp4", subfolder: "", type: "output" }] };
  run.announce();
  assert.equal(shown[1][1], "video", "a video result was handed to an <img>");
});
