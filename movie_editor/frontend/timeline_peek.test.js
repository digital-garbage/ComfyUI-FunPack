// Timeline peek (timeline_peek.js). Run: node --test movie_editor/frontend/timeline_peek.test.js
//
// The hover half is CSS and not testable here. What is testable — and what actually breaks —
// is the drag half: while a file is dragged, the pointer is in drag mode and a strip that
// closes underneath it is a drop target you cannot hit.
const test = require("node:test");
const assert = require("node:assert");

const P = require("./timeline_peek.js");

test("anything carrying a drag payload opens the strip", () => {
  assert.equal(P.isDrag({ dataTransfer: {} }), true);
});

test("a plain pointer event does not", () => {
  assert.equal(P.isDrag({}), false);
  assert.equal(P.isDrag(null), false);
});

test("a drag that turns out to be unusable still opens it", () => {
  // The strip has to open before anything can tell you the drop was refused.
  assert.equal(P.isDrag({ dataTransfer: { types: ["text/plain"] } }), true);
});

test("crossing from one clip to the next keeps it open", () => {
  // dragleave for the element being left fires AFTER dragenter for the one being entered,
  // so a plain boolean closes the strip mid-drag. This is why the tracker counts depth.
  const t = P.makeDragTracker();
  assert.equal(t.enter(), true);          // into the zone
  assert.equal(t.enter(), true);          // into a clip inside it
  assert.equal(t.leave(), true);          // out of the first clip — still inside the zone
  assert.equal(t.leave(), false);         // out of the zone entirely
});

test("leaving the window closes it once, not into negative depth", () => {
  const t = P.makeDragTracker();
  t.enter();
  assert.equal(t.leave(), false);
  assert.equal(t.leave(), false);
  assert.equal(t.depth, 0);               // a stray leave must not owe an extra enter
});

test("a dropped or abandoned drag closes it whatever the depth", () => {
  const t = P.makeDragTracker();
  t.enter(); t.enter(); t.enter();
  assert.equal(t.end(), false);
  assert.equal(t.depth, 0);
});

test("the preference is off unless it was explicitly turned on", () => {
  // No localStorage in node: stored() swallows the failure and reports the plain layout.
  assert.equal(P.get(), false);
});

test("apply is safe without a document", () => {
  assert.equal(P.apply(true, null), true);
  assert.equal(P.apply(false, null), false);
});


test("the header is never a trigger", () => {
  // Reaching for Generate, a pinned button or the Composer must not open the timeline.
  const inHead = { closest: (sel) => (sel === ".zone-head" ? {} : null) };
  const doc = { getElementById: () => ({ contains: () => true }) };
  assert.equal(P.opensOnPointer(inHead, doc), false);
});

test("the strip and the timeline itself both keep it open", () => {
  // The body has to count, or it would shut the instant it opened under the cursor.
  const inBody = { closest: () => null };
  const doc = { getElementById: () => ({ contains: () => true }) };
  assert.equal(P.opensOnPointer(inBody, doc), true);
});

test("something outside the zone opens nothing", () => {
  const elsewhere = { closest: () => null };
  const doc = { getElementById: () => ({ contains: () => false }) };
  assert.equal(P.opensOnPointer(elsewhere, doc), false);
});

test("the closed strip says what it is", () => {
  // The previous closed state was a 3px handle over a clipped lane, which read as breakage.
  assert.match(P.STRIP_LABEL, /timeline/i);
  assert.match(P.STRIP_LABEL, /hover/i);
});

// --- hover intent: crossing the strip on the way to the Dock is not a request -----

function fakeTimers() {
  let queue = [], id = 0;
  return {
    setTimer: (fn) => { queue.push({ id: ++id, fn }); return id; },
    clearTimer: (t) => { queue = queue.filter((q) => q.id !== t); },
    run: () => { const q = queue; queue = []; q.forEach((x) => x.fn()); },
    get pending() { return queue.length; },
  };
}

test("passing over the strip and leaving never opens it", () => {
  const T = fakeTimers();
  let opened = 0;
  const hi = P.makeHoverIntent({ ...T, onOpen: () => opened++ });
  hi.point();                 // pointer crosses the strip
  hi.away();                  // ...and carries on to the Dock
  T.run();                    // whatever was queued fires
  assert.equal(opened, 0);
  assert.equal(hi.isOpen, false);
});

test("staying on it opens it once the delay elapses", () => {
  const T = fakeTimers();
  let opened = 0;
  const hi = P.makeHoverIntent({ ...T, onOpen: () => opened++ });
  hi.point();
  assert.equal(hi.isPending, true);
  T.run();
  assert.equal(opened, 1);
  assert.equal(hi.isOpen, true);
});

test("moving around inside it does not re-arm the delay", () => {
  const T = fakeTimers();
  let opened = 0;
  const hi = P.makeHoverIntent({ ...T, onOpen: () => opened++ });
  hi.point(); T.run();
  hi.point(); hi.point();     // strip -> timeline -> a clip
  T.run();
  assert.equal(opened, 1);
});

test("leaving closes what is open", () => {
  const T = fakeTimers();
  let closed = 0;
  const hi = P.makeHoverIntent({ ...T, onClose: () => closed++ });
  hi.point(); T.run();
  hi.away();
  assert.equal(closed, 1);
  assert.equal(hi.isOpen, false);
});

test("leaving when nothing is open closes nothing", () => {
  const T = fakeTimers();
  let closed = 0;
  const hi = P.makeHoverIntent({ ...T, onClose: () => closed++ });
  hi.away();
  assert.equal(closed, 0);
});

test("a drag skips the delay", () => {
  // Dragging a file onto the timeline is deliberate; waiting for it would be wrong.
  const T = fakeTimers();
  const hi = P.makeHoverIntent(T);
  assert.equal(hi.now(), true);
  assert.equal(hi.isOpen, true);
  assert.equal(T.pending, 0);
});

test("a drag cancels a pending hover rather than opening twice", () => {
  const T = fakeTimers();
  let opened = 0;
  const hi = P.makeHoverIntent({ ...T, onOpen: () => opened++ });
  hi.point();
  hi.now();
  T.run();
  assert.equal(opened, 0);    // the hover open never fired; the drag already opened it
  assert.equal(hi.isOpen, true);
});

// --- a modal that belongs to the timeline holds it open --------------------------

test("a hold keeps it open when the pointer leaves", () => {
  // The rating picker mounts on document.body, so opening it leaves the zone.
  const T = fakeTimers();
  let closed = 0;
  let held = false;
  const hi = P.makeHoverIntent({ ...T, onClose: () => closed++, held: () => held });
  hi.point(); T.run();
  held = true;
  hi.away();
  assert.equal(closed, 0);
  assert.equal(hi.isOpen, true);
});

test("it closes once the hold is gone", () => {
  const T = fakeTimers();
  let closed = 0;
  let held = true;
  const hi = P.makeHoverIntent({ ...T, onClose: () => closed++, held: () => held });
  hi.now();
  hi.away();
  assert.equal(closed, 0);
  held = false;
  hi.away();
  assert.equal(closed, 1);
});

test("two holders do not release each other", () => {
  assert.equal(P.hold("rating-picker"), 1);
  assert.equal(P.hold("something-else"), 2);
  assert.equal(P.release("something-else"), 1);
  assert.equal(P.isHeld(), true);
  assert.equal(P.release("rating-picker"), 0);
  assert.equal(P.isHeld(), false);
});

test("releasing something that never held is not a release", () => {
  P.hold("rating-picker");
  P.release("never-held-this");
  assert.equal(P.isHeld(), true);
  P.release("rating-picker");
  assert.equal(P.isHeld(), false);
});
