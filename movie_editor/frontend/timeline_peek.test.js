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
