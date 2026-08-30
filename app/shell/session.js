// The two things the page does around a run: start one, and take back over one
// that was already going.
//
// Kept out of boot.js because boot.js runs on import and needs a document, so
// nothing in it could be driven by a test -- and both of the faults that lived
// here were ordering faults in exactly that untested wiring: a button still
// live during a round trip, and a socket listening before anyone knew which run
// was ours.

/**
 * onGenerate: ask what to queue, then queue it, once at a time.
 *
 * The button only goes dead when the RUN reaches "queued", and reaching it takes
 * a round trip to ask what to queue. For the whole of that round trip the button
 * was live, so two clicks each posted their own /prompt: two jobs burning GPU
 * time, with the UI following one of them.
 */
export function createGenerator({ run, transport, check }) {
  let starting = false;

  return async function onGenerate() {
    if (starting || run.state.phase === "queued" || run.state.phase === "running") return false;
    starting = true;
    transport.generate.setDisabled(true);
    try {
      let plan;
      try {
        plan = await check({});
      } catch (err) {
        transport.say(`The pipeline could not be read: ${err.message}`);
        return false;
      }

      if (!plan.queueable || !plan.prompt) {
        // The first reason, whichever kind it is. A refusal and an unfinished
        // pipeline both stop the run, and both belong next to Generate.
        const stopping = [...(plan.refused || []), ...(plan.incomplete || [])];
        transport.say(stopping[0] || "This pipeline is not ready to run.");
        return false;
      }

      try {
        await run.start(plan.prompt);
        return true;
      } catch {
        return false;                            // run.state carries the reason
      }
    } finally {
      starting = false;
      // draw() owns the button from here; this only undoes the click-time
      // disable for the paths that never reached a run at all.
      transport.draw(run.state);
    }
  };
}

/**
 * Take back over whatever this tab had running before the page reloaded.
 *
 * Two questions, because a run can be in two places: still in the queue, or
 * already finished. The second is not an edge case -- the socket opens before
 * the queue can answer, and a run ending in that moment leaves the queue before
 * it is asked about, so without this the result is lost and the app says Ready.
 * Only ids this page actually saw on its own socket are asked about, so nothing
 * older than this page load can be resurrected.
 */
export async function reattach(run, id, { runningFor, finishedFor } = {}) {
  try {
    const running = await runningFor(id);
    if (running) {
      if (run.state.phase === "idle") { run.adopt(running); return running; }
      return null;
    }

    const seen = run.seen();
    if (!seen.length || run.state.phase !== "idle") return null;
    const finished = await finishedFor(id, seen);
    if (finished && run.state.phase === "idle") { run.adopt(finished); return finished; }
    return null;
  } catch {
    return null;                                 // nothing to reattach to
  }
}
