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
export function createGenerator({ run, transport, check, ready, slots }) {
  let starting = false;

  return async function onGenerate() {
    // Whatever the page is still working out about runs already in progress
    // finishes first. On a reload during a generation the button is live and
    // the phase still reads "idle" for the length of the queue lookup, so a
    // click in that window queued a SECOND job and orphaned the first -- the
    // adopt that would have found it then refused, because the phase was no
    // longer idle, and the original run's messages were discarded from then on
    // as belonging to nobody.
    if (ready) await ready;
    if (starting || run.state.phase === "queued" || run.state.phase === "running") return false;
    starting = true;
    transport.generate.setDisabled(true);
    try {
      let plan;
      try {
        // The pipeline the user is looking at, or -- before anything has been
        // edited -- none, which asks the server for its own defaults. Sending
        // an empty array instead would be a request to run nothing at all.
        const edited = slots ? slots() : null;
        plan = await check(edited ? { slots: edited } : {});
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
/**
 * Everything the page does on load, in the order it has to happen.
 *
 * Kept here rather than in boot.js because boot.js runs on import and needs a
 * document, so nothing in it can be driven by a test -- and every fault on this
 * path so far has been an ORDERING fault in exactly that untested wiring. The
 * ordering is the thing under test, so the ordering lives where tests can reach
 * it.
 */
export function wire({ run, page, check, id, queuedFor, finishedFor, slots }) {
  // One subscription draws everything a run affects. It lives here rather than
  // beside it in boot.js because subscribe() delivers the CURRENT state at
  // once, and doing that after the button was deliberately disabled handed it
  // straight back -- live-looking, and doing nothing when pressed.
  run.subscribe((state) => {
    page.transport.draw(state);
    // Results go to the BIN, and the bin decides what is on screen. Pointing
    // the viewer at the newest image from here instead put two things in charge
    // of it: every progress message re-showed the newest result, so clicking an
    // older one in the bin held for as long as it took the next message to
    // arrive.
    if (page.bin) page.bin.absorb(state.images);
  });

  // The socket opens at load, not at Generate. A generation queued before a
  // reload keeps running on the server, and a page that only listens once it
  // starts one of its own hears nothing about it.
  run.listen();

  // Not awaited by the caller, but the button waits on it. Until this settles
  // nobody can know whether this page already has a run, and starting another
  // one on the strength of "phase is idle" is starting one on a guess.
  page.transport.hold("Looking for a run already in progress");

  const ready = reattach(run, id, { queuedFor, finishedFor }).then((adopted) => {
    page.transport.release(run.state);
    return adopted;
  });

  const generate = createGenerator({ run, transport: page.transport, check, ready, slots });
  return { ready, generate };
}

export async function reattach(run, id, { queuedFor, finishedFor } = {}) {
  try {
    const queued = await queuedFor(id);
    if (queued) {
      if (run.state.phase === "idle") {
        run.adopt(queued.promptId, { running: queued.running });
        return queued.promptId;
      }
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
