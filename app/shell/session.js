// The two things the page does around a run: start one, and take back over one
// that was already going.

//
// Kept out of boot.js because boot.js runs on import and needs a document, so
// nothing in it could be driven by a test -- and both of the faults that lived
// here were ordering faults in exactly that untested wiring: a button still
// live during a round trip, and a socket listening before anyone knew which run
// was ours.

import { DONE, FAILED, CANCELLED } from "./run.js";

const TERMINAL = new Set([DONE, FAILED, CANCELLED]);

/**
 * Resolves with the phase a run ends at -- done, failed, or cancelled.
 *
 * Subscribe FIRST, before whatever starts the run: subscribing after risks
 * missing the very transition being waited for. subscribe() delivers the
 * CURRENT state immediately, which is not a new transition and is not what
 * this is waiting for -- if a page reload adopts a run that already finished,
 * that first delivery would resolve this before the caller's own "start"
 * step has even run. Only a state delivered AFTER this call counts.
 *
 * The returned promise carries its own `.cancel()`: subscribing before the
 * caller knows whether a run will actually start is the whole point (see
 * above), but a start attempt that gets refused before it ever reaches
 * run.start() -- an incomplete pipeline, a queue that says no -- never
 * transitions the run at all, so nothing would ever resolve this and the
 * subscription would sit on `run` forever. Call `.cancel()` when the attempt
 * this was waiting on turns out never to have started.
 */
export function waitForTerminal(run) {
  let unsubscribe = () => {};
  const promise = new Promise((resolve) => {
    let first = true;
    unsubscribe = run.subscribe((state) => {
      if (first) { first = false; return; }
      if (TERMINAL.has(state.phase)) { unsubscribe(); resolve(state.phase); }
    });
  });
  promise.cancel = () => unsubscribe();
  return promise;
}

/**
 * onGenerate: ask what to queue, then queue it, once at a time.
 *
 * The button only goes dead when the RUN reaches "queued", and reaching it takes
 * a round trip to ask what to queue. For the whole of that round trip the button
 * was live, so two clicks each posted their own /prompt: two jobs burning GPU
 * time, with the UI following one of them.
 */
export function createGenerator({ run, transport, check, ready, slots, values, inputs, extra }) {
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
        plan = await check({
          ...(edited ? { slots: edited } : {}),
          // What the boxes on the main window hold, addressed by slot. Sent
          // apart from `slots` on purpose: the pipeline window owns the
          // structure and these own one value inside it, so neither has to
          // carry the other's.
          ...(inputs ? { inputs: inputs() } : {}),
          // What every panel in the app holds. Sent even when empty: "nothing
          // is set" is an answer, and the server tells the difference between
          // that and a client too old to send any.
          values: values ? values() : {},
        });
      } catch (err) {
        transport.say(`The pipeline could not be read: ${err.message}`);
        return false;
      }

      // True about the pipeline whatever happens next, so it is said before the
      // run is either started or refused -- and it stays up while the run goes.
      transport.warn((plan.notes || [])[0] || null);

      if (!plan.queueable || !plan.prompt) {
        // The first reason, whichever kind it is. A refusal and an unfinished
        // pipeline both stop the run, and both belong next to Generate.
        const stopping = [...(plan.refused || []), ...(plan.incomplete || [])];
        transport.say(stopping[0] || "This pipeline is not ready to run.");
        return false;
      }

      try {
        // Where this run belongs, so a reload mid-generation can still find it
        // -- omitted rather than sent empty, since ComfyUI stores whatever
        // extra_data it is given verbatim.
        const meta = extra ? extra() : null;
        await run.start(plan.prompt, meta ? { extra: meta } : {});
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
export function wire({ run, page, check, id, queuedFor, finishedFor, slots, values, inputs, extra, onAdopt }) {
  // One subscription draws everything a run affects. It lives here rather than
  // beside it in boot.js because subscribe() delivers the CURRENT state at
  // once, and doing that after the button was deliberately disabled handed it
  // straight back -- live-looking, and doing nothing when pressed.
  run.subscribe((state) => {
    page.transport.draw(state);
    // Whether ComfyUI is still there rides on the same state object, so it is
    // drawn by the same subscription: two things watching the connection is two
    // things that can disagree about whether the app is talking to anything.
    if (page.connection) page.connection.draw(state);
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

  const ready = reattach(run, id, { queuedFor, finishedFor, onAdopt }).then((adopted) => {
    page.transport.release(run.state);
    return adopted;
  });

  const generate = createGenerator({ run, transport: page.transport, check, ready, slots, values, inputs, extra });
  return { ready, generate };
}

/**
 * Resolves to `{promptId, sceneId, projectId}` for a run this page adopted, or
 * null. `sceneId`/`projectId` are whatever the run was queued with (see
 * `run.start`'s `extra`) -- null if it predates that, or was queued some other
 * way.
 *
 * `onAdopt(sceneId, projectId)`, when given, fires SYNCHRONOUSLY, before
 * `run.adopt()` -- not after this function returns. A run that already
 * finished while the page was reloading can go straight to DONE the moment it
 * is adopted (whatever the socket buffered gets replayed inside `adopt()`
 * itself), and a listener reading "which scene is this for" only after
 * `reattach` resolves would read it one tick too late, after that DONE
 * already passed the caller's own run.subscribe with nothing to attach it to.
 */
export async function reattach(run, id, { queuedFor, finishedFor, onAdopt } = {}) {
  try {
    const queued = await queuedFor(id);
    if (queued) {
      if (run.state.phase === "idle") {
        const sceneId = queued.sceneId || null;
        const projectId = queued.projectId || null;
        if (onAdopt) onAdopt(sceneId, projectId);
        run.adopt(queued.promptId, { running: queued.running });
        return { promptId: queued.promptId, sceneId, projectId };
      }
      return null;
    }

    const seen = run.seen();
    if (!seen.length || run.state.phase !== "idle") return null;
    let meta = null;
    const finished = await finishedFor(id, seen, { onFound: (extra) => { meta = extra; } });
    if (finished && run.state.phase === "idle") {
      const sceneId = (meta && meta.funpack_scene_id) || null;
      const projectId = (meta && meta.funpack_project_id) || null;
      if (onAdopt) onAdopt(sceneId, projectId);
      run.adopt(finished);
      return { promptId: finished, sceneId, projectId };
    }
    return null;
  } catch {
    return null;                                 // nothing to reattach to
  }
}
