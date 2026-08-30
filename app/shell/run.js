// A generation, from Generate to a picture on screen.
//
// ComfyUI's own queue and its own socket, not a second one alongside them.
// FunPack has no queue of its own and must not grow one: two things tracking
// what is running is two things that can disagree, and the one the user
// believes is whichever spoke last. So this speaks to /prompt, /interrupt and
// /ws exactly as any ComfyUI client does, and holds only what the UI needs to
// draw -- which is a view of the server's state and never a copy of it.
//
// Everything the outside world provides is injected: `fetch`, `connect`, the
// client id. A run can then be driven end to end in a test without a server,
// and the test exercises this file rather than a mock of it.

export const IDLE = "idle";
export const QUEUED = "queued";
export const RUNNING = "running";
export const DONE = "done";
export const FAILED = "failed";
export const CANCELLED = "cancelled";

/** The socket, which is a different question from what a run is doing. */
export const OFFLINE = "offline";     // not opened yet
export const LIVE = "live";
export const LOST = "lost";           // dropped, and trying again

/** How long to wait before the next attempt: 1s, 2s, 4s ... capped at 30. */
export const backoff = (attempt) => Math.min(30_000, 1000 * 2 ** Math.max(0, attempt));

/** Where ComfyUI serves a produced file from. */
export function viewUrl(image, base = "") {
  const q = new URLSearchParams({
    filename: image.filename || "",
    subfolder: image.subfolder || "",
    type: image.type || "output",
  });
  return `${base}/view?${q}`;
}

export function createRun({
  fetch: doFetch = globalThis.fetch,
  connect,
  clientId,
  base = "",
} = {}) {
  const listeners = new Set();
  let state = {
    phase: IDLE, promptId: null, progress: null, images: [], audio: [],
    error: null, node: null,
    // Whether this page can still hear ComfyUI at all. Part of the run state
    // rather than a second thing to subscribe to: one state object is what
    // stops two parts of the UI disagreeing about what is happening.
    connection: OFFLINE,
  };

  const emit = (next) => {
    state = { ...state, ...next };
    for (const fn of listeners) {
      try { fn(state); } catch { /* one listener must not stop the others */ }
    }
  };

  // Only messages carrying OUR prompt id are acted on. ComfyUI broadcasts some
  // of them (an interrupt is broadcast to every client), so a second tab
  // generating would otherwise drive this one's progress bar and, worse, mark
  // this run finished when it was somebody else's that ended.
  //
  // Strict about the id being KNOWN, which it is not for the length of the
  // /prompt round trip. Treating "no id yet" as "everything is mine" adopted a
  // stranger's result during that window -- this run went to Done, showing
  // somebody else's picture, before it had started.
  const mine = (data) => Boolean(state.promptId) && Boolean(data)
    && data.prompt_id === state.promptId;

  // Messages that arrived before the id did. Ours cannot be told apart from
  // anyone else's yet, so they are kept and sorted out once the id is known --
  // dropping them would lose a run that finished inside the round trip.
  // Bounded, because on a busy shared server this fills with other people's
  // traffic and nothing here needs the older end of it.
  const PENDING = 200;
  let pending = [];

  function settle(promptId) {
    emit({ promptId });
    const held = pending;
    pending = [];
    for (const message of held) handle(message);
    // A cancel asked for while the id was unknown was never sent, because a
    // null prompt_id is a GLOBAL interrupt at ComfyUI: it stops whatever any
    // client is running. Now that the run has a name, it can be stopped by it.
    if (cancelWanted) sendCancel();
  }

  function handle(message) {
    const { type, data } = message || {};
    if (!type) return;

    // Not yet named: hold it rather than guess whose it is.
    //
    // Whenever the id is unknown, not only while queueing. A page that reloads
    // during a generation opens its socket first and learns which run is its
    // own a moment later, and everything arriving in between was being dropped
    // -- including the whole finish of a run that ended inside that moment,
    // which then left the app at Ready with the result lost.
    if (!state.promptId) {
      pending.push(message);
      if (pending.length > PENDING) pending.shift();
      return;
    }

    switch (type) {
      case "execution_start":
        if (mine(data)) emit({ phase: RUNNING, progress: null });
        break;

      case "progress_state": {
        if (!mine(data)) break;
        // One entry per active node; the run's progress is the node that is
        // actually working. Summing them would report a percentage of a total
        // nobody can know before the graph has run.
        const nodes = Object.values(data.nodes || {});
        const active = nodes.find((n) => n.state === "running") || nodes[nodes.length - 1];
        if (active && active.max) {
          emit({ phase: RUNNING, node: active.display_node_id ?? active.node_id,
                 progress: { value: active.value || 0, max: active.max } });
        }
        break;
      }

      case "executed": {
        if (!mine(data)) break;
        const output = data.output || {};
        // Appended, not replaced: a graph may save more than once, and the
        // second save is not a correction of the first.
        if (output.images) emit({ images: [...state.images, ...output.images] });
        if (output.audio) emit({ audio: [...state.audio, ...output.audio] });
        break;
      }

      case "execution_success":
        if (mine(data)) emit({ phase: DONE, progress: null, node: null });
        break;

      case "execution_error":
        if (!mine(data)) break;
        emit({
          phase: FAILED, progress: null,
          // ComfyUI's own words. Rewriting them would lose the one thing that
          // says which node and why.
          error: {
            node: data.node_type || data.node_id,
            message: data.exception_message || "the run failed",
            traceback: data.traceback || null,
          },
        });
        break;

      case "execution_interrupted":
        // Broadcast, so the prompt id is the only way to know it was ours.
        if (data && data.prompt_id === state.promptId) {
          emit({ phase: CANCELLED, progress: null, node: null });
        }
        break;

      default:
        break;                                   // not ours to interpret
    }
  }

  // A cancel asked for before the run had a name.
  let cancelWanted = false;

  async function sendCancel() {
    cancelWanted = false;
    const id = state.promptId;

    // ComfyUI's own state-agnostic cancel: it interrupts a job that is running
    // and dequeues one that is only pending, atomically, and answers
    // {"cancelled": false} for an id that has already finished.
    //
    // /interrupt alone was not enough and looked like it was. It only consults
    // the RUNNING half of the queue, so cancelling a job waiting behind another
    // one returned 200, logged "not currently running, skipping interrupt", and
    // let the job run in full when its turn came -- a cancel that did nothing
    // and said nothing, which is worse than a cancel that fails.
    let response;
    try {
      response = await doFetch(`${base}/api/jobs/${encodeURIComponent(id)}/cancel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
    } catch {
      response = null;
    }
    if (response && response.ok) {
      // A job that was only PENDING is dequeued and never runs, so no
      // execution_interrupted is ever sent -- nothing would arrive to move the
      // UI off "Queued", and the run would look stuck forever after a cancel
      // that worked. A RUNNING job is left to the socket, which says when it
      // actually stopped rather than when it was asked to.
      const body = await response.json().catch(() => ({}));
      if (body && body.cancelled && state.phase === QUEUED) {
        emit({ phase: CANCELLED, progress: null, node: null });
      }
      return response;
    }

    // An older ComfyUI without that route. Interrupting is all it can do, and
    // it is right for the case people hit most: a run that is under way.
    return doFetch(`${base}/interrupt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt_id: id }),
    });
  }

  let socket = null;
  let attempt = 0;
  let waiting = null;

  // A CLOSED socket is still an object. "Already listening" therefore cannot be
  // "there is a socket": every drop -- a laptop sleeping, a tunnel to a rented
  // box hiccuping, ComfyUI restarting -- was permanent for that tab, and
  // silent. Progress and results simply stopped arriving while the GPU carried
  // on, and only a manual reload brought them back.
  //
  // readyState is undefined on a stand-in socket, and a stand-in is a live one.
  const alive = (s) => Boolean(s) && (s.readyState === undefined || s.readyState <= 1);

  function listen() {
    if (typeof connect !== "function") return null;
    if (alive(socket)) return socket;

    if (waiting) { clearTimeout(waiting); waiting = null; }
    socket = connect(clientId);
    if (!socket) return null;

    socket.addEventListener("open", () => { attempt = 0; emit({ connection: LIVE }); });
    socket.addEventListener("message", (event) => {
      // Binary frames are previews, which nothing here consumes yet. A parse
      // failure must not take the socket down with it.
      if (typeof event.data !== "string") return;
      try { handle(JSON.parse(event.data)); } catch { /* not for us */ }
    });

    // close AND error: a refused connection fires error without ever closing on
    // some browsers, and a dropped one closes without an error on others.
    const dropped = (which) => () => {
      if (which !== socket) return;              // an older socket letting go
      emit({ connection: LOST });
      retry();
    };
    socket.addEventListener("close", dropped(socket));
    socket.addEventListener("error", dropped(socket));
    return socket;
  }

  function retry() {
    if (waiting) return;
    const wait = backoff(attempt);
    attempt += 1;
    waiting = setTimeout(() => { waiting = null; listen(); }, wait);
    // Not a reason to keep a process alive: under a test runner a pending timer
    // is the difference between a suite that finishes and one that hangs.
    if (waiting && typeof waiting.unref === "function") waiting.unref();
  }

  return {
    get state() { return { ...state }; },

    /** Prompt ids seen on the socket while this page had no run of its own.
     *  Whoever is reattaching can ask ComfyUI which of them belongs here. */
    seen: () => [...new Set(pending.map((m) => m && m.data && m.data.prompt_id)
      .filter(Boolean))],
    subscribe(fn) {
      listeners.add(fn);
      // Guarded like every later delivery. Handing a subscriber the current
      // state is a delivery too, and doing it bare meant one bad listener took
      // down whoever was wiring the UI up -- at startup, before anything is on
      // screen to say so.
      try { fn(state); } catch { /* the same rule as emit */ }
      return () => listeners.delete(fn);
    },
    handle,                                       // for tests and for a socket owned elsewhere
    listen,
    /** Stop trying. Only a page being torn down wants this. */
    stopListening() {
      if (waiting) { clearTimeout(waiting); waiting = null; }
      if (socket && typeof socket.close === "function") socket.close();
      socket = null;
    },

    /** Queue a prompt. Resolves with the prompt id, or throws with the reason. */
    async start(prompt) {
      listen();
      pending = [];
      cancelWanted = false;
      emit({ phase: QUEUED, promptId: null, progress: null, images: [], audio: [],
             error: null, node: null });

      let response;
      try {
        response = await doFetch(`${base}/prompt`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt, client_id: clientId }),
        });
      } catch (err) {
        // The queue could not be reached at all -- no ComfyUI, or the dev
        // server, which serves the app and has no queue behind it.
        emit({ phase: FAILED, error: { message: `the queue could not be reached: ${err.message}` } });
        throw err;
      }

      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        // ComfyUI refuses a bad graph here, BEFORE anything loads a model, and
        // says which node and why. That is the message worth showing.
        const detail = body.error && (body.error.message || body.error.type);
        emit({ phase: FAILED, error: {
          message: detail || `the queue refused this graph (${response.status})`,
          nodes: body.node_errors || null,
        } });
        throw new Error(detail || `queue refused: ${response.status}`);
      }

      settle(body.prompt_id || null);
      return body.prompt_id;
    },

    /** Ask for the run to stop. */
    async cancel() {
      if (state.phase !== QUEUED && state.phase !== RUNNING) return false;
      if (!state.promptId) {
        // Nothing to name yet. Remembered rather than sent: ComfyUI reads a
        // missing prompt_id as "interrupt whatever is running", which on a
        // shared box is someone else's generation, and on this one is not
        // necessarily the run the user is looking at.
        cancelWanted = true;
        return true;
      }
      await sendCancel();
      // Not marked cancelled here: the server says when it actually stopped,
      // and claiming it early is how a UI ends up showing "cancelled" over a
      // run that is still burning GPU time.
      return true;
    },

    /**
     * Take over a run already in flight -- after a reload, the generation the
     * previous page load queued. The id comes from ComfyUI's own queue, so this
     * adopts a real run rather than assuming one.
     */
    adopt(promptId, { running = true } = {}) {
      if (!promptId) return false;
      listen();
      // A job waiting its turn is queued, not running. Saying "working" over a
      // run that has not started is the same small lie as saying "cancelled"
      // over one that has not stopped.
      emit({ phase: running ? RUNNING : QUEUED, images: [], audio: [],
             error: null, node: null });
      settle(promptId);
      return true;
    },

    images: () => state.images.map((image) => viewUrl(image, base)),
    reset() {
      pending = [];
      cancelWanted = false;
      emit({ phase: IDLE, promptId: null, progress: null, images: [],
             audio: [], error: null, node: null });
    },
  };
}
