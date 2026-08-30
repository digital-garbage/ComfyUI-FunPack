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
  const mine = (data) => !state.promptId || !data || data.prompt_id === state.promptId;

  function handle(message) {
    const { type, data } = message || {};
    if (!type) return;

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

  let socket = null;
  function listen() {
    if (socket || typeof connect !== "function") return socket;
    socket = connect(clientId);
    if (socket) {
      socket.addEventListener("message", (event) => {
        // Binary frames are previews, which nothing here consumes yet. A parse
        // failure must not take the socket down with it.
        if (typeof event.data !== "string") return;
        try { handle(JSON.parse(event.data)); } catch { /* not for us */ }
      });
    }
    return socket;
  }

  return {
    get state() { return { ...state }; },
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

    /** Queue a prompt. Resolves with the prompt id, or throws with the reason. */
    async start(prompt) {
      listen();
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

      emit({ promptId: body.prompt_id || null });
      return body.prompt_id;
    },

    /** Ask for the run to stop. */
    async cancel() {
      if (state.phase !== QUEUED && state.phase !== RUNNING) return false;
      await doFetch(`${base}/interrupt`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt_id: state.promptId }),
      });
      // Not marked cancelled here: the server says when it actually stopped,
      // and claiming it early is how a UI ends up showing "cancelled" over a
      // run that is still burning GPU time.
      return true;
    },

    images: () => state.images.map((image) => viewUrl(image, base)),
    reset: () => emit({ phase: IDLE, promptId: null, progress: null, images: [],
                        audio: [], error: null, node: null }),
  };
}
