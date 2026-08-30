// What the server would queue, asked for rather than assembled here.
//
// The graph is built by core from the pipeline's slots, so the app never
// composes one: two builders would be two answers to "what runs", and the one
// the user sees would be whichever was written last. The app asks, and gets
// either a prompt or the reasons there is not one.

const ENDPOINT = "/funpack/api/pipeline";
const NODES = "/funpack/api/nodes";

/**
 * check({slots, values}) -> { slots, refused, incomplete, notes, queueable, prompt }
 *
 * `refused` and `incomplete` are different things and stay apart: refused means
 * the edit did not happen, incomplete means it did and the pipeline still is
 * not ready -- an unset file picker on a fresh install is the normal case, not
 * a failure. `notes` is a third thing again: the pipeline is fine and something
 * about it is worth saying, such as settings that will not be applied.
 */
export async function check({ fetch: doFetch = globalThis.fetch, slots, values,
                              action = "check", slot, node } = {}) {
  // `slots` and an action travel together on purpose: a remove is "take this
  // one out of THIS pipeline", and sending the action without the pipeline the
  // user is looking at would apply it to the server's defaults instead.
  const body = { action };
  if (slots) body.slots = slots;
  // Sent with the pipeline, every time. The server holds no copy of what the
  // panels say: two stores of "what the user picked" is two answers to what a
  // run used, and the one believed would be whichever was written last.
  if (values) body.values = values;
  if (slot !== undefined) body.slot = slot;
  if (node !== undefined) body.node = node;

  const response = await doFetch(ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    // A 400 here is a malformed REQUEST, which is the app's fault and not the
    // user's -- so it is reported as a refusal rather than dressed up as an
    // unfinished pipeline.
    return {
      slots: payload.slots || [],
      refused: payload.problems || payload.refused || [`the pipeline could not be read (${response.status})`],
      incomplete: [],
      notes: [],
      queueable: false,
      prompt: null,
    };
  }
  return {
    slots: payload.slots || [],
    refused: payload.refused || [],
    incomplete: payload.incomplete || [],
    notes: payload.notes || [],
    queueable: Boolean(payload.queueable),
    prompt: payload.prompt || null,
  };
}

/** The pipeline as the server holds it, before anything has been edited. */
export async function load({ fetch: doFetch = globalThis.fetch } = {}) {
  const response = await doFetch(ENDPOINT);
  if (!response.ok) throw new Error(`the pipeline could not be read (${response.status})`);
  const payload = await response.json();
  return {
    slots: payload.slots || [],
    incomplete: payload.incomplete || [],
    queueable: Boolean(payload.queueable),
  };
}

/**
 * describe(classes) -> { [className]: description | null }
 *
 * A null is an answer, not a gap: the slot points at a node this install does
 * not have. The caller asked about it, so it is in the reply saying so.
 */
export async function describe(classes, { fetch: doFetch = globalThis.fetch } = {}) {
  const wanted = [...new Set(classes)].filter(Boolean);
  if (!wanted.length) return {};
  const response = await doFetch(`${NODES}?classes=${encodeURIComponent(wanted.join(","))}`);
  if (!response.ok) throw new Error(`those nodes could not be described (${response.status})`);
  return (await response.json()).nodes || {};
}

/** Installed nodes matching a query, and how many there were before the cut. */
export async function search(query, { fetch: doFetch = globalThis.fetch, limit = 40 } = {}) {
  const response = await doFetch(
    `${NODES}/search?q=${encodeURIComponent(query || "")}&limit=${limit}`);
  if (!response.ok) throw new Error(`nodes could not be searched (${response.status})`);
  const payload = await response.json();
  return { nodes: payload.nodes || [], total: payload.total || 0 };
}
