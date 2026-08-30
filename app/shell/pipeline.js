// What the server would queue, asked for rather than assembled here.
//
// The graph is built by core from the pipeline's slots, so the app never
// composes one: two builders would be two answers to "what runs", and the one
// the user sees would be whichever was written last. The app asks, and gets
// either a prompt or the reasons there is not one.

const ENDPOINT = "/funpack/api/pipeline";

/**
 * check(values) -> { slots, refused, incomplete, queueable, prompt }
 *
 * `refused` and `incomplete` are different things and stay apart: refused means
 * the edit did not happen, incomplete means it did and the pipeline still is
 * not ready -- an unset file picker on a fresh install is the normal case, not
 * a failure.
 */
export async function check({ fetch: doFetch = globalThis.fetch, slots } = {}) {
  const body = { action: "check" };
  if (slots) body.slots = slots;

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
      queueable: false,
      prompt: null,
    };
  }
  return {
    slots: payload.slots || [],
    refused: payload.refused || [],
    incomplete: payload.incomplete || [],
    queueable: Boolean(payload.queueable),
    prompt: payload.prompt || null,
  };
}
