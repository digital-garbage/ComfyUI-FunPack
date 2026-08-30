// This tab's identity to ComfyUI, and its socket.
//
// The id is REMEMBERED, because ComfyUI addresses progress and results to the
// client that queued the run: a reload with a new id hears nothing about the
// generation already in flight -- the run continues, the GPU time is spent, and
// the app sits there saying Ready.
//
// Remembered per TAB, not per browser. ComfyUI keys its socket table by this
// id and evicts the old entry when a new socket arrives with the same one
// (`self.sockets.pop(sid, None)` on connect), so two tabs sharing an id means
// the second tab silently takes the first tab's updates -- the first freezes
// mid-progress while the GPU carries on. Worse, the socket's `finally` pops
// that id again on close, so closing the dead tab disconnects the live one.
// sessionStorage is exactly the right lifetime: it survives a reload of this
// tab, which is what reattach needs, and does not reach another tab.
//
// The one gap: duplicating a tab copies its sessionStorage, so a duplicate
// starts out sharing the id. That is the same collision, from a deliberate act,
// and there is nothing readable on the page that would tell the two apart.

const KEY = "funpack.client-id";

export function clientId() {
  try {
    const kept = window.sessionStorage.getItem(KEY);
    if (kept) return kept;
  } catch { /* private mode: a fresh id per load, which still works */ }

  const fresh = (globalThis.crypto && globalThis.crypto.randomUUID)
    ? globalThis.crypto.randomUUID()
    : `funpack-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  try { window.sessionStorage.setItem(KEY, fresh); } catch { /* not fatal */ }
  return fresh;
}

/** ComfyUI's socket, on whatever origin served this page. */
export function connect(id) {
  const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
  return new WebSocket(`${scheme}//${window.location.host}/ws?clientId=${encodeURIComponent(id)}`);
}

/**
 * What this browser already has in the queue: {promptId, running}, or null.
 *
 * Asked of ComfyUI's own queue rather than remembered locally: a run this page
 * queued may have finished, been cancelled, or been dequeued while the page was
 * gone, and a remembered id would reattach the UI to a run that no longer
 * exists.
 *
 * BOTH halves. `/queue` answers with `queue_running` and `queue_pending`, and a
 * job sits in the second one from the moment /prompt returns until the worker
 * picks it up -- always briefly, and for as long as it takes whenever something
 * is running ahead of it. Reading only the running half meant a reload in that
 * window found nothing, handed Generate back, and let the user queue the same
 * job twice: the first copy then ran unnamed, its messages failing the id check,
 * burning time while the UI narrated the second.
 *
 * Items in both are [number, prompt_id, prompt, extra_data, outputs], and
 * extra_data carries the client_id ComfyUI was given when the run was queued.
 */
export async function queuedFor(id, { fetch: doFetch = globalThis.fetch, base = "" } = {}) {
  let response;
  try {
    response = await doFetch(`${base}/queue`);
  } catch {
    return null;                                 // no queue reachable: nothing to reattach to
  }
  if (!response.ok) return null;

  const body = await response.json().catch(() => ({}));
  // Running first: if this browser somehow has both, the one under way is the
  // one worth showing. Which half it came from is returned as well, because a
  // job waiting its turn is not a job in progress, and telling the user it is
  // working when nothing is happening yet is a small lie the UI does not need
  // to tell.
  for (const [half, running] of [[body.queue_running, true], [body.queue_pending, false]]) {
    for (const item of half || []) {
      if (!Array.isArray(item) || item.length < 4) continue;
      const [, promptId, , extra] = item;
      if (extra && extra.client_id === id && promptId) return { promptId, running };
    }
  }
  return null;
}

/**
 * Which of these runs was this browser's, among ones that have already
 * finished.
 *
 * Only ids seen on this page's own socket are ever asked about, so nothing here
 * can resurrect a result from an hour ago: a run that ended while the page was
 * loading is the case this exists for, and it is indistinguishable from a lost
 * result without it. A history entry's `prompt` is
 * [number, prompt_id, prompt, extra_data, outputs].
 */
export async function finishedFor(id, candidates, { fetch: doFetch = globalThis.fetch, base = "" } = {}) {
  for (const promptId of [...(candidates || [])].reverse()) {
    let response;
    try {
      response = await doFetch(`${base}/history/${encodeURIComponent(promptId)}`);
    } catch {
      return null;                               // no history reachable
    }
    if (!response.ok) continue;

    const body = await response.json().catch(() => ({}));
    const entry = body && body[promptId];
    const prompt = entry && entry.prompt;
    if (Array.isArray(prompt) && prompt.length > 3 && prompt[3]
        && prompt[3].client_id === id) {
      return promptId;
    }
  }
  return null;
}
