// This browser's identity to ComfyUI, and its socket.
//
// The id is REMEMBERED. ComfyUI addresses progress and results to the client
// that queued the run, so a reload with a new id is a reload that hears nothing
// about the generation already in flight -- the run continues, the GPU time is
// spent, and the app sits there saying Ready. v4 learned this the hard way and
// kept a fixed id for exactly this reason.

const KEY = "funpack.client-id";

export function clientId() {
  try {
    const kept = window.localStorage.getItem(KEY);
    if (kept) return kept;
  } catch { /* private mode: a fresh id per load, which still works */ }

  const fresh = (globalThis.crypto && globalThis.crypto.randomUUID)
    ? globalThis.crypto.randomUUID()
    : `funpack-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  try { window.localStorage.setItem(KEY, fresh); } catch { /* not fatal */ }
  return fresh;
}

/** ComfyUI's socket, on whatever origin served this page. */
export function connect(id) {
  const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
  return new WebSocket(`${scheme}//${window.location.host}/ws?clientId=${encodeURIComponent(id)}`);
}
