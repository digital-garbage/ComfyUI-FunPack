// Where on the timeline "now" is, in seconds. One number, shared by whoever draws or acts at it.
export function createPlayhead() {
  let at = 0;
  const heard = new Set();
  return {
    playing: false,       // set by the monitor; read by whatever sounds along with it
    get at() { return at; },
    set(sec) { const next = Math.max(0, sec); if (next === at) return; at = next; heard.forEach((fn) => fn(at)); },
    on(fn) { heard.add(fn); return () => heard.delete(fn); },
  };
}
