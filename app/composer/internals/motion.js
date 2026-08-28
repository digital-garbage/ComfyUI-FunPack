// Durations come from the tokens, so "how long" is answered in one place and
// the reduced-motion guard in tokens.css zeroes every one of them at once.

/** "120ms" | "0.12s" | "0" -> milliseconds. */
export function parseDuration(value) {
  const text = String(value ?? "").trim();
  if (!text) return 0;
  const match = /^(-?[\d.]+)(ms|s)?$/.exec(text);
  if (!match) return 0;
  const n = Number(match[1]);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, match[2] === "s" ? n * 1000 : n);
}

export function duration(step = 1) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(`--dur-${step}`);
  return parseDuration(value);
}

export const prefersReducedMotion = () =>
  Boolean(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

/**
 * Run `fn` after the element's transition, or immediately when motion is off.
 * Always fires: a transitionend that never arrives (the property did not change,
 * the node was hidden) would otherwise strand a teardown forever.
 */
export function afterTransition(node, step, fn) {
  const ms = duration(step);
  if (!ms) { fn(); return () => {}; }
  let done = false;
  const finish = () => { if (done) return; done = true; node.removeEventListener("transitionend", finish); clearTimeout(timer); fn(); };
  const timer = setTimeout(finish, ms + 50);
  node.addEventListener("transitionend", finish);
  return finish;
}
