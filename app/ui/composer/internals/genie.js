// The Genie: a window pours out of the button that opens it and back into it,
// the way a macOS window goes to the Dock.
//
// Web Animations, not CSS classes: the start frame depends on where the button
// is, which only the DOM knows. Without Element.animate, or with reduced motion
// on (the tokens zero every duration), there is no animation and the callback
// runs at once -- so nothing ever waits on motion that is not coming.

import { duration, prefersReducedMotion } from "./motion.js";

const FULL = "polygon(0 0, 100% 0, 100% 100%, 0 100%)";
const IN_MS = 1.6, OUT_MS = 1.25;      // multiples of --dur-3

function frames(node, origin) {
  const a = node.getBoundingClientRect();
  const b = origin.getBoundingClientRect();
  if (!a.width || !b.width) return null;
  const dx = b.left + b.width / 2 - (a.left + a.width / 2);
  const dy = b.top + b.height / 2 - (a.top + a.height / 2);
  const below = dy > 0;
  const funnel = (w) => (below
    ? `polygon(0 0, 100% 0, ${50 + w}% 100%, ${50 - w}% 100%)`
    : `polygon(${50 - w}% 0, ${50 + w}% 0, 100% 100%, 0 100%)`);
  return [
    { transform: `translate(${dx}px, ${dy}px) scale(.05, .03)`, clipPath: funnel(6), opacity: 0.4, offset: 0 },
    { transform: `translate(${dx * 0.3}px, ${dy * 0.3}px) scale(.55, .5)`, clipPath: funnel(26), opacity: 1, offset: 0.55 },
    { transform: "none", clipPath: FULL, opacity: 1, offset: 1 },
  ];
}

function run(node, origin, reverse, done) {
  const source = typeof origin === "function" ? origin() : origin;
  const ms = duration(3) * (reverse ? OUT_MS : IN_MS);
  if (!source || !node.animate || !ms || prefersReducedMotion()) { done(); return; }
  const keys = frames(node, source);
  if (!keys) { done(); return; }
  const anim = node.animate(keys, {
    duration: ms, fill: "both",
    easing: reverse ? "cubic-bezier(.6, 0, .8, .4)" : "cubic-bezier(.2, .6, .3, 1)",
    direction: reverse ? "reverse" : "normal",
  });
  const finish = () => { try { anim.cancel(); } catch { /* already gone */ } done(); };
  anim.onfinish = finish;
  anim.oncancel = () => done();
}

/** Open: the node grows out of `origin` (an element or a function returning one). */
export const genieIn = (node, origin) => run(node, origin, false, () => {});

/** Close: the node shrinks into `origin`, then `done` runs (tear down there). */
export const genieOut = (node, origin, done) => run(node, origin, true, done);
