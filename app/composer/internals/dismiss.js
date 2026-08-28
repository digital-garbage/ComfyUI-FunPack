// One dismissal stack for the whole app.
//
// Each overlay in v4 wired its own outside-click and Escape handler, so Escape
// inside a modal with an open autocomplete closed the modal -- every listener
// fired, and none of them knew about the others. Here there is a single pair of
// document listeners over a stack, and Escape pops the TOP entry only.

const stack = [];
let listening = false;

function onKeyDown(event) {
  if (event.key !== "Escape" || !stack.length) return;
  const top = stack[stack.length - 1];
  if (!top.closeOnEsc) return;
  // stopPropagation matters: without it a parent's own key handler also runs
  // and closes a second layer on one press.
  event.stopPropagation();
  event.preventDefault();
  top.dismiss("escape");
}

function onPointerDown(event) {
  if (!stack.length) return;
  const top = stack[stack.length - 1];
  if (!top.closeOnOutside) return;
  for (const node of top.nodes) {
    if (node && node.contains(event.target)) return;
  }
  top.dismiss("outside");
}

function listen() {
  if (listening) return;
  // Capture phase, so an overlay that stops propagation on its own content
  // cannot accidentally suppress dismissal of something above it.
  document.addEventListener("keydown", onKeyDown, true);
  document.addEventListener("pointerdown", onPointerDown, true);
  listening = true;
}

function unlisten() {
  if (!listening) return;
  document.removeEventListener("keydown", onKeyDown, true);
  document.removeEventListener("pointerdown", onPointerDown, true);
  listening = false;
}

/**
 * push({ nodes, onDismiss, closeOnEsc, closeOnOutside }) -> handle
 *
 * `nodes` are the regions that count as "inside" -- a popover plus the control
 * that opened it, so clicking the trigger again does not dismiss-then-reopen.
 */
export function push({ nodes = [], onDismiss, closeOnEsc = true, closeOnOutside = true } = {}) {
  const entry = {
    nodes: [].concat(nodes).filter(Boolean),
    closeOnEsc,
    closeOnOutside,
    done: false,
    dismiss(reason) {
      if (entry.done) return;
      entry.done = true;
      remove(entry);
      if (onDismiss) onDismiss(reason);
    },
  };
  stack.push(entry);
  listen();
  return {
    dismiss: (reason = "manual") => entry.dismiss(reason),
    // Closing without running onDismiss: the caller is already tearing down.
    release() {
      entry.done = true;
      remove(entry);
    },
    get depth() { return stack.indexOf(entry) + 1; },
    get isTop() { return stack[stack.length - 1] === entry; },
  };
}

function remove(entry) {
  const i = stack.indexOf(entry);
  if (i !== -1) stack.splice(i, 1);
  if (!stack.length) unlisten();
}

export const depth = () => stack.length;

export function _resetDismiss() {
  stack.length = 0;
  unlisten();
}
