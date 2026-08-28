// One overlay root, appended to <body>.
//
// Every floating thing lives here rather than beside whatever opened it. An
// overlay nested inside app layout inherits that layout's stacking context, so
// its z-index is measured against its siblings instead of the page -- which is
// how v4 ended up with 99999 and 200000 and still had two things fighting.

let root = null;

export const ROOT_ID = "composer-overlays";

export function portal() {
  if (root && root.isConnected) return root;
  root = document.getElementById(ROOT_ID);
  if (!root) {
    root = document.createElement("div");
    root.id = ROOT_ID;
    // No transform, filter, backdrop-filter or contain on this element, ever:
    // each of those would create a stacking context and undo the whole point.
    root.style.position = "fixed";
    root.style.inset = "0";
    root.style.pointerEvents = "none";
    root.style.zIndex = "0";
  }
  if (!root.isConnected) document.body.appendChild(root);
  return root;
}

/** Put `node` on the overlay root. Children re-enable their own pointer events. */
export function mount(node) {
  node.style.pointerEvents = "auto";
  portal().appendChild(node);
  return node;
}

export function unmount(node) {
  // Both halves matter: with no root yet, an unparented node's parentNode is
  // also null, and `null === null` would send removeChild to nothing.
  if (root && node && node.parentNode === root) root.removeChild(node);
}

export function _resetPortal() {
  if (root && root.isConnected) root.remove();
  root = null;
}
