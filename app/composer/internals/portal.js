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
    // NO POSITIONING, no transform, no filter, no contain, no z-index -- ever.
    //
    // Each of those makes this element a stacking context, and then the whole
    // ladder is measured INSIDE it rather than against the page: every rung,
    // from a tooltip at 300 to a modal at 800, collapses to whatever this one
    // element resolves to. It said exactly that and then set `position: fixed`,
    // which creates a stacking context all by itself -- so a menu at z 500 was
    // painted under a button at z 1, and the only reason it was not obvious
    // everywhere is that almost nothing in the app claims a z-index at all.
    //
    // Nothing is needed here anyway: every layer mounted into this root is
    // itself `position: fixed`, so a static, zero-sized parent lays nothing out
    // and intercepts nothing.
    root.style.pointerEvents = "none";
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
