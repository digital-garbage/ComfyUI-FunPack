// Reordering a row of id-tagged children by drag, shared by every element that
// has one (the scene strip, the timeline track).
//
// Delegated on the container: cells are torn down and rebuilt on every redraw,
// and listeners bound to them would have to be reattached every time. One set
// here survives all of that.
//
// The dragged item's id lives on `dataTransfer`, not in a closure variable. A
// closure var tracks by NOTHING the browser understands, so nothing keeps it in
// step with the actual drag session: `dragend` fires on the ORIGINAL source
// element, and a redraw mid-drag (an unrelated remove/undo firing on its own
// input channel while the mouse is still held down) can detach that element
// from the tree -- its `dragend` then never bubbles here, and a closure var set
// at dragstart is never cleared, ready to be picked up by the next unrelated
// drop that lands on this row from somewhere else entirely. `dataTransfer` has
// no such gap: it belongs to the actual OS-level drag session, unaffected by
// which DOM node currently receives the events, and a drop from any OTHER
// session simply never had this MIME type set.
const MIME = "text/x-funpack-reorder";

/** wireDragReorder(node, ".cx-cell-selector", (draggedId, targetId) => {}) */
export function wireDragReorder(node, cellSelector, onReorder) {
  const cellOf = (e) => e.target.closest(cellSelector);
  node.addEventListener("dragstart", (e) => {
    const cell = cellOf(e);
    if (!cell || !e.dataTransfer) return;
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData(MIME, cell.dataset.id || "");
    cell.classList.add("cx-dragging");
  });
  node.addEventListener("dragend", (e) => { cellOf(e)?.classList.remove("cx-dragging"); });
  node.addEventListener("dragover", (e) => {
    // `getData` only returns the real value on "drop" (a browser security
    // restriction during drag) -- `types` is what dragover can actually read.
    if (e.dataTransfer && e.dataTransfer.types.includes(MIME)) e.preventDefault();
  });
  node.addEventListener("drop", (e) => {
    if (!e.dataTransfer) return;
    const draggedId = e.dataTransfer.getData(MIME);
    if (!draggedId) return;
    e.preventDefault();
    const cell = cellOf(e);
    const targetId = cell ? cell.dataset.id : null;
    if (onReorder && targetId && targetId !== draggedId) onReorder(draggedId, targetId);
  });
}
