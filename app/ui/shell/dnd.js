// Dragging a media-bin tile onto something. The bin says what it carries; whoever owns a drop target says what that means.
export const MEDIA_DRAG = "application/funpack-media";

/** When a bin tile is dropped on an element matching `selector`, call fn({id, kind}, element, event). -> remover. */
export function onMediaDrop(selector, fn, doc = document) {
  const carries = (e) => e.dataTransfer && [...(e.dataTransfer.types || [])].includes(MEDIA_DRAG);
  const target = (e) => (doc.querySelector('[role="dialog"]') ? null : e.target.closest && e.target.closest(selector));
  const over = (e) => { if (carries(e) && target(e)) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; } };
  const drop = (e) => {
    if (!carries(e)) return;
    const el = target(e);
    if (!el) return;
    let item;
    try { item = JSON.parse(e.dataTransfer.getData(MEDIA_DRAG)); } catch { return; }
    e.preventDefault();
    fn(item, el, e);
  };
  doc.addEventListener("dragover", over);
  doc.addEventListener("drop", drop);
  return () => { doc.removeEventListener("dragover", over); doc.removeEventListener("drop", drop); };
}
