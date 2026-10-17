// Dragging a media-bin tile onto something. The bin says what it carries; whoever owns a drop target says what that means.
export const MEDIA_DRAG = "application/funpack-media";

// Drop targets by name, so a feature never writes a kit class. A raw selector still works (tests use one).
const TARGETS = { "lane:video": ".cx-nle-lane.cx-nle-lane-video", "lane:audio": ".cx-nle-lane.cx-nle-lane-audio", "lane:overlay": ".cx-nle-lane.cx-nle-lane-overlay" };

/** When a bin tile is dropped on a target (a name from TARGETS, or a selector), call
 *  fn({id, kind}, element, event, { clipId, row }). `clipId` is the clip under the drop, `row` the target's place among its kind
 *  (top first). -> remover. */
export function onMediaDrop(target_, fn, doc = document) {
  const selector = TARGETS[target_] || target_;
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
    const clip = e.target.closest(".cx-nle-clip[data-id]");
    fn(item, el, e, { clipId: clip && clip.dataset ? clip.dataset.id : null, get row() { return [...doc.querySelectorAll(selector)].indexOf(el); } });
  };
  doc.addEventListener("dragover", over);
  doc.addEventListener("drop", drop);
  return () => { doc.removeEventListener("dragover", over); doc.removeEventListener("drop", drop); };
}
