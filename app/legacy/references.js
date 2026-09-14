// Reference numbering — the "R1, R2, …" a marked media item shows.
//
// The number is counted WITHIN the item's own kind, because that is how a reference slot
// actually resolves: builder._reference_by_slot takes the marks of one kind, in mark order,
// and picks the nth. "Reference video 1" is the first video marked, not the first mark
// overall.
//
// Numbering the marks globally instead made the two disagree, and the disagreement was not
// cosmetic: mark an image and then a video, and the video is labelled R2 everywhere while
// the only slot that can reach it is "Reference video 1". There was no way to have an R1
// video and an R1 image at the same time, so one of a matched pair always looked like the
// second of something.
(function () {
  "use strict";

  /** Every marked reference of `kind`, in mark order, as media-bin entries. */
  function referencesOfKind(marks, bin, kind) {
    const byId = new Map((bin || []).map((m) => [m.id, m]));
    return (marks || [])
      .map((id) => byId.get(id))
      .filter((m) => m && (m.kind || "image") === kind);
  }

  /** The R number this media shows, counted within its own kind. 0 when it is not marked
   *  (or not in the bin — an id left in the list by a deleted file is not a reference). */
  function referenceNumber(marks, bin, mediaId) {
    const byId = new Map((bin || []).map((m) => [m.id, m]));
    const self = byId.get(mediaId);
    if (!self || !(marks || []).includes(mediaId)) return 0;
    const kind = self.kind || "image";
    let n = 0;
    for (const id of marks) {
      const m = byId.get(id);
      if (!m || (m.kind || "image") !== kind) continue;
      n += 1;
      if (id === mediaId) return n;
    }
    return 0;
  }

  /** How many references share this media's kind — the "of N" half of the badge. */
  function referenceCountOfKind(marks, bin, kind) {
    return referencesOfKind(marks, bin, kind).length;
  }

  const api = { referenceNumber, referenceCountOfKind, referencesOfKind };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.References = api;
})();
