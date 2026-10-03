// Turns a scene's picked reference-media ids into slot overrides for whatever
// pipeline is current -- pulled out of boot.js so it can be tested without
// booting the whole app (boot.js runs start() on import).

/**
 * wireReferences(mediaIds, slots, reservedSlotIds) -> { overrides, unwired }
 *
 * `overrides` is keyed by slot id, merge these into the raw input overrides
 * sent with a run. `unwired` counts references that did not make it into the
 * run -- the pipeline has fewer "assets.reference_N" roles than the scene has
 * references (e.g. added under a bigger preset, then the preset was swapped
 * for a smaller one), two references' roles resolve to the same destination,
 * or a reference's slot is one the caller already assigned to something else
 * (`reservedSlotIds` -- e.g. a slot that also carries the scene's
 * `assets.source_image` role). Any of these silently overwriting or dropping
 * a reference would look like it was used; the caller is the one place that
 * knows this happened and must say so.
 */
export function wireReferences(mediaIds, slots, reservedSlotIds) {
  const pool = slots || [];
  const overrides = {};
  const claimed = new Set(reservedSlotIds || []);
  let unwired = 0;
  mediaIds.forEach((mediaId, i) => {
    const at = `assets.reference_${i + 1}`;
    const slot = pool.find((s) => (s.roles || []).some((r) => r.at === at));
    if (!slot) { unwired += 1; return; }
    // The wire's target is the role's own data (`wireTo`), never a slot id
    // this file would otherwise have to know: the preset declares where its
    // own reference lands, the same way it declares everything else.
    const role = slot.roles.find((r) => r.at === at);
    // A destination -- the slot's own media_id input, or wherever `wireTo`
    // points -- can hold one value. Two references claiming the SAME slot
    // (a preset with two reference roles on one slot) or the SAME wireTo
    // target (two different slots both wired into one downstream input) are
    // the same failure either way: the second silently overwrites the first
    // unless it is counted unwired instead.
    const dest = role && role.wireTo ? `${role.wireTo.slot}.${role.wireTo.input}` : slot.id;
    if (claimed.has(slot.id) || claimed.has(dest)) { unwired += 1; return; }
    claimed.add(slot.id);
    claimed.add(dest);
    overrides[slot.id] = { ...overrides[slot.id], media_id: mediaId };
    if (role && role.wireTo) {
      overrides[role.wireTo.slot] = { ...overrides[role.wireTo.slot],
        [role.wireTo.input]: [slot.id, 0] };
    }
  });
  return { overrides, unwired };
}
