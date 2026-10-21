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
export function wireReferences(mediaIds, slots, reservedSlotIds, pins, kindOf, routeOf, hasSound) {
  const pool = slots || [];
  const overrides = {};
  const claimed = new Set(reservedSlotIds || []);
  let unwired = 0;
  const left = [];       // what was not wired, and the role it looked for: said by name, not just counted
  // Each reference gets its slot number (1-based), so a pinned one keeps its number when a reference
  // before it is switched off; the rest take the free numbers in the order they were added.
  const kind = (id) => (kindOf && kindOf(id)) || "image";
  const numbers = referenceNumbers(mediaIds, pins);
  const seen = {};         // how many of each kind went in: a video or audio input is numbered per kind
  mediaIds.forEach((mediaId, i) => {
    const at = `assets.reference_${numbers[i]}`;
    const slot = pool.find((s) => (s.roles || []).some((r) => r.at === at));
    if (!slot) { unwired += 1; left.push({ id: mediaId, at, why: "no slot with that role" }); return; }
    // A picture's place is the role's own data (`wireTo`): the preset declares where its reference lands. A video's
    // or a sound's place is not stored anywhere: `routeOf(kind, slot, role)` reads it off what the target node offers.
    const role = slot.roles.find((r) => r.at === at);
    const k = kind(mediaId);
    const route = k === "image" ? null : routeOf && routeOf(k, slot, role);
    if (k !== "image" && !route) { unwired += 1; left.push({ id: mediaId, at, why: `this pipeline has no input for a ${k}` }); return; }
    const n = seen[k] = (seen[k] || 0) + 1;
    if (route && n > route.names.length) { unwired += 1; left.push({ id: mediaId, at, why: `the model takes only ${route.names.length} ${k}${route.names.length > 1 ? "s" : ""}` }); return; }
    const wire = route ? { slot: route.slot, input: route.names[n - 1], output: route.output } : role.wireTo && { ...role.wireTo, output: 0 };
    // A destination can hold one value. Two references claiming the SAME slot or the SAME wire target are
    // the same failure: the second silently overwrites the first unless it is counted unwired instead.
    const dest = wire ? `${wire.slot}.${wire.input}` : slot.id;
    if (claimed.has(slot.id) || claimed.has(dest)) { unwired += 1; left.push({ id: mediaId, at, why: "its slot is already taken" }); return; }
    claimed.add(slot.id);
    claimed.add(dest);
    overrides[slot.id] = { ...overrides[slot.id], media_id: mediaId, ...(k === "image" ? {} : { full: true }) };
    if (wire) overrides[wire.slot] = { ...overrides[wire.slot], [wire.input]: [slot.id, wire.output] };
    // A video with a soundtrack also feeds the input paired with its own number (video 1's sound is audio 1).
    const track = k === "video" && hasSound && hasSound(mediaId) && routeOf && routeOf("videosound", slot, role);
    if (track && track.names[n - 1]) overrides[track.slot] = { ...overrides[track.slot], [track.names[n - 1]]: [slot.id, track.output] };
  });
  return { overrides, unwired, left };
}

/**
 * referenceNumbers(mediaIds, pins) -> [n per mediaId], 1-based reference numbers.
 *
 * `pins` is { mediaId: n } as the scene keeps it. A pinned reference is its number, as long as it is in the
 * list; a pin that another pinned reference already holds, or that is out of range, is ignored. Everything
 * else takes the lowest free number, in the order the references were added. With no pins this is simply
 * 1, 2, 3, ... -- the positional wiring this file always did.
 */
export function referenceNumbers(mediaIds, pins) {
  const ids = mediaIds || [], taken = new Set(), out = new Array(ids.length).fill(0);
  ids.forEach((id, i) => {
    const n = (pins || {})[id];
    if (Number.isInteger(n) && n >= 1 && n <= MAX_REFERENCES && !taken.has(n)) { taken.add(n); out[i] = n; }
  });
  let next = 1;
  ids.forEach((_id, i) => {
    if (out[i]) return;
    while (taken.has(next)) next += 1;
    taken.add(next); out[i] = next;
  });
  return out;
}

/**
 * sceneReferences(scene, fixed) -> { ids, pins }: what one scene sends. `fixed` is the project's { mediaId: n }
 * pins, each at the number it was pinned at, for every scene. Pinned references come first.
 */
export function sceneReferences(scene, fixed) {
  const fx = fixed || {};
  const own = (scene.references || []).filter((id) => !(id in fx));
  return { ids: [...Object.keys(fx), ...own], pins: fx };
}

export const MAX_REFERENCES = 16;

const natural = (a, b) => a.localeCompare(b, undefined, { numeric: true });

/**
 * routeFor(kind, target, loader, targetSlot) -> { slot, names, output } | null
 *
 * Where a picked video or sound goes, decided from the nodes themselves: `target` is what the node the
 * picture reference feeds offers ({ inputs: {name: type}, outputs }), `loader` what the reference slot's own node
 * outputs. A video is the picture-typed inputs whose name says video, a sound the audio-typed inputs whose name says
 * audio (not a video's own soundtrack input); the loader's second picture output carries a video's frames, its
 * first audio output the sound. Null when either side has no such place.
 */
export function routeFor(kind, target, loader, targetSlot) {
  const inputs = (target && target.inputs) || {}, outs = (loader && loader.outputs) || [];
  const named = (type, wanted, unwanted) => Object.keys(inputs).filter((name) => {
    const last = name.split(".").pop().toLowerCase();
    return inputs[name] === type && last.includes(wanted) && !(unwanted && last.includes(unwanted));
  }).sort(natural);
  const names = kind === "video" ? named("IMAGE", "video") : kind === "videosound" ? named("AUDIO", "video") : named("AUDIO", "audio", "video");
  const at = kind === "video" ? outs.map((t, i) => (t === "IMAGE" ? i : -1)).filter((i) => i >= 0)[1] : outs.indexOf("AUDIO");
  return names.length && at >= 0 ? { slot: targetSlot, names, output: at } : null;
}
