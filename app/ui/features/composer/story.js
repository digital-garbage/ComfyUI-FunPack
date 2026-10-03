// The Story: every generated scene's prompt as one box, a marker word between scenes. Pure half (applyStory) here;
// the box itself is in index.js. Scene.text stays the truth; the story is a second way to edit it.
import { removeScene } from "../../shell/edits.js";
import { genUnitId, isGenerative, isSubclip } from "../../shell/scenes.js";

const norm = (t) => String(t || "").replace(/\s+/g, " ").trim();
const newId = () => "c" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);

/** The scenes the story shows: one root per generation unit (never video clips, cuts or left-out scenes), in plan order. */
export const storyRoots = (p) => {
  const seen = new Set(), out = [];
  for (const sc of p.scenes || []) {
    if (sc.excluded || !isGenerative(sc) || isSubclip(sc) || seen.has(genUnitId(sc))) continue;
    seen.add(genUnitId(sc)); out.push(sc);
  }
  return out;
};

export const joinStory = (p, marker = "qcut") => storyRoots(p).map((s) => (s.text || "").trim()).join(`\n${marker}\n`);

/** For each new part, the index of the old root that owns its content (-1: none): exact text first, nearest position
 *  breaking ties, then the leftovers in order. A reorder or insert keeps every unchanged scene's render and anchor. */
export function match(parts, roots) {
  const out = parts.map(() => -1), used = roots.map(() => false);
  parts.forEach((t, i) => {
    if (!t) return;
    let best = -1;
    roots.forEach((r, ri) => { if (!used[ri] && r === t && (best < 0 || Math.abs(ri - i) < Math.abs(best - i))) best = ri; });
    if (best >= 0) { out[i] = best; used[best] = true; }
  });
  let at = 0;
  parts.forEach((_, i) => { if (out[i] >= 0) return; while (at < roots.length && used[at]) at += 1; if (at < roots.length) { out[i] = at; used[at] = true; } });
  return out;
}

/** The first scene whose own text holds a cut word: the story would show it as two scenes. */
export const clash = (p, words) => {
  if (!words.length) return null;
  const re = new RegExp(`(?<![\\p{L}\\p{N}_])(?:${words.map((w) => w.trim().split(/\s+/).map((x) => x.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("\\s+")).join("|")})(?![\\p{L}\\p{N}_])`, "iu");
  return storyRoots(p).find((s) => re.test(s.text || "")) || null;
};

/** Make the plan say `parts` (the story split at its markers). Roots that no part owns leave the plan WITH their cuts (a
 *  rendered one keeps a ghost); parts nothing owns become new scenes. Plan order only moves when the parts were reordered.
 *  -> true when anything changed. */
export function applyStory(p, parts) {
  if (!parts.some((t) => String(t).trim())) return false;        // an emptied box is not an instruction to delete everything
  const before = JSON.stringify(p);
  const roots = storyRoots(p);
  const owner = match(parts.map(norm), roots.map((r) => norm(r.text)));
  const unitOf = (r) => p.scenes.filter((s) => genUnitId(s) === genUnitId(r));
  roots.forEach((r, i) => { if (!owner.includes(i)) unitOf(r).sort((a, b) => (b.cut_offset_frames || 0) - (a.cut_offset_frames || 0)).forEach((s) => removeScene(p, s.id)); });
  const kept = parts.map((_, i) => (owner[i] >= 0 ? roots[owner[i]] : null));
  kept.forEach((r, i) => { if (r && norm(r.text) !== norm(parts[i])) unitOf(r).forEach((s) => { if (!isSubclip(s)) s.text = parts[i]; }); });      // a cut's first piece shares the prompt
  const made = new Map();
  parts.forEach((text, i) => { if (!kept[i]) made.set(i, { id: newId(), text, transition_to_next: "", source: { type: "carry" }, excluded: false, frames_mode: "project", fps_mode: "project" }); });
  const owned = owner.filter((o) => o >= 0);
  if (owned.some((o, k) => k && o < owned[k - 1])) {                 // the parts were reordered: whole units move, into the slots the story's scenes held
    const fill = parts.flatMap((_, i) => (kept[i] ? unitOf(kept[i]) : [made.get(i)]));
    const keep = new Set(kept.filter(Boolean).flatMap(unitOf).map((s) => s.id));
    const out = []; let n = 0;
    for (const s of p.scenes) { if (keep.has(s.id)) { if (n < fill.length) out.push(fill[n++]); } else out.push(s); }
    p.scenes = out.concat(fill.slice(n)).filter((s, k, all) => all.indexOf(s) === k);
  } else {
    parts.forEach((_, i) => {                                         // new scenes go right after the previous part's last scene
      if (!made.has(i)) return;
      const prev = parts.slice(0, i).map((__, k) => k).reverse().find((k) => kept[k] || made.has(k));
      const anchor = prev === undefined ? -1 : p.scenes.indexOf(kept[prev] ? unitOf(kept[prev]).pop() : made.get(prev));
      p.scenes.splice(anchor + 1, 0, made.get(i));
    });
  }
  if (p.timeline_manually_ordered && Array.isArray(p.timeline_order)) {
    const have = new Set(p.timeline_order);
    p.timeline_order = p.timeline_order.filter((id) => p.scenes.some((s) => s.id === id)).concat([...made.values()].filter((s) => !have.has(s.id)).map((s) => s.id));
  }
  return JSON.stringify(p) !== before;
}
