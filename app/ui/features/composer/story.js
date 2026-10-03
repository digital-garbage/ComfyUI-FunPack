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

/** Make the plan say `parts` (the story split at its markers). Roots that no part owns leave the plan (a rendered one
 *  keeps a ghost); parts nothing owns become new scenes. -> true when anything changed. */
export function applyStory(p, parts) {
  if (!parts.length) return false;
  const before = JSON.stringify(p);
  const roots = storyRoots(p);
  const owner = match(parts.map(norm), roots.map((r) => norm(r.text)));
  roots.forEach((r, i) => { if (!owner.includes(i)) removeScene(p, r.id); });
  const group = (r) => p.scenes.filter((s) => genUnitId(s) === genUnitId(r));
  const groups = parts.map((text, i) => {
    if (owner[i] >= 0) { const r = roots[owner[i]]; if (norm(r.text) !== norm(text)) r.text = text; return group(r); }
    return [{ id: newId(), text, transition_to_next: "", source: { type: "carry" }, excluded: false, frames_mode: "project", fps_mode: "project" }];
  });
  // The story's scenes take the slots the story's scenes held (video clips and left-out scenes stay where they were), in the parts' order.
  const mine = new Set(groups.flat().map((s) => s.id)), keep = new Set(roots.filter((_, i) => owner.includes(i)).flatMap(group).map((s) => s.id));
  const out = [], fill = groups.flat();
  let queued = 0;
  for (const s of p.scenes) { if (keep.has(s.id)) { if (queued < fill.length) out.push(fill[queued++]); } else out.push(s); }
  out.push(...fill.slice(queued));
  p.scenes = out;
  if (p.timeline_manually_ordered && Array.isArray(p.timeline_order)) {
    const have = new Set(p.timeline_order);
    p.timeline_order = p.timeline_order.filter((id) => out.some((s) => s.id === id)).concat(out.filter((s) => mine.has(s.id) && !have.has(s.id)).map((s) => s.id));
  }
  return JSON.stringify(p) !== before;
}
