// Takes: every render of a scene stays in project.scene_variants; stepping puts another one on the clip.
import { genUnitId, unitRoot } from "./scenes.js";

/** The scene whose renders the unit shares, its takes (oldest first), and where the clip's own render sits among them (-1: none). */
export function takesOf(p, scene) {
  const head = unitRoot(p, genUnitId(scene)) || scene;
  const list = (p.scene_variants || {})[head.id] || [];
  const now = (p.scene_renders || {})[head.id];
  const at = now && now.promptId ? list.findIndex((t) => t.promptId === now.promptId) : -1;
  return { head, list, at };
}

/** Put take number `to` on every clip of the unit, keeping each one's rating with it. -> true when something moved. */
export function pickTake(p, scene, to) {
  const { head, list, at } = takesOf(p, scene);
  if (at < 0 || to < 0 || to >= list.length || to === at) return false;
  list[at].rating = head.rating || "";
  const take = list[to];
  p.scenes.filter((s) => genUnitId(s) === genUnitId(head)).forEach((s) => {
    const r = (p.scene_renders || {})[s.id];
    if (r) p.scene_renders[s.id] = { ...r, media: take.media, ...(take.promptId ? { promptId: take.promptId } : {}), ...(take.secs && r.durationSec !== undefined ? { durationSec: take.secs } : {}) };       // a take made at another length plays at its own
  });
  head.rating = take.rating || "";
  return true;
}

/** The take `dir` steps from the clip's current one. */
export function stepTake(p, scene, dir) {
  return pickTake(p, scene, takesOf(p, scene).at + dir);
}
