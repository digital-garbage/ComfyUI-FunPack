// The cut as the render route wants it: one spec per playing clip, in timeline order. Pure.
import { segments, isVideoClip, effFps } from "../../shell/scenes.js";

/** -> { clips, missing }: `missing` counts scenes that have nothing to play yet (no render). */
export function clipSpecs(p, only) {   // `only`: a Set of scene ids to keep (the picked clips)
  const renders = p.scene_renders || {};
  const clips = [];
  let missing = 0;
  for (const seg of segments(p)) {
    if (seg.kind !== "scene" || (seg.scene.excluded && !seg.scene.removed_from_plan)) continue;       // removed from the plan still plays
    if (only && !only.has(seg.scene.id)) continue;
    const sc = seg.scene, r = renders[sc.id], base = { scene_id: sc.id, dur: seg.dur, fps: effFps(sc, p), gap_after: Math.max(0, +sc.gap_after_sec || 0) };
    if (isVideoClip(sc)) {
      if (!(sc.source && sc.source.media_ref)) { missing += 1; continue; }
      clips.push({ ...base, bin_media_ref: sc.source.media_ref, in: sc.source_in || 0 });
    } else if (r && r.media) {
      clips.push({ ...base, ...r.media, in: (r.inSec || 0) + (sc.source_in || 0) });
    } else missing += 1;
  }
  return { clips, missing };
}
