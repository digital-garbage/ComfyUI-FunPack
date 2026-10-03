// Auto Montage, trailer-style: the lead clip is cut into segments that shrink as it goes, and after each one a segment from a
// pool of other rendered clips is cut in. Every piece reuses media that already exists; nothing is generated. Pure.
import { snapFrames, effFps, seconds, isVideoClip, genUnitId } from "../../shell/scenes.js";

const newId = () => "c" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
const clone = (x) => JSON.parse(JSON.stringify(x));

/** What a clip can play from: {fps, from (seconds into its file), total (seconds), render?} or null when it has nothing yet. */
export function playable(sc, p) {
  const fps = effFps(sc, p);
  if (isVideoClip(sc)) return sc.source && sc.source.media_ref ? { fps, from: sc.source_in || 0, total: seconds(sc, p) } : null;
  const r = (p.scene_renders || {})[sc.id];
  return r && r.media ? { fps, from: (r.inSec || 0) + (sc.source_in || 0), total: seconds(sc, p), render: r } : null;
}

/** Cut a clip's length into [{start, frames}] (frames), each `base * decay^i` long (decay 1 = all equal). */
export function chop(totalFrames, base, decay = 1) {
  const out = [];
  for (let at = 0, i = 0; at < totalFrames - 8; i += 1) {
    const len = Math.min(totalFrames - at, Math.max(9, Math.round(base * decay ** i)));
    out.push({ start: at, frames: snapFrames(len) });
    at += len;
  }
  return out;
}

/** Append the montage to `p` (the scenes, and their renders). -> how many clips were added (0 = nothing could be built). */
export function build(p, { leadId, poolIds, segmentFrames = 100, decay = 1, random = Math.random }) {
  const find = (id) => p.scenes.find((s) => s.id === id);
  const lead = find(leadId), leadSrc = lead && playable(lead, p);
  const pool = (poolIds || []).map(find).filter((s) => s && playable(s, p));
  if (!leadSrc || !pool.length) return 0;
  const base = Math.max(9, Math.round(segmentFrames)), dk = Math.min(1, Math.max(0.3, decay));
  const leadSegs = chop(snapFrames(leadSrc.total * leadSrc.fps), base, dk);
  // Each pool clip is its own queue read front to back: chance only picks WHICH clip to draw from next.
  const queues = pool.map((sc) => ({ sc, src: playable(sc, p), cursor: 0, chunks: chop(snapFrames(playable(sc, p).total * playable(sc, p).fps), base, 1) })).filter((q) => q.chunks.length);
  if (!leadSegs.length || !queues.length) return 0;
  const next = () => {
    let open = queues.filter((q) => q.cursor < q.chunks.length);
    if (!open.length) { queues.forEach((q) => { q.cursor = 0; }); open = queues; }       // every clip used up: each starts over from its own beginning
    const q = open[Math.floor(random() * open.length)];
    return { sc: q.sc, src: q.src, ...q.chunks[q.cursor++] };
  };
  const piece = (sc, src, start, frames) => {
    const out = clone(sc), at = start / src.fps;
    Object.assign(out, { id: newId(), gen_unit_id: genUnitId(sc), frames, rating: "", transition_to_next: "cut", transition_frames: null, frames_mode: "timeline", excluded: false, audio_separated: false, audio_volume: sc.audio_separated ? 1 : sc.audio_volume });
    if (isVideoClip(sc)) Object.assign(out, { source_in: src.from + at, source_dur: frames / src.fps });
    else {
      out.cut_offset_frames = (sc.cut_offset_frames || 0) + start;
      out.source_in = 0;                                    // the render entry below already starts where this piece does
      p.scene_renders = p.scene_renders || {};
      p.scene_renders[out.id] = { media: src.render.media, inSec: src.from + at, renderPrompt: src.render.renderPrompt ? { ...src.render.renderPrompt } : null };
    }
    if (out.cut_offset_frames > 0) out.text = "";           // a later part's prompt belongs to the first part
    return out;
  };
  const built = [];
  leadSegs.forEach((seg) => { built.push(piece(lead, leadSrc, seg.start, seg.frames)); const c = next(); built.push(piece(c.sc, c.src, c.start, c.frames)); });
  built[built.length - 1].transition_to_next = "";
  p.scenes.push(...built);
  return built.length;
}
