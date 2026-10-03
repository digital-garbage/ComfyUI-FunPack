// What a project's scenes are, as numbers: how long each runs, in what order the cut plays them, and where
// each begins on the timeline. Pure -- every function takes the project (and renders) it reads, so the same
// answers come out on the timeline, in the player and in a run.

/** A pipeline role that `drives` the plan's length or rate is fed from the project number the timeline draws. */
export const DRIVEN = { frames: "num_frames_per_scene", fps: "frame_rate" };

export const genUnitId = (sc) => (sc && (sc.gen_unit_id || sc.id)) || "";
export const isVideoClip = (sc) => Boolean(sc && sc.source && sc.source.type === "video");
export const isGenerative = (sc) => Boolean(sc) && !isVideoClip(sc);
/** A timeline cut of one generated clip: only its root owns the prompt, rating and source. */
export const isSubclip = (sc) => Boolean(sc) && (sc.cut_offset_frames || 0) > 0;

/** The unit's root scene (the one that owns text/source), or the first of the unit. */
export function unitRoot(p, unit) {
  const group = (p.scenes || []).filter((s) => genUnitId(s) === unit);
  return group.find((s) => !isSubclip(s)) || group[0] || null;
}
export const unitSceneIds = (p, unit) => (p.scenes || []).filter((s) => genUnitId(s) === unit && !s.excluded).map((s) => s.id);

/** Frame counts a model can make: `base + k * step`. LTX is 8k+1; a family says its own (see the probe). */
export const LTX_GRID = { step: 8, base: 1, fps: null };

export function snapFrames(n, mode = "round", grid = LTX_GRID) {
  const k = (Math.round(Number(n) || 0) - grid.base) / grid.step;
  const k2 = mode === "floor" ? Math.floor(k) : mode === "ceil" ? Math.ceil(k) : Math.round(k);
  return Math.max(grid.step + grid.base, k2 * grid.step + grid.base);
}

// "project" follows the project; a trimmed scene (timeline/custom) owns its number.
export function effFrames(sc, p) {
  if (!sc || (sc.frames_mode || "project") === "project") return p.num_frames_per_scene || 97;
  return sc.frames != null ? sc.frames : p.num_frames_per_scene || 97;
}
export function effFps(sc, p) {
  if (!sc || (sc.fps_mode || "project") === "project") return p.frame_rate || 25;
  return sc.fps != null ? sc.fps : p.frame_rate || 25;
}

/** What the next Generate would make: ignores any render already there. */
export function planSeconds(sc, p) {
  if (isVideoClip(sc) && sc.source_dur != null) return sc.source_dur;
  return effFrames(sc, p) / (effFps(sc, p) || 25);
}

/** How long the clip plays. A render is IMMUTABLE: it keeps the length it was made at, so changing the project
 *  can never cut off its tail -- unless the scene has an explicit length of its own (trim/slip/split). */
export function seconds(sc, p, renders = p.scene_renders || {}) {
  const made = renders[sc.id] && renders[sc.id].durationSec;
  if (made != null && !isVideoClip(sc) && sc.source_dur == null && (sc.frames_mode || "project") === "project") return made;
  return planSeconds(sc, p);
}

/** The cut: scenes in timeline order. Until a person pins an order by hand it simply follows the plan. */
export function orderedScenes(p) {
  const scenes = p.scenes || [];
  if (!p.timeline_manually_ordered) return scenes;
  const byId = new Map(scenes.map((s) => [s.id, s]));
  const seen = new Set();
  const out = [];
  for (const id of p.timeline_order || []) { const s = byId.get(id); if (s && !seen.has(id)) { out.push(s); seen.add(id); } }
  for (const s of scenes) if (!seen.has(s.id)) out.push(s);
  return out;
}

const ghostSeconds = (g, p) => g.durationSec != null ? g.durationSec
  : ((g.frames_mode !== "project" && g.frames != null ? g.frames : p.num_frames_per_scene) || 1)
    / ((g.fps_mode !== "project" && g.fps != null ? g.fps : p.frame_rate) || 25);

/** What the timeline plays, in order, with where each part starts: scenes, the ghosts of removed scenes that
 *  still have a clip, and the pauses between. -> [{ kind, id, scene?, ghost?, start, dur }] */
export function segments(p) {
  const ghosts = p.scene_ghosts || [];
  const after = new Map();
  for (const g of ghosts) {
    const key = g.afterSceneId || "";
    if (!after.has(key)) after.set(key, []);
    after.get(key).push(g);
  }
  const placed = new Set();
  const flat = [];
  const addGhosts = (key) => (after.get(key) || []).forEach((g) => {
    if (!placed.has(g.id)) { placed.add(g.id); flat.push({ kind: "ghost", id: `ghost:${g.id}`, ghost: g, dur: ghostSeconds(g, p) }); }
  });
  addGhosts("");
  for (const sc of orderedScenes(p)) {
    flat.push({ kind: "scene", id: sc.id, scene: sc, dur: seconds(sc, p) });
    addGhosts(sc.id);
    const gap = Math.max(0, +sc.gap_after_sec || 0);
    if (gap > 0.001) flat.push({ kind: "gap", id: `gap:${sc.id}`, dur: gap });
  }
  for (const g of ghosts) addGhosts(g.afterSceneId || "");        // orphans (stale anchor) still show, after the rest
  for (const g of ghosts) if (!placed.has(g.id)) { placed.add(g.id); flat.push({ kind: "ghost", id: `ghost:${g.id}`, ghost: g, dur: ghostSeconds(g, p) }); }
  let at = 0;
  return flat.map((seg) => { const out = { ...seg, start: at }; at += seg.dur; return out; });
}

export const totalSeconds = (p) => { const s = segments(p); return s.length ? s[s.length - 1].start + s[s.length - 1].dur : 0; };

/** Seconds to `HH:MM:SS` (or `MM:SS` under an hour). */
export function clock(sec) {
  const whole = Math.max(0, Math.floor(sec));
  const h = Math.floor(whole / 3600), m = Math.floor(whole / 60) % 60, s = whole % 60;
  const two = (n) => String(n).padStart(2, "0");
  return h ? `${two(h)}:${two(m)}:${two(s)}` : `${two(m)}:${two(s)}`;
}
