// Overlays: text and pictures laid over the cut for a time, on stacked lanes (a higher lane draws on top).
// Pure edits of the project; the render reads project.overlay_tracks / overlay_lanes as they are.
const newId = () => "o" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
const MIN = 0.1;

export const TEXT_DEFAULTS = { font_size: 42, font_family: "arial", color: "#ffffff", opacity: 1, bold: false, italic: false, text_align: "center",
  line_spacing: 1.2, stroke_width: 0, stroke_color: "#000000", shadow: false, shadow_color: "#000000", bg_enabled: false, bg_color: "#000000", bg_opacity: 0.5 };
export const FONTS = ["arial", "helvetica", "georgia", "times", "courier", "verdana", "impact"];

export const lanesOf = (p) => p.overlay_lanes || [];
export const overlaysOf = (p) => p.overlay_tracks || [];
export const find = (p, id) => overlaysOf(p).find((o) => o.id === id);

/** A new lane on top. -> it. */
export function addLane(p) {
  const lane = { id: newId(), label: `Overlay ${lanesOf(p).length + 1}` };
  p.overlay_lanes = [...lanesOf(p), lane];
  return lane;
}

/** Drop a lane and everything on it. */
export function removeLane(p, laneId) {
  if (!lanesOf(p).some((l) => l.id === laneId)) return false;
  p.overlay_lanes = lanesOf(p).filter((l) => l.id !== laneId);
  p.overlay_tracks = overlaysOf(p).filter((o) => o.lane_id !== laneId);
  return true;
}

function put(p, ov, at, laneId) {
  const lane = lanesOf(p).some((l) => l.id === laneId) ? laneId : (lanesOf(p).length ? lanesOf(p)[lanesOf(p).length - 1].id : addLane(p).id);
  const made = { id: newId(), lane_id: lane, start_sec: Math.max(0, at || 0), duration_sec: 3, x: 0.5, y: 0.5, flip_h: false, flip_v: false, opacity: 1, ...ov };
  p.overlay_tracks = [...overlaysOf(p), made];
  return made;
}

export const addText = (p, at, style = {}, laneId) => put(p, { kind: "text", ...TEXT_DEFAULTS, ...style, text: String(style.text || "").trim() || "Title", label: "Text" }, at, laneId);
export const addImage = (p, mediaId, name, at, laneId) => put(p, { kind: "image", media_ref: mediaId, width_px: Math.max(8, Math.round((p.width || 768) * 0.35)), keep_aspect: true, label: name || "Image" }, at, laneId);

export function update(p, id, patch) {
  const o = find(p, id);
  if (!o) return false;
  Object.assign(o, patch);
  return true;
}

export function remove(p, id) {
  if (!find(p, id)) return false;
  p.overlay_tracks = overlaysOf(p).filter((o) => o.id !== id);
  return true;
}

export function move(p, id, deltaSec) {
  const o = find(p, id);
  if (!o || !deltaSec) return false;
  o.start_sec = Math.max(0, (o.start_sec || 0) + deltaSec);
  return true;
}

/** "in" moves the start (the end stays); "out" sets the length. */
export function trim(p, id, edge, deltaSec) {
  const o = find(p, id);
  if (!o || !deltaSec) return false;
  const dur = o.duration_sec || 0;
  if (edge === "out") { o.duration_sec = Math.max(MIN, dur + deltaSec); return true; }
  const d = Math.min(Math.max(deltaSec, -(o.start_sec || 0)), dur - MIN);
  if (!d) return false;
  o.start_sec = (o.start_sec || 0) + d;
  o.duration_sec = dur - d;
  return true;
}

/** Put an overlay on the lane `by` steps up (+) or down (-); -> false at the edge. */
export function restack(p, id, by) {
  const o = find(p, id), lanes = lanesOf(p);
  const to = o && lanes.findIndex((l) => l.id === o.lane_id) + by;
  if (!o || to < 0 || to >= lanes.length || by === 0) return false;
  o.lane_id = lanes[to].id;
  return true;
}

/** Bottom lane first, then by start: the order the render lays them down. */
export function inStackOrder(p) {
  const at = (o) => Math.max(0, lanesOf(p).findIndex((l) => l.id === o.lane_id));
  return [...overlaysOf(p)].sort((a, b) => at(a) - at(b) || (a.start_sec || 0) - (b.start_sec || 0));
}
