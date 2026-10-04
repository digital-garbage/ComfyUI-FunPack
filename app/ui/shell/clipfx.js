// Per-clip effects (stored on scene.effects) and the transition on a clip's outgoing edge. The render reads these as they are.
const ZOOM = { zoom_ratio: 0.15, zoom_frames: 25, zoom_start_frame: 0 };

/** Apply a library effect to a scene. Flips, fill and reverse are switches: applying again turns them off. -> false if the id is unknown. */
export function applyEffect(sc, id, value) {
  const fx = { ...(sc.effects || {}) };
  const zoom = (way) => Object.assign(fx, { zoom: way, zoom_ratio: fx.zoom_ratio ?? ZOOM.zoom_ratio, zoom_frames: fx.zoom_frames ?? ZOOM.zoom_frames, zoom_start_frame: fx.zoom_start_frame ?? ZOOM.zoom_start_frame });
  const ops = {
    zoom_in: () => zoom("in"), zoom_out: () => zoom("out"),
    blur: () => { fx.blur = value; }, fade_in: () => { fx.fade_in = value; }, fade_out: () => { fx.fade_out = value; },
    flip_h: () => { fx.flip_h = !fx.flip_h; }, flip_v: () => { fx.flip_v = !fx.flip_v; },
    fill_frame: () => { fx.fit = fx.fit === "fill" ? "contain" : "fill"; },
    crop: () => { fx.crop_inset = Math.max(0, Math.min(0.4, (+value || 0) / 100)); },
    reverse: () => { fx.reverse = !fx.reverse; },
    reset: () => { for (const k of Object.keys(fx)) delete fx[k]; },
  };
  if (!ops[id]) return false;
  ops[id]();
  sc.effects = fx;
  return true;
}

/** Put a library transition on the clip's outgoing edge. */
export function applyTransition(sc, item, value) {
  if (!item) return false;
  sc.video_transition = item.type || item.id;
  sc.transition_frames = value != null ? Math.round(value) : Math.round((item.param && item.param.default) || 16);
  return true;
}

export const clearTransition = (sc) => { delete sc.video_transition; delete sc.transition_frames; return true; };

const LABELS = [
  ["reverse", () => "◀ reverse"], ["flip_h", () => "⇄ flip"], ["flip_v", () => "⇅ flip"],
  ["fit", (v) => (v === "fill" ? "fill" : null)], ["crop_inset", (v) => (v > 0 ? `crop ${Math.round(v * 100)}%` : null)],
  ["zoom", (v) => (v === "in" ? "zoom in" : v === "out" ? "zoom out" : null)], ["blur", (v) => (v > 0 ? `blur ${(+v).toFixed(2)}` : null)],
  ["fade_in", (v) => (v > 0 ? `fade in ${v}s` : null)], ["fade_out", (v) => (v > 0 ? `fade out ${v}s` : null)],
];

/** What is on this clip, as short tags (so a switch can be seen to be on). */
export function tags(sc) {
  const fx = sc.effects || {}, out = [];
  for (const [key, fmt] of LABELS) { const raw = fx[key]; if (raw !== undefined && raw !== null && raw !== false && raw !== "") { const t = fmt(raw); if (t) out.push(t); } }
  if (sc.video_transition) out.push(`→ ${sc.video_transition}`);
  return out;
}
