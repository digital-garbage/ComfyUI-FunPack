// The project as lanes of clips: what the timeline stage draws. Pure.
import { segments, isVideoClip, effFps } from "../../shell/scenes.js";

const first = (t) => ((t || "").split("\n")[0] || "").slice(0, 80);

/** "liked" / "disliked" / "" from a saved rating word (numbers: 6+ is liked; "-Just forget it-" is none). */
export function mood(rating) {
  const v = String(rating || "").replace(/\|loved$/, "").trim();
  if (!v || v.startsWith("-")) return "";
  return /^\d+$/.test(v) ? (Number(v) >= 6 ? "liked" : "disliked") : v.toLowerCase().startsWith("disliked") ? "disliked" : "";
}

export function videoLane(p, picked, focus, actions = () => [], ghostActions = () => [], tags = () => []) {
  const renders = p.scene_renders || {};
  return {
    id: "video", label: "Video", kind: "video", reorder: true,
    clips: segments(p).map((seg) => {
      if (seg.kind === "gap") return { id: seg.id, start: seg.start, dur: seg.dur, title: "pause", ghost: true };
      if (seg.kind === "ghost") return { id: seg.id, start: seg.start, dur: seg.dur, title: first(seg.ghost.text) || "removed", ghost: true, actions: ghostActions(seg.ghost) };
      const sc = seg.scene;
      const rendered = Boolean(renders[sc.id]);
      return {
        id: sc.id, start: seg.start, dur: seg.dur, trim: true,
        rating: mood(sc.rating) || undefined,
        head: [isVideoClip(sc) ? "▶ video" : rendered ? "✓" : "◌", `${seg.dur.toFixed(1)}s`, ...(mood(sc.rating) ? [mood(sc.rating) === "liked" ? "★" : "✕"] : []), ...tags(sc)],
        title: first(sc.text) || "(empty scene)",
        selected: picked.includes(sc.id), focus: sc.id === focus, excluded: Boolean(sc.excluded),
        actions: actions(sc),
      };
    }),
  };
}

/** Frames at `sec` into scene `sc`, for Split. */
export const framesAt = (sc, p, sec) => Math.round(sec * effFps(sc, p));

/** The sound that came with the picture, one block under each clip. */
export const audioLane = (p, picked) => ({
  id: "audio", label: "Original", kind: "audio",
  clips: segments(p).filter((s) => s.kind === "scene").map((s, i) => ({ id: `a:${s.id}`, start: s.start, dur: s.dur, title: `S${i + 1}`, selected: picked.includes(s.id), excluded: Boolean(s.scene.excluded) })),
});
