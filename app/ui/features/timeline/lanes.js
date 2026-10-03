// The project as lanes of clips: what the timeline stage draws. Pure.
import { segments, isVideoClip, effFps } from "../../shell/scenes.js";

const first = (t) => ((t || "").split("\n")[0] || "").slice(0, 80);

export function videoLane(p, picked, focus, actions = () => []) {
  const renders = p.scene_renders || {};
  return {
    id: "video", label: "Video", kind: "video", reorder: true,
    clips: segments(p).map((seg) => {
      if (seg.kind === "gap") return { id: seg.id, start: seg.start, dur: seg.dur, title: "pause", ghost: true };
      if (seg.kind === "ghost") return { id: seg.id, start: seg.start, dur: seg.dur, title: first(seg.ghost.text) || "removed", ghost: true };
      const sc = seg.scene;
      const rendered = Boolean(renders[sc.id]);
      return {
        id: sc.id, start: seg.start, dur: seg.dur, trim: true,
        head: [isVideoClip(sc) ? "▶ video" : rendered ? "✓" : "◌", `${seg.dur.toFixed(1)}s`],
        title: first(sc.text) || "(empty scene)",
        selected: picked.includes(sc.id), focus: sc.id === focus, excluded: Boolean(sc.excluded),
        actions: actions(sc),
      };
    }),
  };
}

/** Frames at `sec` into scene `sc`, for Split. */
export const framesAt = (sc, p, sec) => Math.round(sec * effFps(sc, p));
