// A clip's effects drawn live on the monitor (the render does the real thing; this follows the same geometry): flips, fill, crop,
// zoom ramp, blur, fades. Computed from the playhead, so it tracks scrubbing as well as playing.
import { segments } from "../../shell/scenes.js";

const num = (v, d = 0) => (Number.isFinite(+v) ? +v : d);

/** The zoom scale `within` seconds into a clip of `dur` seconds at `fps`: the ramp from effects.zoom_scale_at. */
export function zoomScale(fx, within, dur, fps) {
  if (fx.zoom !== "in" && fx.zoom !== "out") return 1;
  const n = Math.max(1, Math.round(dur * fps)), ratio = Math.max(0.01, Math.min(0.5, num(fx.zoom_ratio, 0.15)));
  const start = Math.max(0, Math.min(Math.round(num(fx.zoom_start_frame)), n - 1));
  const length = Math.max(1, Math.min(Math.round(num(fx.zoom_frames, Math.min(Math.max(1, Math.floor(n / 4)), 25))), Math.max(1, n - start)));
  const on = Math.max(0, Math.min(Math.floor(within * fps), n - 1)), end = 1 + ratio;
  if (on < start) return fx.zoom === "in" ? 1 : end;
  if (on >= start + length) return fx.zoom === "in" ? end : 1;
  const t = (on - start) / length;
  return fx.zoom === "in" ? 1 + ratio * t : end - ratio * t;
}

/** CSS for the monitor's video: { transform, filter, objectFit, opacity }. */
export function fxStyle(fx = {}, within = 0, dur = 0, fps = 25) {
  const inset = Math.max(0, Math.min(0.4, num(fx.crop_inset)));
  const scale = zoomScale(fx, within, dur, fps) * (inset > 0 ? 1 / (1 - 2 * inset) : 1);
  const t = [fx.flip_h ? "scaleX(-1)" : "", fx.flip_v ? "scaleY(-1)" : "", scale !== 1 ? `scale(${scale.toFixed(4)})` : ""].filter(Boolean).join(" ");
  let opacity = 1;
  const fi = num(fx.fade_in), fo = num(fx.fade_out);
  if (fi > 0 && within < fi) opacity = Math.max(0, within / fi);
  if (fo > 0 && dur > 0 && within > dur - fo) opacity = Math.min(opacity, Math.max(0, (dur - within) / fo));
  return { transform: t, filter: num(fx.blur) > 0 ? `blur(${(num(fx.blur) * 8).toFixed(1)}px)` : "", objectFit: fx.fit === "fill" ? "cover" : "contain", opacity: opacity < 1 ? opacity.toFixed(3) : "" };
}

export default {
  id: "preview-fx",
  mount: "preview",
  needs: ["project", "playhead"],
  setup({ host, app }) {
    const p = app.project, head = app.playhead;
    function draw() {
      const v = host.querySelector("video"), open = p.project;
      if (!v) return;
      const seg = open && segments(open).find((s) => s.kind === "scene" && head.at >= s.start - 1e-6 && head.at < s.start + s.dur - 1e-6);
      const st = fxStyle(seg ? seg.scene.effects : {}, seg ? head.at - seg.start : 0, seg ? seg.dur : 0, seg ? num(seg.scene.fps_mode === "project" ? open.frame_rate : seg.scene.fps, 25) || 25 : 25);
      Object.assign(v.style, st);
      if (v.parentElement) v.parentElement.style.overflow = "hidden";       // a zoom or crop punches in: the frame stays put
    }
    const off = [head.on(draw), app.on(draw)];
    draw();
    return () => off.forEach((f) => f());
  },
};
