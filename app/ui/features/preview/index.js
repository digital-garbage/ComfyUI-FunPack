// The monitor and its transport: plays the cut, clip after clip, wherever the playhead is. The playhead is the one
// clock: playing moves it, seeking (scrubber, ruler, picking a clip) moves it, and the picture follows it.
import { composer as c } from "../../composer/composer.js";
import { segments, totalSeconds, isVideoClip } from "../../shell/scenes.js";

const pad = (n) => String(Math.floor(n)).padStart(2, "0");
export const timecode = (sec, fps) => { const s = Math.max(0, sec), w = Math.floor(s); return [pad(w / 3600), pad((w % 3600) / 60), pad(w % 60), pad((s - w) * (fps || 25))].join(":"); };

/** Where a clip's picture comes from: {url, from} (`from` = seconds into the file), or null when there is none yet.
 *  A generated clip is played from the server's own trimmed copy, never the raw render: ComfyUI's saver writes the index at
 *  the END of the file, which a browser cannot seek in (a deep seek stalls or blacks the monitor). */
export function sourceOf(sc, open, dur) {
  if (isVideoClip(sc)) return sc.source && sc.source.media_ref ? { url: `/funpack/api/media/${encodeURIComponent(sc.source.media_ref)}/file`, from: sc.source_in || 0 } : null;
  const r = (open.scene_renders || {})[sc.id];
  if (!(r && r.media)) return null;
  const q = new URLSearchParams({ filename: r.media.filename || "", subfolder: r.media.subfolder || "", type: r.media.type || "output", render_in: String(r.inSec || 0), src_in: String(sc.source_in || 0), dur: String(dur) });
  if (sc.effects && sc.effects.reverse) q.set("rev", "1");
  return { url: `/funpack/api/m/render/projects/${encodeURIComponent(open.id)}/preview-segment/${encodeURIComponent(sc.id)}?${q}`, from: 0 };
}

export default {
  id: "preview",
  mount: "preview",
  needs: ["project", "selection", "playhead", "api"],
  setup({ host, app }) {
    const p = app.project, head = app.playhead, sel = app.selection;
    const viewer = c.viewer.media({ kind: "video", empty: "", controls: false, loop: false });
    const empty = c.emptyState.default({ icon: "🎬", title: "No render yet", hint: "Use Generate in the timeline header" });
    const scrub = c.slider.sm({ label: "Playback position", min: 0, max: 1, step: 0.01, onChange: (v) => head.set(v) });
    const stop = c.iconButton.sm({ icon: "⏹", label: "Stop", onClick: () => { pause(); head.set(0); } });
    const play = c.iconButton.sm({ icon: "▶", label: "Play", onClick: () => (playing ? pause() : start()) });
    const time = c.text.sm({ text: timecode(0, 25) });
    const frame = c.button.sm({ label: "📌 Save frame", tone: "ghost", disabled: true, onClick: saveFrame });
    host.append(viewer.node, empty.node, scrub.node, c.toolbar.default({ items: [stop, play, time,
      c.select.sm({ label: "Save frame to", options: [{ value: "", label: "— save to Media bin —" }], value: "" }), frame] }).node);

    let playing = false, shown = null, seg = null, from = 0, quiet = false;
    const open = () => p.project;
    const segAt = (t) => (open() ? segments(open()).find((s) => s.kind === "scene" && t >= s.start - 1e-6 && t < s.start + s.dur - 1e-6) : null);
    const total = () => (open() ? totalSeconds(open()) : 0);

    function show(next) {
      const src = next && sourceOf(next.scene, open(), next.dur);
      const key = src ? `${next.id}|${src.url}` : "";
      if (key === shown) return;
      shown = key; from = src ? src.from : 0;
      viewer.node.hidden = !src; empty.node.hidden = Boolean(src);
      frame.setDisabled(!src);
      viewer.setSource(src ? `${src.url}#t=${src.from}` : null, "video");
      const v = viewer.element;
      if (!v) return;
      v.addEventListener("timeupdate", () => onTime(v));
      v.addEventListener("ended", () => { if (playing && seg) head.set(seg.start + seg.dur); });
      if (playing) v.play().catch(() => {});
    }
    function onTime(v) {
      if (!playing || !seg || v !== viewer.element) return;
      const at = seg.start + v.currentTime - from;
      quiet = true; head.set(Math.min(at, seg.start + seg.dur)); quiet = false;
      if (at >= seg.start + seg.dur - 0.02) head.set(seg.start + seg.dur);     // this clip is done: the next one takes over
    }
    function sync() {
      seg = segAt(head.at);
      show(seg);
      const v = viewer.element;
      if (v && seg && !quiet) {
        const want = from + head.at - seg.start;
        if (Math.abs(v.currentTime - want) > 0.25) v.currentTime = want;
      }
      scrub.input.max = String(total() || 1);
      scrub.setValue(head.at);
      scrub.input.style.setProperty("--fill", `${Math.min(100, (head.at / (total() || 1)) * 100)}%`);       // the slider drew its fill for 0..1
      time.setText(timecode(head.at, open() && open().frame_rate));
      play.node.firstChild.textContent = playing ? "⏸" : "▶"; play.node.title = playing ? "Pause" : "Play";
      if (playing && viewer.element) viewer.element.paused && viewer.element.play().catch(() => {});
      if (!playing && viewer.element) viewer.element.pause();
    }
    // Gaps and clips with no picture have nothing to play, so the clock itself walks the playhead through them.
    const walk = setInterval(() => {
      if (!playing) return;
      const v = viewer.element;
      if (seg && v && !v.paused) onTime(v);                        // timeupdate is too slow to catch a short clip's end
      else head.set(head.at + 0.04);
      if (head.at >= total()) { pause(); head.set(Math.min(head.at, total())); }
    }, 40);
    function start() { if (!total()) return; if (head.at >= total() - 0.02) head.set(0); playing = true; sync(); }
    function pause() { playing = false; sync(); }
    async function saveFrame() {
      const blob = await viewer.captureFrame();
      if (!blob) return c.toast.warn({ text: "No frame to save yet." });
      const { problems } = await app.api.uploadMedia([new File([blob], `${(open() || {}).name || "frame"}-${timecode(head.at, 25).replaceAll(":", "-")}.png`, { type: "image/png" })]);
      if (problems && problems.length) return c.toast.warn({ text: problems.join(" ") });
      app.say("media");
      c.toast.good({ text: "Frame saved to the media bin." });
    }
    // Picking a clip puts the playhead at its start, unless the playhead is already inside it.
    const onSelect = () => {
      const s = open() && segments(open()).find((x) => x.kind === "scene" && x.id === sel.focus);
      if (s && !(head.at >= s.start && head.at < s.start + s.dur)) head.set(s.start); else sync();
    };
    sync();
    const off = [head.on(sync), app.on((what) => {
      if (what === "select") onSelect();
      else if (what === "play.toggle") (playing ? pause() : start());
      else if (what === "play.start") start();
      else if (what === "play.pause") pause();
      else if (what === "open") { pause(); head.set(0); }          // another project: nothing carries over
      else sync();
    })];
    return () => { clearInterval(walk); off.forEach((f) => f()); viewer.destroy(); };
  },
};
