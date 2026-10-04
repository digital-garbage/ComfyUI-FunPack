// The audio lanes on the monitor: each lane's sound plays while the playhead is inside it and the monitor is playing.

const view = (m) => `/view?filename=${encodeURIComponent(m.filename || "")}&subfolder=${encodeURIComponent(m.subfolder || "")}&type=${encodeURIComponent(m.type || "output")}`;
const bin = (id) => `/funpack/api/media/${encodeURIComponent(id)}/file`;

/** Where a lane's sound comes from, and which part of it plays: {url, from, dur} or null. */
export function laneSource(t, open) {
  if (t.kind === "separated") {
    const url = t.pinned_bin_ref ? bin(t.pinned_bin_ref) : t.pinned_media && t.pinned_media.filename ? view(t.pinned_media) : null;
    return url && { url, from: t.pinned_in_sec ?? t.source_in_sec ?? 0, dur: t.pinned_dur ?? t.source_dur ?? 0 };
  }
  if (t.media_ref) return { url: bin(t.media_ref), from: t.source_in_sec || 0, dur: t.source_dur != null ? t.source_dur : 86400 };
  return null;
}

export default {
  id: "preview-lanes",
  mount: "preview",
  needs: ["project", "playhead"],
  setup({ app }) {
    const p = app.project, head = app.playhead, pool = new Map();
    const stopAll = () => pool.forEach((a) => a.pause());
    function tick() {
      const open = p.project, live = new Set();
      if (open && head.playing) {
        for (const t of open.audio_tracks || []) {
          const src = laneSource(t, open);
          if (!src) continue;
          live.add(t.id);
          let a = pool.get(t.id);
          if (!a) { a = new Audio(); a.preload = "auto"; pool.set(t.id, a); }
          if (a.dataset.src !== src.url) { a.dataset.src = src.url; a.src = src.url; }
          const start = t.start_sec || 0, within = head.at - start, inside = within >= 0 && within < src.dur;
          const vol = Math.max(0, Math.min(1, t.volume != null ? +t.volume : 1));
          if (a.volume !== vol && Number.isFinite(vol)) a.volume = vol;
          if (!inside) { if (!a.paused) a.pause(); continue; }
          if (Math.abs(a.currentTime - (src.from + within)) > 0.3) a.currentTime = src.from + within;
          if (a.paused) a.play().catch(() => {});
        }
      }
      for (const [id, a] of pool) if (!live.has(id)) { a.pause(); a.removeAttribute("src"); a.load(); delete a.dataset.src; if (!head.playing) pool.delete(id); }       // not playing: every lane gives its connection back (the 6-per-origin pool)
    }
    const off = [head.on(tick), app.on(() => tick())];
    const timer = setInterval(tick, 200);       // the playhead only moves while playing; this catches pause/stop
    return () => { off.forEach((f) => f()); clearInterval(timer); stopAll(); pool.forEach((a) => { a.removeAttribute("src"); a.load(); }); pool.clear(); };
  },
};
