// The audio lanes laid over the cut (files from the bin, and a clip's separated sound): drawn on the timeline, slid and trimmed there,
// and a window for each lane's volume, start, split and removal.
import { composer as c } from "../../composer/composer.js";
import { moveLane, trimLane, removeTrack, splitTrack } from "../../shell/audio.js";

const ID = "t:";

export const tracksLane = (p) => (p.audio_tracks || []).length ? {
  id: "tracks", label: "Audio", kind: "audio", move: true,
  clips: p.audio_tracks.map((t) => ({ id: ID + t.id, start: t.start_sec || 0, dur: t.pinned_dur ?? t.source_dur ?? 1, title: t.label || "Audio", trim: t.kind === "separated" ? t.pinned_dur != null : t.source_dur != null })),
} : null;

export default {
  id: "audio-tracks",
  mount: "timeline.toolbar",
  needs: ["project", "playhead"],
  setup({ app }) {
    const p = app.project;
    const edit = (fn) => p.edit(fn);
    function open(id) {
      const t = (p.project.audio_tracks || []).find((x) => x.id === id);
      if (!t) return;
      const separated = t.kind === "separated";
      let volume = t.volume != null ? t.volume : 1, start = t.start_sec || 0;
      const body = c.region.stack({ gap: "sm", children: [
        c.hint.default({ text: separated ? "This clip's own sound. It follows its clip; slide or trim it on the timeline." : "A sound over the cut. Slide and trim it on the timeline." }),
        c.field.default({ label: "Volume", control: c.slider.readout({ label: "Volume", value: volume, min: 0, max: 2, step: 0.05, precision: 2, onCommit: (v) => { volume = v; } }) }),
        separated ? null : c.field.default({ label: "Starts at (s)", control: c.number.md({ label: "Starts at", value: start, min: 0, step: 0.1, onChange: (v) => { start = v; } }) }),
      ].filter(Boolean) });
      const win = c.modal.generic({ title: t.label || "Audio track", size: "sm", body });
      win.setFooter({ actions: [
        c.button.sm({ label: "Remove", tone: "danger", onClick: async () => { if (await c.modal.dialogue({ title: "Remove audio track", message: separated ? "Remove this track? The clip gets its sound back." : "Remove this audio track?", tone: "danger", confirmLabel: "Remove" }).result) { edit((pr) => removeTrack(pr, id)); win.close("done"); } } }),
        separated ? null : c.button.sm({ label: "Split at playhead", tone: "ghost", onClick: () => { if (edit((pr) => splitTrack(pr, id, app.playhead.at))) win.close("done"); else c.toast.warn({ text: "Put the playhead inside the track to split it." }); } }),
        c.button.sm({ label: "Cancel", tone: "ghost", onClick: () => win.close("cancel") }),
        c.button.sm({ label: "Save", tone: "primary", onClick: () => { edit((pr) => { const x = pr.audio_tracks.find((y) => y.id === id); if (!x) return false; x.volume = volume; if (!separated) x.start_sec = Math.max(0, start); return true; }); win.close("done"); } }),
      ].filter(Boolean) });
    }
    const entry = {
      owns: (id) => id.startsWith(ID),
      lanes: (pr) => [tracksLane(pr)].filter(Boolean),
      select: (id) => open(id.slice(ID.length)),
      move: (id, d) => edit((pr) => moveLane(pr, id.slice(ID.length), d)),
      trim: (id, edge, d) => edit((pr) => trimLane(pr, id.slice(ID.length), edge, d)),
    };
    app.timelineLanes.push(entry);
    app.say("timeline.lanes");
    return () => { const i = app.timelineLanes.indexOf(entry); if (i >= 0) app.timelineLanes.splice(i, 1); app.say("timeline.lanes"); };
  },
};
