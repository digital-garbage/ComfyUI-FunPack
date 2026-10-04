// "＋ Import" on the timeline toolbar: a video from the media bin as a clip of its own, or a sound as an audio track at the playhead.
import { composer as c } from "../../composer/composer.js";
import { addVideoClip, addAudioTrack } from "../../shell/imports.js";
import { learn } from "../../shell/bin.js";
import { onMediaDrop } from "../../shell/dnd.js";
import { addScene } from "../../shell/edits.js";
import { isVideoClip, isSubclip } from "../../shell/scenes.js";

/** How long a bin file is, from the browser's own reading of it (null when it cannot say). */
const lengthOf = (asset) => new Promise((resolve) => {
  const m = document.createElement(asset.kind === "audio" ? "audio" : "video");
  const done = (v) => { m.removeAttribute("src"); m.load(); resolve(v); };
  m.preload = "metadata";
  m.onloadedmetadata = () => done(Number.isFinite(m.duration) ? m.duration : null);
  m.onerror = () => done(null);
  setTimeout(() => done(null), 8000);
  m.src = `/funpack/api/media/${encodeURIComponent(asset.id)}/file`;
});

export default {
  id: "imports",
  mount: "timeline.toolbar",
  needs: ["project", "api", "playhead"],
  setup({ host, app }) {
    const p = app.project;

    async function choose(kind) {
      let bin;
      try { const all = (await app.api.media()).media || []; learn(all); bin = all.filter((m) => m.kind === kind); } catch (err) { return c.toast.warn({ text: `Could not read the media bin: ${err.message}` }); }
      const win = c.modal.generic({ title: kind === "video" ? "Add video clip" : "Add audio track", size: "sm", body: c.region.stack({ gap: "sm", children: [
        c.hint.default({ text: kind === "video" ? "Video clips play as they are and are skipped by Generate." : "The sound starts at the playhead." }),
        bin.length ? c.checklist.default({ label: "Media", items: bin.map((m) => ({ value: m.id, label: m.name })), values: [], onChange: (v) => { const id = v[v.length - 1]; if (id) { win.close("done"); add(bin.find((m) => m.id === id)); } } })
          : c.hint.default({ text: `The media bin has no ${kind} yet: drop one onto the Assets panel first.` })] }) });
    }

    async function add(asset) {
      const seconds = await lengthOf(asset), at = app.playhead.at;
      if (!p.project) return;
      const made = p.edit((pr) => (asset.kind === "video" ? addVideoClip(pr, asset, seconds) : addAudioTrack(pr, asset, at, seconds)));
      if (!made) return c.toast.warn({ text: "Could not add that." });
      if (asset.kind === "video") p.select(p.scenes[p.scenes.length - 1].id);
      if (seconds == null) c.toast.warn({ text: asset.kind === "video" ? "The file's length could not be read; the clip uses the project's length." : "The file's length could not be read: the lane cannot be trimmed, and plays to the end of the file." });
    }

    const menu = c.button.menu({ label: "＋ Import", tone: "ghost", onClick: () => c.menu.dropdown({ anchor: menu, items: [{ id: "video", label: "Video clip…", disabled: !p.project }, { id: "audio", label: "Audio track…", disabled: !p.project }],
      onPick: (id) => choose(id) }) });
    host.append(menu.node);

    // Dropping a bin tile on the timeline: a video makes a clip, a sound a lane, a picture a scene's anchor (on a clip) or a new scene (beside them).
    const at = (e) => (app.timelineView && app.timelineView.timeAt ? app.timelineView.timeAt(e.clientX) : app.playhead.at);
    const asset = async (item) => { let list = []; try { list = (await app.api.media()).media || []; } catch { /* the drop does nothing */ } return list.find((m) => m.id === item.id); };
    const offs = [
      onMediaDrop(".cx-nle-lane.cx-nle-lane-video", async (item, lane, e) => {
        const a = await asset(item);
        if (!a || !p.project) return;
        const clip = e.target.closest && e.target.closest(".cx-nle-clip[data-id]"), id = clip && clip.dataset.id;
        if (a.kind === "video") return add(a);
        if (a.kind !== "image") return c.toast.warn({ text: "A sound goes on an audio lane." });
        const hit = id && p.scenes.find((s) => s.id === id);
        if (hit) {
          if (isVideoClip(hit) || isSubclip(hit)) return c.toast.warn({ text: "Only a generated scene can start from a picture (not a video clip or a cut part)." });
          p.setScene(id, "source_image", a.id);
          return c.toast.good({ text: "Set as that clip's starting picture." });
        }
        const made = p.edit((pr) => { const sc = addScene(pr, "image"); sc.source_image = a.id; return sc; });
        if (made) p.select(made.id);
      }),
      onMediaDrop(".cx-nle-lane.cx-nle-lane-audio", async (item, lane, e) => {
        const a = await asset(item);
        if (!a || a.kind !== "audio" || !p.project) return a && c.toast.warn({ text: "Only a sound can go on an audio lane." });
        const seconds = await lengthOf(a);
        p.edit((pr) => addAudioTrack(pr, a, at(e), seconds));
      }),
    ];
    return () => { offs.forEach((f) => f()); menu.node.remove(); };
  },
};
