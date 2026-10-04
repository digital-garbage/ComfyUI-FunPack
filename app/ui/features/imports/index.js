// "＋ Import" on the timeline toolbar: a video from the media bin as a clip of its own, or a sound as an audio track at the playhead.
import { composer as c } from "../../composer/composer.js";
import { addVideoClip, addAudioTrack } from "../../shell/imports.js";
import { learn } from "../../shell/bin.js";

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
    return () => menu.node.remove();
  },
};
