// Export and Save to media bin: the picked clips, joined in timeline order, as one file. Nothing else is applied.
import { composer as c } from "../../composer/composer.js";
import { call } from "../../shell/api.js";
import { viewUrl } from "../../shell/run.js";
import { inPlace } from "../../shell/place.js";
import { clipSpecs } from "./clips.js";
import { runJob } from "./job.js";

export default {
  id: "render-selected",
  mount: "timeline.toolbar",
  needs: ["project", "selection"],
  setup({ host, app }) {
    const p = app.project, sel = app.selection;
    const put = inPlace(host);
    let busy = false;
    const tell = (text) => c.toast.warn({ text });

    // -> the joined file's media, or null (already told why)
    async function join() {
      const open = p.project, { clips, missing } = clipSpecs(open, new Set(sel.ids));
      if (!clips.length) return tell("Generate the picked clip(s) first, or pick a video clip that has media."), null;
      if (missing && !(await c.modal.dialogue({ title: "Some clips have no render", message: `${missing} picked clip(s) have no render yet and will be skipped. Continue with ${clips.length}?`, confirmLabel: "Continue" }).result)) return null;
      await p.flush();
      const state = await runJob(`/api/projects/${open.id}/export-clips`, { clips });
      return state.state === "done" ? state.media : (tell(state.detail || "The export failed."), null);
    }
    const run = (work) => async () => {
      if (busy) return;
      busy = true; draw();
      try { await work(); } catch (err) { tell(err.message); } finally { busy = false; draw(); }
    };
    const exportIt = run(async () => {
      const name = p.project.name, media = await join();         // named for the project it was started in
      if (!media) return;
      const link = Object.assign(document.createElement("a"), { href: viewUrl(media), download: `${name}.mp4` });
      link.click();
    });
    const toBin = run(async () => {
      const name = p.project.name, media = await join();
      if (!media) return;
      const { media: entry } = await call("POST", "/api/import-clip", { clip: media, name: `${name}.mp4` });
      app.say("media");
      c.toast.good({ text: `Saved to the media bin as ${entry.name}.` });
    });
    function draw() {
      const off = busy || !p.project || !sel.ids.length;
      put(c.button.sm({ label: "⤓ Export", tone: "ghost", disabled: off, onClick: exportIt, title: "Join the picked clips and download them" }).node,
        c.button.sm({ label: "Save to media bin", tone: "ghost", disabled: off, onClick: toBin, title: "Join the picked clips and keep them in the media bin" }).node);
    }
    draw();
    return app.on(draw);
  },
};
