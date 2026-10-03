// Render: stitch the cut into one file and keep it in the media bin.
import { composer as c } from "../../composer/composer.js";
import { call } from "../../shell/api.js";
import { clipSpecs } from "./clips.js";

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

export default {
  id: "render",
  mount: "timeline.actions",
  needs: ["project"],
  setup({ host, app }) {
    const p = app.project;
    let busy = false, node = null;
    const tell = (text) => c.toast.warn({ text });

    async function render() {
      const open = p.project, { clips, missing } = clipSpecs(open);
      if (!clips.length) return tell("Nothing to render: generate at least one clip first.");
      if (missing) tell(`${missing} scene(s) have no render yet and are left out.`);
      busy = true; draw();
      try {
        await p.flush();
        const { job_id: job } = await call("POST", `/api/projects/${open.id}/render`, { clips });
        let state;
        do { await wait(1500); state = await call("GET", `/api/projects/${open.id}/render/${job}`); } while (state.state === "queued" || state.state === "running");
        if (state.state !== "done") return tell(state.detail || "The render failed.");
        const { media } = await call("POST", "/api/import-clip", { clip: state.media, name: `${open.name}.mp4` });
        if (app.say) app.say("media");                   // the bin shows it now
        c.toast.good({ text: `Rendered: saved to the media bin as ${media.name}.` });
      } catch (err) { tell(err.message); } finally { busy = false; draw(); }
    }
    function draw() {
      const next = c.button.sm({ label: busy ? "Rendering…" : "⧉ Render", tone: "render", disabled: busy || !p.project, onClick: render }).node;
      if (node) node.replaceWith(next); else host.append(next);
      node = next;
    }
    draw();
    return app.on(() => { if (Boolean(p.project) === node.disabled) draw(); });
  },
};
