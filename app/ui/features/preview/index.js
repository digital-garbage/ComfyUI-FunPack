// The monitor: plays the render of the selected scene.
import { composer as c } from "../../composer/composer.js";
import { viewUrl } from "../../shell/run.js";
import { seconds } from "../../shell/scenes.js";

const start = (sc, r, open) => (r.inSec || 0) + (sc.source_in || 0);   // where this clip begins in its render

// A clip with a length of its own (cut, trimmed) stops there; one that follows the project plays its render out.
const end = (sc, r, open) => ((sc.frames_mode || "project") === "project" ? "" : `,${start(sc, r, open) + seconds(sc, open)}`);

export default {
  id: "preview",
  mount: "preview",
  needs: ["project"],
  setup({ host, app }) {
    const p = app.project;
    const viewer = c.viewer.media({ kind: "video", empty: "" });
    const empty = c.emptyState.default({ icon: "🎬", title: "No render yet", hint: "Use Generate in the timeline header" });
    host.append(viewer.node, empty.node);
    let shown = null;
    const draw = () => {
      const sc = p.selected, media = sc && p.project && (p.project.scene_renders || {})[sc.id];
      const src = media && media.media ? `${viewUrl(media.media)}#t=${start(sc, media, p.project)}${end(sc, media, p.project)}` : "";   // a split clip starts where its half does
      if (src === shown) return;          // a keystroke elsewhere must not restart the video
      shown = src;
      viewer.node.hidden = !src; empty.node.hidden = Boolean(src);
      viewer.setSource(src || null, "video", src ? media.media : null);
    };
    draw();
    return app.on(draw);
  },
};
