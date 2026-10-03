// The monitor: plays the render of the selected scene.
import { composer as c } from "../../composer/composer.js";
import { viewUrl } from "../../shell/run.js";

const start = (sc, r, open) => (r.inSec || 0) + (sc.source_in || 0);   // where this clip begins in its render

export default {
  id: "preview",
  mount: "preview",
  needs: ["project"],
  setup({ host, app }) {
    const p = app.project;
    const viewer = c.viewer.media({ kind: "video", empty: "No render yet. Use Generate in the timeline header." });
    host.append(viewer.node);
    let shown = "";
    const draw = () => {
      const sc = p.selected, media = sc && p.project && (p.project.scene_renders || {})[sc.id];
      const src = media && media.media ? `${viewUrl(media.media)}#t=${start(sc, media, p.project)}` : "";   // a split clip starts where its half does
      if (src === shown) return;          // a keystroke elsewhere must not restart the video
      shown = src;
      viewer.setSource(src || null, "video", src ? media.media : null);
    };
    draw();
    return app.on(draw);
  },
};
