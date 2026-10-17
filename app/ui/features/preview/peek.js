// Look at a bin file on the monitor: click a tile and it shows over the picture (a picture, a clip with controls, a sound with controls);
// click it again, press ✕ or Esc and the cut is back. Nothing in the project changes.
import { composer as c } from "../../composer/composer.js";

const url = (id) => `/funpack/api/media/${encodeURIComponent(id)}/file`;

export default {
  id: "media-peek",
  mount: "preview",
  needs: ["project", "playhead", "keys"],
  setup({ host, app }) {
    let now = null;
    const layer = c.viewer.cover({ onClose: () => { now = null; } });
    host.append(layer.node);

    function open(item) {
      layer.close();
      app.say("play.pause");
      now = item.id;
      layer.show({ src: url(item.id), kind: item.kind });
    }
    const off = [app.on((what) => {
      if (what === "media.peek") { const item = app.mediaPeek && app.mediaPeek.item; if (!item) return; if (now === item.id) layer.close(); else open(item); }
      else if (now && (what === "open" || what === "play.start" || what === "play.toggle")) layer.close();
    }), app.keys.bind("escape", () => (now ? (layer.close(), true) : false))];
    return () => { off.forEach((f) => f()); layer.destroy(); };
  },
};
