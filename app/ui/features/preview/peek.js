// Look at a bin file on the monitor: click a tile and it shows over the picture (a picture, a clip with controls, a sound with controls);
// click it again, press ✕ or Esc and the cut is back. Nothing in the project changes.
import { composer as c } from "../../composer/composer.js";

const url = (id) => `/funpack/api/media/${encodeURIComponent(id)}/file`;

export default {
  id: "media-peek",
  mount: "preview",
  needs: ["project", "playhead", "keys"],
  setup({ host, app }) {
    const layer = document.createElement("div");
    layer.style.cssText = "position:absolute;inset:0;z-index:5;display:flex;align-items:center;justify-content:center;background:var(--elev-1,#111);padding:8px;box-sizing:border-box";
    layer.hidden = true;
    host.style.position = "relative";
    host.append(layer);
    let now = null, media = null;

    const close = () => {
      if (media) { media.removeAttribute("src"); if (media.load) media.load(); media = null; }       // give the connection back
      layer.replaceChildren(); layer.hidden = true; now = null;
      return true;
    };
    function open(item) {
      close();
      app.say("play.pause");
      now = item.id;
      const tag = item.kind === "image" ? "img" : item.kind === "audio" ? "audio" : "video";
      media = document.createElement(tag);
      media.src = url(item.id);
      if (tag !== "img") { media.controls = true; media.autoplay = true; }
      media.style.cssText = tag === "audio" ? "inline-size:80%" : "max-inline-size:100%;max-block-size:100%;object-fit:contain";
      const x = c.iconButton.sm({ icon: "✕", label: "Close preview", onClick: close }).node;
      x.style.cssText = "position:absolute;inset-block-start:6px;inset-inline-end:6px";
      layer.replaceChildren(media, x);
      layer.hidden = false;
    }
    const off = [app.on((what) => {
      if (what === "media.peek") { const item = app.mediaPeek && app.mediaPeek.item; if (!item) return; if (now === item.id) close(); else open(item); }
      else if (now && (what === "open" || what === "play.start" || what === "play.toggle")) close();
    }), app.keys.bind("escape", () => (now ? close() : false))];
    return () => { off.forEach((f) => f()); close(); layer.remove(); };
  },
};
