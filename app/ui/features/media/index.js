// The media bin: files brought in (reference images, clips), kept apart from what runs produce.
import { composer as c } from "../../composer/composer.js";

const base = (id) => `/funpack/api/media/${encodeURIComponent(id)}`;

export default {
  id: "media",
  mount: "assets.media",
  needs: ["project", "api"],
  setup({ host, app }) {
    const p = app.project, api = app.api;
    let items = [];
    const tell = (text) => c.toast.warn({ text });

    const gallery = c.gallery.adaptive({
      id: "media", items: [], empty: "No media yet. Drop images or clips here.",
      onActivate: (it) => {                                 // an image becomes the selected scene's source
        const sc = p.selected, item = items.find((m) => m.id === it.id);
        if (!sc || !item) return tell("Select a scene first.");
        if (item.kind !== "image") return tell("Only an image can be a scene's source here.");
        p.setScene(sc.id, "source_image", item.id);
        draw();
      },
      onContext: (it, e) => c.menu.context({ x: e.clientX, y: e.clientY, items: [{ id: "delete", label: "Delete from the bin", danger: true }],
        onPick: async () => { try { await api.deleteMedia(it.id); await refresh(); } catch (err) { tell(err.message); } } }),
    });
    const drop = c.dropzone.default({ label: "Drop or choose files", hint: "images, clips, audio",
      onFiles: async (files) => {
        try { const r = await api.uploadMedia(files); (r.problems || []).forEach(tell); await refresh(); } catch (err) { tell(err.message); }
      } });
    host.append(c.toolbar.default({ items: [c.text.sm({ text: "Media" })] }).node, drop.node, gallery.node);

    let drawn = "";
    function draw(force) {
      const key = `${(p.selected || {}).source_image}|${items.map((m) => m.id)}`;
      if (!force && key === drawn) return;           // typing elsewhere must not rebuild the thumbnails
      drawn = key;
      gallery.setItems(items.map((m) => ({ id: m.id, label: m.name, thumb: m.kind === "audio" ? undefined : `${base(m.id)}/thumb`, badge: m.kind })));
      const sc = p.selected;
      gallery.setValue(sc && sc.source_image ? [sc.source_image] : []);
    }
    async function refresh() {
      try { items = (await api.media()).media || []; } catch (err) { tell(err.message); }
      draw(true);
    }
    refresh();
    return app.on(() => draw());
  },
};
