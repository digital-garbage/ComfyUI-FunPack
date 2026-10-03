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

    const FILTERS = [{ value: "all", label: "All" }, { value: "video", label: "Video" }, { value: "audio", label: "Audio" }, { value: "image", label: "Images" }];
    const SORTS = [{ value: "name", label: "Name A-Z" }, { value: "name-", label: "Name Z-A" }, { value: "kind", label: "Type" }, { value: "added", label: "Date added" }];
    const VIEWS = [{ value: "adaptive", label: "Grid" }, { value: "list", label: "List" }, { value: "icons", label: "Icons" }];
    const DENSITY = [{ value: "0", label: "Auto" }, ...[1, 2, 3, 4].map((n) => ({ value: String(n), label: `${n}×` }))];
    const show = { filter: "all", sort: "name", view: "adaptive", cols: "0" };

    const pick = (it) => {                                  // an image becomes the selected scene's source
      const sc = p.selected, item = items.find((m) => m.id === it.id);
      if (!sc || !item) return tell("Select a scene first.");
      if (item.kind !== "image") return tell("Only an image can be a scene's source here.");
      p.setScene(sc.id, "source_image", item.id);
      draw();
    };
    const context = (it, e) => c.menu.context({ x: e.clientX, y: e.clientY, onPick: (id) => act[id](it),
      items: [{ id: "ref", label: "Use as a reference for the selected scene", disabled: !p.selected || (p.selected.references || []).includes(it.id) }, { id: "rename", label: "Rename…" },
        { id: "export", label: "Save to your computer", disabled: !items.some((m) => m.id === it.id && m.kind !== "audio") }, { separator: true }, { id: "delete", label: "Delete from the bin", danger: true }] });
    const act = {
      ref: (it) => p.setScene(p.selected.id, "references", [...(p.selected.references || []), it.id]),
      rename: async (it) => {
        const name = await c.modal.prompt({ title: "Rename media", label: "Name", value: (items.find((m) => m.id === it.id) || {}).name || "", confirmLabel: "Rename" }).result;
        if (!name || !name.trim()) return;
        try { await api.renameMedia(it.id, name.trim()); await refresh(); } catch (err) { tell(err.message); }
      },
      export: (it) => { const m = items.find((x) => x.id === it.id); Object.assign(document.createElement("a"), { href: `${base(it.id)}/file`, download: m ? m.name : it.id }).click(); },
      delete: async (it) => {
        try {
          await api.deleteMedia(it.id);
          p.edit((pr) => pr.scenes.reduce((hit, s) => {
            const had = s.source_image === it.id || (s.references || []).includes(it.id);
            if (s.source_image === it.id) s.source_image = "";
            if (had) s.references = (s.references || []).filter((r) => r !== it.id);
            return hit || had;
          }, false));    // nothing may keep pointing at it
          await refresh();
        } catch (err) { tell(err.message); }
      },
    };
    const props = { id: "media", items: [], empty: "No media yet. Drop images or clips here.", onActivate: pick, onContext: context };
    const galleries = { adaptive: c.gallery.adaptive(props), list: c.gallery.list(props), icons: c.gallery.icons(props) };
    const shelf = c.region.stack({ gap: "none", children: [galleries.adaptive] });
    const drop = c.dropzone.default({ label: "Drop or choose files", hint: "images, clips, audio",
      onFiles: async (files) => {
        try { const r = await api.uploadMedia(files); (r.problems || []).forEach(tell); await refresh(); } catch (err) { tell(err.message); }
      } });
    const seg = (label, options, key) => c.segmented.sm({ label, options, value: show[key], onChange: (v) => { show[key] = v; draw(true); } });
    const sort = c.select.sm({ label: "Sort", options: SORTS, value: show.sort, onChange: (v) => { show.sort = v; draw(true); } });
    host.append(c.toolbar.default({ items: [c.text.sm({ text: "Media" })] }).node, drop.node, seg("Show", FILTERS, "filter").node, sort.node,
      seg("View", VIEWS, "view").node, seg("Tile size", DENSITY, "cols").node, shelf.node);

    let drawn = "";
    function draw(force) {
      const key = `${(p.selected || {}).source_image}|${items.map((m) => m.id)}`;
      if (!force && key === drawn) return;           // typing elsewhere must not rebuild the thumbnails
      drawn = key;
      const rev = show.sort.endsWith("-") ? -1 : 1, by = show.sort.replace("-", "");
      const shown = items.filter((m) => show.filter === "all" || m.kind === show.filter)
        .sort((x, y) => rev * (by === "added" ? (y.added || 0) - (x.added || 0) : String(x[by === "kind" ? "kind" : "name"]).localeCompare(String(y[by === "kind" ? "kind" : "name"]))));
      const g = galleries[show.view];
      shelf.set([g]);
      if (g.setCols) g.setCols(Number(show.cols));
      g.setItems(shown.map((m) => ({ id: m.id, label: m.name, thumb: m.kind === "audio" ? undefined : `${base(m.id)}/thumb`, badge: m.kind })));
      const sc = p.selected;
      g.setValue(sc && sc.source_image ? [sc.source_image] : []);
    }
    async function refresh() {
      try { items = (await api.media()).media || []; } catch (err) { tell(err.message); }
      draw(true);
    }
    refresh();
    return app.on((what) => (what === "media" ? refresh() : draw()));       // "media": another feature put something in the bin
  },
};
