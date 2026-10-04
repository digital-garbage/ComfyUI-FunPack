// The media bin: files brought in (reference images, clips), kept apart from what runs produce.
import { composer as c } from "../../composer/composer.js";
import { learn } from "../../shell/bin.js";
import { MEDIA_DRAG } from "../../shell/dnd.js";

const base = (id) => `/funpack/api/media/${encodeURIComponent(id)}`;

export default {
  id: "media",
  mount: "assets.media",
  needs: ["project", "api"],
  setup({ host, app }) {
    const p = app.project, api = app.api;
    let items = [];
    const tell = (text) => c.toast.warn({ text });

    const toggle = (id) => { if (chosen.has(id)) chosen.delete(id); else chosen.add(id); draw(true); };
    const FILTERS = [{ value: "all", label: "All" }, { value: "video", label: "Video" }, { value: "audio", label: "Audio" }, { value: "image", label: "Images" }];
    const SORTS = [{ value: "name", label: "Name A-Z" }, { value: "name-", label: "Name Z-A" }, { value: "kind", label: "Type" }, { value: "added", label: "Date added" }];
    const VIEWS = [{ value: "adaptive", label: "Grid" }, { value: "list", label: "List" }, { value: "icons", label: "Icons" }];
    const DENSITY = [{ value: "0", label: "Auto" }, ...[1, 2, 3, 4].map((n) => ({ value: String(n), label: `${n}×` }))];
    const show = { filter: "all", sort: "name", view: "adaptive", cols: "0" };

    let selecting = false;
    const chosen = new Set();
    const peek = (it) => { const item = items.find((m) => m.id === it.id); if (!item) return; app.mediaPeek.item = item; app.say("media.peek"); };       // a click looks at it on the monitor
    const pick = (it) => {                                  // an image becomes the selected scene's resolution source
      const sc = p.selected, item = items.find((m) => m.id === it.id);
      if (!sc || !item) return tell("Select a scene first.");
      if (item.kind !== "image") return tell("Only an image can be a scene's resolution source.");
      p.setScene(sc.id, "source_image", item.id);
      draw();
    };
    const context = (it, e) => c.menu.context({ x: e.clientX, y: e.clientY, onPick: (id) => act[id](it),
      items: [{ id: "look", label: "Look at it on the monitor" }, { id: "res", label: "Use as the selected scene's resolution source", disabled: !p.selected || it.kind === "audio" || !items.some((m) => m.id === it.id && m.kind === "image") }, { id: "ref", label: "Use as a reference for the selected scene", disabled: !p.selected || (p.selected.references || []).includes(it.id) }, { id: "rename", label: "Rename…" },
        { id: "export", label: "Save to your computer", disabled: !items.some((m) => m.id === it.id && m.kind !== "audio") }, { separator: true }, { id: "delete", label: "Delete from the bin", danger: true }] });
    /** Delete these from the bin, and nothing in this project may keep pointing at them. */
    async function gone(ids) {
      const dead = new Set();
      try {
        for (const id of ids) { await api.deleteMedia(id); dead.add(id); }
      } catch (err) { tell(`${err.message}${dead.size ? ` (${dead.size} of ${ids.length} were deleted)` : ""}`); }
      try {
        p.edit((pr) => pr.scenes.reduce((hit, s) => {
          const had = dead.has(s.source_image) || (s.references || []).some((r) => dead.has(r));
          if (dead.has(s.source_image)) s.source_image = "";
          if (had) s.references = (s.references || []).filter((r) => !dead.has(r));
          return hit || had;
        }, false));
        ids.forEach((id) => chosen.delete(id));
      } catch (err) { tell(err.message); }
      await refresh();
    }
    const act = {
      look: (it) => peek(it), res: (it) => pick(it),
      ref: (it) => p.setScene(p.selected.id, "references", [...(p.selected.references || []), it.id]),
      rename: async (it) => {
        const name = await c.modal.prompt({ title: "Rename media", label: "Name", value: (items.find((m) => m.id === it.id) || {}).name || "", confirmLabel: "Rename" }).result;
        if (!name || !name.trim()) return;
        try { await api.renameMedia(it.id, name.trim()); await refresh(); } catch (err) { tell(err.message); }
      },
      export: (it) => { const m = items.find((x) => x.id === it.id); Object.assign(document.createElement("a"), { href: `${base(it.id)}/file`, download: m ? m.name : it.id }).click(); },
      delete: async (it) => {
        const name = (items.find((m) => m.id === it.id) || {}).name || "this file";
        if (!(await c.modal.dialogue({ title: "Delete media", message: `Delete “${name}” from the bin for good? Scenes in other projects that use it will lose it too.`, tone: "danger", confirmLabel: "Delete" }).result)) return;
        await gone([it.id]);
      },
    };
    const props = { id: "media", items: [], empty: "No media yet. Drop images or clips here.", onActivate: (it) => (selecting ? toggle(it.id) : peek(it)), onContext: context, drag: { type: MEDIA_DRAG, data: (it) => ({ id: it.id, kind: (items.find((m) => m.id === it.id) || {}).kind }) } };
    const galleries = { adaptive: c.gallery.adaptive(props), list: c.gallery.list(props), icons: c.gallery.icons(props) };
    const shelf = c.region.stack({ gap: "none", children: [galleries.adaptive] });
    const note = c.text.sm({ text: "" });
    const drop = c.dropzone.default({ label: "Drop or choose files", hint: "images, clips, audio",
      onFiles: async (files) => {
        try { const r = await api.uploadMedia(files, (i, n, name) => note.setText(`Uploading ${i + 1}/${n}: ${name}`)); (r.problems || []).forEach(tell); await refresh(); } catch (err) { tell(err.message); }
        note.setText("");
      } });
    const seg = (label, options, key) => c.segmented.sm({ label, options, value: show[key], onChange: (v) => { show[key] = v; draw(true); } });
    const sort = c.select.sm({ label: "Sort", options: SORTS, value: show.sort, onChange: (v) => { show.sort = v; draw(true); } });
    const selectBtn = c.button.sm({ label: "Select", tone: "ghost", title: "Pick several files, then delete them together", onClick: () => { selecting = !selecting; chosen.clear(); draw(true); } });
    const bulk = c.button.sm({ label: "Delete (0)", tone: "danger", onClick: async () => {
      const ids = [...chosen];
      if (!ids.length || !(await c.modal.dialogue({ title: "Delete media", message: `Delete ${ids.length} file${ids.length > 1 ? "s" : ""} from the bin for good? Scenes in other projects that use them will lose them too.`, tone: "danger", confirmLabel: "Delete" }).result)) return;
      await gone(ids);
    } });
    bulk.node.hidden = true;
    host.append(c.toolbar.default({ items: [c.text.sm({ text: "Media" })], trailing: [selectBtn, bulk] }).node, drop.node, note.node, seg("Show", FILTERS, "filter").node, sort.node,
      seg("View", VIEWS, "view").node, seg("Tile size", DENSITY, "cols").node, shelf.node);

    let drawn = "";
    function draw(force) {
      const key = `${(p.selected || {}).source_image}|${items.map((m) => m.id)}|${selecting}|${[...chosen]}|${show.filter}`;
      if (!force && key === drawn) return;           // typing elsewhere must not rebuild the thumbnails
      drawn = key;
      const visible = new Set(items.filter((m) => show.filter === "all" || m.kind === show.filter).map((m) => m.id));
      for (const id of [...chosen]) if (!visible.has(id)) chosen.delete(id);       // a file out of sight is not picked
      const rev = show.sort.endsWith("-") ? -1 : 1, by = show.sort.replace("-", "");
      const shown = items.filter((m) => show.filter === "all" || m.kind === show.filter)
        .sort((x, y) => rev * (by === "added" ? (y.added || 0) - (x.added || 0) : String(x[by === "kind" ? "kind" : "name"]).localeCompare(String(y[by === "kind" ? "kind" : "name"]))));
      const g = galleries[show.view];
      shelf.set([g]);
      if (g.setCols) g.setCols(Number(show.cols));
      g.setItems(shown.map((m) => ({ id: m.id, label: m.name, thumb: m.kind === "audio" ? undefined : `${base(m.id)}/thumb`, badge: m.kind })));
      const sc = p.selected;
      g.setValue(selecting ? [...chosen] : sc && sc.source_image ? [sc.source_image] : []);
      selectBtn.setLabel(selecting ? "Done" : "Select");
      bulk.node.hidden = !selecting;
      bulk.setLabel(`Delete (${chosen.size})`);
      bulk.setDisabled(!chosen.size);
    }
    async function refresh() {
      try { items = (await api.media()).media || []; learn(items); app.say("bin"); } catch (err) { tell(err.message); }
      draw(true);
    }
    refresh();
    return app.on((what) => (what === "media" ? refresh() : draw()));       // "media": another feature put something in the bin
  },
};
