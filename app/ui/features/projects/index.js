// The Assets zone's project list: every project, the open one marked, "New" to start another.
import { composer as c } from "../../composer/composer.js";
import { list } from "../../shell/project.js";

export default {
  id: "projects",
  mount: "assets",
  needs: ["project"],
  setup({ host, app }) {
    const p = app.project;
    const rows = c.gallery.rows({ items: [], onActivate: (item) => p.open(item.id) });
    const news = c.button.sm({ label: "＋ New", onClick: async () => {
      const name = await c.modal.prompt({ title: "New project", label: "Name", value: "Untitled" }).result;
      if (name) await p.newProject(name.trim() || "Untitled");
    } });
    host.append(c.toolbar.default({ items: [c.text.sm({ text: "Projects" })], trailing: [news] }).node, rows.node);

    let known = [];
    const draw = () => {
      rows.setItems(known.map((r) => ({ id: r.id, label: r.name, hint: r.scene_count == null ? "" : `${r.scene_count}▦` })));
      if (p.project) rows.setValue([p.project.id]);
    };
    const refresh = () => list().then((all) => { known = all; draw(); }).catch(() => draw());
    refresh();
    return app.on((what) => { if (what === "open") refresh(); else draw(); });
  },
};
