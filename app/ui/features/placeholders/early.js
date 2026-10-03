// Stand-ins that sit BEFORE a real feature in the same row (see index.js).
import { composer as c } from "../../composer/composer.js";

// v4's rating block: "Scene N · Rate…", shown only while the selected scene has a render. Inert until ratings land.
const rating = { id: "placeholder:rating", mount: "timeline.actions", needs: ["project"],
  setup({ host, app }) {
    const p = app.project;
    const label = c.text.sm({ text: "" }), rate = c.button.sm({ label: "Rate…", tone: "ghost", disabled: true });
    const box = c.toolbar.default({ items: [label, rate] });
    host.append(box.node);
    const draw = () => {
      const sc = p.selected, has = sc && p.project && (p.project.scene_renders || {})[sc.id];
      box.node.hidden = !has;
      if (has) label.setText ? label.setText(`Scene ${p.scenes.indexOf(sc) + 1}`) : (label.node.textContent = `Scene ${p.scenes.indexOf(sc) + 1}`);
    };
    draw();
    return app.on(draw);
  } };

export default [rating, { id: "placeholder:sampler", mount: "timeline.status",
  setup: ({ host }) => host.append(c.button.sm({ label: "⏱ Sampler", tone: "neutral", disabled: true }).node) }];
