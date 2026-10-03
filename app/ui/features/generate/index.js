// Generate / Selected: queue the plan's scenes one after another and put each result on its clip.
import { composer as c } from "../../composer/composer.js";
import { isGenerative, genUnitId, effFrames, effFps } from "../../shell/scenes.js";
import { buildInputs } from "./inputs.js";

const expand = (body) => fetch("/funpack/api/prompt/expand", { method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ ...body, seed: Math.floor(Math.random() * 2 ** 31) || 1 }) }).then((r) => (r.ok ? r.json() : null));

export default {
  id: "generate",
  mount: "timeline.actions",
  needs: ["project", "pipeline", "generate"],
  setup({ host, app }) {
    const p = app.project, g = app.generate;
    let busy = false;

    const record = (pid, id, image) => {
      if (!image || !p.project || p.project.id !== pid) return;
      p.edit((pr) => {
        const sc = pr.scenes.find((s) => s.id === id);
        if (!sc) return false;
        (pr.scene_renders ||= {})[id] = { media: { filename: image.filename, subfolder: image.subfolder || "", type: image.type || "output" },
          durationSec: effFrames(sc, pr) / effFps(sc, pr), inSec: 0 };
        return true;
      });
    };

    async function runAll(scenes) {
      if (busy) return;
      busy = true; draw();
      const pid = p.project.id, seen = new Set();
      try {
        for (const sc of scenes) {
          if (!isGenerative(sc) || sc.excluded || seen.has(genUnitId(sc))) continue;
          seen.add(genUnitId(sc));
          const { inputs, unwired } = await buildInputs({ project: p.project, scene: sc, slots: app.pipeline.slots(), expand });
          if (unwired) c.toast.warn({ text: `${unwired} reference(s) did not fit this pipeline and are not used.` });
          const done = g.waitForTerminal();             // listening before the run starts, so a fast one is not missed
          if (!(await g.generate({ sceneId: sc.id, projectId: pid, inputs }))) { done.cancel(); break; }
          const end = await done;
          if (end !== g.DONE) break;
          const images = g.run.state.images;
          record(pid, sc.id, images[images.length - 1]);
        }
      } finally { busy = false; draw(); }
    }

    let nodes = [];
    function draw() {
      const open = p.project;
      const go = c.button.sm({ label: busy ? "Generating…" : "▶ Generate", tone: "primary", disabled: busy || !open,
        onClick: () => runAll(p.scenes) });
      const one = c.button.sm({ label: "Selected", disabled: busy || !p.selected, onClick: () => runAll([p.selected]) });
      const stop = c.button.sm({ label: "■ Stop", tone: "danger", disabled: !busy, onClick: () => g.cancel() });
      const next = [go, one, stop].map((b) => b.node);
      if (nodes.length) nodes.forEach((n, i) => n.replaceWith(next[i])); else host.append(...next);     // in place: keeps its spot in the row
      nodes = next;
    }
    draw();
    return app.on(draw);
  },
};
