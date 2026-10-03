// Generate / Selected / Stop: queue generated units one after another and put each result on its clips.
import { composer as c } from "../../composer/composer.js";
import { isGenerative, genUnitId, unitRoot, effFrames, effFps } from "../../shell/scenes.js";
import { buildInputs } from "./inputs.js";

const expand = (body) => fetch("/funpack/api/prompt/expand", { method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ ...body, seed: Math.floor(Math.random() * 2 ** 31) || 1 }) }).then((r) => (r.ok ? r.json() : null));

/** The render a scene gets from its unit's one clip: a cut half plays from where its half begins. */
export const renderFor = (sc, p, media, unitSec) => ({ media, inSec: (sc.cut_offset_frames || 0) / effFps(sc, p),
  ...((sc.frames_mode || "project") === "project" ? { durationSec: unitSec } : {}) });

export default {
  id: "generate",
  mount: "timeline.actions",
  needs: ["project", "pipeline", "generate"],
  setup({ host, app }) {
    const p = app.project, g = app.generate;
    const tell = (text) => c.toast.warn({ text });
    g.on("say", tell); g.on("warn", tell);          // the pipeline check's refusals: said where the person is looking
    let busy = false, stopped = false;

    const record = (pid, unit, image) => p.project && p.project.id === pid && p.edit((pr) => {
      const group = pr.scenes.filter((s) => genUnitId(s) === unit);
      if (!group.length) return false;
      const secs = group.reduce((t, s) => t + effFrames(s, pr) / effFps(s, pr), 0);
      const media = { filename: image.filename, subfolder: image.subfolder || "", type: image.type || "output" };
      group.forEach((s) => { (pr.scene_renders ||= {})[s.id] = renderFor(s, pr, media, secs); });
      return true;
    });

    async function runUnits(units) {
      if (busy) return;
      busy = true; stopped = false; draw();
      const pid = p.project.id;
      try {
        for (const unit of units) {
          if (stopped) break;
          if (!p.project || p.project.id !== pid) { tell("Stopped: another project was opened."); break; }
          const root = unitRoot(p.project, unit);
          if (!root || root.excluded || !isGenerative(root)) continue;        // removed, left out, or not made by the model
          const { inputs, unwired } = await buildInputs({ project: p.project, scene: root, slots: app.pipeline.slots(), expand });
          if (stopped) break;
          if (unwired) tell(`${unwired} reference(s) did not fit this pipeline and are not used.`);
          const done = g.waitForTerminal();             // listening before the run starts, so a fast one is not missed
          if (!(await g.generate({ sceneId: root.id, projectId: pid, inputs }))) { done.cancel(); break; }
          const end = await done;
          if (end === g.CANCELLED) break;
          const images = g.run.state.images;
          if (end !== g.DONE) { tell("Generation failed. The log has ComfyUI's message."); break; }
          if (!images.length) { tell("ComfyUI finished without a result (a cached run makes none). Change the prompt or seed and try again."); break; }
          record(pid, unit, images[images.length - 1]);
        }
      } finally { busy = false; draw(); }
    }
    const unitsOf = (scenes) => [...new Set(scenes.map(genUnitId))];

    let nodes = [];
    function draw() {
      const open = p.project;
      const go = c.button.sm({ label: busy ? "Generating…" : "▶ Generate", tone: "primary", disabled: busy || !open,
        onClick: () => runUnits(unitsOf(p.scenes)) });
      const one = c.button.sm({ label: "Selected", disabled: busy || !p.selected, onClick: () => runUnits(unitsOf([p.selected])) });
      const stop = c.button.sm({ label: "■ Stop", tone: "danger", disabled: !busy, onClick: () => { stopped = true; g.cancel(); } });
      const next = [go, one, stop].map((b) => b.node);
      if (nodes.length) nodes.forEach((n, i) => n.replaceWith(next[i])); else host.append(...next);     // in place: keeps its spot in the row
      nodes = next;
    }
    draw();
    return app.on(draw);
  },
};
