// What a run is doing, beside Generate: the step it is on, a bar, and how long it has been going (this page's own clock).
import { composer as c } from "../../composer/composer.js";
import { QUEUED, RUNNING } from "../../shell/run.js";

export const elapsed = (ms) => { const s = Math.max(0, Math.floor(ms / 1000)); return `${String(Math.floor(s / 60)).padStart(2, "0")}m ${String(s % 60).padStart(2, "0")}s`; };

export default {
  id: "generate-progress",
  mount: "timeline.actions",
  needs: ["generate"],
  setup({ host, app }) {
    const g = app.generate;
    const bar = c.progress.inline({ label: "Generating" });
    bar.setHidden(true);
    host.append(bar.node);
    let since = 0;
    const draw = () => {
      const s = g.state(), working = s.phase === QUEUED || s.phase === RUNNING;
      if (working && !since) since = Date.now();
      if (!working) since = 0;
      bar.setHidden(!working);
      if (!working) { return; }
      const pr = s.progress;
      bar.setValue(pr && pr.max ? (pr.value / pr.max) * 100 : 0);
      bar.setText(`${s.phase === QUEUED ? "Queued" : pr && pr.max ? `Step ${pr.value}/${pr.max}` : "Running"} · ${elapsed(Date.now() - since)}`);

    };
    const off = g.subscribe(draw);
    const timer = setInterval(() => { if (since) draw(); }, 1000);       // the clock ticks between ComfyUI's messages
    draw();
    return () => { off(); clearInterval(timer); bar.destroy(); };
  },
};
