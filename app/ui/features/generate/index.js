// Generate / Selected / Stop: queue generated units one after another and put each result on its clips.
import { composer as c } from "../../composer/composer.js";
import { isGenerative, isVideoClip, genUnitId, unitRoot, effFrames, effFps } from "../../shell/scenes.js";
import { QUEUED, RUNNING } from "../../shell/run.js";
import { inPlace } from "../../shell/place.js";
import { buildInputs, rolesAt } from "./inputs.js";
import { offer } from "../../shell/actions.js";

const MAX_TAKES = 24;
const expand = (body) => fetch("/funpack/api/prompt/expand", { method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ ...body, seed: Math.floor(Math.random() * 2 ** 31) || 1 }) }).then((r) => (r.ok ? r.json() : null));

/** The render a scene gets from its unit's one clip: a cut half plays from where its half begins. */
export const renderFor = (sc, p, media, unitSec, promptId) => ({ media, ...(promptId ? { promptId } : {}), inSec: (sc.cut_offset_frames || 0) / effFps(sc, p),
  ...((sc.frames_mode || "project") === "project" ? { durationSec: unitSec } : {}) });

export default {
  id: "generate",
  mount: "timeline.actions",
  needs: ["project", "pipeline", "generate", "api", "selection"],
  setup({ host, app }) {
    const p = app.project, g = app.generate;
    let said = false;
    const tell = (text) => { said = true; c.toast.warn({ text }); };
    g.on("say", tell); g.on("warn", tell);
    g.on("hold", () => { held = true; draw(); }); g.on("release", () => { held = false; draw(); });   // while the page asks ComfyUI whether a run is already going          // the pipeline check's refusals: said where the person is looking
    let busy = false, stopped = false, held = false;

    const record = (pid, unit, image, frames, promptId) => { app.lastRun.n += 1; return p.editFor(pid, (pr) => {
      const group = pr.scenes.filter((s) => genUnitId(s) === unit);
      if (!group.length) return false;
      const secs = (frames || group.reduce((t, s) => t + effFrames(s, pr), 0) - (group.length - 1)) / (pr.frame_rate || 25);   // as queued, not as the project reads now
      const media = { filename: image.filename, subfolder: image.subfolder || "", type: image.type || "output" };
      const head = group.find((s) => !(s.cut_offset_frames > 0)) || group[0];
      const takes = ((pr.scene_variants ||= {})[head.id] ||= []);        // every render stays as a take: the one that was on the clip keeps its rating
      const was = (pr.scene_renders || {})[head.id];
      const sameTake = (t) => was && (was.promptId ? t.promptId === was.promptId : t.media.filename === (was.media || {}).filename);
      const old = was && was.media && takes.find(sameTake);
      if (old) old.rating = head.rating || "";
      else if (was && was.media) takes.push({ media: was.media, ...(was.promptId ? { promptId: was.promptId } : {}), rating: head.rating || "", ...(was.durationSec ? { secs: was.durationSec } : {}) });       // a render from before takes is a take too
      takes.push({ media, ...(promptId ? { promptId } : {}), rating: "", secs });
      while (takes.length > MAX_TAKES) { const at = takes.findIndex((t) => !t.rating); takes.splice(at < 0 || at === takes.length - 1 ? 0 : at, 1); }       // the oldest unrated goes first; a rated take stays as long as anything else can go
      group.forEach((s) => {
        (pr.scene_renders ||= {})[s.id] = renderFor(s, pr, media, secs, promptId);
        if (!isVideoClip(s)) s.source_in = 0;
        if (!(s.cut_offset_frames > 0)) s.rating = "";       // a new render has not been rated yet      // a fresh render was made at this length: an earlier trim's window no longer applies
      });
      return true;
    }).catch(() => tell("The result could not be saved to its project.")); };

    // `times` > 1: the same shots again, only the seed differing (the prompt's random picks are drawn once), so the takes can be compared.
    async function runUnits(units, times = 1) {
      if (busy) { if (times > 1) tell("A run is already going: wait for it, or Stop it."); return; }
      busy = true; stopped = false; draw();
      const pid = p.project.id;
      let made = 0;
      said = false;
      try {
        // A project's pipeline still going in is the one this run uses, not the last one's.
        if (app.pipeline.settled && !(await app.pipeline.settled())) { tell("Not started: this project's pipeline is still loading. ComfyUI is slow to answer; try again in a moment."); return; }
        if (app.pipelineOwned && !app.pipelineOwned()) { tell(`Not started: ${(app.pipelineWhy && app.pipelineWhy()) || "this project's pipeline is not loaded yet."}`); return; }
        // The pipeline as it is NOW, at the click: every shot of this run uses it, whatever is opened or edited meanwhile.
        const frozen = structuredClone(app.pipeline.slots() || []);
        if (times > 1 && !rolesAt(frozen, "generation.seed").length) { tell("This pipeline has no seed input the app controls, so the takes would all come out the same."); return; }
        const snap = times > 1 ? structuredClone(p.project) : null;        // takes differ in the seed alone: later edits to the project do not reach them
        const drawn = new Map();
        const same = times > 1 ? (body) => { const key = JSON.stringify(body); if (!drawn.has(key)) drawn.set(key, expand(body)); return drawn.get(key); } : expand;
        for (const unit of Array.from({ length: times }, () => units).flat()) {
          if (stopped) break;
          if (!p.project || p.project.id !== pid) { tell("Stopped: another project was opened."); break; }
          const root = unitRoot(snap || p.project, unit);
          const group = (snap || p.project).scenes.filter((s) => genUnitId(s) === unit);
          if (!root || group.every((s) => s.excluded) || !isGenerative(root)) continue;        // removed, all left out, or not made by the model
          // The unit is made once at the length of its clips together (each cut adds one shared frame).
          const frames = group.reduce((t, s) => t + effFrames(s, snap || p.project), 0) - (group.length - 1);
          const { inputs, unwired, noPrompt, notes } = await buildInputs({ project: snap || p.project, scene: root, slots: frozen, expand: same, frames, hooks: app.inputHooks, prefix: (app.promptPrefix || []).flatMap((f) => { try { return f(root); } catch { return []; } }) });
          app.lastRun.typed = (root.text || "").trim();        // what a Chat comment made now would be about
          if (stopped) break;
          if (noPrompt && !made) tell("This pipeline has no prompt input, so the scene text is not sent.");
          notes.forEach(tell);
          if (unwired) tell(`${unwired} reference(s) did not fit this pipeline and are not used.`);
          const done = g.waitForTerminal();             // listening before the run starts, so a fast one is not missed
          said = false;
          if (!(await g.generate({ sceneId: root.id, projectId: pid, inputs, slots: frozen }))) {
            done.cancel();
            if (!said) tell((g.run.state.error && g.run.state.error.message) || "Could not queue the run. Is ComfyUI running, and is a run already going?");
            break;
          }
          if (stopped) g.cancel();                      // Stop landed while this one was being queued
          if (!made) app.api.newTasteGeneration().catch(() => {});        // a run really started: the last run's unrated clips can no longer be paired with what it learned
          made += 1;
          const end = await done;
          if (end === g.CANCELLED) break;
          const images = g.run.state.images;
          if (end !== g.DONE) { tell("Generation failed. The log has ComfyUI's message."); break; }
          if (!images.length) { tell("ComfyUI finished without a result. Try again."); break; }
          await record(pid, unit, images[images.length - 1], frames, g.run.state.promptId);       // the next shot may continue from this one: it must be on the clip first
        }
        if (!made && !stopped && !said) tell("Nothing to generate: every scene is left out or is a video clip.");
      } finally { busy = false; draw(); }
    }
    app.runner.units = (units, times) => runUnits(units, times);
    const pickedScenes = () => { const ids = new Set(app.selection.ids); const hit = p.scenes.filter((s) => ids.has(s.id)); return hit.length ? hit : p.selected ? [p.selected] : []; };       // every picked clip, else the one in focus
    const unitsOf = (scenes) => [...new Set(scenes.map(genUnitId))];

    const put = inPlace(host);
    let drawn = "";
    function draw() {
      const open = p.project;
      const picked = open ? pickedScenes().length : 0, key = `${busy}|${held}|${g.state().phase}|${Boolean(open)}|${picked}`;
      if (key === drawn) return;          // progress ticks must not rebuild the buttons
      drawn = key;
      const working = busy || held || g.state().phase === QUEUED || g.state().phase === RUNNING;     // also a run this page did not start
      const go = c.button.sm({ label: working ? "Generating…" : "▶ Generate", tone: "primary", disabled: working || !open,
        onClick: () => runUnits(unitsOf(p.scenes)) });
      const one = c.button.sm({ label: picked > 1 ? `Selected (${picked})` : "Selected", disabled: working || !picked, onClick: () => runUnits(unitsOf(pickedScenes())) });
      const stop = c.button.sm({ label: "■ Stop", tone: "danger", onClick: () => { stopped = true; g.cancel(); } });
      stop.node.hidden = !working;
      put(go.node, one.node, stop.node);
    }
    // A run found after a reload belongs to this page's last session: its result still goes on its clips.
    const claim = ({ sceneId, projectId }) => {
      const sc = p.project && p.project.id === projectId && p.project.scenes.find((s) => s.id === sceneId);
      if (!sc) return false;
      const unit = genUnitId(sc);
      const finish = (end) => {
        const im = g.run.state.images;
        if (end === g.DONE && im.length) record(projectId, unit, im[im.length - 1], undefined, g.run.state.promptId);
        else if (end === g.FAILED) tell((g.run.state.error && g.run.state.error.message) || "A run from before the reload failed.");
        else if (end !== g.CANCELLED) tell("A run from before the reload ended without a result that could be attached. Generate again.");
        busy = false; draw();
      };
      const now = g.run.state.phase;                 // a run that ended while the page reloaded is over already: nothing to wait for
      if (now === g.DONE || now === g.FAILED || now === g.CANCELLED) { finish(now); return true; }
      busy = true; draw();                           // Stop works on it, Generate waits
      g.waitForTerminal().then(finish);
      return true;
    };
    // The page may know the run before it has opened a project: wait for the project, then claim or give up.
    g.on("adopt", (a) => {
      if (claim(a)) return;
      const off = app.on(() => {                      // until its project opens, or the run is over
        if (!p.project) return;
        if (claim(a)) return off();
        const phase = g.run.state.phase;
        if (phase === g.DONE || phase === g.FAILED || phase === g.CANCELLED) {
          off();
          tell("A run from before the reload finished, but its project or scene is not open here, so its result was not attached.");
        }
      });
    });
    draw();
    const offRun = g.subscribe(draw);
    const offApp = app.on((what) => { if (what === "generate.scene" && !busy && p.selected) runUnits(unitsOf([p.selected]));
      else if (what === "generate.selected" && !busy && pickedScenes().length) runUnits(unitsOf(pickedScenes())); else draw(); });
    const offers = [offer(app, { id: "generate-all", label: "Generate", icon: "▶", run: () => { if (!busy && p.project) runUnits(unitsOf(p.scenes)); } }),
      offer(app, { id: "generate-selected", label: "Generate selected", icon: "▶", run: () => { if (!busy && pickedScenes().length) runUnits(unitsOf(pickedScenes())); } })];
    return () => { offRun(); offApp(); offers.forEach((f) => f()); };
  },
};
