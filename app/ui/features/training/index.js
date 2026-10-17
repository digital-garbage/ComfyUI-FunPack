// "Training": make a LoRA from your own media and rated results. An add-on (Settings > Modules); the window is absent until it is on.
// Four steps in one window: Data (optional), Tags, Train, Results. A training set is saved, so each step can be left and come back to.
import { composer as c } from "../../composer/composer.js";
import { dataStep } from "./data.js";
import { applyTagging, tagsStep } from "./tags.js";
import { monitor, notice, settingsOf, trainStep } from "./train.js";
import { resultsStep } from "./results.js";
import { makeSession, srcKey } from "./state.js";

const VIDEO = /\.(mp4|webm|mov|mkv)$/i;

const TABS = [{ value: "data", label: "1  Data" }, { value: "tags", label: "2  Tags" }, { value: "train", label: "3  Train" }, { value: "results", label: "Results" }];
const POLL = 1000;
const QUEUE_WAIT = 60000;

const smooth = (values) => {                       // a gentle moving average, so the trend shows through a noisy loss
  const w = Math.max(3, Math.round(values.length / 40)), out = [];
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) { sum += values[i]; if (i >= w) sum -= values[i - w]; out.push(sum / Math.min(i + 1, w)); }
  return out;
};

export default {
  id: "training",
  mount: "timeline.status",
  needs: ["api", "dom", "pipeline", "project", "generate"],
  setup({ host, app }) {
    const api = app.api;
    const session = makeSession(api);
    const ui = {
      tab: "data", openBin: false, openRated: false, clips: {}, picked: new Set(), ratedPicked: new Set(), bin: [], rated: { results: [], counts: {} }, ratedKind: "liked", include: ["main"], selected: new Set(), filter: "all", tag: "", limit: 48,
      tagger: { id: "wd_vit_v3", list: [], job: null, general: 0.35, limit: 0, characters: false, mode: "add", scope: "untagged" },
      advanced: false, runs: [], chosenRun: "", record: null,
      run: { shown: false, active: false, state: "idle", loss: [], smooth: [], step: 0, total: 0, elapsed: 0, message: "", error: "", lora: "", before: "", runId: "", paused: null },
      previews: [], holdPause: false, awaitStart: false, peek: (item) => peek(item),
      pause: () => pause(), resume: () => launch(ui.run.runId), restore: (id) => { ui.run.loss = []; ui.run.runId = id; ui.record = null; return launch(id); }, previewNow: () => previewNow(),
      monitorBox: null, redraw: () => draw(), pollTaggers: () => pollTaggers(true), loadRecord: (id) => loadRecord(id), loadRuns: () => loadRuns(),
      start: () => start(), stop: () => stop(),
      /** Ask for the frame counts of clips not known yet; `then` runs when new numbers arrived. */
      async needClips(srcs, then) {
        const fresh = srcs.filter((x) => !(srcKey(x) in ui.clips));
        if (!fresh.length) return;
        fresh.forEach((x) => { ui.clips[srcKey(x)] = null; });
        try { Object.assign(ui.clips, (await api.trainClips(fresh)).clips || {}); } catch { fresh.forEach((x) => { delete ui.clips[srcKey(x)]; }); return; }
        if (then) then();
      },
    };
    let win = null, body = null, timers = [], offSession = null, tagPoll = null, runPoll = null;

    const button = c.button.sm({ label: "🎓 Training", tone: "neutral", title: "Make a LoRA from your media and rated results", onClick: () => (win ? win.close() : open()) });
    button.node.hidden = true;
    host.append(button.node);
    api.trainState().then((s) => { button.node.hidden = !(s && s.enabled); }).catch(() => {});

    const peek = (item) => {
      const s = item.src;
      const src = s.kind === "bin" ? `/funpack/api/media/${encodeURIComponent(s.id)}/file` : `/view?filename=${encodeURIComponent(s.filename)}&subfolder=${encodeURIComponent(s.subfolder || "")}&type=${encodeURIComponent(s.type || "output")}`;
      const viewer = c.viewer.media({ src, kind: item.media_kind === "video" ? "video" : "image" });
      const m = c.modal.generic({ title: item.name || "Preview", body: viewer, onClose: () => viewer.destroy && viewer.destroy() });
      return m;
    };

    // --- the steps
    const steps = {
      data: () => dataStep({ app, session, go, ui }),
      tags: () => tagsStep({ app, session, go, ui, peek }),
      train: () => trainStep({ app, session, go, ui }),
      results: () => resultsStep({ app, ui }),
    };
    function go(tab) { ui.tab = tab; if (tab === "results") loadRuns(); draw(); }
    function draw() {
      if (!win) return;
      const tabs = c.tabs.underline({ label: "Training steps", tabs: TABS, value: ui.tab, onChange: go });
      const note = session.error ? c.inlineError.default({ text: session.error }) : null;
      body.set([tabs, note, steps[ui.tab]()].filter(Boolean));
    }

    // --- the Bin and rated results, read when the window opens
    async function loadSources() {
      try { ui.bin = (await api.media()).media || []; } catch { ui.bin = []; }
      try { ui.rated = await api.trainRated(ui.include); } catch { ui.rated = { results: [], counts: {} }; }
      draw();
    }
    async function loadRuns() { try { ui.runs = await api.trainRuns(); } catch { ui.runs = []; } if (ui.runs.length) loadRecord(ui.chosenRun || ui.runs[0].run); }
    async function loadRecord(id) { try { ui.record = await api.trainRunRecord(id); } catch { ui.record = null; } if (ui.tab === "results") draw(); }

    // --- tagger downloads and the tagging job
    async function pollTaggers(fast) {
      clearTimeout(tagPoll);
      let again = false;
      try {
        const d = await api.trainTaggers();
        ui.tagger.list = d.taggers || [];
        const was = ui.tagger.job && ui.tagger.job.state;
        const job = await api.trainAutotagState();
        ui.tagger.job = job;
        if (was === "running" && job.state === "done") {
          const n = applyTagging(session, job, ui.tagger.mode);
          c.toast.info({ text: `Tagged ${n} item${n === 1 ? "" : "s"}.` });
          await session.flush();
        } else if (was === "running" && job.state === "error") c.toast.warn({ text: job.error });
        again = job.state === "running" || ui.tagger.list.some((t) => t.state === "downloading");
      } catch { /* the page shows what it had */ }
      if (win && ui.tab === "tags") draw();
      if (win && (again || fast === true)) tagPoll = setTimeout(() => pollTaggers(again ? undefined : false), again ? POLL : POLL * 2);
    }

    // --- a run
    const setRun = (patch) => Object.assign(ui.run, patch);
    const start = () => launch("");
    async function launch(resumeId) {
      const doc = session.doc;
      if (!doc) return;
      await session.flush();
      const settings = settingsOf(doc);
      settings.trigger = doc.trigger || "";
      settings.postfix = doc.postfix || "";
      if (resumeId) settings.resume = resumeId;
      else { ui.previews = []; settings.resume = ""; }
      ui.holdPause = false;
      setRun(resumeId ? { shown: true, active: true, state: "queued", message: "Carrying on. The models load again first.", error: "", paused: null }
        : { shown: true, active: true, state: "queued", loss: [], smooth: [], step: 0, total: settings.steps, elapsed: 0, message: "Checking the pipeline", error: "", lora: "", runId: "", paused: null });
      draw();
      let before = "";
      ui.staleStarted = 0;
      try {
        const last = await api.trainRun(0);
        if (resumeId) ui.staleStarted = last.started || 0;      // a failed or stopped run being restored still shows as the current record until the new one begins
        else before = last.run || "";
      } catch { /* first run */ }
      try {
        const plan = await api.trainPlan(session.name, app.pipeline.slots(), settings);
        if (plan.problems && plan.problems.length) throw new Error(plan.problems.join("; "));
        setRun({ message: resumeId ? "Sent to ComfyUI. It loads the models again, which can take a while." : "Sent to ComfyUI. It loads the models first, which can take a while." });
        draw();
        await api.queueGraph(plan.prompt);
      } catch (err) { setRun({ active: false, state: resumeId ? "paused" : "error", error: err.message, message: "" }); draw(); return; }
      ui.before = before;
      ui.awaitStart = Boolean(resumeId);              // until the new run begins, the record still shows the pause it is carrying on from
      watch();
    }
    function refreshLive() {                          // the graph lives in Results, a short notice in Train
      if (!ui.monitorBox) return;
      if (ui.tab === "results") ui.monitorBox.set([monitor(ui)].filter(Boolean));
      else if (ui.tab === "train") ui.monitorBox.set([notice(ui, go)].filter(Boolean));
    }
    function watch() {
      clearTimeout(runPoll);
      const queuedAt = Date.now();
      const tick = async () => {
        if (!ui.run.active) return;
        if (ui.run.state === "queued" && Date.now() - queuedAt > QUEUE_WAIT && !ui.run.message.includes("log")) {
          setRun({ message: "ComfyUI has not started the run yet. It may be busy with another job or loading models; if this stays, look at ComfyUI's log." });
          refreshLive();
        }
        let snap = null;
        try { snap = await api.trainRun(ui.run.loss.length); } catch { /* try again */ }
        if (snap && snap.run && snap.run !== ui.before && !(ui.awaitStart && (snap.state === "paused" || (ui.staleStarted && snap.started === ui.staleStarted)))) {
          ui.awaitStart = false;
          const loss = snap.loss_from === ui.run.loss.length ? ui.run.loss.concat(snap.loss) : ui.run.loss;
          setRun({ state: snap.state, step: snap.step, total: snap.total || ui.run.total, elapsed: snap.elapsed, message: snap.message, loss, smooth: smooth(loss), error: snap.error || "", lora: snap.lora || "", runId: snap.run, paused: snap.paused || null });
          if (snap.state === "paused") {
            setRun({ active: false });
            draw();
            if (snap.paused && snap.paused.reason === "preview" && !ui.holdPause) previewNow(true);
            return;
          }
          if (["done", "error", "stopped"].includes(snap.state)) {
            setRun({ active: false });
            if (snap.state === "done") c.toast.info({ text: `Done: ${snap.lora}` });
            draw();
            loadRuns();
            loadRecord(snap.run);                    // a failure offers Restore and continue again
            return;
          }
          refreshLive();
        }
        runPoll = setTimeout(tick, POLL);
      };
      runPoll = setTimeout(tick, POLL);
    }
    async function stop() {
      const waiting = ui.run.state === "paused" || ui.run.state === "previewing";
      setRun({ message: "Stopping" });
      if (ui.run.state === "previewing") { ui.stopAfterPreview = true; try { app.generate.cancel(); } catch { /* it was over */ } }
      try { await api.trainStop(); } catch { /* it ends anyway */ }
      if (!waiting) { try { await fetch("/interrupt", { method: "POST" }); } catch { /* the training step check ends it */ } }
      if (ui.run.state === "queued" || waiting) { setRun({ active: false, state: "stopped", message: "" }); draw(); }
    }
    async function pause() {
      ui.holdPause = true;                            // a pause the person asked for stays: no preview step then carries on by itself
      setRun({ message: ui.run.state === "previewing" ? "It will wait after this preview" : "Pausing at the end of this step" });
      if (ui.run.state === "training") { try { await api.trainPause(); } catch { /* the run goes on */ } }
      refreshLive();
    }
    /** A picture or clip made by the pipeline with the LoRA so far; when `auto`, training carries on afterwards. */
    async function previewNow(auto) {
      const paused = ui.run.paused;
      if (!paused || !paused.preview) return;
      ui.stopAfterPreview = false;
      setRun({ state: "previewing", message: `Making a preview with the LoRA at step ${paused.step}` });
      draw();
      let out;
      try { out = app.runner && app.runner.preview ? await app.runner.preview({ lora: paused.preview }) : { error: "Previews need the app's Generate." }; } catch (err) { out = { error: err.message }; }
      if (out.image) {
        const im = out.image;
        ui.previews.push({ id: `p${ui.previews.length + 1}`, step: paused.step, src: { kind: "output", type: im.type || "output", subfolder: im.subfolder || "", filename: im.filename }, media_kind: VIDEO.test(im.filename) ? "video" : "image" });
      } else if (!ui.stopAfterPreview) c.toast.warn({ text: out.error || "The preview could not be made." });
      if (ui.stopAfterPreview) { draw(); return; }
      setRun({ state: "paused", message: "" });
      if (auto && !ui.holdPause) launch(ui.run.runId); else draw();
    }
    // A run already going when the window opens (page reloaded): pick it up.
    async function resume() {
      try {
        const snap = await api.trainRun(0);
        if (snap.state === "paused") {
          ui.before = "";
          setRun({ shown: true, active: false, state: "paused", loss: snap.loss, smooth: smooth(snap.loss), step: snap.step, total: snap.total, message: "", runId: snap.run, paused: snap.paused });
          ui.holdPause = true;                        // found after a reload: the person decides
          ui.tab = "train";
        } else if (snap.state === "training") {
          ui.before = "";
          setRun({ shown: true, active: true, state: "training", runId: snap.run, loss: snap.loss, smooth: smooth(snap.loss), step: snap.step, total: snap.total, elapsed: snap.elapsed, message: snap.message });
          ui.tab = "train";
          watch();
        }
      } catch { /* nothing to resume */ }
    }

    function open() {
      if (win) return;
      body = c.region.stack({ gap: "md", fill: true });
      win = c.floating.window({ id: "training", title: "Training", subtitle: "make a LoRA", body, width: 780, height: 720, x: 240, y: 40,
        onClose: () => { win = null; session.flush(); clearTimeout(tagPoll); } });
      offSession = session.on(draw);
      session.refresh().then(async () => {
        if (!session.name) {
          const remembered = (() => { try { return localStorage.getItem("fp.training.set") || ""; } catch { return ""; } })();
          const pick = session.sets.find((x) => x.name === remembered) || session.sets[0];
          if (pick) await session.open(pick.name);
        }
        draw();
      });
      loadSources();
      pollTaggers();
      resume();
      draw();
    }
    const offOpen = session.on(() => { try { if (session.name) localStorage.setItem("fp.training.set", session.name); } catch { /* private mode */ } });

    return () => { clearTimeout(tagPoll); clearTimeout(runPoll); timers.forEach(clearTimeout); if (offSession) offSession(); offOpen(); if (win) win.close(); };
  },
};
