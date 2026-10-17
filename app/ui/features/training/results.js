// Step 4, Results: the runs that finished, with their loss graphs.
import { composer as c } from "../../composer/composer.js";
import { clock, fmt, monitor } from "./train.js";

const STATES = { done: "Finished", error: "Failed", stopped: "Stopped", training: "Running" };

export function resultsStep({ app, ui }) {
  const runs = ui.runs.filter((r) => !(ui.run.active && r.run === ui.run.runId));      // the run going on is shown above, not a second time from its old record
  const live = c.region.stack({ gap: "md", children: [monitor(ui)].filter(Boolean) });      // the run going on now, redrawn on its own
  ui.monitorBox = live;
  if (!runs.length) return ui.run.shown ? live : c.emptyState.default({ title: "No runs yet", hint: "A run you start in the Train step is kept here, with its loss graph." });
  const chosen = runs.find((r) => r.run === ui.chosenRun) || runs[0];
  const rows = runs.map((r) => ({ id: r.run, label: `${r.name || "LoRA"}  ·  ${STATES[r.state] || r.state}`, hint: `${r.set} · ${r.steps || 0} steps${r.final_loss != null ? ` · loss ${fmt(r.final_loss)}` : ""}` }));
  const list = c.gallery.rows({ items: rows, selection: [chosen.run], onActivate: (item) => { ui.chosenRun = item.id; ui.loadRecord(item.id); ui.redraw(); } });
  const rec = ui.record && ui.record.run === chosen.run ? ui.record : null;
  const chart = c.chart.line({ series: rec && rec.loss ? [{ name: "loss", values: rec.loss, tone: "accent" }] : [], xLabel: "step", yLabel: "loss", height: "md", empty: rec ? "No loss was recorded" : "Loading" });
  const facts = rec ? [
    `${STATES[rec.state] || rec.state}`, `${rec.steps || 0} of ${rec.total || 0} steps`, rec.seconds ? clock(rec.seconds) : null,
    rec.final_loss != null ? `final loss ${fmt(rec.final_loss)}` : null, rec.set ? `from "${rec.set}"` : null,
  ].filter(Boolean).join(" · ") : "";
  const s = (rec && rec.settings) || {};
  const settings = rec ? ["algorithm", "rank", "learning_rate", "optimizer", "loss", "resolution", "batch_size", "use"].filter((k) => s[k] !== undefined).map((k) => `${k.replace("_", " ")} ${s[k]}`).join(" · ") : "";
  return c.region.stack({ gap: "lg", children: [
    live,
    list,
    c.group.default({ label: chosen.name || "LoRA", rows: [c.region.stack({ gap: "lg", children: [
      c.hint.default({ text: facts }),
      rec && rec.lora ? c.banner.info({ text: `File: ${rec.lora}. Pick it in a LoRA loader to use it.` }) : null,
      rec && rec.error ? c.inlineError.default({ text: rec.error }) : null,
      rec && rec.resumable && ["error", "stopped"].includes(rec.state) ? c.banner.info({ text: `The state at step ${rec.steps || 0} or earlier is kept. Restore and continue picks the training up from it with the Train step's current settings.` }) : null,
      chart, c.hint.default({ text: settings }),
      c.toolbar.default({ items: [...(rec && rec.lora ? [download(app, rec)] : []), ...(rec && rec.resumable && !ui.run.active && !["training", "queued", "previewing", "paused"].includes(ui.run.state) ? [c.button.sm({ label: "Restore and continue", tone: "primary", title: "Carry on from the saved state", onClick: async () => { await ui.restore(chosen.run); } })] : [])], trailing: [c.button.sm({ label: "Delete this record", tone: "ghost", title: "Removes the record only, not the LoRA file", onClick: async () => { await app.api.trainForgetRun(chosen.run); ui.chosenRun = ""; await ui.loadRuns(); ui.redraw(); } })] }),
    ].filter(Boolean) })] }),
  ] });
}

/** Saves the finished LoRA through the browser, to wherever the person keeps files: an instance's own disk is gone when it is. */
function download(app, rec) {
  return c.button.sm({ label: "Download LoRA", tone: "primary", title: `Save ${rec.lora} to this computer`, onClick: async () => {
    try {
      const res = await fetch(app.api.trainLoraUrl(rec.run));
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).why || `The server refused (${res.status}).`);
      app.dom.saveBlob(await res.blob(), rec.lora);
    } catch (err) { c.toast.warn({ text: err.message }); }
  } });
}
