// Step 3, Train: the settings, Start and Stop. The graph, previews and progress of a run are in the Results step.
import { composer as c } from "../../composer/composer.js";
import { stillOf } from "./state.js";

/** The node's own defaults. A set remembers what was changed from them. */
export const DEFAULTS = {
  name: "funpack_lora", use: "all", preview_every: 0, save_every: 0, steps: 500, learning_rate: 0.0001, rank: 16, algorithm: "LoRA", optimizer: "AdamW", loss: "MSE",
  batch_size: 1, grad_accumulation: 1, resolution: 512, aspect_buckets: false, frames: 33, fps: 0, start: "start", seed: 0,
  training_dtype: "bf16", lora_dtype: "bf16", gradient_checkpointing: true, offloading: false, bypass: false, swap_blocks: 0,
};
export const PRESETS = [
  { value: "check", label: "Quick check", steps: 100, rank: 8 },
  { value: "standard", label: "Standard", steps: 500, rank: 16 },
  { value: "long", label: "Thorough", steps: 2000, rank: 32 },
];
const opts = (list) => list.map((v) => ({ value: v, label: v }));

/** The settings for a run: the defaults, then what the set remembers. */
export const settingsOf = (doc) => ({ ...DEFAULTS, ...((doc && doc.settings) || {}), use: "all" });      // the Data step decides what is in; a set made before that remembers an old choice

const fmt = (n) => (Number.isFinite(n) ? (Math.abs(n) >= 0.01 ? n.toFixed(4) : n.toPrecision(3)) : "");
const clock = (s) => { s = Math.round(s || 0); return s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min` : s >= 60 ? `${Math.floor(s / 60)} min ${s % 60} s` : `${s} s`; };
export { fmt, clock };

export function trainStep({ app, session, go, ui }) {
  const doc = session.doc;
  if (!doc) return c.emptyState.default({ title: "No set open", hint: "Pick or make a training set in the Data step.", action: c.button.sm({ label: "Back to data", tone: "primary", onClick: () => go("data") }) });
  const v = settingsOf(doc);
  const set = (key) => (value) => { session.edit((d) => { d.settings = { ...(d.settings || {}), [key]: value }; }); };
  const num = (key, label, hint, o = {}) => c.field.default({ label, hint, control: c.number.md({ label, value: v[key], onChange: set(key), ...o }) });
  const pickOf = (key, label, hint, options) => c.field.default({ label, hint, control: c.select.md({ label, options, value: v[key], onChange: set(key) }) });
  const flag = (key, label, hint) => c.toggle.default({ label, hint, checked: v[key], onChange: set(key) });

  const items = doc.items;
  const used = items;
  const slots = (app.pipeline && app.pipeline.slots && app.pipeline.slots()) || [];
  const hasAudioVae = slots.some((s) => s.id === "audio_vae");
  const caption = used.filter((i) => i.caption.trim()).length;

  const name = c.field.default({ label: "Name of the LoRA", hint: "Saved in your loras folder. An earlier file is never overwritten.", control: c.input.md({ label: "Name", value: v.name, onCommit: set("name") }) });
  const preset = c.segmented.sm({ label: "Preset", options: PRESETS, value: (PRESETS.find((p) => p.steps === v.steps && p.rank === v.rank) || {}).value, onChange: (key) => {
    const p = PRESETS.find((x) => x.value === key);
    session.edit((d) => { d.settings = { ...(d.settings || {}), steps: p.steps, rank: p.rank }; });
    ui.redraw();
  } });

  const advanced = c.collapsible.default({ label: "More settings", open: ui.advanced, onToggle: (o) => { ui.advanced = o; }, body: c.region.stack({ gap: "sm", children: [
    pickOf("algorithm", "Adapter", "LoRA is the common one.", opts(["LoRA", "LoHa", "LoKr", "OFT"])),
    pickOf("optimizer", "Optimizer", "How the weights are updated.", opts(["AdamW", "Adam", "SGD", "RMSprop"])),
    pickOf("loss", "Loss", "How the error is measured.", opts(["MSE", "L1", "Huber", "SmoothL1"])),
    num("batch_size", "Batch size", "Samples per step. H3 always uses 1.", { min: 1, max: 64, step: 1 }),
    num("grad_accumulation", "Gradient accumulation", "Steps added up before the weights change: a bigger batch for less memory.", { min: 1, max: 256, step: 1 }),
    pickOf("training_dtype", "Training precision", "bf16 is right for nearly everything.", opts(["bf16", "fp32", "none"])),
    pickOf("lora_dtype", "LoRA precision", "How the file is stored.", opts(["bf16", "fp32"])),
    num("seed", "Seed", "The same seed repeats the same run.", { min: 0, step: 1 }),
    flag("gradient_checkpointing", "Save memory (gradient checkpointing)", "Slower, but needs far less memory."),
    num("swap_blocks", "Blocks kept in system memory", "Block swap: this many transformer blocks wait in system memory and come to the card one layer at a time. Slower; each block frees about 0.75 GiB (H3 has 50). 0 keeps the whole model on the card.", { min: 0, max: 128, step: 1 }),
    flag("offloading", "Keep weights in system memory (ComfyUI's own)", "Slower. It brought ComfyUI down with H3 on a rental: prefer Blocks kept in system memory."),
    flag("bypass", "Bypass mode", "Applies the LoRA by hooks, for quantized models."),
  ] }) });

  const run = ui.run;
  const live = c.region.stack({ gap: "md", children: [notice(ui, go)].filter(Boolean) });      // redrawn on its own while a run goes on: the settings above stay as they are
  ui.monitorBox = live;
  const waiting = run.state === "paused" || run.state === "previewing";       // between steps: nothing is training, the run is still open
  const busy = run.active || run.state === "previewing";
  const start = c.button.lg({ label: busy ? "Training…" : "Start training", tone: "primary", disabled: busy || waiting || !used.length, onClick: () => ui.start() });
  start.node.hidden = waiting;
  const stop = c.button.lg({ label: "Stop", tone: "danger", onClick: () => ui.stop() });
  stop.node.hidden = !busy && !waiting;
  const pause = c.button.lg({ label: run.state === "previewing" ? "Stay paused" : "Pause", tone: "neutral", disabled: ui.holdPause && run.state !== "paused", onClick: () => ui.pause() });
  pause.node.hidden = !(run.state === "training" || run.state === "previewing");
  const resume = c.button.lg({ label: "Resume", tone: "primary", onClick: () => ui.resume() });
  resume.node.hidden = run.state !== "paused";
  const previewNow = c.button.lg({ label: "Preview now", tone: "neutral", title: "Make a picture or clip with the LoRA so far", onClick: () => ui.previewNow() });
  previewNow.node.hidden = !(run.state === "paused" && run.paused && run.paused.preview);

  return c.region.stack({ gap: "md", children: [
    c.hint.default({ text: used.length ? `${used.length} item${used.length > 1 ? "s" : ""} will be learned from (${caption} with tags). ${used.length - caption ? "Items without a caption are learned from with only the prefix and postfix." : ""}`.trim()
      : `"${session.name}" is empty. Add media in the Data step.` }),
    hasAudioVae ? c.banner.info({ text: "This pipeline has an audio VAE, so it trains picture and sound together (H3). One clip at a time, so training is slow and needs a lot of memory. musubi-tuner's H3 recipe: learning rate 0.0001, rank 16, batch 1, bf16, gradient checkpointing, plus block swap. Its plain flow-matching loss undoes the few-step quality of a distilled (turbo) model; musubi adds a training adapter, guidance loss or teacher matching for that, which this trainer does not have." }) : null,
    c.group.default({ label: "Basics", rows: [name, preset,
      num("steps", "Steps", "More steps learn more and can overdo it.", { min: 1, max: 100000, step: 10 }),
      num("learning_rate", "Learning rate", "How big each step is. Lower is safer and slower.", { min: 0.0000001, max: 1, step: 0.0001, precision: 7 }),
      num("rank", "Rank", "How much it can learn. Higher holds more detail and makes a bigger file.", { min: 1, max: 128, step: 1 }),
      num("save_every", "Save a copy every (steps)", "0 saves only the finished LoRA. Otherwise a copy of the LoRA so far goes to your loras folder every this many steps, named like yourname_step200, so you can pick the best moment.", { min: 0, max: 100000, step: 10 }),
      num("preview_every", "Preview every (steps)", "0 never. Otherwise training waits every this many steps while your pipeline makes a picture or clip with the LoRA so far (the picked scene, the project's own settings), then carries on. Needs a LoRA loader in the pipeline; each wait loads the models again.", { min: 0, max: 100000, step: 10 })] }),
    c.group.default({ label: "Pictures and clips", rows: [
      num("resolution", "Size", "Pixels per side. Everything is fitted to about this area.", { min: 64, max: 2048, step: 32 }),
      flag("aspect_buckets", "Keep each shape", "A few size groups instead of one shape for all. Not used for H3."),
      num("frames", "Frames per clip", "For video models. H3 rounds to its own grid (17 x n + 5).", { min: 1, max: 1025, step: 1 }),
      num("fps", "Frames per second", "0 uses each clip's own rate (H3: 24).", { min: 0, max: 120, step: 0.5 }),
      pickOf("start", "Take frames from", "", [{ value: "start", label: "The start of the clip" }, { value: "middle", label: "The middle" }])] }),
    advanced,
    c.toolbar.default({ items: [c.button.md({ label: "Back", tone: "ghost", onClick: () => go("tags") })], trailing: [stop, pause, previewNow, resume, start] }),
    live,
  ].filter(Boolean) });
}

/** What the Train step says about a run: where to look, and the error if it failed. */
export function notice(ui, go) {
  const run = ui.run;
  if (!run.shown) return null;
  const where = c.button.sm({ label: "Open Results", tone: "ghost", onClick: () => go("results") });
  if (run.error) return c.region.stack({ gap: "sm", children: [c.inlineError.default({ text: run.error }), where] });
  const text = { done: "Training finished.", stopped: "Training stopped.", paused: `Training paused at step ${run.step || 0}.`, previewing: "Making a preview." }[run.state] || "Training started.";
  return c.region.stack({ gap: "sm", children: [c.banner.info({ text: `${text} Check graphs and previews in the Results tab.` }), where] });
}

/** Progress, the message, and the loss graph. */
export function monitor(ui) {
  const run = ui.run;
  if (!run.shown) return null;
  const chart = c.chart.line({ series: [{ name: "loss", values: run.loss, tone: "accent" }, ...(run.smooth.length > 1 ? [{ name: "smoothed", values: run.smooth, tone: "good" }] : [])],
    xLabel: "step", yLabel: "loss", height: "md", empty: "The first steps are on their way" });
  const done = run.step || 0, total = run.total || 1;
  const lines = [
    run.state === "training" || run.state === "queued" ? `Step ${done} of ${total}${run.elapsed ? ` · ${clock(run.elapsed)}` : ""}${done && run.elapsed ? ` · about ${clock((run.elapsed / done) * (total - done))} left` : ""}` : null,
    run.message || null,
  ].filter(Boolean);
  const previews = ui.previews && ui.previews.length ? c.gallery.adaptive({ id: "training-previews", cols: 3, selection: [],
    items: ui.previews.map((p) => ({ id: p.id, label: `Step ${p.step}`, thumb: stillOf(p), badge: p.media_kind })),
    onActivate: (item) => ui.peek(ui.previews.find((p) => p.id === item.id)) }) : null;
  const heading = { done: "Finished", error: "Stopped with an error", stopped: "Stopped", paused: `Paused at step ${done}`, previewing: `Making a preview at step ${done}` }[run.state] || "Training";
  return c.group.default({ label: heading, rows: [c.region.stack({ gap: "md", children: [
    run.state === "previewing" ? c.progress.indeterminate({ label: "Making a preview" }) : null,
    run.state === "training" ? c.progress.bar({ value: done, max: total, label: "Training progress" }) : run.state === "queued" ? c.progress.indeterminate({ label: "Waiting for ComfyUI" }) : null,
    lines.length ? c.hint.default({ text: lines.join(" · ") }) : null,
    run.error ? c.inlineError.default({ text: run.error }) : null,
    run.lora ? c.banner.info({ text: `Saved as ${run.lora}. Pick it in a LoRA loader to try it.` }) : null,
    previews,
    chart,
    run.loss.length ? c.hint.default({ text: `Latest loss ${fmt(run.loss[run.loss.length - 1])}. It should fall at first, then wobble; a LoRA is rarely better after the wobble stops shrinking.` }) : null,
  ].filter(Boolean) })] });
}
