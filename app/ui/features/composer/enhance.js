// Enhance: a language model rewrites the finished prompt before it is encoded. These are the inputs of the pipeline's
// enhancer node, edited through the shared pipeline state; off, or on failure, the prompt runs as typed.
import { composer as c } from "../../composer/composer.js";

const NODE = "FunPackEnhancePrompt";
const DIALS = [["temperature", 0.7, 0, 2, 0.05], ["top_p", 0.92, 0, 1, 0.01], ["top_k", 50, 1, 1000, 1], ["min_p", 0.05, 0, 1, 0.01], ["repetition_penalty", 1.3, 1, 2, 0.05], ["presence_penalty", 0, 0, 2, 0.05]];

export function enhance(app, own) {
  const ps = app.pipeline, page = c.region.stack({ gap: "sm" });
  let defaults = null, last = null, alive = true;
  own(() => { alive = false; });
  const slot = () => (ps.slots() || []).find((s) => s.node === NODE) || null;
  const set = async (name, value) => { const sl = slot(); if (sl) { await ps.save({ inputs: { [sl.id]: { [name]: value } } }); draw(); } };
  const field = (label, control, hint) => c.field.default({ label, control, hint });
  const num = (label, v, lo, hi, step, name) => field(label, c.number.md({ label, value: v, min: lo, max: hi, step, onChange: (x) => set(name, x) }));

  function readout() {
    if (!last) return [c.label.section({ text: "Last run" }), c.hint.default({ text: "Nothing yet. The rewrite of a run shows here." })];
    const part = (title, body) => (body ? c.collapsible.default({ label: title, body: c.code.block({ text: body, label: title }) }) : null);
    return [c.label.section({ text: "Last run" }), c.hint.default({ text: last.status || "" }), part("Before", last.before),
      c.field.default({ label: "After", control: c.text.sm({ text: last.after || "" }) }), part("Model's thinking", last.thinking), part("Sent to the model (chat run)", last.sent)].filter(Boolean);
  }
  function draw() {
    if (!alive) return;
    const sl = slot();
    if (!sl) return page.set([c.emptyState.default({ icon: "✎", title: "No prompt enhancer here", hint: "Pick a preset that has one, or add the “FunPack Enhance Prompt” node in Settings ▸ Models & Pipeline." })]);
    const v = sl.inputs || {}, val = (n, d) => (v[n] === undefined ? d : v[n]);
    const on = val("enabled", false), greedy = val("greedy", false);
    page.set([
      c.label.section({ text: "Prompt enhancer" }),
      c.hint.default({ text: "A language model rewrites the finished prompt before it is encoded. Off, or if it fails, your prompt is used as typed. (word:1.5) and [a@2-4] markup does not survive a rewrite." }),
      c.toggle.default({ label: "Enhance prompt", checked: on, onChange: (x) => set("enabled", x) }),
      ...(!on ? readout() : [
        field("Instructions", c.textarea.md({ label: "Instructions", rows: 6, value: val("instructions", ""), placeholder: defaults ? defaults.instructions : "(built-in instructions)", onCommit: (x) => set("instructions", x) }), "Empty uses the built-in instructions."),
        num("Max tokens", val("max_length", 400), 32, 4096, 8, "max_length"),
        c.toggle.default({ label: "Use the picture", hint: "Sends the scene's picture to the model along with the prompt.", checked: val("use_image", true), onChange: (x) => set("use_image", x) }),
        c.toggle.default({ label: "Thinking", hint: "Let a thinking model reason first. Its reasoning eats the token budget.", checked: val("thinking", false), onChange: (x) => set("thinking", x) }),
        c.toggle.default({ label: "Greedy (always the likeliest word)", checked: greedy, onChange: (x) => set("greedy", x) }),
        ...(greedy ? [] : DIALS.map(([n, d, lo, hi, st]) => num(n, val(n, d), lo, hi, st, n))),
        num("Seed", val("seed", 0), 0, 2 ** 53, 1, "seed"),
        c.hint.default({ text: "Seed 0 = a fresh rewrite every Generate; any other number repeats the same rewrite." }),
        c.label.section({ text: "Reference" }),
        field("Lorebook / shortcut files", c.textarea.md({ label: "Lorebook files", rows: 2, value: val("lorebooks", ""), placeholder: "one path per line", onCommit: (x) => set("lorebooks", x) }), "SillyTavern lorebooks or FunPack shortcut files, on the machine running ComfyUI."),
        field("Reference intro", c.textarea.md({ label: "Reference intro", rows: 2, value: val("reference_intro", ""), placeholder: defaults ? defaults.reference_intro : "", onCommit: (x) => set("reference_intro", x) })),
        ...readout()]),
    ]);
  }
  draw();
  ps.ensureLoaded().then(draw);
  // Another project's pipeline, or an edit made elsewhere: redraw, but never under a field being typed in.
  const typing = () => page.node.contains(document.activeElement);
  own(ps.subscribe(() => { if (!typing()) draw(); }));
  app.api.enhancerDefaults().then((d) => { defaults = d; draw(); }).catch(() => {});
  const fetchRuns = () => app.api.enhancerRuns().then((r) => { last = (r.runs || []).slice(-1)[0] || null; if (!typing()) draw(); }).catch(() => {});
  fetchRuns();
  // a run that ends while this tab is open shows its prompt here without reopening it
  let phase = null;
  if (app.generate && app.generate.subscribe) own(app.generate.subscribe((st) => {
    const was = phase; phase = st && st.phase;
    if (was && was !== phase && (phase === "done" || phase === "failed")) fetchRuns();
  }));
  return page;
}
