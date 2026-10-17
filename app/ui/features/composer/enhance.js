// Enhance: a language model rewrites the finished prompt before it is encoded. These are the inputs of the pipeline's
// enhancer node, edited through the shared pipeline state; off, or on failure, the prompt runs as typed.
import { composer as c } from "../../composer/composer.js";

const NODE = "FunPackEnhancePrompt";
const DIALS = [["temperature", 0.7, 0, 2, 0.05, "Higher gives more varied wording, lower more predictable."], ["top_p", 0.92, 0, 1, 0.01, "Picks only among the likeliest words whose chances add up to this share."],
  ["top_k", 50, 1, 1000, 1, "Picks only among this many likeliest words."], ["min_p", 0.05, 0, 1, 0.01, "Drops words much less likely than the likeliest one; higher drops more."],
  ["repetition_penalty", 1.3, 1, 2, 0.05, "Above 1 discourages repeating words."], ["presence_penalty", 0, 0, 2, 0.05, "Above 0 discourages words already used."]];

export function enhance(app, own) {
  const ps = app.pipeline, page = c.region.stack({ gap: "sm" });
  let defaults = null, last = null, alive = true, encoders = { files: [], families: [] };
  own(() => { alive = false; });
  const slot = () => (ps.slots() || []).find((s) => s.node === NODE) || null;
  const set = async (name, value) => { const sl = slot(); if (sl) { await ps.save({ inputs: { [sl.id]: { [name]: value } } }); draw(); } };
  const field = (label, control, hint) => c.field.default({ label, control, hint });
  const num = (label, v, lo, hi, step, name, hint) => field(label, c.number.md({ label, value: v, min: lo, max: hi, step, onChange: (x) => set(name, x) }), hint);

  function readout() {
    if (!last) return [c.label.section({ text: "Last run" }), c.hint.default({ text: "Nothing yet. The rewrite of a run shows here." })];
    const part = (title, body) => (body ? c.collapsible.default({ label: title, body: c.code.block({ text: body, label: title }) }) : null);
    return [c.label.section({ text: "Last run" }), c.hint.default({ text: last.status || "" }), part("Before", last.before),
      c.field.default({ label: "After", control: c.text.sm({ text: last.after || "" }) }), part("Model's thinking", last.thinking), part("Sent to the model (chat run)", last.sent)].filter(Boolean);
  }
  const choices = (name) => (name === "encoder_file" ? encoders.files : encoders.families);
  const pick = (name, label, first, hint, dflt = "") => {
    const cur = (slot().inputs || {})[name] || dflt, list = choices(name);
    const opts = [...(first ? [{ value: "", label: first }] : []), ...[...list, ...(cur && !list.includes(cur) ? [cur] : [])].map((v) => ({ value: v, label: String(v) }))];
    return field(label, c.select.md({ label, value: cur, options: opts, onChange: (v) => set(name, v) }), hint);
  };
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
        pick("encoder_file", "Encoder", "The model's own encoder", "The text encoder that writes the rewrite. Models like H3's own encoder cannot write prose: pick a chat model such as Gemma 4 here."),
        ...((val("encoder_file", "") && choices("encoder_family").length) ? [pick("encoder_family", "Encoder family", "", "Which family that file is for.", "stable_diffusion")] : []),
        field("Instructions", c.textarea.md({ label: "Instructions", rows: 6, value: val("instructions", ""), placeholder: defaults ? defaults.instructions : "(built-in instructions)", onCommit: (x) => set("instructions", x) }), "Empty uses the built-in instructions."),
        num("Max tokens", val("max_length", 400), 32, 4096, 8, "max_length", "The longest the rewrite may be."),
        c.toggle.default({ label: "Use the picture", hint: "Sends the scene's picture to the model along with the prompt.", checked: val("use_image", true), onChange: (x) => set("use_image", x) }),
        c.toggle.default({ label: "Thinking", hint: "Let a thinking model reason first. Its reasoning eats the token budget.", checked: val("thinking", false), onChange: (x) => set("thinking", x) }),
        c.toggle.default({ label: "Greedy (always the likeliest word)", hint: "The same rewrite every time; the dials below are not used.", checked: greedy, onChange: (x) => set("greedy", x) }),
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
  app.api.enhancerEncoders().then((r) => { encoders = { files: r.files || [], families: r.families || [] }; draw(); }).catch(() => {});
  // Another project's pipeline, or an edit made elsewhere: redraw, but never under a field being typed in.
  const typing = () => app.dom.typingIn(page.node);
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
