// Engine settings: Studio, Chain Sampler, continuity (moved out of Project inspector).
// A section of the unified Settings window with its OWN inner sidebar — categories are
// always visible (Overview · Studio: Refinement/Adjustments/Sampler · Chain Sampler:
// Continuity/Timing/Guidance/Decode/Experimental), no long scrolling card list.
(function () {
  const { el, clear } = window.dom;
  const S = window.Store;
  const API = window.MovieEditorAPI;

  // Project ids whose legacy Distilled-Flow ALG switch has been folded into the
  // sampler-wide alg_anchor this session. Module scope, not render scope: every store
  // write re-renders this pane synchronously, so the guard has to outlive a render or
  // the migration re-enters itself and recurses until the stack blows.
  const _ALG_MIGRATED = new Set();

  // Simple mode has no rating UI, so every setting that is a no-op without a trained
  // refinement key / rated history is hidden there rather than just made harder to find.
  // Read per render, not once: the mode switch flips it live.
  const EASY = () => !!window.FunPackMode?.isSimple();
  const RATING_GATED_KNOBS = new Set([
    "embed_guidance", "embed_guidance_source", "embed_guidance_strength",
    "score_slider", "score_slider_strength", "taste_nearest_prompt",
    "output_guidance", "output_guidance_strength",
    "trajectory_guidance", "trajectory_guidance_strength",
    "dynashift", "dynashift_strength", "dynashift_threshold",
    "h3_repr_steering", "h3_repr_steering_strength", "h3_repr_steering_block",
  ]);
  const RATING_GATED_STUDIO = new Set(["reference_injection", "value_guidance", "steer_mode", "absolute_strength"]);

  let _mounted = null; // { scroller (pane, set per render), content (shell root) }
  let unsub = null;
  let _editing = false;
  let view = "overview";
  function setView(v) { view = v; render(); }

  // macOS-style row: title on the left, control on the right. Append into a .sw-rows group.
  // `hint` is ONE short sentence saying what the setting does — that is all most rows ever
  // show. Everything else (cost figures, sampler caveats, failure modes) goes in `detail`,
  // which stays collapsed behind a Details link: nobody reads a paragraph per checkbox, but
  // the paragraph is still the only place some of those numbers exist.
  function field(labelText, control, hint, detail) {
    const row = el("div", "sw-row eng-field");
    const main = el("div", "sw-row-main");
    main.append(el("div", "sw-row-title", labelText));
    if (hint) main.append(el("div", "sw-row-hint", hint));
    if (detail) {
      const body = el("div", "sw-row-hint eng-detail", detail);
      body.hidden = true;
      const more = el("button", "eng-more", "Details");
      more.type = "button";
      more.onclick = () => {
        body.hidden = !body.hidden;
        more.textContent = body.hidden ? "Details" : "Hide details";
      };
      main.append(more, body);
    }
    row.append(main, control);
    return row;
  }

  function toggleField(labelText, checkbox, hint) {
    checkbox.style.width = "auto";
    return field(labelText, checkbox, hint);
  }

  function group(parent, label) {
    if (label) parent.append(el("div", "sw-rows-label", label));
    const g = el("div", "sw-rows");
    parent.append(g);
    return g;
  }

  function hintEl(text) { return el("div", "sw-hint", text); }

  function slotLabelFor(st, slotId) {
    if (!slotId || slotId === "funpack") return null;
    const slot = (st.models?.slots || []).find((s) => s.id === slotId);
    return slot ? (slot.label || slot.node_class || slot.id) : slotId;
  }

  function activeSceneCount(p) {
    return (p.scenes || []).filter((s) => !s.excluded).length;
  }

  // ── FunPack Studio: refiner fields ─────────────────────────────────────────
  const STUDIO_REFINER_ESSENTIALS = [
    { name: "vision_conditioning", label: "Vision conditioning", default: true,
      hint: "Lets Studio look at your anchor image and write what it sees into the prompt. Turn it off if the prompt should stand on its own." },
    { name: "reference_injection", label: "Reference injection", default: false,
      hint: "Pushes the reference image's own attention into the identity blocks while sampling, so the face holds harder. Only does something on i2v scenes that have a source image." },
  ];
  const STUDIO_REFINER_ADVANCED = [
    { name: "value_guidance", label: "Value guidance", kind: "bool", default: true,
      hint: "Moves the prompt toward what your ratings say you like, before sampling starts. Learning happens either way — this only decides whether it gets applied." },
    { name: "steer_mode", label: "Steer mode", kind: "combo", choices: ["relative", "absolute", "both"], default: "relative",
      hint: "'relative' finds the best conditioning for THIS prompt; 'absolute' pulls toward your global taste whatever the prompt says; 'both' layers them." },
    { name: "absolute_strength", label: "Absolute strength", kind: "float", default: 0.6, min: 0, max: 1, step: 0.05,
      dependsOn: "steer_mode", dependsVals: ["absolute", "both"],
      hint: "How hard Absolute mode pulls toward your global taste. 0.6 is visible without overriding the prompt; higher overrides it more." },
    { name: "temporal_style", label: "Temporal style", kind: "combo",
      choices: ["natural", "auto", "accelerate", "decelerate", "loop", "freeze", "pulse", "rapid_start", "rapid_end", "rapid_start_end"], default: "natural",
      hint: "Lies to the model about the frame rate to change how motion feels — faster, heavier, looping, frozen. Free. 'auto' and 'pulse' pick per scene and need the Chain Sampler." },
    { name: "split_transition_placement", label: "Transition placement", kind: "combo",
      choices: ["start", "end", "silent"], default: "start",
      hint: "Where a transition sentence lands when a prompt is split into scenes: the start of the next scene, the end of the previous one, or neither." },
    { name: "negative_erase", label: "Use the negative prompt", kind: "bool", default: false,
      hint: "EXPERIMENTAL. MiniMax H3 runs at CFG 1, so it never reads the negative prompt and what you type there does nothing. This encodes it anyway and takes its direction out of the positive conditioning instead. Unproven: expect concrete things ('a hat', 'red') to work better than vague quality words." },
    { name: "negative_erase_strength", label: "Negative strength", kind: "float", default: 0.5, min: 0, max: 2, step: 0.05,
      dependsOn: "negative_erase", dependsVals: [true],
      hint: "1.0 removes the negative's component completely; below that is partial, above pushes past it into the opposite. Start at 0.5 — this changes the prompt the model sees, so it can lose the prompt as well as the thing you did not want." },
    { name: "negative_erase_mode", label: "Negative mode", kind: "combo", choices: ["project", "subtract"], default: "project",
      dependsOn: "negative_erase", dependsVals: [true],
      hint: "'project' removes only the part of each word that points at the negative and leaves the rest alone. 'subtract' moves every word by the same amount whether or not it had anything to do with it — closer to what CFG does, and blunter." },
    { name: "negative_erase_renorm", label: "Keep prompt strength", kind: "bool", default: true,
      dependsOn: "negative_erase", dependsVals: [true],
      hint: "Puts each word back to its original strength after the change, so the result reads as 'less of that thing' rather than 'quieter prompt'. Off is the raw result." },
    { name: "h3_phrase_emphasis", label: "Rating-driven phrase emphasis (H3)", kind: "bool", default: false,
      hint: "EXPERIMENTAL, unvalidated. Boosts the attention paid to phrases the rating said were MISSING, by biasing their attention logits in H3's packed stream. Needs a rated run first. Turn it off if generations drift away from the prompt — and note it forces SLA to run dense." },
    { name: "h3_phrase_variability", label: "Phrase emphasis variability", kind: "float", default: 0.0, min: 0.0, max: 1.0, step: 0.05,
      dependsOn: "h3_phrase_emphasis", dependsVals: [true],
      hint: "0 = only what the rating has cemented gets extra attention, nothing else. 1 = that emphasis is switched off entirely, so untrained phrasing and actions compete on equal footing and can bleed into the scene — including ones you have rated against before. Same bias channel as the toggle above, just scaled down." },
    { name: "prompt_enhance", label: "Enhance the prompt with an LLM", kind: "bool", default: false,
      hint: "Before anything is generated, the fully expanded prompt is rewritten by a language model and the result is what the video is made from. Needs a text encoder that can GENERATE text wired into the Studio node's advisor_clip (or a FunPack Advisor LLM node); without one it says so and leaves the prompt alone.",
      detail: "Runs after shortcuts and $variables resolve, before encoding — so the model is handed exactly the text that would otherwise have been used, and nothing downstream needs to know it happened. On a multi-scene timeline each scene is enhanced separately: the scene list is authoritative on count, and rewriting them as one paragraph could come back with a different number of scenes. One generation per distinct scene text, so a long timeline costs real time." },
    { name: "prompt_enhance_system", label: "Enhancer instructions", kind: "textarea", rows: 10, default: "",
      dependsOn: "prompt_enhance", dependsVals: [true],
      placeholder: "Leave empty to use the built-in instructions.",
      hint: "The system prompt the enhancer runs under. Empty uses FunPack's built-in one, which expands detail and adds a soundscape without inventing characters, camera moves or speech." },
    { name: "prompt_enhance_max_length", label: "Enhancer length limit", kind: "int", default: 400, min: 32, max: 4096, step: 32,
      dependsOn: "prompt_enhance", dependsVals: [true],
      hint: "Maximum tokens the enhancer may write. Higher allows a richer prompt and costs more time." },
    { name: "prompt_enhance_temperature", label: "Enhancer temperature", kind: "float", default: 0.7, min: 0.01, max: 2.0, step: 0.05,
      dependsOn: "prompt_enhance", dependsVals: [true],
      hint: "How freely the enhancer writes. Lower sticks closer to your wording, higher invents more detail." },
    { name: "prompt_enhance_top_p", label: "Enhancer top-p", kind: "float", default: 0.92, min: 0.0, max: 1.0, step: 0.01,
      dependsOn: "prompt_enhance", dependsVals: [true],
      hint: "Nucleus sampling cutoff for the enhancer. Lower is more predictable wording." },
    { name: "prompt_enhance_thinking", label: "Enhancer thinking mode", kind: "bool", default: false,
      dependsOn: "prompt_enhance", dependsVals: [true],
      hint: "Let the model reason before answering, if it supports it. Slower, and the reasoning is stripped from the result." },
  ];

  function parseStudioSettings(p) {
    const si = p.studio_inputs || {};
    let cur = {};
    try { cur = JSON.parse(si.studio_settings || "{}"); } catch (_) {}
    const rf = (cur.refiner && typeof cur.refiner === "object") ? cur.refiner : {};
    return { si, cur, rf };
  }

  function countStudioRefine(p) {
    const { rf } = parseStudioSettings(p);
    let n = 0;
    [...STUDIO_REFINER_ESSENTIALS, ...STUDIO_REFINER_ADVANCED].forEach((f) => {
      if (EASY() && RATING_GATED_STUDIO.has(f.name)) return;
      const cur = rf[f.name] != null ? rf[f.name] : f.default;
      if (cur !== f.default) n++;
    });
    if (!EASY() && (p.refinement_key || "default") !== "default") n++;
    return n;
  }

  function persistStudioRefiner(patch, now) {
    const { cur } = parseStudioSettings(S.get().project);
    const rf = (cur.refiner && typeof cur.refiner === "object") ? cur.refiner : {};
    const next = JSON.stringify({ ...cur, refiner: { ...rf, ...patch } });
    if (now) S.setStudioInputNow("studio_settings", next);
    else S.setStudioInput("studio_settings", next);
  }

  // Conditioning adjustments — universal per-phrase steering. Stored as the FunPackStudio
  // `adjustments` widget (JSON [{phrase, strength}]), a top-level studio_input the builder
  // wires straight onto the node. Each phrase is CLIP-encoded and shifts conditioning toward
  // (+) / away (−) from it on EVERY generation, regardless of prompt. The backend ignores
  // blank/zero rows, so we persist the list as-is (no need to prune while editing).
  function parseAdjustments(p) {
    const si = (p && p.studio_inputs) || {};
    try {
      const v = JSON.parse(si.adjustments || "[]");
      return Array.isArray(v) ? v.map((i) => ({ phrase: String(i.phrase || ""), strength: +i.strength || 0 })) : [];
    } catch (_) { return []; }
  }

  function adjIsActive(i) {
    return !!String(i.phrase || "").trim() && Math.abs(+i.strength || 0) > 1e-6;
  }

  function persistAdjustments(items, now) {
    const json = JSON.stringify(items);
    if (now) S.setStudioInputNow("adjustments", json);
    else S.setStudioInput("adjustments", json);
  }

  function renderStudioRefinerBool(parentGroup, rf, f) {
    const cur = rf[f.name] != null ? rf[f.name] : f.default;
    const ctrl = el("input"); ctrl.type = "checkbox"; ctrl.checked = !!cur;
    ctrl.dataset.k = "rf-" + f.name;
    ctrl.onchange = () => persistStudioRefiner({ [f.name]: ctrl.checked }, true);
    parentGroup.append(toggleField(f.label, ctrl, f.hint));
  }

  function renderStudioRefinerField(parentGroup, rf, f) {
    if (f.dependsOn) {
      const depVal = rf[f.dependsOn] != null ? rf[f.dependsOn] : STUDIO_REFINER_ADVANCED.find((x) => x.name === f.dependsOn)?.default;
      if (!(f.dependsVals || []).includes(depVal)) return;
    }
    const val = rf[f.name] != null ? rf[f.name] : f.default;
    let ctrl;
    if (f.kind === "bool") {
      ctrl = el("input"); ctrl.type = "checkbox"; ctrl.checked = !!val; ctrl.style.width = "auto";
      ctrl.dataset.k = "rf-" + f.name;
      ctrl.onchange = () => persistStudioRefiner({ [f.name]: ctrl.checked }, true);
    } else if (f.kind === "combo") {
      ctrl = el("select"); ctrl.dataset.k = "rf-" + f.name;
      (f.choices || []).forEach((c) => { const o = new Option(c, c); if (c === val) o.selected = true; ctrl.append(o); });
      ctrl.onchange = () => persistStudioRefiner({ [f.name]: ctrl.value }, true);
    } else if (f.kind === "textarea") {
      // A system prompt is paragraphs, not a value. Quiet while typing (no repaint under the
      // caret, same rule the sampler's text knobs follow), committed on blur.
      ctrl = el("textarea"); ctrl.dataset.k = "rf-" + f.name;
      ctrl.rows = f.rows || 8;
      ctrl.value = val != null ? String(val) : "";
      if (f.placeholder) ctrl.placeholder = f.placeholder;
      ctrl.style.width = "100%";
      ctrl.style.resize = "vertical";
      ctrl.style.fontFamily = "inherit";
      ctrl.onchange = () => persistStudioRefiner({ [f.name]: ctrl.value }, true);
    } else {
      ctrl = el("input"); ctrl.type = "number";
      if (f.step != null) ctrl.step = String(f.step);
      if (f.min != null) ctrl.min = String(f.min);
      if (f.max != null) ctrl.max = String(f.max);
      ctrl.value = val; ctrl.dataset.k = "rf-" + f.name;
      ctrl.oninput = () => persistStudioRefiner({ [f.name]: parseFloat(ctrl.value || "0") }, false);
    }
    parentGroup.append(field(f.label, ctrl, f.hint, f.detail));
  }

  // ── Chain Sampler knobs ────────────────────────────────────────────────────
  const SAMPLER_KNOBS = [
    { name: "frame_overlap",         label: "Frame overlap",         kind: "int",   default: 16,    min: 0, max: 512, step: 8,
      hint: "Copies this many frames from the previous scene into the next one so the join doesn't show. 0 turns blending off, which is known to look bad together with Carry i2v guides." },
    { name: "transition_duration",   label: "Transition duration",   kind: "int",   default: 16,    min: 0, max: 128, step: 2,
      hint: "Adds this much extra fade on each side of a scene boundary. 0 turns every transition effect off." },
    { name: "use_same_seed",         label: "Same seed per scene",   kind: "bool",  default: false,
      hint: "Gives every scene the same seed instead of one each. Makes scenes resemble each other more, and makes the run repeatable when you also set a fixed seed below." },
    { name: "carry_i2v_guides",      label: "Carry i2v guides",      kind: "bool",  default: false, lockMulti: true,
      hint: "Shows each scene the protected frames of the one before it, so the look carries down the chain. Costs guide tokens (slightly slower scenes)." },
    { name: "carry_overlap_through_anchor", label: "Carry context through an anchor change", kind: "bool", default: false,
      hint: "When a scene starts from its own anchor image, still carry the previous scene's tail into the frames after it — so the background and environment survive the change of subject.",
      detail: "Without this an anchored scene is a hard cut with no carried context. The anchor's own first frame is never touched either way, so the cut still reads as a cut; only the frames after it keep the old scene's surroundings. No effect on scenes without their own anchor. Needs Frame overlap above 0." },
    { name: "cfg",                   label: "CFG",                   kind: "float", default: 1.0,   min: 0, max: 20,  step: 0.1,
      hint: "How hard the model is pushed toward the prompt. LTX and H3 are distilled and want 1.0 — raising it burns the image instead of improving prompt-following." },
    { name: "embed_guidance",        label: "Embed guidance",        kind: "bool",  default: false,
      hint: "Nudges every step toward what your ratings say you like. Costs 20-30% more time, and only does something with a refinement key and enough liked generations to have learned a direction." },
    { name: "embed_guidance_source", label: "Embed mode",            kind: "combo", choices: ["relative", "absolute"], default: "relative", dependsOn: "embed_guidance",
      hint: "Which learned direction to use: 'relative' is what worked for prompts like this one, 'absolute' is your overall taste regardless of prompt." },
    { name: "embed_guidance_strength", label: "Embed strength",      kind: "float", default: 0.02,  min: 0.005, max: 0.1, step: 0.005, dependsOn: "embed_guidance",
      hint: "How hard each step is nudged. It applies at every step so it compounds — 0.01-0.03 is the usable band, above that the prompt starts losing." },
    { name: "score_slider",          label: "Score slider",          kind: "bool",  default: false,
      hint: "A stronger version of Embed guidance that steers the prediction itself instead of the prompt. Doubles the cost of the late steps, needs 3+ liked generations, and affects video only." },
    { name: "score_slider_strength", label: "Slider strength (eta)", kind: "float", default: 1.0,   min: 0, max: 3, step: 0.25, dependsOn: "score_slider",
      hint: "How hard to push along the learned taste axis. 1.0 is a clear, safe push; up to 3.0 pushes harder; 0 is off." },
    { name: "taste_nearest_prompt",  label: "Per-prompt taste direction", kind: "bool", default: false,
      hint: "Steers each scene toward what you liked on SIMILAR prompts, instead of one global average.",
      detail: "Sources Embed guidance / Score slider from the directions learned on the prompts NEAREST this scene's, instead of one global average — a cosine lookup, no extra model pass. Falls back to the global direction when nothing rated is close." },
    { name: "output_guidance",       label: "Output guidance",       kind: "bool",  default: false,
      hint: "Applies your learned taste to what the model predicts instead of to the prompt. Almost free, but it trains a separate memory and needs its own 10+ rated generations before it does anything." },
    { name: "output_guidance_strength", label: "Output guidance strength", kind: "float", default: 0.02, min: 0.005, max: 0.1, step: 0.005, dependsOn: "output_guidance",
      hint: "How hard the prediction is corrected each step. Same scale as Embed strength — start there and adjust." },
    { name: "trajectory_guidance",   label: "Steer the whole generation", kind: "bool", default: false,
      hint: "Applies your ratings from the very start of a generation, not just near the end. Needs 10+ rated generations per quarter before any part of it acts.",
      detail: "Every other rating-driven setting only acts over the last half of a generation \u2014 after the motion and the layout have already been settled. This learns a separate memory for each quarter of the run and applies each one in its own window, so a rating about movement finally has somewhere to land. Measured across 51 rated runs, the early part carried about 88% of the signal the late part does. Costs the same as Output guidance (nothing you would notice). A quarter without enough ratings stays inert and says so." },
    { name: "trajectory_guidance_strength", label: "Whole-generation strength", kind: "float", default: 0.02, min: 0.005, max: 0.1, step: 0.005, dependsOn: "trajectory_guidance",
      hint: "How hard each step is corrected. Same scale as Output guidance strength, but not eased off near the end \u2014 so it bites harder at the same number." },
    { name: "decode_noise_scale",    label: "Decode noise scale",    kind: "float", default: 0.0,   min: 0, max: 1,   step: 0.01,
      hint: "Adds fine detail and grain back while decoding. 0 is a clean decode, ~0.025 is a gentle restore. Free, and affects the video only — not the latent." },
    { name: "decode_timestep",       label: "Decode timestep",       kind: "float", default: 0.05,  min: 0, max: 1,   step: 0.01,
      hint: "How much freedom the decoder gets while adding that detail. Higher looks more detailed but drifts further from what was actually generated. Only used when Decode noise scale is above 0." },
    { name: "decode_tile_size",      label: "Decode tile size",      kind: "int",   default: 0,     min: 0, max: 4096, step: 64,
      hint: "Decodes the video in tiles instead of all at once, to fit in less VRAM. 0 is off — set it to 512 if decoding runs out of memory." },
    { name: "joyai_memory",          label: "JoyAI-Echo memory",     kind: "bool",  default: false,
      hint: "Keeps a bank of frames from earlier shots and shows them to every new scene, so a character stays the same across the whole video. Costs guide tokens (slower scenes), and takes over from Mid-scene guide." },
    { name: "joyai_memory_size",     label: "Memory size",           kind: "int",   default: 7,     min: 1, max: 32, step: 1, dependsOn: "joyai_memory",
      hint: "How many remembered frames each scene gets. More holds identity better over a long video and makes every scene slower." },
    { name: "joyai_fix_frames",      label: "Pinned anchors",        kind: "int",   default: 3,     min: 0, max: 16, step: 1, dependsOn: "joyai_memory",
      hint: "How many opening scenes stay in the bank forever as a fixed anchor. Everything past them is a rolling window of the most recent shots." },
    { name: "joyai_frame_select",    label: "Frame select",          kind: "combo", choices: ["center", "first", "random"], default: "center", dependsOn: "joyai_memory",
      hint: "Which frame of a finished scene gets remembered — its middle, its first, or a random one." },
    { name: "joyai_memory_strength", label: "Memory strength",       kind: "float", default: 0.3,   min: 0.25, max: 10.0, step: 0.05, dependsOn: "joyai_memory",
      hint: "How hard remembered frames pull. 0.25-0.5 is the audio-safe band; higher holds the character harder but can degrade audio and stiffen motion." },
    { name: "joyai_audio_memory",    label: "Paired audio memory",   kind: "bool",  default: false, dependsOn: "joyai_memory",
      hint: "Carries voice and ambience across shots too, not just the face. This deliberately changes the audio, which nothing else here does. Needs JoyAI-Echo memory on." },
    { name: "v2a_grad_scale",        label: "Video→audio coupling", kind: "float", default: 1.0, min: 0.0, max: 4.0, step: 0.25, dependsOn: "joyai_audio_memory",
      hint: "How much the carried audio follows the new shot's picture. 1.0 is the model's own behaviour and costs nothing; JoyAI uses 2.0; 0 makes audio ignore the video." },
    { name: "alg_anchor",            label: "Blur the i2v anchor (ALG)", kind: "bool", default: false,
      hint: "Blurs the anchor's fine detail during the first, noisiest steps — the usual fix for an anchored scene that barely moves. No effect without an anchor image." },
    { name: "alg_anchor_strength",   label: "Anchor blur strength", kind: "float", default: 2.0, min: 1.0, max: 4.0, step: 0.1, dependsOn: "alg_anchor",
      hint: "How blurry the anchor gets while it's blurred. The paper says 2.5; 2.0 held character likeness noticeably better here." },
    { name: "alg_anchor_sigma_threshold", label: "Anchor blur sigma threshold", kind: "float", default: 0.975, min: 0.5, max: 0.999, step: 0.005, dependsOn: "alg_anchor",
      hint: "How long the anchor stays blurred before switching to sharp. Higher = a shorter blurred window." },
    { name: "alg_blur_guides",       label: "Blur i2v guides and JoyAI memory", kind: "bool", default: false,
      hint: "The same blur for guide and JoyAI-memory frames, so they steer composition without pasting their own detail in. No effect on a scene with no guide frames." },
    { name: "alg_guide_blur_strength", label: "Guide blur strength", kind: "float", default: 2.0, min: 1.0, max: 4.0, step: 0.1, dependsOn: "alg_blur_guides",
      hint: "How blurry those frames get while they're blurred. Higher = looser guidance and more freedom to move." },
    { name: "alg_guide_blur_sigma_threshold", label: "Guide blur sigma threshold", kind: "float", default: 0.975, min: 0.5, max: 0.999, step: 0.005, dependsOn: "alg_blur_guides",
      hint: "How long they stay blurred before switching to sharp. Higher = a shorter blurred window." },
    { name: "h3_block_repeat", label: "Repeat block(s) (H3, experimental)", kind: "text", default: "",
      hint: "Runs the named block(s) on their own output before passing it on — a second pass that never leaves latent space. Blank = off. A block, a range, or a list: 40 | 38-42 | 10,40,44.",
      detail: "MEASURED 2026-09-04, same seed/prompt/reference throughout — this does NOT sharpen, it re-rolls the shot (detail stayed within ±5% everywhere useful while the picture itself moved 17-60%). Treat it as a depth-indexed variation generator with a quality gradient, not a detail pass.\n\nWHAT EACH BAND DID (one extra pass unless noted):\n• 0-10 — destroys everything: coloured grid, no characters. Never.\n• 10, 15 — indistinguishable from plain. Pointless, pure wasted compute.\n• 11-20 — video detailed with good motion, but AUDIO AWFUL. At two passes video still holds while audio is replaced with unprompted content and the requested sounds vanish. Video-only work.\n• 20-30 — mild and mostly positive. 25 resolved distinct pupils where plain barely renders them; 30 gave motion on point with good audio. Safe band.\n• 31-41 — a STYLE lever, not a quality lever. It reliably shifts the look toward the prompt's style (skin texture, outlines; replicated across two reference images) and guarantees NOTHING about motion, anatomy or coherence — those are still a re-roll. The audio damage is constant: unprompted dialogue on every run of this band. ONE PASS ONLY: 31-40 and 35-41 both collapsed at two.\n• 42-46 — 44 and 45 showed no significant difference at one pass, and still degrade at two. Inert at one pass is NOT headroom: everything past 40 dislikes a second pass whether or not the first did anything.\n• 47-49 — 47 fuzzy and slow, 48 destroys detail and shifts colour, 49 is noise with broken audio. Never.\n\nAudio breaks at shallower depths than video, so check it separately. Most cells above are a single observation judged by eye; the 31-41 two-pass failure is the firmest result (seen twice).\n\nCost is one extra block forward per repeat (~2% of a step each) — no VAE round trip, no re-noising, no extra sampler step. The blocks after a repeated one were trained on its normal output, so a twice-processed stream is out of distribution for them, which is what the failures above look like." },
    { name: "h3_block_repeat_span_loop", label: "Loop the span instead of each block", kind: "bool", default: false, dependsOn: "h3_block_repeat",
      hint: "Runs 31,32…40 then goes back to 31 and runs it again, instead of running each block twice in a row.",
      detail: "Off, a repeat of 31-40 runs 31,31,32,32…40,40 — every block receives its OWN doubled output, which is a distribution the next block has never seen, so a ten-block span creates ten of those seams. That matches what the sweep found: two scattered blocks survive two extra passes while the full range collapses at the same count. On, it runs 31,32…40,31,32…40 — every block still gets input from its normal predecessor and there is exactly ONE seam, at the wrap from last back to first. Same number of extra block evaluations either way, so the same cost. Needs a contiguous range like 31-40; a scattered list (31,34,37,40) is refused with a note rather than silently looping the blocks you left out." },
    { name: "h3_block_repeat_video_only", label: "Repeat video rows only", kind: "bool", default: false, dependsOn: "h3_block_repeat",
      hint: "Keeps the extra pass for the picture and leaves the soundtrack on a single pass.",
      detail: "H3 puts video, text and audio in one sequence, so a plain repeat doubles the block for the audio rows too — the leading suspect for the unprompted dialogue that band 31-40 produces on every run of it. This restores text and audio rows to their single-pass value afterwards. TWO limits, both measured: it does not undo audio's influence on the second pass (attention had already mixed the twice-processed audio into the video rows before they were restored), and it does not keep audio clean DOWNSTREAM — the blocks after the repeated ones still process audio rows while attending to twice-processed video rows, so audio can be contaminated after the fact. It protects the audio at the exit of the repeated blocks, not all the way to the output, which is why enabling it did not save the audio in span-loop tests. Off by default so the measured sweep results stay reproducible — turn it on and compare against the same seed to see whether the trade goes away." },
    { name: "h3_block_repeat_times", label: "Extra passes per repeated block", kind: "int", default: 1, min: 1, max: 4,
      hint: "1 = the block runs twice in total. Higher pushes further from what the following blocks were trained to receive — measured: everything past ~30 collapses at 2 (including blocks that did nothing at 1), while 11-20 survives it on video but not on audio." },
    { name: "h3_block_repeat_last_steps", label: "Only repeat part of the schedule", kind: "int", default: 0, min: -50, max: 50, dependsOn: "h3_block_repeat",
      hint: "0 = repeat on every step (the original behaviour). Positive N = the final N steps only. Negative N = the first |N| steps only, then stop.",
      detail: "Every sweep result above was run with this at 0 — the repeat firing on EVERY step, including the earliest, near-pure-noise ones where the block is still deciding layout and motion, not detail. That is the likely reason repeat re-rolls the shot instead of sharpening it: doubling a block's say over what the shot even IS, before it exists yet, is a different operation than doubling its say over how a settled shot looks. A positive value confines the repeat to the tail, after structure is already locked in by the earlier steps. A negative value confines it to the HEAD instead — applying while structure is still forming, then leaving the untouched rest of the schedule to resolve whatever came out, rather than the perturbed step being close to the last word on the output. Neither direction carries over any finding from this page automatically. Counted in STEPS from either end of the schedule, not sigma, so the window lands on the same point of denoising regardless of step count or sampler." },
    { name: "bounded_attention_enabled", label: "Bounded attention (experimental, untested)", kind: "bool", default: false,
      hint: "Stops two people in one frame swapping each other's features, by letting each half of the frame see only its own sentence. Nearly free, and does nothing unless the scene prompt has two sentences describing two subjects. Built but never run — the left/right split is fixed, so it suits two subjects side by side and nothing else." },
    { name: "dynashift",             label: "DynaShift (steer off bad gens)", kind: "bool", default: false,
      hint: "Steers away from generations you rated bad — a negative prompt built from your ratings instead of from text. Nearly free, and needs a refinement key plus some bad ratings already banked." },
    { name: "dynashift_strength",    label: "DynaShift strength",    kind: "float", default: 0.3, min: 0.05, max: 1.0, step: 0.05, dependsOn: "dynashift",
      hint: "How much of the matched bad direction is removed per step. 0.3 is a gentle nudge; 1.0 removes it outright each step." },
    { name: "dynashift_threshold",   label: "DynaShift match threshold", kind: "float", default: 0.6, min: 0.3, max: 0.95, step: 0.05, dependsOn: "dynashift",
      hint: "How closely a frame must resemble a banked bad one before steering starts. Lower is more aggressive and more likely to push away from content that was actually fine." },
    { name: "h3_repr_steering",      label: "REINS (representation steering, H3, experimental)", kind: "bool", default: false,
      hint: "EXPERIMENTAL, unvalidated, H3 only. Reaches inside the model instead of around it: captures each steered block's video-row hidden state every generation, and once a block has 3+ liked and 3+ disliked runs of its OWN, adds that block's mean liked-minus-disliked difference back into its own output on every later run. No architectural ceiling like the attention-bias mechanisms have — push the strength too far and coherence can break with no warning." },
    { name: "h3_repr_steering_strength", label: "REINS strength", kind: "float", default: 0.05, min: 0.0, max: 2.0, step: 0.01, dependsOn: "h3_repr_steering",
      hint: "Fraction of each steered block's own hidden-state norm added along its learned direction — the same strength applies to every block named below. Start low and push it deliberately high on a same-seed test to confirm it moves the video at all before trusting a small value." },
    { name: "h3_repr_steering_block", label: "REINS steer at block(s)", kind: "text", default: "25", dependsOn: "h3_repr_steering",
      hint: "TEST-ONLY: which block(s) actually get steered, instead of always block 25 — every block 0-49 is captured read-only regardless. A single block, a range, or a comma list: 25 | 31-40 | 4,5,6 — same syntax as Repeat block(s). Each named block steers with its OWN learned direction from its OWN liked/disliked history at that block; a block without enough data just captures, same as any unnamed candidate block.",
      detail: "Blocks 40-49 carry the most residual movement (48+49 alone about a third of it), so a steering push has the most to act on there. Blocks 0-1 are structurally distinct rather than just 'quiet early blocks' — 0 is the raw input projection, 1 does the first cross-patch mixing — expect steering there to behave differently in kind, not just 'weaker', than the middle of the stack." },
    { name: "h3_av_decouple", label: "Video/audio attention decoupling (H3, experimental)", kind: "float", default: 0.0, min: 0.0, max: 50.0, step: 0.5,
      hint: "EXPERIMENTAL, unvalidated, H3 only. Weakens how much video and audio attend to each other, for the failure pattern where one comes out clean and the other doesn't. 0 = off. Not rating-driven — a manual dial, same as Repeat block(s). This is a raw attention-logit penalty, not a 0-1 fraction — try 1-5 first, then 10-20.",
      detail: "H3 has no separate cross-attention to mask — video, text and audio share one joint self-attention. Instead of building a mask over that whole sequence (unaffordable at H3's token count), this runs the same attention as separate video-query and audio-query passes, each carrying a penalty on the other modality's attention weight. Same total cost as the ordinary single pass. There's no way to read H3's actual trained attention scale from here (the checkpoint isn't on this machine), so this number can't be pre-calibrated — v1 tried normalizing it to 0-1 against a guessed constant and live testing came back flat across the whole range, which is why it's now the raw penalty for you to search directly. Large enough to fully exclude the other modality likely also kills legitimate audio/video sync — search up from a small value, not down from a large one." },
    { name: "identity_transfer_enabled", label: "Best-FaceID compatibility", kind: "bool", default: false,
      hint: "Feeds the identity pin image the way Best-FaceID identity LoRAs expect it. Needs an Identity pin set.",
      detail: "Replaces Continuity's identity-pin guide with separate, non-rendered reference tokens plus an optional ArcFace projector below. Load the LoRA itself in Models. No effect without an identity pin." },
    { name: "source_id", label: "Source-phase id", kind: "float", default: 2.0, min: 0.0, max: 8.0, step: 1.0, dependsOn: "identity_transfer_enabled",
      hint: "Matches the LoRA's training convention (ltx-trainer used 2). 0 disables the rotation." },
    { name: "phase_scale", label: "Phase scale", kind: "float", default: 1.0, min: 0.0, max: 4.0, step: 0.1, dependsOn: "identity_transfer_enabled",
      hint: "Multiplies Source-phase id before the rotation is applied. Leave it at 1.0 unless you are matching a LoRA trained with an unusual convention." },
    { name: "id_strength", label: "ArcFace token strength", kind: "float", default: 1.0, min: 0.0, max: 50.0, step: 0.5, dependsOn: "identity_transfer_enabled",
      hint: "Only used when an ArcFace projector is set below. Weak channel — push high (5-20) to test." },
    { name: "arcface_mode", label: "ArcFace detection mode", kind: "combo", choices: ["auto_adjust", "as_is", "disable"], default: "auto_adjust", dependsOn: "identity_transfer_enabled",
      hint: "What to do when the face detector can't get a clean crop of the pin image: 'auto_adjust' fixes the crop, 'as_is' uses it anyway, 'disable' skips the ArcFace channel entirely." },
    { name: "debug_log", label: "Debug log", kind: "bool", default: false, dependsOn: "identity_transfer_enabled",
      hint: "Print per-scene identity-transfer shape/status logs to the ComfyUI console." },
    { name: "cut_opening_frames", label: "Cut the opening (frames)", kind: "int", default: 0, min: 0, max: 512, step: 8,
      hint: "Trims this many frames off the FRONT of the finished clip, so an i2v render reads as t2v. The scene comes out shorter.",
      detail: "The anchor is generated at full strength, then this many frames are dropped from the FRONT of the finished clip, so an i2v render reads as t2v without weakening the anchor. Nothing is regrown: the scene comes out shorter and the audio is cropped to match. 8 removes only the anchor and is usually too little; 48 worked on a 768x768x305@30 chain. Skipped on continuation scenes and scenes carrying guides." },
    { name: "second_pass_upscale", label: "Resample factor", kind: "float", min: 1.0, max: 4.0, step: 0.05, default: 2.0,
      hint: "Cost is the SQUARE of this when upscaling — 2x is four times the pixels for pass 2, 4x is sixteen.",
      detail: "Only upsamplers that take a factor honour it: the LTX one is a fixed 2x network and reports that it ignored the value; MiniMax H3's resizer takes anything from 1.0 to 4.0. On 'sharpen' it is how far up the latent goes before coming straight back, so it buys detail rather than resolution." },
    { name: "second_pass_op", label: "Between-pass operation", kind: "combo", choices: ["none", "sharpen", "upscale_2x"], default: "none",
      hint: "'sharpen' adds detail almost free; 'upscale_2x' doubles the output resolution at 3-5x the cost of pass 2.",
      detail: "Both need the LTX 2.3 latent upsampler (~1 GB, found or downloaded once); without it pass 2 still runs and the report says the operation was skipped. 'sharpen' resamples straight back to size — no video-model calls, and it cannot fix wrong structure. 'upscale_2x' keeps the 2x, so pass 2 costs 3-5x; the i2v anchor is upscaled with the clip and stays pinned, but guide keyframes are dropped for pass 2. Video only." },
    { name: "context_windows", label: "Context windows (long scenes)", kind: "bool", default: false,
      hint: "Renders very long scenes as overlapping windows instead of one pass. Slower short, faster past ~300 frames.",
      detail: "Denoises a scene longer than the model's comfortable window as overlapping windows — core's own mechanism, audio-aware on LTX. Engages only past the window length below. About 1.45x the per-frame work at the defaults, roughly break-even near 200 frames and a win past ~300. Needs ComfyUI v0.29.0 or newer." },
    { name: "context_window_length", label: "Window length (frames)", kind: "int", default: 145, min: 9, max: 2049, step: 8, dependsOn: "context_windows",
      hint: "Window size in frames — and the threshold: shorter scenes skip windowing entirely.",
      detail: "Keep it at or under the length the model already generates well in one pass — the point is to stay inside that range while the scene as a whole goes past it." },
    { name: "context_window_overlap", label: "Window overlap (frames)", kind: "int", default: 40, min: 0, max: 512, step: 8, dependsOn: "context_windows",
      hint: "Frames shared between windows: too low shows a seam, too high wastes compute.",
      detail: "This is the only thing carrying motion and appearance across a window boundary, and also the only extra compute windowing costs." },
    { name: "context_window_schedule", label: "Window schedule", kind: "combo", choices: ["standard_uniform", "standard_static", "looped_uniform", "batched"], default: "standard_uniform", dependsOn: "context_windows",
      legacy: { uniform_standard: "standard_uniform", static_standard: "standard_static", uniform_looped: "looped_uniform" },
      hint: "Where the window cut points fall each step. The default is the safest.",
      detail: "'standard_uniform' shifts the cut points between steps so a boundary never bakes in. 'standard_static' is cheapest but a bad boundary stays bad. 'looped_uniform' wraps the end into the start. 'batched' has no overlap logic." },
    { name: "context_window_fuse", label: "Window blend", kind: "combo", choices: ["pyramid", "relative", "flat", "overlap-linear"], default: "pyramid", dependsOn: "context_windows",
      hint: "How overlapping windows are blended. Change it if boundaries look ghosted.",
      detail: "'pyramid' (default) fades each window toward its edges so seams go soft. 'flat' averages equally (can smear). Not the setting for boundaries that look merely misaligned." },
    { name: "context_window_freenoise", label: "FreeNoise blending", kind: "bool", default: true, dependsOn: "context_windows",
      hint: "Makes windows blend better by correlating their starting noise. Free — leave it on.",
      detail: "Shuffles rather than redraws the starting noise between windows, so overlapping regions begin from correlated noise. A one-time permutation, and core's own default because it measurably improves how windows blend. Turn off only to A/B whether it's helping." },
    { name: "context_window_retain_first", label: "Pin anchor in every window", kind: "bool", default: false, dependsOn: "context_windows",
      hint: "Keeps the anchor frame in every window. Turn on if later windows drift; off again if motion stalls.",
      detail: "Keeps latent frame 0 inside every window instead of just the first. Off by default because on a continuation scene frame 0 is the carried tail of the previous scene, not the anchor — pinning it everywhere can make the scene read as static." },
  ];
  const SAMPLER_KNOB_MAP = Object.fromEntries(SAMPLER_KNOBS.map((k) => [k.name, k]));

  const CONTINUITY_DEFAULTS = {
    auto_enabled: true,
    identity_pin_ref: null,
    identity_pin_strength: 0.35,
    prior_scene_guides: true,
    prior_scene_strength: 0.35,
    guide_decay: 0.85,
    solo_scene_guides: true,
  };

  function normContinuitySettings(p) {
    const cs = (p && p.continuity_settings) || {};
    return {
      auto_enabled: cs.auto_enabled !== false,
      identity_pin_ref: cs.identity_pin_ref || null,
      identity_pin_strength: cs.identity_pin_strength != null ? +cs.identity_pin_strength : CONTINUITY_DEFAULTS.identity_pin_strength,
      prior_scene_guides: cs.prior_scene_guides !== false,
      prior_scene_strength: cs.prior_scene_strength != null ? +cs.prior_scene_strength : CONTINUITY_DEFAULTS.prior_scene_strength,
      guide_decay: cs.guide_decay != null ? +cs.guide_decay : CONTINUITY_DEFAULTS.guide_decay,
      solo_scene_guides: cs.solo_scene_guides !== false,
    };
  }

  function patchContinuitySettings(patch) {
    const p = S.get().project;
    S.patchProject({ continuity_settings: { ...(p.continuity_settings || {}), ...patch } });
  }

  function normGuideSettings(p) {
    const gs = (p && p.guide_settings) || {};
    return { stack_enabled: !!gs.stack_enabled, accumulate_prior: !!gs.accumulate_prior };
  }

  function patchGuideSettings(patch) {
    const p = S.get().project;
    S.patchProject({ guide_settings: { ...(p.guide_settings || {}), ...patch } });
  }

  function _depSatisfied(name, value, si) {
    const depVal = si[name] != null ? si[name] : SAMPLER_KNOB_MAP[name]?.default;
    return value !== undefined ? depVal === value : !!depVal;
  }

  function knobVisible(k, si, st) {
    if (EASY() && RATING_GATED_KNOBS.has(k.name)) return false;
    // The loaded model cannot use this one. Hide it rather than offer a control and then
    // explain per run that it does nothing — what is not wired does not appear. The stored
    // value survives, so switching family back brings the setting back with it.
    if (st && window.PipelineCaps?.familyInertInputs(st).has(k.name)) return false;
    // dependsOn/dependsValue: single condition (dependsValue absent = plain truthy
    // gate, as every existing boolean dependsOn already relies on).
    if (k.dependsOn && !_depSatisfied(k.dependsOn, k.dependsValue, si)) return false;
    // deps: an AND'd list, for knobs gated by more than one condition (e.g. the
    // feature toggle AND a mode combo equaling a specific option) — dependsOn
    // alone can't express two conditions without one silently overriding the other.
    if (Array.isArray(k.deps)) {
      for (const d of k.deps) {
        if (!_depSatisfied(d.name, d.value, si)) return false;
      }
    }
    return true;
  }

  // A knob left at its default is not a setting, it is the absence of one. Storing it anyway
  // meant the project file accumulated every dial the user ever touched, with no way to undo
  // that except typing the default back in — and even then the key stayed, so the value
  // outlived the refinement key it came from and looked like learned state that would not
  // clear. Writing the default REMOVES the key, so "back to default" really is back to
  // nothing, and a project only ever records what was deliberately changed.
  function writeSamplerInput(k, value, immediate) {
    if (value === k.default) {
      S.unsetSamplerInput(k.name);
      if (immediate) S.flushSave?.();
      return;
    }
    (immediate ? S.setSamplerInputNow : S.setSamplerInput)(k.name, value);
  }

  function renderSamplerKnob(parentGroup, st, k, si, multiScene) {
    if (!knobVisible(k, si, st)) return;
    const val = si[k.name] != null ? si[k.name] : k.default;
    const gs = normGuideSettings(st.project);
    const cs = normContinuitySettings(st.project);
    const forced = k.lockMulti && multiScene && !gs.stack_enabled && cs.auto_enabled;
    let ctrl;
    if (k.kind === "bool") {
      ctrl = el("input"); ctrl.type = "checkbox";
      ctrl.checked = forced ? true : !!val;
      ctrl.style.width = "auto";
      ctrl.dataset.k = "si-" + k.name;
      if (forced) {
        ctrl.disabled = true;
        ctrl.title = "Auto-enabled by Auto continuity for multi-scene carry runs";
      } else {
        ctrl.onchange = () => writeSamplerInput(k, ctrl.checked, true);
      }
    } else if (k.kind === "combo") {
      ctrl = el("select"); ctrl.dataset.k = "si-" + k.name;
      // A renamed choice: show what the stored value MEANS, and write the new name back, so
      // the dropdown never quietly displays a different setting from the one that will run.
      const shown = (k.legacy && k.legacy[val]) || val;
      if (shown !== val) S.setSamplerInput(k.name, shown);
      (k.choices || []).forEach((c) => { const o = el("option", null, c); o.value = c; if (c === shown) o.selected = true; ctrl.append(o); });
      ctrl.onchange = () => writeSamplerInput(k, ctrl.value, true);
    } else if (k.kind === "text") {
      ctrl = el("input"); ctrl.type = "text";
      ctrl.value = val != null ? String(val) : "";
      if (k.placeholder) ctrl.placeholder = k.placeholder;
      ctrl.dataset.k = "si-" + k.name;
      // Quiet while typing (no repaint under the caret), commit on blur/Enter.
      ctrl.oninput = () => writeSamplerInput(k, ctrl.value, false);
      ctrl.onchange = () => writeSamplerInput(k, ctrl.value, true);
    } else {
      ctrl = el("input"); ctrl.type = "number";
      if (k.step != null) ctrl.step = String(k.step);
      if (k.min != null) ctrl.min = String(k.min);
      if (k.max != null) ctrl.max = String(k.max);
      ctrl.value = val; ctrl.dataset.k = "si-" + k.name;
      ctrl.oninput = () => {
        const v = k.kind === "int" ? parseInt(ctrl.value || "0", 10) : parseFloat(ctrl.value || "0");
        writeSamplerInput(k, v, false);
      };
    }
    parentGroup.append(field(k.label + (forced ? " (auto)" : ""), ctrl, k.hint, k.detail));
  }

  function renderKnobList(parentGroup, st, names) {
    const si = st.project.sampler_inputs || {};
    const multiScene = activeSceneCount(st.project) > 1;
    names.forEach((name) => {
      const k = SAMPLER_KNOB_MAP[name];
      if (k) renderSamplerKnob(parentGroup, st, k, si, multiScene);
    });
  }

  // ── views (inner-sidebar categories) ───────────────────────────────────────
  const CHAIN_VIEW_KNOBS = {
    chain_continuity: ["carry_i2v_guides", "carry_overlap_through_anchor"],
    chain_timing: ["frame_overlap", "transition_duration", "use_same_seed", "cut_opening_frames"],
    chain_guidance: ["cfg", "embed_guidance", "embed_guidance_source", "embed_guidance_strength", "score_slider", "score_slider_strength", "taste_nearest_prompt", "output_guidance", "output_guidance_strength", "trajectory_guidance", "trajectory_guidance_strength", "dynashift", "dynashift_strength", "dynashift_threshold", "h3_repr_steering", "h3_repr_steering_strength", "h3_repr_steering_block", "h3_av_decouple"],
    chain_decode: ["decode_noise_scale", "decode_timestep", "decode_tile_size"],
    chain_experimental: ["context_windows", "context_window_length", "context_window_overlap", "context_window_schedule", "context_window_fuse", "context_window_freenoise", "context_window_retain_first", "joyai_memory", "joyai_memory_size", "joyai_fix_frames", "joyai_frame_select", "joyai_memory_strength", "joyai_audio_memory", "v2a_grad_scale", "alg_blur_guides", "alg_guide_blur_strength", "alg_guide_blur_sigma_threshold", "bounded_attention_enabled", "h3_block_repeat", "h3_block_repeat_span_loop", "h3_block_repeat_video_only", "h3_block_repeat_times", "h3_block_repeat_last_steps", "identity_transfer_enabled", "source_id", "phase_scale", "id_strength", "arcface_mode", "debug_log"],
  };

  function countChainView(p, id, st) {
    const si = p.sampler_inputs || {};
    // A knob the model cannot use is hidden, so a badge counting it would advertise a
    // change the user cannot see or undo from this pane.
    const inert = st ? (window.PipelineCaps?.familyInertInputs(st) || new Set()) : new Set();
    let n = 0;
    (CHAIN_VIEW_KNOBS[id] || []).forEach((name) => {
      if (EASY() && RATING_GATED_KNOBS.has(name)) return;
      if (inert.has(name)) return;
      const k = SAMPLER_KNOB_MAP[name];
      if (k && si[name] != null && si[name] !== k.default) n++;
    });
    if (id === "chain_timing" && si.seed != null) n++;
    return n;
  }

  function viewList(st) {
    const p = st.project;
    const PC = window.PipelineCaps;
    const studioOn = PC?.usesFunpackStudio(st);
    const chainOn = PC?.usesChainSampler(st);
    const out = [{ id: "overview", group: "", title: "Overview", icon: "◎" }];
    if (studioOn) {
      out.push(
        { id: "studio_refine", group: "FunPack Studio", title: "Refinement", icon: "✦", badge: countStudioRefine(p) || null },
        { id: "studio_adjust", group: "FunPack Studio", title: "Adjustments", icon: "±", badge: parseAdjustments(p).filter(adjIsActive).length || null },
        { id: "studio_sampler", group: "FunPack Studio", title: "Sampler algorithm", icon: "∿" },
      );
    }
    if (chainOn) {
      out.push(
        { id: "chain_continuity", group: "Chain Sampler", title: "Continuity", icon: "∞", badge: countChainView(p, "chain_continuity", st) || null },
        { id: "chain_timing", group: "Chain Sampler", title: "Timing & Seed", icon: "⏱", badge: countChainView(p, "chain_timing", st) || null },
        { id: "chain_guidance", group: "Chain Sampler", title: "Guidance", icon: "◇", badge: countChainView(p, "chain_guidance", st) || null },
        { id: "chain_decode", group: "Chain Sampler", title: "Decode", icon: "▣", badge: countChainView(p, "chain_decode", st) || null },
        { id: "chain_experimental", group: "Chain Sampler", title: "Experimental", icon: "⚗", badge: countChainView(p, "chain_experimental", st) || null },
      );
    }
    return out;
  }

  function renderOverview(pane, st) {
    const p = st.project;
    const PC = window.PipelineCaps;
    const studioOn = PC?.usesFunpackStudio(st);
    const chainOn = PC?.usesChainSampler(st);
    const slots = (st.models?.slots || []);

    if (studioOn || chainOn) {
      const presetBar = el("div", "engine-preset-bar");
      presetBar.append(el("span", "engine-preset-label", "Preset"));
      Object.entries(S.ENGINE_PRESETS || {}).forEach(([key, pr]) => {
        const b = el("button", "btn ghost tiny", pr.label || key);
        b.onclick = () => S.applyEnginePreset(key);
        presetBar.append(b);
      });
      pane.append(presetBar);
    }

    const summary = el("div", "engine-modal-summary");
    const parts = [];
    parts.push(studioOn ? "FunPack Studio" : (slotLabelFor(st, p.conditioning_slot) || p.conditioning_slot));
    parts.push(chainOn ? "Chain Sampler" : (slotLabelFor(st, p.sampler_slot) || p.sampler_slot));
    summary.textContent = "Generating with " + parts.join(" + ");
    pane.append(summary);

    if (!studioOn && !chainOn) {
      pane.append(hintEl("Neither FunPack Studio nor Chain Sampler is active — pick them below or wire custom nodes in Models & Pipeline."));
    }

    const g = group(pane, "Pipeline nodes");
    const condSel = el("select"); condSel.dataset.k = "pj-cond";
    [["funpack", "FunPack Studio"], ...slots.map((s) => [s.id, s.label || s.node_class || s.id])]
      .forEach(([v, lbl]) => { const o = new Option(lbl, v); if ((p.conditioning_slot || "funpack") === v) o.selected = true; condSel.append(o); });
    condSel.onchange = () => S.setConditioningSlot(condSel.value);
    g.append(field("Conditioning", condSel,
      "Which node turns your prompt into conditioning. Picking anything but FunPack Studio hides the Studio settings below."));
    const sampSel = el("select"); sampSel.dataset.k = "pj-samp";
    [["funpack", "FunPack Chain Sampler"], ...slots.map((s) => [s.id, s.label || s.node_class || s.id])]
      .forEach(([v, lbl]) => { const o = new Option(lbl, v); if ((p.sampler_slot || "funpack") === v) o.selected = true; sampSel.append(o); });
    sampSel.onchange = () => S.setSamplerSlot(sampSel.value);
    g.append(field("Sampler", sampSel,
      "Which node does the actual sampling. Picking anything but the FunPack Chain Sampler hides the Chain Sampler settings below."));

    if (!studioOn) pane.append(hintEl("Custom conditioning node — wire and tune it in Models & Pipeline. FunPack Studio categories are hidden."));
    if (!chainOn) pane.append(hintEl("Custom sampler node — wire and tune it in Models & Pipeline. Chain Sampler categories are hidden."));

    const foot = el("div", "sw-hint eng-foot");
    foot.append(el("span", null, "Unet / VAE / linked inputs: "));
    const modelsLink = el("button", "btn ghost tiny", "Models & Pipeline…");
    modelsLink.type = "button";
    modelsLink.onclick = () => window.ModelsModal.open();
    foot.append(modelsLink);
    pane.append(foot);
  }

  function renderStudioRefine(pane, st) {
    const p = st.project;
    const { rf } = parseStudioSettings(p);

    if (!EASY()) {
      // Refinement key — project-level (feeds Studio / Chain Sampler / SaveRefinementLatent).
      // "default" uses the keyless store; a custom name trains/loads its own key. Shortcuts
      // bound to a non-default key layer per-scene training on top of this.
      const gKey = group(pane, "Session");
      const keyCtrl = el("input"); keyCtrl.type = "text"; keyCtrl.dataset.k = "refinement_key";
      keyCtrl.placeholder = "default"; keyCtrl.value = p.refinement_key || "default";
      keyCtrl.onchange = () => S.patchProject({ refinement_key: (keyCtrl.value || "").trim() || "default" });
      gKey.append(field("Refinement key", keyCtrl,
        "Names the learning session your ratings train. Everything rating-driven below reads from it; \"default\" is the shared one."));
    }

    const gEss = group(pane, "Essentials");
    STUDIO_REFINER_ESSENTIALS.filter((f) => !EASY() || !RATING_GATED_STUDIO.has(f.name))
      .forEach((f) => renderStudioRefinerBool(gEss, rf, f));

    const gAdv = group(pane, EASY() ? "Prompt shaping" : "Refinement");
    STUDIO_REFINER_ADVANCED.filter((f) => !EASY() || !RATING_GATED_STUDIO.has(f.name))
      .forEach((f) => renderStudioRefinerField(gAdv, rf, f));

    if (EASY()) {
      pane.append(hintEl(
        "Studio runs in Prompt-only mode here: it shapes and splits the prompt, nothing more. "
        + "Rating-dependent controls are hidden — Simple mode has no rating UI to feed them. "
        + "Switch to Editor for the full learned refiner."));
    } else {
      pane.append(hintEl("Scene text and transitions come from the timeline. Advisor, LoRA, and batch training remain in the ComfyUI Studio popup on the graph."));
    }
  }

  function renderStudioAdjust(pane, st) {
    pane.append(hintEl(
      "Pulls every generation toward (+) or away from (−) a phrase, whatever the prompt says. "
      + "Typical range −0.3 to +0.3."));
    const items = parseAdjustments(st.project);
    const g = group(pane, "Phrases");
    if (!items.length) g.append(el("div", "sw-row eng-empty-row", "No adjustments. Add a phrase below."));
    items.forEach((item, idx) => {
      const row = el("div", "sw-row studio-adjust-row");
      const phrase = el("input"); phrase.type = "text"; phrase.placeholder = "phrase or word";
      phrase.value = item.phrase; phrase.dataset.k = "adj-phrase-" + idx; phrase.style.flex = "1";
      phrase.oninput = () => { items[idx].phrase = phrase.value; persistAdjustments(items, false); };
      const strength = el("input"); strength.type = "number";
      strength.step = "0.05"; strength.min = "-1"; strength.max = "1";
      strength.value = item.strength; strength.dataset.k = "adj-str-" + idx; strength.style.width = "72px";
      strength.title = "Positive steers toward the phrase, negative away.";
      strength.oninput = () => { items[idx].strength = parseFloat(strength.value || "0"); persistAdjustments(items, false); };
      const del = el("button", "btn ghost tiny", "✕"); del.type = "button"; del.title = "Remove";
      del.onclick = () => { items.splice(idx, 1); persistAdjustments(items, true); };
      row.append(phrase, strength, del);
      g.append(row);
    });
    const add = el("button", "btn ghost tiny eng-add", "+ Add phrase"); add.type = "button";
    add.onclick = () => {
      items.push({ phrase: "", strength: 0.1 });
      persistAdjustments(items, true);
      setTimeout(() => {
        const inputs = document.querySelectorAll('[data-k^="adj-phrase-"]');
        inputs[inputs.length - 1]?.focus();
      }, 0);
    };
    pane.append(add);
  }

  function renderStudioSampler(pane, st) {
    const { cur: curSettings } = parseStudioSettings(st.project);
    const samplers = curSettings.samplers || null;
    const box = el("div", "eng-sampler-panel");
    pane.append(box);
    function persistSamplers(updatedSamplers, quiet) {
      const { cur } = parseStudioSettings(S.get().project);
      const next = JSON.stringify({ ...cur, samplers: updatedSamplers });
      if (quiet) S.setStudioInput("studio_settings", next);
      else S.setStudioInputNow("studio_settings", next);
    }

    // ALG used to have two switches: the Distilled Flow panel's own alg_enabled, and the
    // chain sampler's alg_anchor, which is the same guidance on ANY sampler and already
    // drove the Distilled Flow one whenever it was on. Two controls for one behaviour, and
    // which of them won depended on which you had set — so the Editor now shows only
    // alg_anchor. The node input still exists for hand-built graphs; what moves is the
    // Editor's control. A project carrying the old switch is migrated once, with its
    // strength and threshold, rather than left running ALG with nothing on screen to
    // turn it off.
    function migrateDistilledAlg(cfg) {
      const pid = String(S.get()?.project?.id || "");
      const dc = cfg?.high?.distilled;
      if (!dc || !dc.alg_enabled) return _ALG_MIGRATED.has(pid);
      // Mark and clear BEFORE writing anything: each setter notifies the store, which
      // re-renders this pane synchronously and comes straight back through here.
      _ALG_MIGRATED.add(pid);
      const strength = dc.alg_strength;
      const threshold = dc.alg_sigma_threshold;
      dc.alg_enabled = false;
      // Out of the render pass entirely — persisting mid-render is what caused the
      // recursion above, and a deferred write costs one extra repaint instead.
      setTimeout(() => {
        S.setSamplerInputNow("alg_anchor", true);
        if (strength != null) S.setSamplerInputNow("alg_anchor_strength", Number(strength));
        if (threshold != null) S.setSamplerInputNow("alg_anchor_sigma_threshold", Number(threshold));
        persistSamplers(cfg, false);
      }, 0);
      return true;
    }
    const algMigrated = migrateDistilledAlg(samplers);

    try {
      window.SamplerPanel.render(box, samplers,
        (s) => persistSamplers(s, true),
        (s) => persistSamplers(s, false));
    } catch (e) {
      const err = hintEl("Studio sampler panel failed to render: " + e.message);
      err.style.color = "var(--danger)";
      box.append(err);
    }
    // The anchor blur lives HERE, with the sampler, not under Experimental: it is a
    // property of how the scene is sampled and the first place anyone looks for it is the
    // pane where they picked the sampler. It runs on whatever sampler is selected above —
    // inside the loop on Distilled Flow, through a denoiser proxy on everything else — so
    // it is never hidden behind a particular choice up there.
    const algG = group(pane, "Anchor blur (ALG)");
    renderKnobList(algG, st, ["alg_anchor", "alg_anchor_strength", "alg_anchor_sigma_threshold"]);
    if (algMigrated) {
      const moved = hintEl("Moved here from the Distilled Flow panel's own ALG switch — it was "
        + "the same blur, and this one works on every sampler. Your strength and threshold "
        + "came with it; nothing changed about how the scene samples.");
      moved.style.color = "var(--accent)";
      algG.append(moved);
    }
    algG.append(hintEl("The same blur for guide and JoyAI-memory frames is under Experimental, "
      + "with its own strength and window."));
    // Enable and schedule are in the panel above, with the sampler they belong to; this is
    // the optional operation applied to the latent between the two passes.
    const g = group(pane, "Second pass");
    renderKnobList(g, st, ["second_pass_op"]);
    const si = st.project.sampler_inputs || {};
    if (si.second_pass_op && si.second_pass_op !== "none") {
      renderKnobList(g, st, ["second_pass_upscale"]);
      // Both operations are one forward of the same trained latent upsampler segmented
      // detailing uses, so the model choice belongs here too — otherwise picking 'sharpen'
      // silently depends on a file the user was never shown.
      renderDetailUpsampler(pane, si, "Between-pass operation: upsampler model");
    }
  }

  function renderChainContinuity(pane, st) {
    const p = st.project;
    const multiScene = activeSceneCount(p) > 1;
    const cs = normContinuitySettings(p);
    const gs = normGuideSettings(p);

    pane.append(hintEl(cs.auto_enabled
      ? (gs.stack_enabled
        ? "Auto continuity: mid-scene guide only — custom guide stack overrides auto guide lists."
        : "Auto continuity builds the guides for you each run — identity pin, prior-scene guides, mid-scene anchor — based on how the scenes are chained.")
      : "Auto continuity off — use manual Chain Sampler knobs and optional custom guide stack below."));

    const g = group(pane, "Auto continuity");
    const autoCb = el("input"); autoCb.type = "checkbox"; autoCb.checked = cs.auto_enabled;
    autoCb.dataset.k = "cs-auto";
    autoCb.onchange = () => patchContinuitySettings({ auto_enabled: autoCb.checked });
    g.append(toggleField("Auto continuity (recommended)", autoCb,
      "Builds the guides for every run itself, so characters and places hold across scenes. Turn it off only to drive the guide stack by hand."));

    const pinRow = el("div", "sw-row eng-field eng-stack");
    const pinMain = el("div", "sw-row-main");
    pinMain.append(el("div", "sw-row-title", "Identity pin (all scenes)"));
    pinMain.append(el("div", "sw-row-hint",
      "One image every scene is pulled toward, so the same face carries the whole video."));
    pinRow.append(pinMain);
    const pin = window.MediaPicker.create({
      value: cs.identity_pin_ref,
      mediaBin: st.mediaBin,
      noneLabel: "— no identity pin —",
      onChange: (v) => patchContinuitySettings({ identity_pin_ref: v }),
    });
    if (!cs.auto_enabled) pin.classList.add("disabled");
    pinRow.append(pin);
    g.append(pinRow);

    const gAdv = group(pane, "Advanced");
    const mk = (label, checked, key, opts = {}) => {
      const cb = el("input"); cb.type = "checkbox"; cb.checked = checked;
      cb.disabled = !cs.auto_enabled || !!opts.disabled;
      if (opts.title) cb.title = opts.title;
      cb.onchange = () => patchContinuitySettings({ [key]: cb.checked });
      gAdv.append(toggleField(label, cb, opts.hint));
    };
    mk("Borrow prior-scene guides", cs.prior_scene_guides, "prior_scene_guides",
      { hint: "Lets each scene look at frames from the one before it, so the look carries down the chain." });
    mk("Prior guides on solo mixed runs", cs.solo_scene_guides, "solo_scene_guides",
      { hint: "Does the same when you render a single scene out of a mixed timeline. Off, that scene uses only its own anchor." });
    const num = (label, val, key, min, max, step, hint) => {
      const i = el("input"); i.type = "number"; i.min = String(min); i.max = String(max); i.step = String(step);
      i.value = val; i.disabled = !cs.auto_enabled; i.dataset.k = "cs-" + key;
      i.oninput = () => patchContinuitySettings({ [key]: parseFloat(i.value || "0") });
      gAdv.append(field(label, i, hint));
    };
    num("Pin strength", cs.identity_pin_strength, "identity_pin_strength", 0.0, 1.0, 0.05,
      "How hard the identity pin pulls. Higher holds the face better and follows the prompt less.");
    num("Prior guide strength", cs.prior_scene_strength, "prior_scene_strength", 0.0, 1.0, 0.05,
      "How hard borrowed frames from earlier scenes pull. Higher keeps the look, lower lets each scene be its own shot.");
    num("Guide decay / scene", cs.guide_decay, "guide_decay", 0.5, 1, 0.05,
      "How much weaker guides get with each scene further down the chain. 1.0 keeps them at full strength the whole way.");

    const gMan = group(pane, "Manual");
    renderKnobList(gMan, st, CHAIN_VIEW_KNOBS.chain_continuity);
    const stackCb = el("input"); stackCb.type = "checkbox"; stackCb.checked = gs.stack_enabled;
    stackCb.dataset.k = "gs-stack";
    stackCb.onchange = () => patchGuideSettings({ stack_enabled: stackCb.checked });
    gMan.append(toggleField("Custom guide stack", stackCb,
      "Lets you choose each scene's guide frames yourself, in the Scene inspector, instead of having them built for you. "
      + (gs.stack_enabled
        ? "Scenes you leave empty fall back to scene 1's first frame."
        : (cs.auto_enabled
          ? "Auto continuity is supplying them right now."
          : "Without it, multi-scene carry runs get one guide from scene 1."))));
    if (gs.stack_enabled) {
      const accCb = el("input"); accCb.type = "checkbox"; accCb.checked = gs.accumulate_prior;
      accCb.dataset.k = "gs-accum";
      accCb.onchange = () => patchGuideSettings({ accumulate_prior: accCb.checked });
      gMan.append(toggleField("Stack guides from all prior scenes", accCb,
        "Gives each scene the guides of every scene before it, not just its own. Holds the look harder and makes later scenes slower."));
    } else if (multiScene) {
      pane.append(hintEl("Carry i2v guides is auto-enabled for multi-scene runs."));
    }
  }

  function renderChainTiming(pane, st) {
    const si = st.project.sampler_inputs || {};
    const g = group(pane, "Timing");
    renderKnobList(g, st, CHAIN_VIEW_KNOBS.chain_timing);
    const gSeed = group(pane, "Seed");
    const ctrl = el("input");
    ctrl.type = "number"; ctrl.min = "0";
    ctrl.placeholder = "Random each run";
    ctrl.value = si.seed != null ? String(si.seed) : "";
    ctrl.dataset.k = "si-seed";
    ctrl.oninput = () => {
      const raw = ctrl.value.trim();
      if (raw === "") S.unsetSamplerInput("seed");
      else S.setSamplerInput("seed", parseInt(raw, 10));
    };
    gSeed.append(field("Seed (optional)", ctrl,
      "Empty = randomize every Generate. Fixed value = reproducible runs (pair with Same seed per scene)."));
    pane.append(hintEl(
      "Transition duration controls in-decode boundary fades (from transition library). Post-render crossfades are per-clip in the scene inspector."));
  }

  function renderChainKnobsView(pane, st, id, label, hint) {
    if (hint) pane.append(hintEl(hint));
    const g = group(pane, label);
    renderKnobList(g, st, CHAIN_VIEW_KNOBS[id]);
  }

  // ── Segmented detailing: the upsampler model list is live (models/latent_upscale_models
  // on the server), so like the ArcFace projector below it can't be a static SAMPLER_KNOBS
  // combo. The chain sampler node's own spec already carries the choices (with "None").
  let _detailUpsamplerChoices = null;

  async function ensureDetailUpsamplerChoices() {
    if (_detailUpsamplerChoices) return _detailUpsamplerChoices;
    try {
      const spec = await API.nodeSpec("FunPackLTXAVSceneChainSampler");
      const w = (spec?.inputs || []).find((i) => i.name === "detail_upsampler");
      _detailUpsamplerChoices = (w && w.choices) || ["None"];
    } catch (_) {
      _detailUpsamplerChoices = ["None"];
    }
    render();
    return _detailUpsamplerChoices;
  }

  // Two features load this same file: segmented detailing and the second pass's
  // between-pass operation (sharpen / upscale_2x). It used to render only under
  // Experimental and only while detailing was on, so someone using 'sharpen' alone got no
  // upsampler control and no warning that a ~1 GB download was involved.
  function renderDetailUpsampler(pane, si, title) {
    const g = group(pane, title || "Latent upsampler model");
    if (!_detailUpsamplerChoices) {
      g.append(hintEl("Loading upsampler list…"));
      ensureDetailUpsamplerChoices();
      return;
    }
    const sel = el("select"); sel.dataset.k = "si-detail_upsampler";
    const cur = si.detail_upsampler && si.detail_upsampler !== "None" ? si.detail_upsampler : "auto";
    _detailUpsamplerChoices.forEach((c) => {
      const o = el("option", null, c); o.value = c;
      if (c === cur) o.selected = true;
      sel.append(o);
    });
    sel.onchange = () => S.setSamplerInputNow(sel.value);
    // An upsampler has to take latents the same width as the model's, so on H3 the LTX
    // file is not a fallback and 'auto' deliberately will not fetch it. Promising a
    // download that cannot happen sends people to wait at the console for nothing.
    const h3 = !!(window.PipelineCaps && window.PipelineCaps.isH3(S.get()));
    g.append(field("Latent upsampler", sel,
      h3 ? "'auto' uses an installed upsampler; it never downloads the LTX one on H3."
         : "'auto' picks the newest installed one, or downloads the official file (~1 GB, once).",
      h3 ? "MiniMax H3's latents are 24-channel, so it needs an upsampler trained for H3 — the LTX file is 128-channel and is refused by name rather than failing mid-render."
         : "The LTX 2.3 spatial upsampler from models/latent_upscale_models — the official two-stage workflows use the same file."));
    if (_detailUpsamplerChoices.length <= 1) {
      g.append(hintEl(h3
        ? "Nothing installed in models/latent_upscale_models yet. H3 needs a 24-channel upsampler — put one there and it appears in this list; the operation is skipped, with the reason, until then."
        : "Nothing installed in models/latent_upscale_models yet — the first detailed run downloads the official upsampler automatically (watch the ComfyUI console)."));
    }
  }

  // ── Best-FaceID ArcFace projector: the one identity-transfer field that needs a
  // live (server-fetched) choice list, so it can't live in the static SAMPLER_KNOBS combo
  // shape. Reuses the same loras folder listing LoraLoaderModelOnly exposes.
  let _loraChoices = null;

  async function ensureLoraChoices() {
    if (_loraChoices) return _loraChoices;
    try {
      const spec = await API.nodeSpec("LoraLoaderModelOnly");
      const w = (spec?.inputs || []).find((i) => i.name === "lora_name");
      _loraChoices = (w && w.choices) || [];
    } catch (_) {
      _loraChoices = [];
    }
    render();
    return _loraChoices;
  }

  function renderChainExperimental(pane, st) {
    renderChainKnobsView(pane, st, "chain_experimental", "Experimental",
      "Research techniques — off by default; expect quality/overhead trade-offs.");
    const si = st.project.sampler_inputs || {};
    if (!si.identity_transfer_enabled) return;
    const g = group(pane, "Best-FaceID: ArcFace projector");
    if (!_loraChoices) {
      g.append(hintEl("Loading projector list…"));
      ensureLoraChoices();
      return;
    }
    const sel = el("select"); sel.dataset.k = "si-identity_projector";
    const noneOpt = el("option", null, "None"); noneOpt.value = "None";
    if (!si.identity_projector || si.identity_projector === "None") noneOpt.selected = true;
    sel.append(noneOpt);
    _loraChoices.forEach((c) => {
      const o = el("option", null, c); o.value = c;
      if (c === si.identity_projector) o.selected = true;
      sel.append(o);
    });
    sel.onchange = () => S.setSamplerInputNow("identity_projector", sel.value);
    g.append(field("ArcFace projector", sel,
      "Optional second identity channel — the reference tokens above carry most of it even at None."));
  }

  function renderPane(pane, st) {
    const p = st.project;
    if (!p) { pane.append(el("div", "pj-meta", "No project open.")); return; }
    if (st.models?.disable_core) {
      pane.append(hintEl("Built-in FunPack pipeline is disabled — Studio and Chain Sampler settings are unavailable. Tune your workflow in Models & Pipeline."));
      const link = el("button", "btn ghost tiny", "Models & Pipeline…");
      link.type = "button";
      link.onclick = () => window.ModelsModal.open();
      pane.append(link);
      return;
    }
    switch (view) {
      case "studio_refine": return renderStudioRefine(pane, st);
      case "studio_adjust": return renderStudioAdjust(pane, st);
      case "studio_sampler": return renderStudioSampler(pane, st);
      case "chain_continuity": return renderChainContinuity(pane, st);
      case "chain_timing": return renderChainTiming(pane, st);
      case "chain_guidance": return renderChainKnobsView(pane, st, "chain_guidance", "Guidance",
        EASY() ? "Rating-dependent guidance (embed guidance, score slider, output guidance, taste retrieval, "
          + "DynaShift) is hidden in Simple mode — switch to Editor for those." : null);
      case "chain_decode": return renderChainKnobsView(pane, st, "chain_decode", "Decode");
      case "chain_experimental": return renderChainExperimental(pane, st);
      default: return renderOverview(pane, st);
    }
  }

  function renderContent(container, st) {
    // No project / built-in pipeline off: single message pane, no category sidebar.
    if (!st.project || st.models?.disable_core) {
      const solo = el("div", "models-pane");
      renderPane(solo, st);
      container.append(solo);
      return;
    }
    const views = viewList(st);
    if (!views.some((v) => v.id === view)) view = "overview";
    const cols = el("div", "models-cols");
    const side = el("div", "models-side");
    let lastGroup = null;
    views.forEach((v) => {
      if (v.group && v.group !== lastGroup) side.append(el("div", "mn-group", v.group));
      lastGroup = v.group || lastGroup;
      side.append(window.SettingsWindow.navItem({
        icon: v.icon, label: v.title, badge: v.badge,
        active: view === v.id, onClick: () => setView(v.id),
      }));
    });
    const pane = el("div", "models-pane eng-pane");
    renderPane(pane, st);
    cols.append(side, pane);
    container.append(cols);
  }

  // Protect a text/number/range field being typed into from autosave rebuilds, but let
  // checkboxes and selects rebuild immediately so their dependent controls (absolute
  // strength, embed mode/strength, mid-scene strength, …) appear right after the toggle.
  function shouldProtect(a, scope) {
    if (!a || !a.dataset || !a.dataset.k || !scope || !scope.contains(a)) return false;
    if (a.tagName === "TEXTAREA") return true;
    if (a.tagName !== "INPUT") return false;
    const t = (a.type || "text").toLowerCase();
    return t !== "checkbox" && t !== "radio";
  }

  function render() {
    if (!_mounted) return;
    const { content } = _mounted;
    if (_editing) {
      if (shouldProtect(document.activeElement, content)) return;
      _editing = false;
    }
    const prevPane = content.querySelector(".models-pane");
    const scrollTop = prevPane ? prevPane.scrollTop : 0;
    clear(content);
    renderContent(content, S.get());
    const pane = content.querySelector(".models-pane");
    if (pane) pane.scrollTop = scrollTop;
  }

  /** Human name for a category, group included — "Sampler algorithm" alone appears under
   *  more than one group, and a pin's tooltip has to be unambiguous. */
  function viewLabel(id) {
    // Guarded: this runs from mount(), and viewList reads project state that may not be
    // there yet. A throw here would blank the whole section over a tooltip.
    try {
      const found = (viewList(S.get()) || []).find((v) => v.id === id);
      if (!found) return null;
      return found.group ? `${found.group} ▸ ${found.title}` : found.title;
    } catch (_) { return null; }
  }

  function mount(body, ctx) {
    const content = el("div", "models-mount eng-mount");
    body.append(content);
    _mounted = { content };
    // A pinned button can name a category to open straight into. A category that is not in
    // the list for THIS pipeline (Studio panes with no Studio wired) falls back to the
    // overview — the sidebar visibly does not offer it, so the reason is on screen.
    const wanted = ctx && ctx.sub;
    view = (wanted && viewLabel(wanted)) ? wanted : "overview";

    content.addEventListener("focusin", (e) => {
      const t = e.target;
      if (t && t.dataset && t.dataset.k) _editing = true;
    });
    content.addEventListener("focusout", (e) => {
      const t = e.target;
      if (!(t && t.dataset && t.dataset.k)) return;
      _editing = false;
      setTimeout(() => { if (!_editing) render(); }, 60);
    });

    unsub = S.subscribe(() => render());
    const offMode = window.FunPackMode?.onChange(() => render());
    render();
    return () => {
      if (offMode) offMode();
      if (unsub) { unsub(); unsub = null; }
      _mounted = null;
      _editing = false;
    };
  }

  window.SettingsWindow.register({
    id: "engine", group: "Generation", order: 1, title: "Engine", flush: true,
    subtitle: "FunPack Studio, Chain Sampler, and continuity for the open project.",
    keywords: "studio chain sampler refinement key seed cfg guidance continuity guides presets "
      + "embed dynashift joyai decode transition overlap steer adjustments temporal",
    iconBg: "linear-gradient(180deg,#ffb64d,#e07f1f)",
    icon: '<svg viewBox="0 0 16 16" width="13" height="13"><path d="M9.2 1.3 3 9h4.1l-1 5.7L12.9 7H8.5l.7-5.7z" fill="#fff"/></svg>',
    mount,
    // Pin the CATEGORY that is open, not the section — "Engine" is one click from anywhere
    // already; "Sampler algorithm" is the one that costs a hunt through the sidebar.
    pinTarget: () => {
      if (!view || view === "overview") return null;
      const label = viewLabel(view);
      if (!label) return null;
      return { kind: "section", id: "engine", sub: view, label: `Engine ▸ ${label}` };
    },
  });

  window.EngineSettingsModal = {
    open: () => window.SettingsWindow.open("engine"),
    close: () => window.SettingsWindow.close(),
  };
})();
