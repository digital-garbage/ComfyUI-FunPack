// Easy Gen's own slimmed "Generation" Settings section — modeled on the
// field(label, control, hint) row pattern in movie_editor/frontend/engine_settings.js,
// but limited to plain Project-level fields (no scene-chain/continuity/overlay
// knobs, which only make sense with multiple scenes in the full Editor).
(function () {
  const { el } = window.dom;

  function field(labelText, control, hint) {
    const row = el("div", "sw-row eng-field");
    const main = el("div", "sw-row-main");
    main.append(el("div", "sw-row-title", labelText));
    if (hint) main.append(el("div", "sw-row-hint", hint));
    row.append(main, control);
    return row;
  }

  function numberInput(value, onChange, opts = {}) {
    const inp = el("input", "sw-input");
    inp.type = "number";
    if (opts.min != null) inp.min = opts.min;
    if (opts.step != null) inp.step = opts.step;
    inp.value = value;
    inp.style.width = "90px";
    inp.onchange = () => onChange(Number(inp.value));
    return inp;
  }

  function textInput(value, onChange, placeholder) {
    const inp = el("input", "sw-input");
    inp.type = "text";
    inp.value = value || "";
    if (placeholder) inp.placeholder = placeholder;
    inp.style.width = "220px";
    inp.onchange = () => onChange(inp.value);
    return inp;
  }

  function mount(body) {
    const S = window.Store;
    const wrap = el("div", "sw-stack");
    body.append(wrap);

    function render() {
      body.innerHTML = "";
      body.append(wrap);
      wrap.innerHTML = "";
      const st = S.get();
      const p = st.project;
      if (!p) {
        wrap.append(el("div", "sw-hint", "Open or create a project first."));
        return;
      }
      const commit = async (mut) => { mut(p); await S.save(); };

      wrap.append(el("div", "sw-rows-label", "Output"));
      const out = el("div", "sw-rows");
      out.append(field("Width", numberInput(p.width, (v) => commit((pr) => pr.width = v), { min: 64, step: 16 })));
      out.append(field("Height", numberInput(p.height, (v) => commit((pr) => pr.height = v), { min: 64, step: 16 })));
      out.append(field("Frames", numberInput(p.num_frames_per_scene, (v) => commit((pr) => pr.num_frames_per_scene = v), { min: 9, step: 8 }),
        "Length of the generated clip, in frames."));
      out.append(field("Frame rate", numberInput(p.frame_rate, (v) => commit((pr) => pr.frame_rate = v), { min: 1 })));
      wrap.append(out);

      wrap.append(el("div", "sw-rows-label", "Prompt"));
      const pr = el("div", "sw-rows");
      pr.append(field("Negative prompt", textInput(p.negative_prompt, (v) => commit((proj) => proj.negative_prompt = v)),
        "Passed to the negative conditioning for every run."));
      wrap.append(pr);

      wrap.append(el("div", "sw-hint",
        "Studio/Chain Sampler/continuity controls, including Seed, live under Engine ▸ Timing "
        + "& Seed (empty = a new random seed every Generate). Studio always runs in Prompt-only "
        + "mode here — no refinement key, since Easy Gen has no rating UI to train one. For the "
        + "full learned refiner, use the Cutting Room or the ComfyUI node graph."));
    }

    const unsub = S.subscribe(render);
    render();
    return () => unsub();
  }

  window.SettingsWindow.register({
    id: "generation", group: "Generation", order: 1, title: "Generation",
    subtitle: "Output size, length, and negative prompt for this project.",
    keywords: "width height frames frame rate negative prompt generation seed engine",
    iconBg: "linear-gradient(180deg,#ffb64d,#e07f1f)",
    icon: '<svg viewBox="0 0 16 16" width="13" height="13"><path d="M9.2 1.3 3 9h4.1l-1 5.7L12.9 7H8.5l.7-5.7z" fill="#fff"/></svg>',
    mount,
  });
})();
