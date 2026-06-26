// Studio sampler settings panel — ported from web/funpack_studio.js renderSampler().
// Renders two pass sections (High / Low) into a given container element.
// Exposes window.SamplerPanel.render(container, samplers, save, saveNow).
//   save(samplers)    — debounced/quiet, for numeric inputs
//   saveNow(samplers) — immediate notify + re-render, for selects / checkboxes
(function () {
  const { el } = window.dom;

  const SAMPLER_TYPES = ["Hybrid Euler 2S", "Distilled Flow", "KSampler"];
  const VELOCITY_BIAS_MODES = ["off", "capture", "apply", "capture_and_apply"];
  const MOTION_PULSE_MODES = ["off", "balanced", "aggressive", "custom"];
  const KSAMPLER_NAMES = ["euler", "euler_ancestral", "dpm_2", "dpm_2_ancestral", "dpmpp_2m", "dpmpp_sde", "ddim", "uni_pc"];

  function defaultHybrid() {
    return {
      eta: 1.0, eta_final: 1.0, normalize_strength: 0.0, normalize_start_sigma: 0.9,
      s_noise: 1.0, high_quality_pct: 0.35, correction_blend: 1.0, quality_sharpness: 0.0,
      motion_pulse_mode: "off", motion_pulse_start_pct: 0.3, motion_pulse_count: 2,
      motion_pulse_spacing_pct: 0.22, motion_pulse_strength: 0.85,
      velocity_bias_mode: "off", velocity_bias_strength: 0.0,
      velocity_bias_source: "mean", velocity_refinement_key: "default",
      rescue_mode: false, rescue_threshold: 0.15, rescue_strength: 0.2,
    };
  }

  function defaultDistilled() {
    return {
      order: 2, final_correction_steps: 1, ab2_ramp: false,
      normalize_strength: 0.0, normalize_start_sigma: 0.9, s_noise: 0.0,
      velocity_bias_mode: "off", velocity_bias_strength: 0.0,
      velocity_bias_source: "mean", velocity_refinement_key: "default",
      rescue_mode: false, rescue_threshold: 0.15, rescue_strength: 0.2,
      alg_enabled: false, alg_strength: 2.5, alg_sigma_threshold: 0.94,
    };
  }

  function defaultPass(type) {
    return {
      type: type || "Hybrid Euler 2S", sigmas: "",
      hybrid: defaultHybrid(), distilled: defaultDistilled(),
      ksampler_name: "euler",
    };
  }

  function defaultSamplers() {
    return { high: defaultPass("Hybrid Euler 2S"), low: defaultPass("Distilled Flow") };
  }

  function deepMerge(def, over) {
    if (!over || typeof over !== "object") return def;
    const out = Object.assign({}, def);
    for (const k of Object.keys(over)) {
      if (over[k] !== null && typeof over[k] === "object" && !Array.isArray(over[k]) && typeof def[k] === "object")
        out[k] = deepMerge(def[k], over[k]);
      else if (over[k] !== undefined)
        out[k] = over[k];
    }
    return out;
  }

  // ── micro DOM helpers (match inspector style) ────────────────────────────────

  function row(container, labelText, ctrl) {
    const f = el("label", "field");
    f.append(el("span", null, labelText));
    f.append(ctrl);
    container.append(f);
    return ctrl;
  }

  function hint(container, text) {
    container.append(el("div", "insp-hint", text));
  }

  function sectionTag(container, text) {
    const t = el("div", "insp-tag sp-sub"); t.textContent = text; container.append(t);
  }

  function numCtrl(val, min, max, step, dk, onChange) {
    const i = el("input"); i.type = "number";
    i.value = Number.isFinite(+val) ? +val : 0;
    if (min != null) i.min = String(min);
    if (max != null) i.max = String(max);
    if (step != null) i.step = String(step);
    if (dk) i.dataset.k = dk;
    i.oninput = () => onChange(parseFloat(i.value) || 0);
    return i;
  }

  function intCtrl(val, min, max, dk, onChange) {
    const i = el("input"); i.type = "number"; i.step = "1";
    i.value = Math.round(+val) || 0;
    if (min != null) i.min = String(min);
    if (max != null) i.max = String(max);
    if (dk) i.dataset.k = dk;
    i.oninput = () => onChange(parseInt(i.value) || 0);
    return i;
  }

  function selCtrl(values, selected, dk, onChange) {
    const s = el("select");
    for (const v of values) {
      const o = el("option", null, v); o.value = v; if (v === selected) o.selected = true; s.append(o);
    }
    if (dk) s.dataset.k = dk;
    s.onchange = () => onChange(s.value);
    return s;
  }

  function checkCtrl(checked, dk, onChange) {
    const i = el("input"); i.type = "checkbox"; i.checked = !!checked; i.style.width = "auto";
    if (dk) i.dataset.k = dk;
    i.onchange = () => onChange(i.checked);
    return i;
  }

  function textCtrl(value, placeholder, dk, onChange) {
    const i = el("input"); i.type = "text"; i.value = value || ""; i.placeholder = placeholder || "";
    if (dk) i.dataset.k = dk;
    i.oninput = () => onChange(i.value);
    return i;
  }

  // ── velocity-bias / rescue block (shared by Hybrid and Distilled) ──

  function renderVelocityBlock(c, dk, sub, save, saveNow) {
    const vbm = selCtrl(VELOCITY_BIAS_MODES, sub.velocity_bias_mode || "off", dk + "-vbm",
      (v) => { sub.velocity_bias_mode = v; saveNow(); });
    row(c, "velocity bias mode", vbm);
    if (sub.velocity_bias_mode && sub.velocity_bias_mode !== "off") {
      row(c, "velocity bias strength",
        numCtrl(sub.velocity_bias_strength, 0, 3, 0.05, dk + "-vbs",
          (v) => { sub.velocity_bias_strength = v; save(); }));
      row(c, "bias source",
        selCtrl(["mean", "nearest"], sub.velocity_bias_source || "mean", dk + "-vbsrc",
          (v) => { sub.velocity_bias_source = v; save(); }));
      row(c, "velocity key",
        textCtrl(sub.velocity_refinement_key, "default", dk + "-vrk",
          (v) => { sub.velocity_refinement_key = v; save(); }));
    }
    const rsm = selCtrl(["off", "on"], sub.rescue_mode ? "on" : "off", dk + "-rsm",
      (v) => { sub.rescue_mode = (v === "on"); saveNow(); });
    row(c, "rescue mode", rsm);
    if (sub.rescue_mode) {
      row(c, "rescue threshold",
        numCtrl(sub.rescue_threshold, 0, 1, 0.01, dk + "-rst",
          (v) => { sub.rescue_threshold = v; save(); }));
      row(c, "rescue strength",
        numCtrl(sub.rescue_strength, 0, 0.5, 0.01, dk + "-rss",
          (v) => { sub.rescue_strength = v; save(); }));
      if (!sub.velocity_bias_mode || sub.velocity_bias_mode === "off") {
        row(c, "bias source",
          selCtrl(["mean", "nearest"], sub.velocity_bias_source || "mean", dk + "-rsrc",
            (v) => { sub.velocity_bias_source = v; save(); }));
      }
    }
  }

  // ── per-pass rendering ────────────────────────────────────────────────────────

  function renderPass(container, passKey, label, cfg, save, saveNow) {
    sectionTag(container, label);

    const typeCtrl = selCtrl(SAMPLER_TYPES, cfg.type || "Hybrid Euler 2S", "sp-" + passKey + "-type",
      (v) => { cfg.type = v; saveNow(); });
    row(container, "Sampler", typeCtrl);

    const sigCtrl = textCtrl(cfg.sigmas, "e.g. 20.0, 14.0, 8.0, 4.0, 1.0, 0.0", "sp-" + passKey + "-sig",
      (v) => { cfg.sigmas = v; save(); });
    row(container, "Sigmas", sigCtrl);
    hint(container, "Comma-separated floats. Leave empty to let the sampler decide.");

    const dk = "sp-" + passKey;

    if (cfg.type === "Hybrid Euler 2S") {
      const hc = cfg.hybrid;
      sectionTag(container, "Hybrid Euler 2S");

      row(container, "normalize strength",
        numCtrl(hc.normalize_strength, 0, 1, 0.05, dk + "-hn",
          (v) => { hc.normalize_strength = v; save(); }));
      hint(container, "Video-only latent normalization (anti-overbake). 0 = off. 0.5 = gentle. Audio untouched.");
      row(container, "normalize start sigma",
        numCtrl(hc.normalize_start_sigma, 0, 1, 0.025, dk + "-hns",
          (v) => { hc.normalize_start_sigma = v; save(); }));

      // Variability macro
      const macroWrap = el("div", "sp-macro");
      const macroLbl = el("div", "sp-macro-title", "Variability macro");
      macroWrap.append(macroLbl);
      const presetRow = el("div", "sp-macro-presets");
      [["Chill", 0.1], ["Spicy", 0.5], ["Chaos", 0.9]].forEach(([name, val]) => {
        const b = el("button", "btn ghost tiny", name); b.type = "button";
        b.onclick = () => { applyVariabilityMacro(hc, val); saveNow(); };
        presetRow.append(b);
      });
      macroWrap.append(presetRow);
      const valDisplay = el("span", "sp-macro-val", (+hc.variability || 0).toFixed(2));
      const slider = el("input"); slider.type = "range"; slider.min = "0"; slider.max = "1"; slider.step = "0.05";
      slider.value = String(Number.isFinite(+hc.variability) ? +hc.variability : 0.5);
      slider.oninput = () => { valDisplay.textContent = (+slider.value).toFixed(2); };
      slider.onchange = () => { applyVariabilityMacro(hc, parseFloat(slider.value)); saveNow(); };
      const sliderWrap = el("div", "sp-macro-slider"); sliderWrap.append(slider, valDisplay);
      macroWrap.append(sliderWrap);
      hint(macroWrap, "Chill = clean + reproducible · Spicy = lively · Chaos = wild. Sets eta / velocity bias / motion pulses below.");
      container.append(macroWrap);

      [["eta", 0, 1, 0.01, dk + "-eta"],
       ["eta_final", 0, 1, 0.01, dk + "-etaf"],
       ["s_noise", 0, 10, 0.01, dk + "-sn"],
       ["high_quality_pct", 0, 1, 0.01, dk + "-hqp"],
       ["correction_blend", 0, 1, 0.01, dk + "-cb"],
       ["quality_sharpness", 0, 1, 0.01, dk + "-qs"]].forEach(([k, mn, mx, st, dkk]) => {
        row(container, k.replace(/_/g, " "),
          numCtrl(hc[k], mn, mx, st, dkk, (v) => { hc[k] = v; save(); }));
      });

      const mpm = selCtrl(MOTION_PULSE_MODES, hc.motion_pulse_mode || "off", dk + "-mpm",
        (v) => { hc.motion_pulse_mode = v; saveNow(); });
      row(container, "motion pulse mode", mpm);
      if (hc.motion_pulse_mode !== "off") {
        [["motion_pulse_start_pct", 0, 0.95, 0.01, dk + "-mpsp"],
         ["motion_pulse_spacing_pct", 0.04, 0.45, 0.01, dk + "-mpspg"],
         ["motion_pulse_strength", 0, 1, 0.01, dk + "-mpstr"]].forEach(([k, mn, mx, st, dkk]) => {
          row(container, k.replace(/_/g, " "),
            numCtrl(hc[k], mn, mx, st, dkk, (v) => { hc[k] = v; save(); }));
        });
        row(container, "motion pulse count",
          intCtrl(hc.motion_pulse_count, 1, 6, dk + "-mpc",
            (v) => { hc.motion_pulse_count = v; save(); }));
      }

      renderVelocityBlock(container, dk + "-h", hc, save, saveNow);

    } else if (cfg.type === "Distilled Flow") {
      const dc = cfg.distilled;
      sectionTag(container, "Distilled Flow");
      row(container, "order",
        intCtrl(dc.order, 1, 2, dk + "-dord", (v) => { dc.order = v; save(); }));
      row(container, "AB2 ramp",
        checkCtrl(!!dc.ab2_ramp, dk + "-dab2", (v) => { dc.ab2_ramp = v; save(); }));
      hint(container, "Graduated 2nd order: ramp AB2 0->1 across schedule. Helps low-step distilled runs.");
      row(container, "normalize strength",
        numCtrl(dc.normalize_strength, 0, 1, 0.05, dk + "-dn",
          (v) => { dc.normalize_strength = v; save(); }));
      hint(container, "Video-only latent normalization. 0 = plain euler. Audio untouched.");
      row(container, "normalize start sigma",
        numCtrl(dc.normalize_start_sigma, 0, 1, 0.025, dk + "-dns",
          (v) => { dc.normalize_start_sigma = v; save(); }));
      row(container, "final correction steps",
        intCtrl(dc.final_correction_steps, 0, 3, dk + "-dfcs",
          (v) => { dc.final_correction_steps = v; save(); }));
      row(container, "s_noise",
        numCtrl(dc.s_noise, 0, 0.5, 0.01, dk + "-dsn",
          (v) => { dc.s_noise = v; save(); }));
      renderVelocityBlock(container, dk + "-d", dc, save, saveNow);

      sectionTag(container, "ALG (experimental)");
      row(container, "enabled",
        checkCtrl(!!dc.alg_enabled, dk + "-dalge", (v) => { dc.alg_enabled = v; saveNow(); }));
      hint(container, "Adaptive Low-Pass Guidance (arXiv:2506.08456). Blurs the i2v anchor frame during the earliest steps so the model can't shortcut to a near-static video that just matches it. No effect without an i2v anchor.");
      if (dc.alg_enabled) {
        row(container, "blur strength",
          numCtrl(dc.alg_strength, 1.01, 4, 0.1, dk + "-dalgs",
            (v) => { dc.alg_strength = v; save(); }));
        row(container, "sigma threshold",
          numCtrl(dc.alg_sigma_threshold, 0.5, 0.999, 0.005, dk + "-dalgt",
            (v) => { dc.alg_sigma_threshold = v; save(); }));
      }

    } else if (cfg.type === "KSampler") {
      sectionTag(container, "KSampler");
      row(container, "sampler name",
        selCtrl(KSAMPLER_NAMES, cfg.ksampler_name || "euler", dk + "-ks",
          (v) => { cfg.ksampler_name = v; save(); }));
    }
  }

  // ── variability macro (matches Studio exactly) ────────────────────────────────

  function applyVariabilityMacro(hc, v) {
    v = Math.max(0, Math.min(1, v));
    hc.variability = +v.toFixed(2);
    hc.eta = +v.toFixed(2);
    hc.eta_final = +(Math.max(0, (v - 0.6) / 0.4) * 0.4).toFixed(2);
    hc.s_noise = 1.0;
    hc.high_quality_pct = 0.35;
    hc.quality_sharpness = +(0.15 + v * 0.25).toFixed(2);
    if (v < 0.3) {
      hc.velocity_bias_mode = "off";
    } else {
      hc.velocity_bias_mode = "apply";
      hc.velocity_bias_source = "nearest";
      hc.velocity_bias_strength = +(((v - 0.3) / 0.7) * 1.2).toFixed(2);
    }
    hc.motion_pulse_mode = v < 0.7 ? "off" : (v < 0.9 ? "balanced" : "aggressive");
  }

  // ── public API ────────────────────────────────────────────────────────────────

  function render(container, samplers, save, saveNow) {
    const s = deepMerge(defaultSamplers(), samplers || {});

    function mkSave(quiet) {
      return () => quiet ? save(s) : saveNow(s);
    }

    // Only the high-pass sampler is wired in the Movie Editor graph
    // (studio outputs 4+5 → chain sampler; outputs 6+7 are not connected).
    renderPass(container, "high", "Sampler", s.high, mkSave(true), mkSave(false));
  }

  window.SamplerPanel = { render, defaultSamplers, defaultPass };
})();
