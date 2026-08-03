import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "FunPackStudio";
const NONE_SENTINEL = "-None-";
const HIDDEN_WIDGETS = new Set(["studio_settings", "adjustments"]);
const LORA_TYPES = ["general", "action", "style", "quality", "character"];
const ADVISOR_DTYPES = ["bfloat16", "float16", "float32"];
const REFINER_MODES = ["Refine", "Prompt only", "Learning"];
const ADVISOR_MODES = ["Off", "Only diagnostics", "Only prompt", "Full"];
const TEMPORAL_STYLES = ["natural", "auto", "accelerate", "decelerate", "loop", "freeze", "pulse", "rapid_start", "rapid_end", "rapid_start_end"];
const CATEGORY_ORDER = ["action", "camera", "subject", "appearance", "environment", "style", "quality", "details"];
const TABS = ["Session", "Shortcuts", "Split markers", "Refiner", "Advisor", "LoRA", "Sampler", "Adjustments", "Timeline"];
const SAMPLER_TYPES = ["Hybrid Euler 2S", "Distilled Flow", "KSampler"];
const MOTION_PULSE_MODES = ["off", "balanced", "aggressive", "custom"];
const VELOCITY_BIAS_MODES = ["off", "capture", "apply", "capture_and_apply"];
// Fallback only — the real list is whatever THIS ComfyUI has. The backend passes the name
// straight to comfy.samplers.sampler_object(), so a hardcoded set hides samplers it can
// already run (res_multistep, which ComfyUI's own MiniMax H3 templates use, was missing).
const KSAMPLER_NAMES_FALLBACK = ["euler", "euler_ancestral", "dpm_2", "dpm_2_ancestral", "dpmpp_2m", "dpmpp_sde", "ddim", "uni_pc"];
let KSAMPLER_NAMES = KSAMPLER_NAMES_FALLBACK;
api.fetchApi("/object_info/KSampler")
  .then((r) => r.json())
  .then((defs) => {
    const names = defs?.KSampler?.input?.required?.sampler_name?.[0];
    if (Array.isArray(names) && names.length) KSAMPLER_NAMES = names;
  })
  .catch(() => {});

let activePanel = null;
let studioShortcutData = null;
let studioTransitionData = null;

// ─── helpers ─────────────────────────────────────────────────────────────────

function widgetByName(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function getSettings(node) {
  try { return JSON.parse(String(widgetByName(node, "studio_settings")?.value || "{}")); }
  catch { return {}; }
}

function saveSettings(node, settings) {
  const w = widgetByName(node, "studio_settings");
  if (w) w.value = JSON.stringify(settings);
  node.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}

function getAdjustments(node) {
  try { const v = JSON.parse(String(widgetByName(node, "adjustments")?.value || "[]")); return Array.isArray(v) ? v : []; }
  catch { return []; }
}

function saveAdjustments(node, items) {
  const w = widgetByName(node, "adjustments");
  if (w) w.value = JSON.stringify(items);
  node.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}

function deepMerge(target, src) {
  const out = { ...target };
  for (const k of Object.keys(src)) {
    if (src[k] && typeof src[k] === "object" && !Array.isArray(src[k]) && typeof target[k] === "object")
      out[k] = deepMerge(target[k], src[k]);
    else
      out[k] = src[k];
  }
  return out;
}

function defaultSettings() {
  return {
    refinement_key: "",
    overrides: { refinement_key: false, feedback_prompt: false, user_intent_prompt: false, negative_prompt: false },
    refiner: { mode: "Refine", advisor_mode: "Off", advisor_thinking: true, im_feeling_lucky: false, reset_session: false, feedback_prompt: "", user_intent_prompt_override: "", negative_prompt: "", temporal_style: "natural", split_by_transitions: false, split_transition_placement: "start", reference_injection: false, vision_conditioning: true, value_guidance: true, steer_mode: "relative", absolute_strength: 0.6 },
    advisor_llm: { enabled: false, model_path: "huihui-ai/Huihui-Qwen3-8B-abliterated-v2", dtype: "bfloat16" },
    loras: [],
    loras_config: { mode: "ltx2", per_block: false },
    samplers: {
      high: { type: "Hybrid Euler 2S", sigmas: "", hybrid: { eta: 1.0, eta_final: 1.0, normalize_strength: 0.0, normalize_start_sigma: 0.9, s_noise: 1.0, high_quality_pct: 0.35, correction_blend: 1.0, quality_sharpness: 0.0, motion_pulse_mode: "off", motion_pulse_start_pct: 0.3, motion_pulse_count: 2, motion_pulse_spacing_pct: 0.22, motion_pulse_strength: 0.85, velocity_bias_mode: "off", velocity_bias_strength: 0.0, velocity_bias_source: "mean", velocity_refinement_key: "default", rescue_mode: false, rescue_threshold: 0.15, rescue_strength: 0.2 }, distilled: { order: 2, final_correction_steps: 1, ab2_ramp: false, normalize_strength: 0.0, normalize_start_sigma: 0.9, s_noise: 0.0, velocity_bias_mode: "off", velocity_bias_strength: 0.0, velocity_bias_source: "mean", velocity_refinement_key: "default", rescue_mode: false, rescue_threshold: 0.15, rescue_strength: 0.2 }, normalizing: { normalize_strength: 0.5, normalize_start_sigma: 0.9, velocity_bias_mode: "off", velocity_bias_strength: 0.0, velocity_bias_source: "mean", velocity_refinement_key: "default", rescue_mode: false, rescue_threshold: 0.15, rescue_strength: 0.2 }, ksampler_name: "euler" },
      low:  { type: "Distilled Flow",  sigmas: "", hybrid: { eta: 1.0, eta_final: 1.0, normalize_strength: 0.0, normalize_start_sigma: 0.9, s_noise: 1.0, high_quality_pct: 0.35, correction_blend: 1.0, quality_sharpness: 0.0, motion_pulse_mode: "off", motion_pulse_start_pct: 0.3, motion_pulse_count: 2, motion_pulse_spacing_pct: 0.22, motion_pulse_strength: 0.85, velocity_bias_mode: "off", velocity_bias_strength: 0.0, velocity_bias_source: "mean", velocity_refinement_key: "default", rescue_mode: false, rescue_threshold: 0.15, rescue_strength: 0.2 }, distilled: { order: 2, final_correction_steps: 1, ab2_ramp: false, normalize_strength: 0.0, normalize_start_sigma: 0.9, s_noise: 0.0, velocity_bias_mode: "off", velocity_bias_strength: 0.0, velocity_bias_source: "mean", velocity_refinement_key: "default", rescue_mode: false, rescue_threshold: 0.15, rescue_strength: 0.2 }, normalizing: { normalize_strength: 0.5, normalize_start_sigma: 0.9, velocity_bias_mode: "off", velocity_bias_strength: 0.0, velocity_bias_source: "mean", velocity_refinement_key: "default", rescue_mode: false, rescue_threshold: 0.15, rescue_strength: 0.2 }, ksampler_name: "euler" },
    },
  };
}

function overrideToggle(settings, key, label) {
  if (!settings.overrides) settings.overrides = {};
  const { wrap, inp } = toggleEl(settings.overrides[key] || false, label);
  inp.addEventListener("change", () => { settings.overrides[key] = inp.checked; });
  wrap.className += " funpack-studio-override-toggle";
  return wrap;
}

function linkedRefinementKey(node) {
  const input = (node.inputs || []).find((i) => i.name === "refinement_key_input");
  const linkId = Array.isArray(input?.link) ? input.link[0] : input?.link;
  if (linkId == null) return "";
  const link = app.graph?.links?.[linkId];
  const src = link ? app.graph?.getNodeById?.(link.origin_id) : null;
  if (!src) return "";
  const sel = String(widgetByName(src, "refinement_key")?.value || "").trim();
  const typed = String(widgetByName(src, "key_name")?.value || "").trim();
  return (sel && sel !== NONE_SENTINEL ? sel : typed) || "";
}

function hideWidget(widget) {
  if (!widget) return;
  widget.__funpackHidden = true;
  widget.hidden = true;
  widget.options = widget.options || {};
  widget.options.hidden = true;
  widget.computeSize = () => [0, -4];
  widget.computedHeight = 0;
  widget.type = "hidden";
  for (const key of ["element", "inputEl", "textElement", "parentEl"]) {
    const el = widget[key];
    if (el?.style) { el.style.display = "none"; el.style.visibility = "hidden"; el.style.pointerEvents = "none"; }
    if (el) el.hidden = true;
  }
}

// ─── API calls ────────────────────────────────────────────────────────────────

async function fetchShortcuts() {
  try {
    const params = new URLSearchParams({ cache_bust: Date.now() });
    const res = await api.fetchApi(`/funpack/shortcuts?${params}`, { cache: "no-store" });
    if (!res.ok) return null;
    studioShortcutData = await res.json();
    return studioShortcutData;
  } catch { return null; }
}

async function saveShortcut(payload) {
  const res = await api.fetchApi("/funpack/shortcuts/shortcut", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "save", ...payload }),
  });
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || `HTTP ${res.status}`); }
  studioShortcutData = await res.json();
  return studioShortcutData;
}

async function deleteShortcut(name) {
  const res = await api.fetchApi("/funpack/shortcuts/shortcut", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "delete", name }),
  });
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || `HTTP ${res.status}`); }
  studioShortcutData = await res.json();
  return studioShortcutData;
}

async function exportShortcuts() {
  const params = new URLSearchParams({ cache_bust: Date.now() });
  const res = await api.fetchApi(`/funpack/shortcuts/export?${params}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "funpack_shortcuts.json";
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function importShortcuts(onDone, onError) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".json,application/json";
  input.onchange = async () => {
    const file = input.files?.[0];
    if (!file) return;
    try {
      const data = JSON.parse(await file.text());
      const res = await api.fetchApi("/funpack/shortcuts/import", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || `HTTP ${res.status}`); }
      studioShortcutData = await res.json();
      onDone?.();
    } catch (e) {
      onError?.(e);
    }
  };
  input.click();
}

async function fetchTransitions() {
  try {
    const params = new URLSearchParams({ cache_bust: Date.now() });
    const res = await api.fetchApi(`/funpack/transitions?${params}`, { cache: "no-store" });
    if (!res.ok) return null;
    studioTransitionData = await res.json();
    return studioTransitionData;
  } catch { return null; }
}

async function saveTransition(payload) {
  const res = await api.fetchApi("/funpack/transitions/transition", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "save", ...payload }),
  });
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || `HTTP ${res.status}`); }
  studioTransitionData = await res.json();
  return studioTransitionData;
}

async function deleteTransition(name) {
  const res = await api.fetchApi("/funpack/transitions/transition", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "delete", name }),
  });
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || `HTTP ${res.status}`); }
  studioTransitionData = await res.json();
  return studioTransitionData;
}

async function exportTransitions() {
  const params = new URLSearchParams({ cache_bust: Date.now() });
  const res = await api.fetchApi(`/funpack/transitions/export?${params}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "funpack_promptsplit.json";
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function importTransitions(onDone, onError) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".json,application/json";
  input.onchange = async () => {
    const file = input.files?.[0];
    if (!file) return;
    try {
      const data = JSON.parse(await file.text());
      const res = await api.fetchApi("/funpack/transitions/import", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || `HTTP ${res.status}`); }
      studioTransitionData = await res.json();
      onDone?.();
    } catch (e) {
      onError?.(e);
    }
  };
  input.click();
}

async function exportValueFunction(key) {
  if (!key) throw new Error("No refinement key set.");
  const params = new URLSearchParams({ key, cache_bust: Date.now() });
  const res = await api.fetchApi(`/funpack/value_function/export?${params}`, { cache: "no-store" });
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || `HTTP ${res.status}`); }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `funpack_vf_${key}.pt`;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function importValueFunction(key, onDone, onError) {
  if (!key) { onError?.(new Error("No refinement key set.")); return; }
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".pt";
  input.onchange = async () => {
    const file = input.files?.[0];
    if (!file) return;
    try {
      const buf = await file.arrayBuffer();
      const params = new URLSearchParams({ key });
      const res = await api.fetchApi(`/funpack/value_function/import?${params}`, {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: buf,
      });
      if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || `HTTP ${res.status}`); }
      const data = await res.json();
      onDone?.(data);
    } catch (e) {
      onError?.(e);
    }
  };
  input.click();
}

async function fetchLoras() {
  try {
    const res = await api.fetchApi("/funpack/available_loras", { cache: "no-store" });
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data.loras) ? data.loras : [];
  } catch { return []; }
}

async function fetchPhraseMemory(key) {
  if (!key) return [];
  try {
    const res = await api.fetchApi(`/funpack/phrase_memory?key=${encodeURIComponent(key)}`, { cache: "no-store" });
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data.phrases) ? data.phrases : [];
  } catch { return []; }
}

// ─── DOM helpers ──────────────────────────────────────────────────────────────

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function btn(label, cls = "") {
  const b = el("button", `funpack-studio-btn ${cls}`.trim(), label);
  b.type = "button";
  return b;
}

function textInput(value, placeholder, cls = "") {
  const i = el("input", `funpack-studio-input ${cls}`.trim());
  i.type = "text";
  i.value = String(value || "");
  i.placeholder = placeholder || "";
  return i;
}

function numInput(value, min, max, step, cls = "") {
  const i = el("input", `funpack-studio-input ${cls}`.trim());
  i.type = "number";
  i.value = String(Number.isFinite(+value) ? +value : 0);
  i.min = String(min); i.max = String(max); i.step = String(step);
  return i;
}

function selectEl(values, selected, cls = "") {
  const s = el("select", `funpack-studio-select ${cls}`.trim());
  for (const v of values) {
    const o = el("option", "", v);
    o.value = v;
    o.selected = v === selected;
    s.append(o);
  }
  return s;
}

function toggleEl(checked, label) {
  const wrap = el("label", "funpack-studio-toggle");
  const inp = el("input");
  inp.type = "checkbox";
  inp.checked = Boolean(checked);
  const span = el("span", "", label);
  wrap.append(inp, span);
  return { wrap, inp };
}

function row(label, control, cls = "") {
  const r = el("div", `funpack-studio-row ${cls}`.trim());
  const lbl = el("span", "funpack-studio-row-label", label);
  r.append(lbl, control);
  return r;
}

function sectionTitle(text) { return el("div", "funpack-studio-section-title", text); }

// One macro that orchestrates the stochastic-variability knobs of the Hybrid sampler.
// Chill = clean/reproducible, Chaos = wild. Writes the existing hybrid fields (still
// individually editable afterward). Keeps high_quality_pct > 0 deliberately: disabling
// the deterministic finisher while ancestral noise runs produces a noisy mess.
function applyVariabilityMacro(hc, v) {
  v = Math.max(0, Math.min(1, v));
  hc.variability = +v.toFixed(2);
  hc.eta = +v.toFixed(2);                                   // ancestral noise = video variability
  hc.eta_final = +(Math.max(0, (v - 0.6) / 0.4) * 0.4).toFixed(2); // late noise only at high chaos
  hc.s_noise = 1.0;
  hc.high_quality_pct = 0.35;                               // keep the deterministic finisher
  hc.quality_sharpness = +(0.15 + v * 0.25).toFixed(2);     // recover detail as noise rises
  if (v < 0.3) {
    hc.velocity_bias_mode = "off";
  } else {                                                  // remembered-action spice past Spicy
    hc.velocity_bias_mode = "apply";
    hc.velocity_bias_source = "nearest";
    hc.velocity_bias_strength = +(((v - 0.3) / 0.7) * 1.2).toFixed(2);
  }
  hc.motion_pulse_mode = v < 0.7 ? "off" : (v < 0.9 ? "balanced" : "aggressive");
}

// Legible readout of what the current Hybrid config actually does (not runtime telemetry).
function buildFeatureSummary(hc) {
  const items = [];
  const eta = +hc.eta || 0, ef = +hc.eta_final || 0;
  items.push(eta > 0
    ? ["Ancestral noise", `eta ${eta.toFixed(2)}${ef < eta ? " → " + ef.toFixed(2) : ""} (video only)`, true]
    : ["Ancestral noise", "off (deterministic)", false]);
  const vbOn = (hc.velocity_bias_mode || "off") !== "off" && (+hc.velocity_bias_strength || 0) > 0;
  items.push(vbOn
    ? ["Velocity spice", `${hc.velocity_bias_mode} ${(+hc.velocity_bias_strength).toFixed(2)} (${hc.velocity_bias_source || "mean"})`, true]
    : ["Velocity spice", "off", false]);
  items.push(hc.rescue_mode
    ? ["Rescue", `on (thr ${(+hc.rescue_threshold || 0).toFixed(2)} / str ${(+hc.rescue_strength || 0).toFixed(2)})`, true]
    : ["Rescue", "off", false]);
  const hqp = +hc.high_quality_pct || 0, sharp = +hc.quality_sharpness || 0;
  items.push(["Quality pass", hqp > 0 ? `${Math.round(hqp * 100)}%${sharp > 0 ? " + sharpen " + sharp.toFixed(2) : ""}` : "off", hqp > 0]);
  const mpm = hc.motion_pulse_mode || "off";
  items.push(["Motion pulses", mpm, mpm !== "off"]);

  const box = el("div");
  box.style.cssText = "margin:0 0 10px 0; padding:8px; border:1px solid rgba(180,190,200,0.2); border-radius:6px;";
  const t = el("div", "", "Active features (this config)");
  t.style.cssText = "font-size:11px; font-weight:600; opacity:0.8; margin-bottom:4px;";
  box.append(t);
  for (const [label, value, on] of items) {
    const r = el("div");
    r.style.cssText = `display:flex; justify-content:space-between; font-size:11px; padding:1px 0; opacity:${on ? "1" : "0.45"};`;
    const l = el("span", "", (on ? "🟢 " : "⚪ ") + label);
    const val = el("span", "", value);
    val.style.cssText = "opacity:0.85;";
    r.append(l, val);
    box.append(r);
  }
  const a = el("div", "", "Audio auto-held deterministic on LTXAV (ancestral noise + steering confined to video).");
  a.style.cssText = "font-size:10px; opacity:0.6; margin-top:5px;";
  box.append(a);
  return box;
}

function splitLines(value) {
  return String(value || "")
    .split(/\n+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

// ─── Panel ────────────────────────────────────────────────────────────────────

function closePanel() { activePanel?.remove(); activePanel = null; }

function showError(panel, msg) {
  const tgt = panel?.querySelector("[data-role='error']");
  if (tgt) tgt.textContent = msg;
}

// ─── Batch Training ─────────────────────────────────────────────────────────
const _fpBatchProcessing = new Set();   // node ids whose armed batch is currently running

function batchKey(node) {
  try { return (getSettings(node)?.refinement_key) || linkedRefinementKey(node) || ""; }
  catch { return ""; }
}

async function refreshBatchButton(node) {
  const w = (node.widgets || []).find((x) => x._fpBatchBtn);
  if (!w) return;
  let label = "Batch Training (off)";
  try {
    const n = parseInt(getSettings(node)?.refiner?.batch_variants || 1, 10) || 1;
    if (_fpBatchProcessing.has(node.id)) {
      label = "Batch Training (Processing…)";
    } else {
      let pending = false;
      const key = batchKey(node);
      if (key) {
        try {
          const res = await api.fetchApi(`/funpack/batch/list?key=${encodeURIComponent(key)}`, { cache: "no-store" });
          pending = !!(await res.json()).found;
        } catch {}
      }
      if (pending) label = "Batch Training (Finished — rate)";
      else if (n > 1) label = `Batch Training (armed ×${n})`;
      else label = "Batch Training (off)";
    }
  } catch {}
  w.name = label;
  node.setDirtyCanvas?.(true, true);
}

function refreshAllBatchButtons() {
  try {
    for (const node of (app.graph?._nodes || [])) {
      if (node?.comfyClass === NODE_NAME || node?.type === NODE_NAME) refreshBatchButton(node);
    }
  } catch {}
}

function openBatchPanel(node) {
  closePanel();
  injectStyles();
  const root = el("div", "funpack-studio-panel");
  root.style.cssText = "position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);z-index:10000;width:min(460px,calc(100vw - 24px));";
  document.body.append(root);
  activePanel = root;

  const header = el("div", "funpack-studio-header");
  header.append(el("div", "funpack-studio-title", "Batch Training"));
  const closeBtn = btn("Close");
  closeBtn.addEventListener("click", () => { closePanel(); refreshBatchButton(node); });
  header.append(closeBtn);
  root.append(header);

  const errorEl = el("div", "funpack-studio-error");
  root.append(errorEl);
  const body = el("div", "funpack-studio-body");
  root.append(body);

  const key = batchKey(node);

  async function render() {
    body.replaceChildren();
    errorEl.textContent = "";
    if (!key) {
      body.append(el("div", "funpack-studio-hint", "Set a refinement key (Studio → Session) or wire refinement_key_input to use Batch Training."));
      return;
    }
    let data = { found: false, labels: [] };
    try {
      const res = await api.fetchApi(`/funpack/batch/list?key=${encodeURIComponent(key)}`, { cache: "no-store" });
      data = await res.json();
    } catch (e) {
      body.append(el("div", "funpack-studio-hint", `Batch list failed: ${e.message}`));
      return;
    }
    if (data.found) renderRatingList(data);
    else renderConfig();
  }

  function renderConfig() {
    const settings = getSettings(node);
    const rf = settings.refiner = settings.refiner || {};
    const n = parseInt(rf.batch_variants || 1, 10) || 1;
    body.append(el("div", "funpack-studio-hint",
      "A batch is a set of generations made in one run. Regular = everything frozen except the seed (and shortcut options). Interactive Guessing = freeze EVERYTHING including the seed and sweep the conditioning strength up or down across the batch, so you can see exactly where it breaks. Set it up, Arm, then press ComfyUI's Generate. All videos come out the Chain Sampler's IMAGES output; reopen this window to rate them."));

    const sizeInput = numInput(n > 1 ? n : 10, 2, 64, 1);
    body.append(row("Batch size", sizeInput));

    if (!rf.batch_mode) rf.batch_mode = "Regular";
    const modeSelect = selectEl(["Regular", "Interactive Guessing"], rf.batch_mode);
    modeSelect.addEventListener("change", () => { rf.batch_mode = modeSelect.value; saveSettings(node, settings); render(); });
    body.append(row("Mode", modeSelect));

    let dirSelect = null, rangeInput = null, freezeToggle = null;
    if (rf.batch_mode === "Interactive Guessing") {
      if (!rf.guess_direction) rf.guess_direction = "up";
      dirSelect = selectEl(["up", "down"], rf.guess_direction);
      body.append(el("div", "funpack-studio-hint", "up = amplify the conditioning's spread toward overbake (find your safe ceiling). down = dampen it (find how flat is too flat)."));
      body.append(row("Direction", dirSelect));
      if (rf.guess_range === undefined) rf.guess_range = 1.0;
      rangeInput = numInput(rf.guess_range, 0.1, 2.0, 0.1);
      body.append(el("div", "funpack-studio-hint", "How far the last rung goes: up → 1.0+range (e.g. 1.0 = up to 2.0×). down → 1.0−range (capped above 0). The ladder ramps linearly from the untouched base to this."));
      body.append(row("Range", rangeInput));
      if (rf.guess_freeze_seed === undefined) rf.guess_freeze_seed = true;
      freezeToggle = toggleEl(!!rf.guess_freeze_seed, "Freeze noise seed across rungs");
      freezeToggle.inp.addEventListener("change", () => { rf.guess_freeze_seed = freezeToggle.inp.checked; saveSettings(node, settings); });
      body.append(el("div", "funpack-studio-hint", "On: every rung shares the same noise, so conditioning strength is the ONLY variable (cleanest for learning the safe ceiling). Off: each rung gets a different seed — varied compositions too."));
      body.append(row("Freeze seed", freezeToggle.wrap));
    }

    if (rf.batch_learning === undefined) rf.batch_learning = true;
    const learnToggle = toggleEl(!!rf.batch_learning, "Learn from ratings on Submit");
    learnToggle.inp.addEventListener("change", () => { rf.batch_learning = learnToggle.inp.checked; saveSettings(node, settings); });
    body.append(el("div", "funpack-studio-hint", rf.batch_mode === "Interactive Guessing"
      ? "On: rating the ladder learns your safe-spread ceiling and auto-caps future steering for this key. Off: pure generation, nothing learned."
      : "On: rating trains the value function + repair memory. Off: pure generation — make variations without teaching anything."));
    body.append(row("Learning", learnToggle.wrap));

    body.append(el("div", "funpack-studio-hint", n > 1
      ? `Armed (${rf.batch_mode}): the next Generate produces ${n}.`
      : "Not armed — Generate runs a single normal generation."));
    const actions = el("div", "");
    actions.style.cssText = "display:flex;gap:6px;margin-top:8px;";
    const armBtn = btn(n > 1 ? "Re-arm" : "Arm batch");
    armBtn.addEventListener("click", () => {
      const v = Math.max(2, Math.min(64, parseInt(sizeInput.value, 10) || 0));
      const s = getSettings(node); s.refiner = s.refiner || {};
      s.refiner.batch_variants = v;
      s.refiner.batch_mode = modeSelect.value;
      s.refiner.guess_mode = (modeSelect.value === "Interactive Guessing");
      if (dirSelect) s.refiner.guess_direction = dirSelect.value;
      if (rangeInput) s.refiner.guess_range = Math.max(0.1, Math.min(2.0, parseFloat(rangeInput.value) || 1.0));
      if (freezeToggle) s.refiner.guess_freeze_seed = freezeToggle.inp.checked;
      s.refiner.batch_learning = learnToggle.inp.checked;
      saveSettings(node, s);
      refreshBatchButton(node); render();
    });
    const offBtn = btn("Disarm", "secondary");
    offBtn.addEventListener("click", () => {
      const s = getSettings(node); s.refiner = s.refiner || {};
      s.refiner.batch_variants = 1; s.refiner.guess_mode = false;
      saveSettings(node, s);
      refreshBatchButton(node); render();
    });
    actions.append(armBtn, offBtn);
    body.append(actions);
  }

  function renderRatingList(data) {
    const labels = data.labels || [];
    const ratings = {};
    body.append(el("div", "funpack-studio-hint",
      `Finished batch ${data.created} — rate each, then Submit to teach the value function + repair memory. Forget discards the whole batch.`));
    data.items.forEach((it) => {
      const r = el("div", "");
      r.style.cssText = "display:flex;align-items:center;gap:6px;margin:2px 0;";
      const vtag = (it.guess_factor != null) ? `[×${it.guess_factor}] ` : ((it.variant != null) ? `[v${it.variant}] ` : "");
      const lab = it.prompt ? it.prompt : (it.preview || it.id);
      const name = el("span", "", `#${it.index ?? 0} ${vtag}${lab} (seed ${it.seed})`);
      name.title = it.preview || it.id;
      name.style.cssText = "flex:1;font-size:12px;opacity:0.85;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
      const prev = btn("Preview", "secondary");
      let imgEl = null;
      prev.addEventListener("click", () => {
        if (imgEl) { imgEl.remove(); imgEl = null; return; }
        const url = `/view?filename=${encodeURIComponent(it.preview)}&type=temp&subfolder=${encodeURIComponent(data.subfolder)}`;
        imgEl = el("img", ""); imgEl.src = url; imgEl.style.cssText = "max-width:100%;border-radius:6px;margin:4px 0;"; r.after(imgEl);
      });
      if (!it.preview) prev.disabled = true;
      const sel = selectEl(["— rate —", ...labels], "— rate —");
      sel.addEventListener("change", () => { if (sel.value && sel.value !== "— rate —") ratings[it.id] = sel.value; else delete ratings[it.id]; });
      r.append(name, prev, sel);
      body.append(r);
    });
    const actions = el("div", "");
    actions.style.cssText = "display:flex;gap:6px;margin-top:8px;";
    const submitBtn = btn("Submit");
    const forgetBtn = btn("Forget batch", "secondary");
    submitBtn.addEventListener("click", async () => {
      if (Object.keys(ratings).length === 0) { errorEl.textContent = "Rate at least one generation first."; return; }
      submitBtn.disabled = true;
      try {
        const learn = (getSettings(node)?.refiner?.batch_learning !== false);
        const res = await api.fetchApi("/funpack/batch/submit", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key, stamp: data.stamp, ratings, learn }) });
        const out = await res.json();
        if (out.error) throw new Error(out.error);
        errorEl.textContent = `Submitted — trained on ${out.trained}.${out.info ? " " + out.info : ""}`;
        refreshBatchButton(node); render();
      } catch (e) { errorEl.textContent = `Submit failed: ${e.message}`; submitBtn.disabled = false; }
    });
    forgetBtn.addEventListener("click", async () => {
      try {
        const res = await api.fetchApi("/funpack/batch/forget", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key, stamp: data.stamp }) });
        const out = await res.json(); if (out.error) throw new Error(out.error);
        refreshBatchButton(node); render();
      } catch (e) { errorEl.textContent = `Forget failed: ${e.message}`; }
    });
    actions.append(submitBtn, forgetBtn);
    body.append(actions);
  }

  render();
}

function openPanel(node) {
  closePanel();
  injectStyles();

  const root = el("div", "funpack-studio-panel");
  root.style.cssText = "position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);z-index:10000;";
  document.body.append(root);
  activePanel = root;

  const settings = deepMerge(defaultSettings(), getSettings(node));
  const adjItems = getAdjustments(node).map((i) => ({ ...i }));
  const tabKey = `funpack_studio_tab_${node.id}`;
  let activeTab = localStorage.getItem(tabKey) || "Session";
  if (!TABS.includes(activeTab)) activeTab = "Session";

  // Header
  const header = el("div", "funpack-studio-header");
  const titleEl = el("div", "funpack-studio-title", "FunPack Studio");
  const closeBtn = btn("Close");
  closeBtn.addEventListener("click", () => {
    saveSettings(node, settings);
    saveAdjustments(node, adjItems.filter((i) => String(i.phrase || "").trim()));
    closePanel();
  });
  header.append(titleEl, closeBtn);
  root.append(header);

  const errorEl = el("div", "funpack-studio-error");
  errorEl.dataset.role = "error";
  root.append(errorEl);

  // Tab bar
  const tabBar = el("div", "funpack-studio-tabs");
  const body = el("div", "funpack-studio-body");
  root.append(tabBar, body);

  function switchTab(name) {
    activeTab = name;
    localStorage.setItem(tabKey, name);
    for (const t of tabBar.querySelectorAll(".funpack-studio-tab"))
      t.classList.toggle("active", t.dataset.tab === name);
    renderTab(name);
  }

  for (const t of TABS) {
    const tabBtn = el("button", "funpack-studio-tab", t);
    tabBtn.type = "button";
    tabBtn.dataset.tab = t;
    tabBtn.addEventListener("click", () => switchTab(t));
    tabBar.append(tabBtn);
  }

  // ── TAB RENDERERS ──────────────────────────────────────────────────────────

  function renderTab(name) {
    body.replaceChildren();
    errorEl.textContent = "";
    if (name === "Session") renderSession();
    else if (name === "Shortcuts") renderShortcuts();
    else if (name === "Split markers") renderSplitMarkers();
    else if (name === "Refiner") renderRefiner();
    else if (name === "Advisor") renderAdvisor();
    else if (name === "LoRA") renderLora();
    else if (name === "Sampler") renderSampler();
    else if (name === "Adjustments") renderAdjustments();
    else if (name === "Timeline") renderTimeline();
  }

  // SESSION ──────────────────────────────────────────────────────────────────
  function renderSession() {
    body.append(sectionTitle("Refinement Session"));

    const keyInput = textInput(settings.refinement_key, "session key name");
    keyInput.addEventListener("input", () => {
      settings.refinement_key = keyInput.value.trim();
    });

    const linkedKey = linkedRefinementKey(node);
    if (linkedKey) {
      const hint = el("div", "funpack-studio-hint", `Linked key from node input: ${linkedKey}`);
      body.append(hint);
    }

    body.append(row("Session key", keyInput));
    body.append(overrideToggle(settings, "refinement_key",
      "Override - use popup key even when refinement_key_input is connected"));

    const resetToggle = toggleEl(settings.refiner.reset_session, "Reset session on next run");
    resetToggle.inp.addEventListener("change", () => { settings.refiner.reset_session = resetToggle.inp.checked; });
    body.append(row("Reset", resetToggle.wrap));
  }

  // SHORTCUTS ────────────────────────────────────────────────────────────────
  function renderShortcuts() {
    body.append(sectionTitle("Shortcuts"));
    if (!studioShortcutData) {
      body.append(el("div", "funpack-studio-empty", "Loading shortcuts..."));
      fetchShortcuts().then(() => renderTab("Shortcuts"));
      return;
    }

    if (!root.__shortcutDrafts) {
      const rows = Array.isArray(studioShortcutData?.shortcuts) ? studioShortcutData.shortcuts : [];
      root.__shortcutDrafts = JSON.parse(JSON.stringify(rows));
    }
    const drafts = root.__shortcutDrafts;

    const toolbar = el("div", "funpack-studio-footer");
    const addBtn = btn("+ Add shortcut", "primary");
    addBtn.addEventListener("click", () => {
      drafts.unshift({ name: "", enabled: true, triggers: [], replacements: [] });
      renderTab("Shortcuts");
    });
    const refreshBtn = btn("Refresh");
    refreshBtn.addEventListener("click", async () => {
      await fetchShortcuts();
      delete root.__shortcutDrafts;
      renderTab("Shortcuts");
    });
    const importBtn = btn("Import");
    importBtn.addEventListener("click", () => importShortcuts(
      () => { delete root.__shortcutDrafts; renderTab("Shortcuts"); },
      (e) => showError(root, e.message),
    ));
    const exportBtn = btn("Export");
    exportBtn.addEventListener("click", async () => {
      try { await exportShortcuts(); }
      catch (e) { showError(root, e.message); }
    });
    toolbar.append(addBtn, refreshBtn, importBtn, exportBtn);
    body.append(toolbar);

    const list = el("div", "funpack-studio-shortcut-list");
    body.append(list);
    if (!drafts.length) {
      list.append(el("div", "funpack-studio-empty", "No shortcuts configured."));
    }

    drafts.forEach((item, index) => {
      const isSaved = !!item.key;
      if (!("_expanded" in item)) item._expanded = !isSaved;

      const rowEl = el("div", "funpack-studio-shortcut-row");

      if (!item._expanded) {
        const summary = el("div", "funpack-studio-shortcut-summary");
        const label = el("span", "funpack-studio-shortcut-label", item.name || "(unnamed)");
        const badge = el("span", item.enabled !== false ? "funpack-studio-badge-on" : "funpack-studio-badge-off",
          item.enabled !== false ? "on" : "off");
        summary.append(label, badge);
        summary.addEventListener("click", () => { item._expanded = true; renderTab("Shortcuts"); });
        rowEl.append(summary);
        list.append(rowEl);
        return;
      }

      const top = el("div", "funpack-studio-shortcut-top");
      const nameInput = textInput(item.name || "", "Shortcut name");
      nameInput.addEventListener("input", () => { item.name = nameInput.value; });
      const enabled = toggleEl(item.enabled !== false, "Enabled");
      enabled.inp.addEventListener("change", () => { item.enabled = enabled.inp.checked; });
      top.append(nameInput, enabled.wrap);

      const triggerArea = el("textarea", "funpack-studio-textarea short");
      triggerArea.placeholder = "Activation phrases, one per line";
      triggerArea.value = (item.triggers || []).join("\n");
      triggerArea.addEventListener("input", () => { item.triggers = splitLines(triggerArea.value); });

      const replacementArea = el("textarea", "funpack-studio-textarea short");
      replacementArea.placeholder = "Replacement phrases, one per line";
      replacementArea.value = (item.replacements || []).join("\n");
      replacementArea.addEventListener("input", () => { item.replacements = splitLines(replacementArea.value); });

      const actions = el("div", "funpack-studio-shortcut-actions");
      const saveBtn = btn("Save", "primary");
      saveBtn.addEventListener("click", async () => {
        try {
          await saveShortcut({
            original_name: item.key || item.name,
            name: item.name,
            enabled: item.enabled !== false,
            triggers: splitLines(triggerArea.value),
            replacements: splitLines(replacementArea.value),
          });
          await fetchShortcuts();
          delete root.__shortcutDrafts;
          renderTab("Shortcuts");
        } catch (e) { showError(root, e.message); }
      });
      const collapseBtn = btn("Collapse");
      collapseBtn.addEventListener("click", () => { item._expanded = false; renderTab("Shortcuts"); });
      const delBtn = btn("Delete", "danger");
      delBtn.addEventListener("click", async () => {
        try {
          if (item.key || item.name) {
            await deleteShortcut(item.name || item.key);
            await fetchShortcuts();
          }
          drafts.splice(index, 1);
          delete root.__shortcutDrafts;
          renderTab("Shortcuts");
        } catch (e) { showError(root, e.message); }
      });
      actions.append(saveBtn, collapseBtn, delBtn);
      rowEl.append(top, triggerArea, replacementArea, actions);
      list.append(rowEl);
    });
  }

  // SPLIT MARKERS ────────────────────────────────────────────────────────────
  function renderSplitMarkers() {
    body.append(sectionTitle("Custom Split Markers"));
    body.append(el("div", "funpack-studio-hint",
      "These extend the built-in split word list. " +
      "Any phrase listed here is recognized as a scene boundary when Split prompt by transitions is enabled. " +
      "Split markers only divide the generation prompt — they do not add video dissolves on a timeline. " +
      "To substitute the phrase in prompt text, create a Shortcut with the same trigger."
    ));

    if (!studioTransitionData) {
      body.append(el("div", "funpack-studio-empty", "Loading split markers..."));
      fetchTransitions().then(() => renderTab("Split markers"));
      return;
    }

    if (!root.__transitionDrafts) {
      const rows = Array.isArray(studioTransitionData?.transitions) ? studioTransitionData.transitions : [];
      root.__transitionDrafts = JSON.parse(JSON.stringify(rows));
    }
    const drafts = root.__transitionDrafts;

    const toolbar = el("div", "funpack-studio-footer");
    const addBtn = btn("+ Add split marker", "primary");
    addBtn.addEventListener("click", () => {
      drafts.unshift({ name: "", trigger: "", enabled: true });
      renderTab("Split markers");
    });
    const refreshBtn = btn("Refresh");
    refreshBtn.addEventListener("click", async () => {
      await fetchTransitions();
      delete root.__transitionDrafts;
      renderTab("Split markers");
    });
    const importBtn = btn("Import");
    importBtn.addEventListener("click", () => importTransitions(
      () => { delete root.__transitionDrafts; renderTab("Split markers"); },
      (e) => showError(root, e.message),
    ));
    const exportBtn = btn("Export");
    exportBtn.addEventListener("click", async () => {
      try { await exportTransitions(); }
      catch (e) { showError(root, e.message); }
    });
    toolbar.append(addBtn, refreshBtn, importBtn, exportBtn);
    body.append(toolbar);

    const list = el("div", "funpack-studio-shortcut-list");
    body.append(list);
    if (!drafts.length) {
      list.append(el("div", "funpack-studio-empty", "No custom split markers configured."));
    }

    drafts.forEach((item, index) => {
      const isSaved = !!item.key;
      if (!("_expanded" in item)) item._expanded = !isSaved;

      const rowEl = el("div", "funpack-studio-shortcut-row");

      if (!item._expanded) {
        const summary = el("div", "funpack-studio-shortcut-summary");
        const label = el("span", "funpack-studio-shortcut-label", item.name || item.trigger || "(unnamed)");
        const trigger = el("span", "funpack-studio-shortcut-trigger", item.trigger || "");
        const badge = el("span", item.enabled !== false ? "funpack-studio-badge-on" : "funpack-studio-badge-off",
          item.enabled !== false ? "on" : "off");
        summary.append(label, trigger, badge);
        summary.addEventListener("click", () => { item._expanded = true; renderTab("Split markers"); });
        rowEl.append(summary);
        list.append(rowEl);
        return;
      }

      const top = el("div", "funpack-studio-shortcut-top");
      const nameInput = textInput(item.name || "", "Label (display only)");
      nameInput.addEventListener("input", () => { item.name = nameInput.value; });
      const enabled = toggleEl(item.enabled !== false, "Enabled");
      enabled.inp.addEventListener("change", () => { item.enabled = enabled.inp.checked; });
      top.append(nameInput, enabled.wrap);

      const triggerInput = textInput(item.trigger || "", "Trigger phrase");
      triggerInput.addEventListener("input", () => { item.trigger = triggerInput.value; });

      const placementSel = selectEl(["global", "start", "end", "silent"], item.placement || "global");
      placementSel.addEventListener("change", () => { item.placement = placementSel.value; });

      const actions = el("div", "funpack-studio-shortcut-actions");
      const saveBtn = btn("Save", "primary");
      saveBtn.addEventListener("click", async () => {
        try {
          await saveTransition({
            original_name: item.key || item.name,
            name: item.name || item.trigger,
            trigger: item.trigger,
            placement: item.placement || "global",
            enabled: item.enabled !== false,
          });
          await fetchTransitions();
          delete root.__transitionDrafts;
          renderTab("Split markers");
        } catch (e) { showError(root, e.message); }
      });
      const collapseBtn = btn("Collapse");
      collapseBtn.addEventListener("click", () => { item._expanded = false; renderTab("Split markers"); });
      const delBtn = btn("Delete", "danger");
      delBtn.addEventListener("click", async () => {
        try {
          if (item.key || item.name || item.trigger) {
            await deleteTransition(item.name || item.trigger || item.key);
            await fetchTransitions();
          }
          drafts.splice(index, 1);
          delete root.__transitionDrafts;
          renderTab("Split markers");
        } catch (e) { showError(root, e.message); }
      });
      actions.append(saveBtn, collapseBtn, delBtn);
      rowEl.append(top, triggerInput, row("Placement override", placementSel), actions);
      list.append(rowEl);
    });
  }

  // REFINER ──────────────────────────────────────────────────────────────────
  function renderRefiner() {
    body.append(sectionTitle("Execution"));

    const modeSelect = selectEl(REFINER_MODES, settings.refiner.mode);
    modeSelect.addEventListener("change", () => { settings.refiner.mode = modeSelect.value; });
    body.append(row("Mode", modeSelect));

    const advisorSelect = selectEl(ADVISOR_MODES, settings.refiner.advisor_mode);
    advisorSelect.addEventListener("change", () => { settings.refiner.advisor_mode = advisorSelect.value; });
    body.append(row("Advisor mode", advisorSelect));

    const thinkToggle = toggleEl(settings.refiner.advisor_thinking, "Advisor thinking");
    thinkToggle.inp.addEventListener("change", () => { settings.refiner.advisor_thinking = thinkToggle.inp.checked; });
    body.append(row("Thinking", thinkToggle.wrap));

    const luckyToggle = toggleEl(settings.refiner.im_feeling_lucky, "I'm Feeling Lucky");
    luckyToggle.inp.addEventListener("change", () => { settings.refiner.im_feeling_lucky = luckyToggle.inp.checked; });
    body.append(row("Lucky", luckyToggle.wrap));

    body.append(sectionTitle("Generation"));
    body.append(el("div", "funpack-studio-hint",
      "Temporal style controls how the model perceives motion timing via RoPE frame_rate manipulation. " +
      "natural = no change. auto = per-scene director picks motion energy from each scene's prompt (needs the Scene Chain Sampler). " +
      "accelerate = faster, snappier motion. decelerate = heavier, slower motion. " +
      "loop = circular temporal coords (experimental). freeze = highly compressed time (experimental). " +
      "pulse = repeated ease-down motion holds per scene (needs the Scene Chain Sampler). " +
      "Attention anchors from Perfect-rated outputs are automatically captured and injected into future runs."));
    if (!settings.refiner.temporal_style) settings.refiner.temporal_style = "natural";
    const temporalSelect = selectEl(TEMPORAL_STYLES, settings.refiner.temporal_style);
    temporalSelect.addEventListener("change", () => { settings.refiner.temporal_style = temporalSelect.value; });
    body.append(row("Temporal style", temporalSelect));

    const splitToggle = toggleEl(!!settings.refiner.split_by_transitions, "Split prompt by transitions");
    splitToggle.inp.addEventListener("change", () => { settings.refiner.split_by_transitions = splitToggle.inp.checked; });
    body.append(el("div", "funpack-studio-hint",
      "Detect transition words and output one conditioning entry per scene for FunPack LTXAV Scene Chain Sampler. " +
      "Leave off for normal single-conditioning workflows."));
    body.append(row("Split by transitions", splitToggle.wrap));

    if (!settings.refiner.split_transition_placement) settings.refiner.split_transition_placement = "start";
    const placementSelect = selectEl(["start", "end", "silent"], settings.refiner.split_transition_placement);
    placementSelect.addEventListener("change", () => { settings.refiner.split_transition_placement = placementSelect.value; });
    body.append(el("div", "funpack-studio-hint",
      "start: split marker opens the new scene (\"cut to — she runs\"). " +
      "end: split marker closes the previous scene (\"she walks — cut to\"). " +
      "silent: split happens but the marker phrase is removed entirely from the output. " +
      "Custom split markers can override this per-entry in the Split markers tab."));
    body.append(row("Transition placement", placementSelect));

    if (settings.refiner.reference_injection === undefined) settings.refiner.reference_injection = false;
    const refInjToggle = toggleEl(!!settings.refiner.reference_injection, "Reference injection");
    refInjToggle.inp.addEventListener("change", () => { settings.refiner.reference_injection = refInjToggle.inp.checked; });
    body.append(el("div", "funpack-studio-hint",
      "Injects hidden state maps from the i2v reference frame into subsequent scenes to reduce character drift. " +
      "Only meaningful when carry_i2v_guides is also enabled in the Scene Chain Sampler - without it the injection has no reliable reference to anchor from."));
    body.append(row("Reference injection", refInjToggle.wrap));

    if (settings.refiner.vision_conditioning === undefined) settings.refiner.vision_conditioning = true;
    const visCondToggle = toggleEl(!!settings.refiner.vision_conditioning, "Vision conditioning");
    visCondToggle.inp.addEventListener("change", () => { settings.refiner.vision_conditioning = visCondToggle.inp.checked; });
    body.append(el("div", "funpack-studio-hint", "Use the reference image as visual context when encoding the prompt with Gemma3. Disable to test without vision conditioning."));
    body.append(row("Vision conditioning", visCondToggle.wrap));

    if (settings.refiner.value_guidance === undefined) settings.refiner.value_guidance = true;
    const valueGuidanceToggle = toggleEl(!!settings.refiner.value_guidance, "Apply value guidance");
    valueGuidanceToggle.inp.addEventListener("change", () => { settings.refiner.value_guidance = valueGuidanceToggle.inp.checked; });
    body.append(el("div", "funpack-studio-hint", "The value function always trains in the background on rated generations (cheap, no extra diffusion). This toggle applies the learned reward direction to conditioning (ascent + search) once 10+ gens are rated — on by default. Turn off for strict prompt fidelity; learning still accumulates either way. The chain sampler's 'embed_guidance' uses the same trained function per-step."));
    body.append(row("Value guidance", valueGuidanceToggle.wrap));

    if (!settings.refiner.steer_mode) settings.refiner.steer_mode = "relative";
    const steerSelect = selectEl(["relative", "absolute", "both"], settings.refiner.steer_mode);
    steerSelect.addEventListener("change", () => { settings.refiner.steer_mode = steerSelect.value; });
    body.append(el("div", "funpack-studio-hint",
      "Relative: per-prompt steering — finds the best conditioning for THIS prompt (the default; learns from prompt-specific ratings). " +
      "Absolute: pulls conditioning toward a global learned taste that applies to ANY prompt; in this mode a rating means 'this has/lacks details I like in general' (Perfect = 'I love this, give me more of it everywhere'), independent of the prompt. " +
      "Both: layer the global taste prior under the per-prompt fit."));
    body.append(row("Steer mode", steerSelect));

    if (settings.refiner.absolute_strength === undefined) settings.refiner.absolute_strength = 0.6;
    const absStrengthInput = numInput(settings.refiner.absolute_strength, 0.0, 2.0, 0.05);
    absStrengthInput.addEventListener("change", () => {
      const v = parseFloat(absStrengthInput.value);
      settings.refiner.absolute_strength = Number.isFinite(v) ? Math.max(0, Math.min(2, v)) : 0.6;
    });
    body.append(el("div", "funpack-studio-hint",
      "How hard Absolute/Both pull toward the global taste direction. 0.6 is visible but non-destructive; raise to override the prompt more strongly."));
    body.append(row("Absolute strength", absStrengthInput));

    const vfActiveKey = () => settings.refinement_key || linkedRefinementKey(node);
    const vfExportBtn = btn("Export", "secondary");
    const vfImportBtn = btn("Import", "secondary");
    vfExportBtn.addEventListener("click", async () => {
      try { await exportValueFunction(vfActiveKey()); }
      catch (e) { showError(root, `Value function export failed: ${e.message}`); }
    });
    vfImportBtn.addEventListener("click", () => {
      importValueFunction(vfActiveKey(),
        (data) => showError(root, `Value function imported (${data.n_trained} samples, buffer ${data.buffer}).`),
        (e) => showError(root, `Value function import failed: ${e.message}`),
      );
    });
    const vfBtnWrap = el("div", "");
    vfBtnWrap.style.cssText = "display:flex;gap:6px;";
    vfBtnWrap.append(vfExportBtn, vfImportBtn);
    body.append(row("Value function", vfBtnWrap));

    body.append(sectionTitle("Negative prompt"));
    body.append(el("div", "funpack-studio-hint",
      "Encoded via CLIP and output as negative conditioning. Skipped when negative_conditioning input is connected."));
    body.append(overrideToggle(settings, "negative_prompt",
      "Override - use popup value even when negative_prompt input is connected"));
    const negArea = el("textarea", "funpack-studio-textarea short");
    negArea.value = settings.refiner.negative_prompt || "";
    negArea.placeholder = "Negative prompt text (e.g. blurry, low quality, noise)...";
    negArea.addEventListener("input", () => { settings.refiner.negative_prompt = negArea.value; });
    body.append(negArea);

    body.append(sectionTitle("Feedback"));
    body.append(overrideToggle(settings, "feedback_prompt",
      "Override - use popup value even when feedback_prompt input is connected"));
    const fbArea = el("textarea", "funpack-studio-textarea short");
    fbArea.value = settings.refiner.feedback_prompt || "";
    fbArea.placeholder = "Feedback: describe what was wrong with the previous output...";
    fbArea.addEventListener("input", () => { settings.refiner.feedback_prompt = fbArea.value; });
    body.append(fbArea);

    body.append(sectionTitle("Intent"));
    body.append(overrideToggle(settings, "user_intent_prompt",
      "Override - use popup value even when user_intent_prompt input is connected"));
    const intentArea = el("textarea", "funpack-studio-textarea short");
    intentArea.value = settings.refiner.user_intent_prompt_override || "";
    intentArea.placeholder = "Intent override (overrides the user_intent_prompt node input)...";
    intentArea.addEventListener("input", () => { settings.refiner.user_intent_prompt_override = intentArea.value; });
    body.append(intentArea);
  }

  // ADVISOR ──────────────────────────────────────────────────────────────────
  function renderAdvisor() {
    body.append(sectionTitle("Advisor LLM"));

    const enableToggle = toggleEl(settings.advisor_llm.enabled, "Enable Advisor LLM");
    enableToggle.inp.addEventListener("change", () => {
      settings.advisor_llm.enabled = enableToggle.inp.checked;
      renderTab("Advisor");
    });
    body.append(row("Enable", enableToggle.wrap));

    if (!settings.advisor_llm.enabled) {
      body.append(el("div", "funpack-studio-hint",
        "Disabled. When enabled, Studio loads the specified HuggingFace model and uses it as the advisor CLIP (same cache as the standalone Advisor LLM node)."));
      return;
    }

    const pathInput = textInput(settings.advisor_llm.model_path,
      "huihui-ai/Huihui-Qwen3-8B-abliterated-v2");
    pathInput.addEventListener("input", () => { settings.advisor_llm.model_path = pathInput.value; });
    body.append(row("Model path", pathInput));

    const dtypeSelect = selectEl(ADVISOR_DTYPES, settings.advisor_llm.dtype);
    dtypeSelect.addEventListener("change", () => { settings.advisor_llm.dtype = dtypeSelect.value; });
    body.append(row("Dtype", dtypeSelect));

    body.append(el("div", "funpack-studio-hint",
      "Model is loaded on first run and cached. bfloat16 recommended for CUDA, float32 for CPU. Also set Advisor mode in the Refiner tab."));
  }

  // LORA ─────────────────────────────────────────────────────────────────────
  function renderLora() {
    body.append(sectionTitle("LoRA Settings"));
    if (!settings.loras_config) settings.loras_config = { mode: "ltx2", per_block: false };
    const modeSelect = selectEl(["ltx2", "wan"], settings.loras_config.mode);
    modeSelect.addEventListener("change", () => { settings.loras_config.mode = modeSelect.value; });
    body.append(row("Model type", modeSelect));
    const perBlockToggle = toggleEl(settings.loras_config.per_block, "Per-block application");
    perBlockToggle.inp.addEventListener("change", () => { settings.loras_config.per_block = perBlockToggle.inp.checked; });
    body.append(row("Per-block", perBlockToggle.wrap));
    body.append(el("div", "funpack-studio-hint",
      "Studio applies LoRAs internally: session weight suggestions are read first, then LoRAs are loaded, then the direction patch is applied on top. An external lora_stack input bypasses this entirely."));
    body.append(sectionTitle("LoRA List"));

    const list = el("div", "funpack-studio-lora-list");

    function renderLoraRows(allLoras) {
      list.replaceChildren();
      const loras = Array.isArray(settings.loras) ? settings.loras : [];
      if (!loras.length) {
        list.append(el("div", "funpack-studio-empty", "No LoRAs configured."));
      }
      for (let idx = 0; idx < loras.length; idx++) {
        const entry = loras[idx];
        const rowEl = el("div", "funpack-studio-lora-row");

        const nameSelect = selectEl(["None", ...allLoras], entry.name || "None", "lora-name");
        nameSelect.value = (allLoras.includes(entry.name) ? entry.name : "None");
        nameSelect.addEventListener("change", () => { entry.name = nameSelect.value; });

        const typeSelect = selectEl(LORA_TYPES, entry.type || "general", "lora-type");
        typeSelect.addEventListener("change", () => { entry.type = typeSelect.value; });

        const mwInput = numInput(entry.model_weight ?? 1.0, -2, 2, 0.05, "lora-weight");
        mwInput.title = "Model weight";
        mwInput.addEventListener("input", () => { entry.model_weight = parseFloat(mwInput.value) || 0; });

        const cwInput = numInput(entry.clip_weight ?? 1.0, -2, 2, 0.05, "lora-weight");
        cwInput.title = "CLIP weight";
        cwInput.addEventListener("input", () => { entry.clip_weight = parseFloat(cwInput.value) || 0; });

        const delBtn = btn("×", "danger compact");
        delBtn.title = "Remove";
        delBtn.addEventListener("click", () => { loras.splice(idx, 1); renderLoraRows(allLoras); });

        rowEl.append(nameSelect, typeSelect, mwInput, cwInput, delBtn);
        list.append(rowEl);
      }
    }

    body.append(list);

    const footer = el("div", "funpack-studio-footer");
    const addBtn = btn("+ Add LoRA", "primary");
    addBtn.addEventListener("click", async () => {
      if (!Array.isArray(settings.loras)) settings.loras = [];
      const allLoras = await fetchLoras();
      settings.loras.push({ name: "None", type: "general", model_weight: 1.0, clip_weight: 1.0 });
      renderLoraRows(allLoras);
    });
    footer.append(addBtn);
    body.append(footer);

    fetchLoras().then((allLoras) => renderLoraRows(allLoras));
  }

  // SAMPLER ──────────────────────────────────────────────────────────────────
  function renderSampler() {
    body.replaceChildren();
    errorEl.textContent = "";
    if (!settings.samplers) settings.samplers = defaultSettings().samplers;

    function renderPassSection(passKey, label) {
      const cfg = settings.samplers[passKey];
      body.append(sectionTitle(label));

      // Type selector
      const typeSelect = selectEl(SAMPLER_TYPES, cfg.type || "Hybrid Euler 2S");
      typeSelect.addEventListener("change", () => { cfg.type = typeSelect.value; renderSampler(); });
      body.append(row("Sampler", typeSelect));

      // Sigmas
      const sigmasInput = el("input", "funpack-studio-input");
      sigmasInput.type = "text";
      sigmasInput.value = cfg.sigmas || "";
      sigmasInput.placeholder = "e.g. 20.0, 14.0, 8.0, 4.0, 1.0, 0.0";
      sigmasInput.addEventListener("input", () => { cfg.sigmas = sigmasInput.value; });
      body.append(row("Sigmas", sigmasInput));
      body.append(el("div", "funpack-studio-hint",
        "Comma-separated floats. Leave empty to let the sampler decide (pass in sigmas externally)."));

      if (cfg.type === "Hybrid Euler 2S") {
        const hc = cfg.hybrid;
        body.append(sectionTitle("Hybrid Euler 2S settings"));
        if (hc.normalize_strength === undefined) hc.normalize_strength = 0.0;
        const hNorm = numInput(hc.normalize_strength, 0, 1, 0.05);
        hNorm.addEventListener("input", () => { hc.normalize_strength = parseFloat(hNorm.value); });
        body.append(el("div", "funpack-studio-hint", "Video-only latent normalization (anti-overbake / oversaturation / colour drift) stacked on the RF loop. 0 = off. 0.5 = gentle. Audio is never touched (LTXAV/CONST path)."));
        body.append(row("normalize strength", hNorm));
        if (hc.normalize_start_sigma === undefined) hc.normalize_start_sigma = 0.9;
        const hNormS = numInput(hc.normalize_start_sigma, 0, 1, 0.025);
        hNormS.addEventListener("input", () => { hc.normalize_start_sigma = parseFloat(hNormS.value); });
        body.append(row("normalize start sigma", hNormS));

        // --- Variability macro: one control over the stochastic knobs below ---
        const macroWrap = el("div");
        macroWrap.style.cssText = "margin:0 0 8px 0; padding:8px; border:1px solid rgba(88,166,214,0.35); border-radius:6px; background:rgba(88,166,214,0.08);";
        const macroTitle = el("div", "", "Variability macro");
        macroTitle.style.cssText = "font-weight:600; margin-bottom:5px;";
        macroWrap.append(macroTitle);
        const presets = el("div");
        presets.style.cssText = "display:flex; gap:6px; margin-bottom:6px;";
        [["Chill", 0.1], ["Spicy", 0.5], ["Chaos", 0.9]].forEach(([name, val]) => {
          const b = btn(name);
          b.addEventListener("click", () => { applyVariabilityMacro(hc, val); renderSampler(); });
          presets.append(b);
        });
        macroWrap.append(presets);
        const slider = el("input");
        slider.type = "range"; slider.min = "0"; slider.max = "1"; slider.step = "0.05";
        slider.value = String(Number.isFinite(+hc.variability) ? +hc.variability : 0.5);
        slider.style.cssText = "flex:1;";
        const macroVal = el("span", "", (+slider.value).toFixed(2));
        macroVal.style.cssText = "font-size:11px; opacity:0.8; margin-left:8px; min-width:28px;";
        slider.addEventListener("input", () => { macroVal.textContent = (+slider.value).toFixed(2); });
        slider.addEventListener("change", () => { applyVariabilityMacro(hc, parseFloat(slider.value)); renderSampler(); });
        const sliderRow = el("div");
        sliderRow.style.cssText = "display:flex; align-items:center;";
        sliderRow.append(slider, macroVal);
        macroWrap.append(sliderRow);
        const macroHint = el("div", "", "Chill = clean & reproducible · Spicy = lively · Chaos = wild. Sets eta / sharpen / velocity spice / motion pulses below (still editable). Audio stays clean automatically.");
        macroHint.style.cssText = "font-size:11px; opacity:0.7; margin-top:5px;";
        macroWrap.append(macroHint);
        body.append(macroWrap);

        // --- What this config actually does ---
        body.append(buildFeatureSummary(hc));

        [["eta", 0, 1, 0.01], ["eta_final", 0, 1, 0.01], ["s_noise", 0, 10, 0.01],
         ["high_quality_pct", 0, 1, 0.01], ["correction_blend", 0, 1, 0.01],
         ["quality_sharpness", 0, 1, 0.01]].forEach(([k, mn, mx, st]) => {
          const inp = numInput(hc[k], mn, mx, st);
          inp.addEventListener("input", () => { hc[k] = parseFloat(inp.value); });
          body.append(row(k.replace(/_/g, " "), inp));
        });
        const mpm = selectEl(MOTION_PULSE_MODES, hc.motion_pulse_mode || "off");
        mpm.addEventListener("change", () => { hc.motion_pulse_mode = mpm.value; renderSampler(); });
        body.append(row("motion pulse mode", mpm));
        if (hc.motion_pulse_mode !== "off") {
          [["motion_pulse_start_pct", 0, 0.95, 0.01], ["motion_pulse_spacing_pct", 0.04, 0.45, 0.01],
           ["motion_pulse_strength", 0, 1, 0.01]].forEach(([k, mn, mx, st]) => {
            const inp = numInput(hc[k], mn, mx, st);
            inp.addEventListener("input", () => { hc[k] = parseFloat(inp.value); });
            body.append(row(k.replace(/_/g, " "), inp));
          });
          const mpcInp = numInput(hc.motion_pulse_count, 1, 6, 1);
          mpcInp.addEventListener("input", () => { hc.motion_pulse_count = parseInt(mpcInp.value); });
          body.append(row("motion pulse count", mpcInp));
        }
        const vbm = selectEl(VELOCITY_BIAS_MODES, hc.velocity_bias_mode || "off");
        vbm.addEventListener("change", () => { hc.velocity_bias_mode = vbm.value; renderSampler(); });
        body.append(row("velocity bias mode", vbm));
        if (hc.velocity_bias_mode !== "off") {
          const vbs = numInput(hc.velocity_bias_strength, 0, 3.0, 0.05);
          vbs.addEventListener("input", () => { hc.velocity_bias_strength = parseFloat(vbs.value); });
          body.append(row("velocity bias strength", vbs));
          const vbsrc = selectEl(["mean", "nearest"], hc.velocity_bias_source || "mean");
          vbsrc.addEventListener("change", () => { hc.velocity_bias_source = vbsrc.value; });
          body.append(row("bias source", vbsrc));
          const vbsrcNote = document.createElement("div");
          vbsrcNote.style.cssText = "font-size:11px;opacity:0.7;margin:0 0 4px 0;";
          vbsrcNote.textContent = "mean = averaged good direction (legacy). nearest = single best-matching prompt cluster, preserves one real good gen's detail (less softening). Also affects rescue.";
          body.append(vbsrcNote);
          const vrk = textInput(hc.velocity_refinement_key, "default");
          vrk.addEventListener("input", () => { hc.velocity_refinement_key = vrk.value; });
          body.append(row("velocity key", vrk));
          const vrkNote = document.createElement("div");
          vrkNote.style.cssText = "font-size:11px;opacity:0.7;margin:0 0 4px 0;";
          vrkNote.textContent = "Blank or 'default' = follow the refinement key wired into Studio (capture and rescue then share one bucket).";
          body.append(vrkNote);
        }
        const rsm = selectEl(["off", "on"], hc.rescue_mode ? "on" : "off");
        rsm.addEventListener("change", () => { hc.rescue_mode = (rsm.value === "on"); renderSampler(); });
        body.append(row("rescue mode", rsm));
        if (hc.rescue_mode) {
          const rst = numInput(hc.rescue_threshold, 0, 1, 0.01);
          rst.addEventListener("input", () => { hc.rescue_threshold = parseFloat(rst.value); });
          body.append(row("rescue threshold", rst));
          const rss = numInput(hc.rescue_strength, 0, 0.5, 0.01);
          rss.addEventListener("input", () => { hc.rescue_strength = parseFloat(rss.value); });
          body.append(row("rescue strength", rss));
          if (hc.velocity_bias_mode === "off") {
            // Source governs rescue too; only render here when the velocity-bias block above didn't.
            const rsrc = selectEl(["mean", "nearest"], hc.velocity_bias_source || "mean");
            rsrc.addEventListener("change", () => { hc.velocity_bias_source = rsrc.value; });
            body.append(row("bias source", rsrc));
            const rsrcNote = document.createElement("div");
            rsrcNote.style.cssText = "font-size:11px;opacity:0.7;margin:0 0 4px 0;";
            rsrcNote.textContent = "mean = blended good direction (legacy). nearest = single best-matching prompt cluster, preserves one real good gen's detail.";
            body.append(rsrcNote);
          }
          const note = document.createElement("div");
          note.style.cssText = "font-size:11px;opacity:0.7;margin:2px 0 4px 0;";
          note.textContent = "Rating-gated: learns automatically from your ratings while on (good = steer toward, Awful = steer away). No-op until a few gens for this prompt are rated. Session reset clears it.";
          body.append(note);
        }
      } else if (cfg.type === "Distilled Flow") {
        const dc = cfg.distilled;
        body.append(sectionTitle("Distilled Flow settings"));
        const orderInp = numInput(dc.order, 1, 2, 1);
        orderInp.addEventListener("input", () => { dc.order = parseInt(orderInp.value); });
        body.append(row("order", orderInp));
        if (dc.ab2_ramp === undefined) dc.ab2_ramp = false;
        const ab2RampToggle = toggleEl(!!dc.ab2_ramp, "Ramp 2nd order across schedule");
        ab2RampToggle.inp.addEventListener("change", () => { dc.ab2_ramp = ab2RampToggle.inp.checked; });
        body.append(el("div", "funpack-studio-hint", "Graduated 2nd order (free, order=2 only): ramp the AB2 contribution 0→1 across the schedule — early/noisy steps stay near 1st-order euler (less overshoot), late/detail steps get full AB2. Helps low-step distilled runs."));
        body.append(row("AB2 ramp", ab2RampToggle.wrap));
        if (dc.normalize_strength === undefined) dc.normalize_strength = 0.0;
        const dNorm = numInput(dc.normalize_strength, 0, 1, 0.05);
        dNorm.addEventListener("input", () => { dc.normalize_strength = parseFloat(dNorm.value); });
        body.append(el("div", "funpack-studio-hint", "Video-only latent normalization (anti-overbake / oversaturation / colour drift) stacked on this ODE. 0 = off. 0.5 = gentle. Audio is never touched."));
        body.append(row("normalize strength", dNorm));
        if (dc.normalize_start_sigma === undefined) dc.normalize_start_sigma = 0.9;
        const dNormS = numInput(dc.normalize_start_sigma, 0, 1, 0.025);
        dNormS.addEventListener("input", () => { dc.normalize_start_sigma = parseFloat(dNormS.value); });
        body.append(row("normalize start sigma", dNormS));
        const fcsInp = numInput(dc.final_correction_steps, 0, 3, 1);
        fcsInp.addEventListener("input", () => { dc.final_correction_steps = parseInt(fcsInp.value); });
        body.append(row("final correction steps", fcsInp));
        const snInp = numInput(dc.s_noise, 0, 0.5, 0.01);
        snInp.addEventListener("input", () => { dc.s_noise = parseFloat(snInp.value); });
        body.append(row("s_noise", snInp));
        const dVbm = selectEl(VELOCITY_BIAS_MODES, dc.velocity_bias_mode || "off");
        dVbm.addEventListener("change", () => { dc.velocity_bias_mode = dVbm.value; renderSampler(); });
        body.append(row("velocity bias mode", dVbm));
        const dFewNote = document.createElement("div");
        dFewNote.style.cssText = "font-size:11px;opacity:0.7;margin:0 0 4px 0;";
        dFewNote.textContent = "Shares the same memory as the Hybrid sampler. Few-step distilled schedules may only land on a velocity target or two, so apply/rescue fire less often than on an 8-step run.";
        body.append(dFewNote);
        if (dc.velocity_bias_mode && dc.velocity_bias_mode !== "off") {
          const dVbs = numInput(dc.velocity_bias_strength, 0, 3.0, 0.05);
          dVbs.addEventListener("input", () => { dc.velocity_bias_strength = parseFloat(dVbs.value); });
          body.append(row("velocity bias strength", dVbs));
          const dVbsrc = selectEl(["mean", "nearest"], dc.velocity_bias_source || "mean");
          dVbsrc.addEventListener("change", () => { dc.velocity_bias_source = dVbsrc.value; });
          body.append(row("bias source", dVbsrc));
          const dVbsrcNote = document.createElement("div");
          dVbsrcNote.style.cssText = "font-size:11px;opacity:0.7;margin:0 0 4px 0;";
          dVbsrcNote.textContent = "mean = averaged good direction (legacy). nearest = single best-matching prompt cluster, preserves one real good gen's detail (less softening). Also affects rescue.";
          body.append(dVbsrcNote);
          const dVrk = textInput(dc.velocity_refinement_key, "default");
          dVrk.addEventListener("input", () => { dc.velocity_refinement_key = dVrk.value; });
          body.append(row("velocity key", dVrk));
          const dVrkNote = document.createElement("div");
          dVrkNote.style.cssText = "font-size:11px;opacity:0.7;margin:0 0 4px 0;";
          dVrkNote.textContent = "Blank or 'default' = follow the refinement key wired into Studio (capture and rescue then share one bucket).";
          body.append(dVrkNote);
        }
        const dRsm = selectEl(["off", "on"], dc.rescue_mode ? "on" : "off");
        dRsm.addEventListener("change", () => { dc.rescue_mode = (dRsm.value === "on"); renderSampler(); });
        body.append(row("rescue mode", dRsm));
        if (dc.rescue_mode) {
          const dRst = numInput(dc.rescue_threshold, 0, 1, 0.01);
          dRst.addEventListener("input", () => { dc.rescue_threshold = parseFloat(dRst.value); });
          body.append(row("rescue threshold", dRst));
          const dRss = numInput(dc.rescue_strength, 0, 0.5, 0.01);
          dRss.addEventListener("input", () => { dc.rescue_strength = parseFloat(dRss.value); });
          body.append(row("rescue strength", dRss));
          if (!dc.velocity_bias_mode || dc.velocity_bias_mode === "off") {
            const dRsrc = selectEl(["mean", "nearest"], dc.velocity_bias_source || "mean");
            dRsrc.addEventListener("change", () => { dc.velocity_bias_source = dRsrc.value; });
            body.append(row("bias source", dRsrc));
          }
          const dRnote = document.createElement("div");
          dRnote.style.cssText = "font-size:11px;opacity:0.7;margin:2px 0 4px 0;";
          dRnote.textContent = "Rating-gated: learns automatically from your ratings while on (good = steer toward, Awful = steer away). No-op until a few gens for this prompt are rated. Session reset clears it.";
          body.append(dRnote);
        }
      } else if (cfg.type === "KSampler") {
        body.append(sectionTitle("KSampler settings"));
        // Keep a saved name that isn't in the live list (older/newer ComfyUI) selectable
        // rather than silently swapping it for the first option.
        const _ksCur = cfg.ksampler_name || "euler";
        const ksSelect = selectEl(
          KSAMPLER_NAMES.includes(_ksCur) ? KSAMPLER_NAMES : KSAMPLER_NAMES.concat([_ksCur]), _ksCur);
        ksSelect.addEventListener("change", () => { cfg.ksampler_name = ksSelect.value; });
        body.append(row("sampler name", ksSelect));
      }
    }

    renderPassSection("high", "High Pass");
    renderPassSection("low",  "Low Pass");
  }

  // ADJUSTMENTS ──────────────────────────────────────────────────────────────
  function renderAdjustments() {
    body.append(sectionTitle("Conditioning Adjustments"));
    body.append(el("div", "funpack-studio-hint",
      "Each phrase is encoded by CLIP. Positive strength pushes conditioning toward that phrase, negative away. Typical range: -0.3 to +0.3."));

    const list = el("div", "funpack-studio-adj-list");

    function renderAdjRows() {
      list.replaceChildren();
      if (!adjItems.length) {
        list.append(el("div", "funpack-studio-empty", "No adjustments. Add a phrase or click a session chip below."));
      }
      for (let idx = 0; idx < adjItems.length; idx++) {
        const item = adjItems[idx];
        const rowEl = el("div", "funpack-studio-adj-row");
        const phraseInput = textInput(item.phrase, "phrase or word", "adj-phrase");
        phraseInput.addEventListener("input", () => { item.phrase = phraseInput.value; });
        const strengthInput = numInput(item.strength ?? 0.1, -1, 1, 0.05, "adj-strength");
        strengthInput.addEventListener("input", () => { item.strength = parseFloat(strengthInput.value) || 0; });
        const delBtn = btn("×", "danger compact");
        delBtn.addEventListener("click", () => { adjItems.splice(idx, 1); renderAdjRows(); });
        rowEl.append(phraseInput, strengthInput, delBtn);
        list.append(rowEl);
      }
    }
    renderAdjRows();
    body.append(list);

    const footer = el("div", "funpack-studio-footer");
    const addBtn = btn("+ Add phrase", "primary");
    addBtn.addEventListener("click", () => {
      adjItems.push({ phrase: "", strength: 0.1 });
      renderAdjRows();
      list.querySelectorAll(".adj-phrase")[adjItems.length - 1]?.focus();
    });
    const clearBtn = btn("Clear all", "danger");
    clearBtn.addEventListener("click", () => { adjItems.length = 0; renderAdjRows(); });
    footer.append(addBtn, clearBtn);
    body.append(footer);

    // Session phrase bank
    const key = settings.refinement_key || linkedRefinementKey(node);
    if (key) {
      body.append(sectionTitle(`Session phrases  (${key})`));
      const search = el("input", "funpack-studio-search");
      search.type = "search";
      search.placeholder = "Search learned phrases";
      body.append(search);
      const bank = el("div", "funpack-studio-bank");
      body.append(bank);

      fetchPhraseMemory(key).then((phrases) => {
        const renderBank = () => {
          const q = search.value.toLowerCase().trim();
          const filtered = q ? phrases.filter((p) => p.text.toLowerCase().includes(q)) : phrases;
          bank.replaceChildren();
          if (!filtered.length) {
            bank.append(el("div", "funpack-studio-empty", phrases.length ? "No matches." : "No learned phrases yet."));
            return;
          }
          const byGroup = new Map();
          for (const p of filtered) {
            const cat = CATEGORY_ORDER.includes(p.category) ? p.category : "details";
            if (!byGroup.has(cat)) byGroup.set(cat, []);
            byGroup.get(cat).push(p);
          }
          for (const cat of CATEGORY_ORDER) {
            const g = byGroup.get(cat);
            if (!g?.length) continue;
            bank.append(el("div", "funpack-studio-cat-label", cat));
            const chipRow = el("div", "funpack-studio-chip-row");
            for (const p of g) {
              const chip = el("button", "funpack-studio-chip", p.text);
              chip.type = "button";
              chip.title = `${p.text}  (seen ${p.evidence}x)`;
              chip.addEventListener("click", () => {
                if (!adjItems.find((i) => i.phrase.trim().toLowerCase() === p.text.toLowerCase())) {
                  adjItems.push({ phrase: p.text, strength: 0.1 });
                  renderAdjRows();
                }
              });
              chipRow.append(chip);
            }
            bank.append(chipRow);
          }
        };
        renderBank();
        search.addEventListener("input", renderBank);
      });
    }
  }

  // ── Timeline ───────────────────────────────────────────────────────────────
  function resolveInputValue(n, inputName) {
    const w = widgetByName(n, inputName);
    if (w) return String(w.value || "");
    const idx = (n.inputs || []).findIndex(i => i.name === inputName);
    if (idx === -1) return "";
    const linkId = n.inputs[idx].link;
    if (!linkId) return "";
    const link = app.graph?.links?.[linkId];
    if (!link) return "";
    const src = app.graph.getNodeById(link.origin_id);
    if (!src) return "";
    return String(src.widgets?.[0]?.value || "");
  }

  async function renderTimeline() {
    const prompt = resolveInputValue(node, "positive_prompt").trim();
    if (!prompt) {
      body.append(el("div", "funpack-studio-hint", "No prompt — type or connect a positive_prompt to preview the timeline."));
      return;
    }
    const loading = el("div", "funpack-studio-hint", "Parsing…");
    body.append(loading);
    let data;
    try {
      const res = await api.fetchApi("/funpack/parse_timeline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          seed: widgetByName(node, "seed")?.value ?? 0,
          refinement_key: settings.refinement_key || linkedRefinementKey(node) || "",
        }),
      });
      data = await res.json();
    } catch (e) {
      loading.textContent = "Error: " + (e.message || e);
      return;
    }
    body.replaceChildren();
    const refreshBtn = btn("Refresh", "secondary");
    refreshBtn.style.cssText = "align-self:flex-start;margin-bottom:4px;";
    refreshBtn.addEventListener("click", () => { body.replaceChildren(); renderTimeline(); });
    body.append(refreshBtn);

    const { anchor, scenes, transitions } = data;
    const trByScene = {};
    for (const t of transitions) trByScene[t.after_scene] = t;

    body.append(el("div", "funpack-studio-hint",
      scenes.length === 1 ? "1 scene — no split markers found." : `${scenes.length} scenes detected.`));

    if (anchor) {
      const anchorBox = el("div", "funpack-timeline-anchor");
      anchorBox.append(el("div", "funpack-timeline-badge", "Anchor (all scenes)"));
      anchorBox.append(el("div", "funpack-timeline-text", anchor));
      body.append(anchorBox);
    }

    const rail = el("div", "funpack-timeline-rail");
    for (const scene of scenes) {
      const box = el("div", "funpack-timeline-box");
      box.append(el("div", "funpack-timeline-badge", `Scene ${scene.index + 1}`));
      box.append(el("div", "funpack-timeline-text", scene.text || "(empty)"));
      rail.append(box);
      const tr = trByScene[scene.index];
      if (tr) {
        const conn = el("div", "funpack-timeline-connector");
        conn.append(el("div", "funpack-timeline-arrow", "→"));
        const label = tr.trigger || "split";
        conn.append(el("div", "funpack-timeline-phrase", label));
        rail.append(conn);
      }
    }
    body.append(rail);
  }

  // ── auto-save on any field change ─────────────────────────────────────────
  let autoSaveTimer = null;
  const scheduleAutoSave = () => {
    clearTimeout(autoSaveTimer);
    autoSaveTimer = setTimeout(() => {
      saveSettings(node, settings);
      saveAdjustments(node, adjItems.filter((i) => String(i.phrase || "").trim()));
    }, 600);
  };
  root.addEventListener("input", scheduleAutoSave);
  root.addEventListener("change", scheduleAutoSave);

  // ── initial render ─────────────────────────────────────────────────────────
  switchTab("Session");
}

// ─── Node face widget ─────────────────────────────────────────────────────────

function fitString(ctx, text, maxWidth) {
  text = String(text ?? "");
  if (ctx.measureText(text).width <= maxWidth) return text;
  let lo = 0, hi = text.length;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    ctx.measureText(text.slice(0, mid) + "...").width <= maxWidth ? lo = mid : hi = mid - 1;
  }
  return text.slice(0, lo) + "...";
}

// ─── Setup ────────────────────────────────────────────────────────────────────

function setupNode(node) {
  const hideInternal = () => {
    for (const w of node.widgets || []) {
      if (HIDDEN_WIDGETS.has(w.name)) hideWidget(w);
    }
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
  };
  hideInternal();
  window.requestAnimationFrame?.(hideInternal);
  window.setTimeout(hideInternal, 0);
  window.setTimeout(hideInternal, 250);

  node.widgets = (node.widgets || []).filter((w) => w.name !== "funpack_studio_open");
  node.addWidget("button", "Open Studio", "funpack_studio_open", () => {
    openPanel(node);
  }, { serialize: false });

  // Batch Training: opens its own window to arm a batch or rate a finished one. It never
  // queues — ComfyUI's Generate runs an armed batch. The label reflects live state.
  node.widgets = (node.widgets || []).filter((w) => !w._fpBatchBtn && w.name !== "funpack_studio_batch");
  const batchBtn = node.addWidget("button", "Batch Training (off)", "funpack_studio_batch",
    () => openBatchPanel(node), { serialize: false });
  batchBtn._fpBatchBtn = true;
  refreshBatchButton(node);

  node.setDirtyCanvas?.(true, true);
}

// ─── Styles ───────────────────────────────────────────────────────────────────

let stylesInjected = false;
function injectStyles() {
  if (stylesInjected) return;
  stylesInjected = true;
  const style = document.createElement("style");
  style.textContent = `
    .funpack-studio-panel {
      width: min(520px, calc(100vw - 24px));
      max-height: min(680px, calc(100vh - 24px));
      display: flex; flex-direction: column;
      padding: 10px;
      border: 1px solid rgba(180,190,200,0.35); border-radius: 8px;
      background: rgba(28,30,34,0.99);
      box-shadow: 0 20px 56px rgba(0,0,0,0.5);
      color: #ddd; font: 12px sans-serif; box-sizing: border-box;
    }
    .funpack-studio-panel * { box-sizing: border-box; }
    .funpack-studio-header {
      display: flex; justify-content: space-between; align-items: center;
      padding-bottom: 8px; margin-bottom: 4px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .funpack-studio-title { font-weight: 700; font-size: 15px; letter-spacing: 0.3px; }
    .funpack-studio-error { min-height: 16px; color: #ff9f9f; padding: 3px 0; font-size: 11px; }
    .funpack-studio-tabs {
      display: flex; gap: 3px; padding: 4px 0;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      flex-shrink: 0; overflow-x: auto;
    }
    .funpack-studio-tab {
      padding: 5px 10px; border: 1px solid transparent; border-radius: 5px;
      background: transparent; color: #aaa; cursor: pointer; font: 11px sans-serif;
      white-space: nowrap;
    }
    .funpack-studio-tab:hover { color: #eee; background: rgba(255,255,255,0.06); }
    .funpack-studio-tab.active { color: #eaffef; border-color: rgba(100,210,140,0.6); background: #244832; }
    .funpack-studio-body {
      flex: 1; overflow-y: auto; padding: 8px 2px; min-height: 0;
      display: flex; flex-direction: column; gap: 5px;
    }
    .funpack-studio-section-title {
      color: #58a6d6; font-weight: 700; font-size: 10px; text-transform: uppercase;
      margin: 8px 0 3px; letter-spacing: 0.4px;
    }
    .funpack-studio-hint { color: #9da6b0; font-size: 11px; line-height: 1.4; }
    .funpack-studio-empty { color: #9da6b0; padding: 6px 0; }
    .funpack-studio-row {
      display: grid; grid-template-columns: 110px minmax(0,1fr);
      gap: 8px; align-items: center;
    }
    .funpack-studio-row-label { color: #b8c0ca; }
    .funpack-studio-input, .funpack-studio-select {
      min-height: 28px; padding: 5px 7px;
      border: 1px solid rgba(180,190,200,0.28); border-radius: 5px;
      background: #17191d; color: #f2f2f2; outline: none; width: 100%;
    }
    .funpack-studio-textarea {
      width: 100%; min-height: 100px; resize: vertical;
      padding: 7px; border: 1px solid rgba(180,190,200,0.28); border-radius: 5px;
      background: #101216; color: #f4f4f4; line-height: 1.4; outline: none; font: 12px sans-serif;
    }
    .funpack-studio-textarea.short { min-height: 60px; }
    .funpack-studio-toggle {
      display: inline-flex; align-items: center; gap: 6px;
      min-height: 28px; color: #d9dee5; cursor: pointer;
    }
    .funpack-studio-toggle input[type=checkbox] { width: 14px; height: 14px; cursor: pointer; }
    .funpack-studio-footer {
      display: flex; gap: 7px; padding-top: 8px;
      border-top: 1px solid rgba(255,255,255,0.08); flex-shrink: 0; margin-top: 6px;
    }
    .funpack-studio-btn {
      min-height: 26px; padding: 4px 10px;
      border: 1px solid rgba(180,190,200,0.35); border-radius: 5px;
      background: #22252a; color: #eee; cursor: pointer; white-space: nowrap;
    }
    .funpack-studio-btn:hover { background: #2b3037; }
    .funpack-studio-btn.primary { border-color: rgba(100,210,140,0.6); background: #244832; }
    .funpack-studio-btn.secondary { border-color: rgba(100,160,220,0.5); background: #1e2d3e; }
    .funpack-studio-btn.danger { border-color: rgba(255,130,130,0.45); background: #472626; }
    .funpack-studio-btn.compact { min-height: 24px; padding: 2px 7px; }
    .funpack-studio-lora-list { display: flex; flex-direction: column; gap: 5px; }
    .funpack-studio-lora-row {
      display: grid; grid-template-columns: minmax(0,1.8fr) 80px 54px 54px 28px;
      gap: 5px; align-items: center;
    }
    .funpack-studio-shortcut-list {
      display: grid;
      gap: 10px;
    }
    .funpack-studio-shortcut-row {
      display: grid;
      gap: 7px;
      padding: 8px;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 6px;
      background: rgba(255,255,255,0.035);
    }
    .funpack-studio-shortcut-top {
      display: grid;
      grid-template-columns: minmax(0,1fr) auto;
      gap: 8px;
      align-items: center;
    }
    .funpack-studio-shortcut-actions {
      display: flex;
      gap: 7px;
      justify-content: flex-end;
      flex-wrap: wrap;
    }
    .funpack-studio-shortcut-summary {
      display: flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      padding: 2px 0;
    }
    .funpack-studio-shortcut-summary:hover .funpack-studio-shortcut-label {
      text-decoration: underline;
    }
    .funpack-studio-shortcut-label {
      font-weight: 500;
      flex: 1;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .funpack-studio-shortcut-trigger {
      font-size: 0.82em;
      color: rgba(255,255,255,0.45);
      font-style: italic;
      white-space: nowrap;
    }
    .funpack-studio-badge-on {
      font-size: 0.75em;
      padding: 1px 6px;
      border-radius: 10px;
      background: rgba(80,180,100,0.25);
      color: #7ddb8f;
      white-space: nowrap;
    }
    .funpack-studio-badge-off {
      font-size: 0.75em;
      padding: 1px 6px;
      border-radius: 10px;
      background: rgba(255,255,255,0.07);
      color: rgba(255,255,255,0.35);
      white-space: nowrap;
    }
    .lora-name, .lora-type { min-height: 28px; padding: 4px 6px;
      border: 1px solid rgba(180,190,200,0.28); border-radius: 5px;
      background: #17191d; color: #f2f2f2; outline: none; width: 100%; }
    .lora-weight { min-height: 28px; padding: 4px 5px; text-align: right;
      border: 1px solid rgba(180,190,200,0.28); border-radius: 5px;
      background: #17191d; color: #f2f2f2; outline: none; width: 100%; }
    .funpack-studio-adj-list { display: flex; flex-direction: column; gap: 5px; }
    .funpack-studio-adj-row {
      display: grid; grid-template-columns: minmax(0,1fr) 70px 28px;
      gap: 6px; align-items: center;
    }
    .adj-phrase { min-height: 28px; padding: 5px 7px;
      border: 1px solid rgba(180,190,200,0.28); border-radius: 5px;
      background: #17191d; color: #f2f2f2; outline: none; width: 100%; }
    .adj-strength { min-height: 28px; padding: 5px 5px; text-align: right;
      border: 1px solid rgba(180,190,200,0.28); border-radius: 5px;
      background: #17191d; color: #f2f2f2; outline: none; width: 100%; }
    .funpack-studio-search {
      width: 100%; min-height: 28px; padding: 5px 7px;
      border: 1px solid rgba(180,190,200,0.28); border-radius: 5px;
      background: #17191d; color: #f2f2f2; outline: none;
    }
    .funpack-studio-bank { max-height: 180px; overflow-y: auto; }
    .funpack-studio-cat-label {
      color: #58a6d6; font-weight: 700; font-size: 10px;
      text-transform: uppercase; margin: 6px 0 3px;
    }
    .funpack-studio-chip-row { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 2px; }
    .funpack-studio-chip {
      padding: 3px 8px; border: 1px solid rgba(255,255,255,0.14); border-radius: 5px;
      background: rgba(255,255,255,0.07); color: #eee; cursor: pointer; font: 11px sans-serif;
    }
    .funpack-studio-chip:hover { background: rgba(100,210,140,0.25); border-color: rgba(100,210,140,0.5); }
    .funpack-studio-override-toggle { color: #9da6b0; font-size: 11px; margin-bottom: 3px; }
    .funpack-studio-override-toggle span { color: #9da6b0; }
    .funpack-studio-banner {
      padding: 7px 10px; border-radius: 5px; font-size: 11px; line-height: 1.5;
      background: rgba(88,166,214,0.12); border: 1px solid rgba(88,166,214,0.35); color: #c8dff0;
    }
    .funpack-studio-tab-link { color: #58a6d6; text-decoration: underline; cursor: pointer; }
    .funpack-studio-tab-link:hover { color: #8dcff5; }
    .funpack-timeline-anchor {
      border: 1px solid rgba(88,166,214,0.35); border-radius: 6px;
      background: rgba(22,45,62,0.5); padding: 8px; margin-bottom: 6px;
    }
    .funpack-timeline-rail {
      display: flex; flex-direction: row; align-items: flex-start;
      gap: 0; overflow-x: auto; padding-bottom: 6px;
    }
    .funpack-timeline-box {
      flex: 0 0 150px; min-height: 90px;
      border: 1px solid rgba(100,210,140,0.35); border-radius: 6px;
      background: rgba(36,72,50,0.35); padding: 8px;
      display: flex; flex-direction: column; gap: 5px;
    }
    .funpack-timeline-badge {
      font-size: 9px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.5px; color: #7ddb8f;
    }
    .funpack-timeline-text {
      font-size: 10px; color: #cdd5df; line-height: 1.4;
      overflow: hidden; display: -webkit-box;
      -webkit-line-clamp: 6; -webkit-box-orient: vertical;
    }
    .funpack-timeline-connector {
      flex: 0 0 auto; display: flex; flex-direction: column;
      align-items: center; justify-content: flex-start;
      padding: 10px 4px 0; gap: 3px; min-width: 52px; max-width: 80px;
    }
    .funpack-timeline-arrow { color: #58a6d6; font-size: 14px; }
    .funpack-timeline-phrase {
      font-size: 9px; color: #9da6b0; text-align: center; line-height: 1.3;
      display: flex; flex-direction: column; align-items: center; gap: 3px;
    }
    .funpack-timeline-effect {
      font-size: 8px; padding: 1px 5px; border-radius: 8px;
      background: rgba(88,166,214,0.18); color: #8dcff5;
      border: 1px solid rgba(88,166,214,0.3); white-space: nowrap;
    }
  `;
  document.head.append(style);
}

// ─── Extension ────────────────────────────────────────────────────────────────

app.registerExtension({
  name: "funpack.studio",
  setup() {
    // Drive the Batch Training button label: Processing while an armed batch runs, then
    // Finished (and one-shot disarm so the next Generate is normal).
    const armedStudioNodes = () => (app.graph?._nodes || []).filter(
      (n) => (n?.comfyClass === NODE_NAME || n?.type === NODE_NAME) &&
             parseInt(getSettings(n)?.refiner?.batch_variants || 1, 10) > 1);
    api.addEventListener("execution_start", () => {
      for (const n of armedStudioNodes()) _fpBatchProcessing.add(n.id);
      refreshAllBatchButtons();
    });
    const onEnd = () => {
      if (_fpBatchProcessing.size === 0) return;
      for (const n of (app.graph?._nodes || [])) {
        if (!_fpBatchProcessing.has(n.id)) continue;
        _fpBatchProcessing.delete(n.id);
        const s = getSettings(n);   // one-shot: disarm after the batch ran
        if (s?.refiner && parseInt(s.refiner.batch_variants || 1, 10) > 1) {
          s.refiner.batch_variants = 1;
          s.refiner.guess_mode = false;
          saveSettings(n, s);
        }
      }
      setTimeout(refreshAllBatchButtons, 400);   // give the manifest a moment to land
    };
    api.addEventListener("execution_success", onEnd);
    api.addEventListener("execution_error", onEnd);
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    const orig = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      orig?.apply(this, arguments);
      setupNode(this);
    };
    const origCfg = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      origCfg?.apply(this, arguments);
      setupNode(this);
    };
  },
});
