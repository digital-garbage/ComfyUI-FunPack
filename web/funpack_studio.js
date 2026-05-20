import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "FunPackStudio";
const NONE_SENTINEL = "-None-";
const HIDDEN_WIDGETS = new Set(["studio_settings", "adjustments"]);
const LORA_TYPES = ["general", "action", "style", "quality", "character"];
const ADVISOR_DTYPES = ["bfloat16", "float16", "float32"];
const REFINER_MODES = ["Refine", "Prompt only", "Learning"];
const ADVISOR_MODES = ["Off", "Only diagnostics", "Only prompt", "Full"];
const TEMPORAL_STYLES = ["natural", "accelerate", "decelerate", "loop", "freeze"];
const SB_MODES = ["Pass-through", "Manual", "Auto", "Learning"];
const CATEGORY_ORDER = ["action", "camera", "subject", "appearance", "environment", "style", "quality", "details"];
const TABS = ["Session", "Scene", "Refiner", "Advisor", "LoRA", "Sampler", "Adjustments"];
const SAMPLER_TYPES = ["Hybrid Euler 2S", "Distilled Flow", "KSampler"];
const MOTION_PULSE_MODES = ["off", "balanced", "aggressive", "custom"];
const VELOCITY_BIAS_MODES = ["off", "capture", "apply", "capture_and_apply"];
const KSAMPLER_NAMES = ["euler", "euler_ancestral", "dpm_2", "dpm_2_ancestral", "dpmpp_2m", "dpmpp_sde", "ddim", "uni_pc"];

let activePanel = null;
let studioSceneData = null;

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
    scene_builder: { mode: "Pass-through", scene: NONE_SENTINEL, scene_name: "", aliases: "", scene_positive: "", scene_negative: "" },
    refiner: { mode: "Refine", advisor_mode: "Off", advisor_thinking: true, prompt_repair: true, im_feeling_lucky: false, reset_session: false, feedback_prompt: "", user_intent_prompt_override: "", negative_prompt: "", temporal_style: "natural", split_by_transitions: false },
    advisor_llm: { enabled: false, model_path: "huihui-ai/Huihui-Qwen3-8B-abliterated-v2", dtype: "bfloat16" },
    loras: [],
    loras_config: { mode: "ltx2", per_block: false },
    samplers: {
      high: { type: "Hybrid Euler 2S", sigmas: "", hybrid: { eta: 1.0, eta_final: 1.0, s_noise: 1.0, high_quality_pct: 0.35, correction_blend: 1.0, motion_pulse_mode: "off", motion_pulse_start_pct: 0.3, motion_pulse_count: 2, motion_pulse_spacing_pct: 0.22, motion_pulse_strength: 0.85, velocity_bias_mode: "off", velocity_bias_strength: 0.0, velocity_refinement_key: "default", velocity_aspect_bucket: "any" }, distilled: { order: 2, final_correction_steps: 1, s_noise: 0.0 }, ksampler_name: "euler" },
      low:  { type: "Distilled Flow",  sigmas: "", hybrid: { eta: 1.0, eta_final: 1.0, s_noise: 1.0, high_quality_pct: 0.35, correction_blend: 1.0, motion_pulse_mode: "off", motion_pulse_start_pct: 0.3, motion_pulse_count: 2, motion_pulse_spacing_pct: 0.22, motion_pulse_strength: 0.85, velocity_bias_mode: "off", velocity_bias_strength: 0.0, velocity_refinement_key: "default", velocity_aspect_bucket: "any" }, distilled: { order: 2, final_correction_steps: 1, s_noise: 0.0 }, ksampler_name: "euler" },
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

async function fetchScenes(key = "") {
  try {
    const params = new URLSearchParams({ cache_bust: Date.now() });
    if (key) params.set("key", key);
    const res = await api.fetchApi(`/funpack/scenes?${params}`, { cache: "no-store" });
    if (!res.ok) return null;
    studioSceneData = await res.json();
    return studioSceneData;
  } catch { return null; }
}

async function saveScene(key, payload) {
  const params = key ? `?key=${encodeURIComponent(key)}` : "";
  const res = await api.fetchApi(`/funpack/scenes/scene${params}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "save", ...payload }),
  });
  if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.error || `HTTP ${res.status}`); }
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

// ─── Panel ────────────────────────────────────────────────────────────────────

function closePanel() { activePanel?.remove(); activePanel = null; }

function showError(panel, msg) {
  const tgt = panel?.querySelector("[data-role='error']");
  if (tgt) tgt.textContent = msg;
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
    else if (name === "Scene") renderScene();
    else if (name === "Refiner") renderRefiner();
    else if (name === "Advisor") renderAdvisor();
    else if (name === "LoRA") renderLora();
    else if (name === "Sampler") renderSampler();
    else if (name === "Adjustments") renderAdjustments();
  }

  // SESSION ──────────────────────────────────────────────────────────────────
  function renderSession() {
    body.append(sectionTitle("Refinement Session"));

    const keyInput = textInput(settings.refinement_key, "session key name");
    keyInput.addEventListener("input", () => { settings.refinement_key = keyInput.value.trim(); });

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

    body.append(sectionTitle("Scene Builder"));
    const modeSelect = selectEl(SB_MODES, settings.scene_builder.mode);
    modeSelect.addEventListener("change", () => { settings.scene_builder.mode = modeSelect.value; renderSession(); });
    body.append(row("Mode", modeSelect));

    if (settings.scene_builder.mode !== "Pass-through") {
      const hint = el("div", "funpack-studio-hint",
        "Scene Builder active - prompt is built from the Scene tab, not from the positive_prompt input.");
      body.append(hint);
      const toScene = btn("Open Scene tab →", "secondary");
      toScene.addEventListener("click", () => switchTab("Scene"));
      body.append(toScene);
    } else {
      const hint = el("div", "funpack-studio-hint",
        "Pass-through - the positive_prompt connected to the node is used as-is.");
      body.append(hint);
    }
  }

  // SCENE ────────────────────────────────────────────────────────────────────
  function renderScene() {
    body.append(sectionTitle("Scene Builder"));

    if (settings.scene_builder.mode === "Pass-through") {
      body.append(el("div", "funpack-studio-hint",
        "Scene Builder is in Pass-through mode. Change mode in the Session tab to enable scene construction."));
      return;
    }

    const key = settings.refinement_key || linkedRefinementKey(node);

    // Scene name + scene selector
    const sceneNames = studioSceneData?.scenes || [NONE_SENTINEL];
    const sceneSelect = selectEl(sceneNames, settings.scene_builder.scene || NONE_SENTINEL);
    sceneSelect.addEventListener("change", () => {
      settings.scene_builder.scene = sceneSelect.value;
      const scene = studioSceneData?.data?.scenes?.[sceneSelect.value];
      if (scene) {
        settings.scene_builder.scene_name = scene.name || sceneSelect.value;
        settings.scene_builder.aliases = Array.isArray(scene.aliases) ? scene.aliases.join(", ") : "";
        settings.scene_builder.scene_positive = scene.positive_text || "";
        settings.scene_builder.scene_negative = scene.negative_text || "";
        renderScene();
      }
    });
    body.append(row("Saved scene", sceneSelect));

    const nameInput = textInput(settings.scene_builder.scene_name, "Scene name");
    nameInput.addEventListener("input", () => { settings.scene_builder.scene_name = nameInput.value; });
    body.append(row("Scene name", nameInput));

    const aliasInput = textInput(settings.scene_builder.aliases, "Aliases, comma separated");
    aliasInput.addEventListener("input", () => { settings.scene_builder.aliases = aliasInput.value; });
    body.append(row("Aliases", aliasInput));

    body.append(sectionTitle("Positive prompt"));
    const posArea = el("textarea", "funpack-studio-textarea");
    posArea.value = settings.scene_builder.scene_positive || "";
    posArea.placeholder = "Positive prompt phrases...";
    posArea.addEventListener("input", () => { settings.scene_builder.scene_positive = posArea.value; });
    body.append(posArea);

    // Memory phrase bank
    const memItems = studioSceneData?.memory || [];
    if (memItems.length) {
      body.append(sectionTitle("Phrase bank - click to insert"));
      const search = el("input", "funpack-studio-search");
      search.type = "search";
      search.placeholder = "Search phrases";
      body.append(search);

      const bank = el("div", "funpack-studio-bank");
      const renderBank = () => {
        const q = search.value.toLowerCase().trim();
        const filtered = q ? memItems.filter((m) => String(m.text || "").toLowerCase().includes(q)) : memItems;
        bank.replaceChildren();
        const byGroup = new Map();
        for (const m of filtered) {
          const cat = CATEGORY_ORDER.includes(m.category) ? m.category : "details";
          if (!byGroup.has(cat)) byGroup.set(cat, []);
          byGroup.get(cat).push(m);
        }
        for (const cat of CATEGORY_ORDER) {
          const g = byGroup.get(cat);
          if (!g?.length) continue;
          bank.append(el("div", "funpack-studio-cat-label", cat));
          const chipRow = el("div", "funpack-studio-chip-row");
          for (const m of g) {
            const chip = el("button", "funpack-studio-chip", m.text);
            chip.type = "button";
            chip.addEventListener("click", () => {
              const cur = posArea.value.trim();
              posArea.value = cur ? `${cur}, ${m.text}` : m.text;
              settings.scene_builder.scene_positive = posArea.value;
            });
            chipRow.append(chip);
          }
          bank.append(chipRow);
        }
      };
      renderBank();
      search.addEventListener("input", renderBank);
      body.append(bank);
    }

    // Save/delete scene buttons
    const footer = el("div", "funpack-studio-footer");
    const refreshBtn = btn("Refresh");
    refreshBtn.addEventListener("click", async () => {
      await fetchScenes(key);
      renderScene();
    });
    const saveBtn = btn("Save scene", "primary");
    saveBtn.addEventListener("click", async () => {
      try {
        await saveScene(key, {
          name: settings.scene_builder.scene_name,
          aliases: settings.scene_builder.aliases,
          mode: settings.scene_builder.mode,
          positive_text: settings.scene_builder.scene_positive,
          negative_text: settings.scene_builder.scene_negative,
        });
        await fetchScenes(key);
        renderScene();
      } catch (e) { showError(root, e.message); }
    });
    footer.append(refreshBtn, saveBtn);
    body.append(footer);

    // Fetch scenes if not loaded
    if (!studioSceneData) {
      fetchScenes(key).then(() => renderScene());
    }
  }

  // REFINER ──────────────────────────────────────────────────────────────────
  function renderRefiner() {
    const sbMode = settings.scene_builder?.mode || "Pass-through";
    const sbActive = sbMode !== "Pass-through";
    if (sbActive) {
      const banner = el("div", "funpack-studio-banner");
      banner.innerHTML = `Scene Builder is active (<b>${sbMode}</b> mode) — positive prompt is built in the <a href="#" class="funpack-studio-tab-link">Scene tab</a>, not from the node input.`;
      banner.querySelector("a").addEventListener("click", (e) => { e.preventDefault(); switchTab("Scene"); });
      body.append(banner);
    }

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

    const repairToggle = toggleEl(settings.refiner.prompt_repair, "Prompt repair");
    repairToggle.inp.addEventListener("change", () => { settings.refiner.prompt_repair = repairToggle.inp.checked; });
    body.append(row("Prompt repair", repairToggle.wrap));

    const luckyToggle = toggleEl(settings.refiner.im_feeling_lucky, "I'm Feeling Lucky");
    luckyToggle.inp.addEventListener("change", () => { settings.refiner.im_feeling_lucky = luckyToggle.inp.checked; });
    body.append(row("Lucky", luckyToggle.wrap));

    body.append(sectionTitle("Generation"));
    body.append(el("div", "funpack-studio-hint",
      "Temporal style controls how the model perceives motion timing via RoPE frame_rate manipulation. " +
      "natural = no change. accelerate = faster, snappier motion. decelerate = heavier, slower motion. " +
      "loop = circular temporal coords (experimental). freeze = highly compressed time (experimental). " +
      "Attention anchors from Perfect-rated outputs are automatically captured and injected into future runs."));
    if (!settings.refiner.temporal_style) settings.refiner.temporal_style = "natural";
    const temporalSelect = selectEl(TEMPORAL_STYLES, settings.refiner.temporal_style);
    temporalSelect.addEventListener("change", () => { settings.refiner.temporal_style = temporalSelect.value; });
    body.append(row("Temporal style", temporalSelect));

    const splitToggle = toggleEl(!!settings.refiner.split_by_transitions, "Split prompt by transitions");
    splitToggle.inp.addEventListener("change", () => { settings.refiner.split_by_transitions = splitToggle.inp.checked; });
    body.append(el("div", "funpack-studio-hint",
      "Detect transition words in the prompt (then, suddenly, cut to...) and encode each scene segment as a separate conditioning entry. " +
      "The character description (text before the first comma) is prepended to every segment to keep the subject consistent. " +
      "Use together with FunPack Context Transition Windows."));
    body.append(row("Split by transitions", splitToggle.wrap));

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
    if (sbActive) {
      body.append(el("div", "funpack-studio-hint",
        "Intent is derived from the Scene Builder prompt when Scene Builder is active."));
    }
    body.append(overrideToggle(settings, "user_intent_prompt",
      "Override - use popup value even when user_intent_prompt input is connected"));
    const intentArea = el("textarea", "funpack-studio-textarea short");
    intentArea.value = settings.refiner.user_intent_prompt_override || "";
    intentArea.placeholder = "Intent override (overrides the user_intent_prompt node input)...";
    intentArea.disabled = sbActive;
    if (sbActive) intentArea.style.opacity = "0.45";
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
        [["eta", 0, 1, 0.01], ["eta_final", 0, 1, 0.01], ["s_noise", 0, 10, 0.01],
         ["high_quality_pct", 0, 1, 0.01], ["correction_blend", 0, 1, 0.01]].forEach(([k, mn, mx, st]) => {
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
          const vbs = numInput(hc.velocity_bias_strength, 0, 0.35, 0.01);
          vbs.addEventListener("input", () => { hc.velocity_bias_strength = parseFloat(vbs.value); });
          body.append(row("velocity bias strength", vbs));
          const vrk = textInput(hc.velocity_refinement_key, "default");
          vrk.addEventListener("input", () => { hc.velocity_refinement_key = vrk.value; });
          body.append(row("velocity key", vrk));
          const vab = textInput(hc.velocity_aspect_bucket, "any");
          vab.addEventListener("input", () => { hc.velocity_aspect_bucket = vab.value; });
          body.append(row("aspect bucket", vab));
        }
      } else if (cfg.type === "Distilled Flow") {
        const dc = cfg.distilled;
        body.append(sectionTitle("Distilled Flow settings"));
        const orderInp = numInput(dc.order, 1, 2, 1);
        orderInp.addEventListener("input", () => { dc.order = parseInt(orderInp.value); });
        body.append(row("order", orderInp));
        const fcsInp = numInput(dc.final_correction_steps, 0, 3, 1);
        fcsInp.addEventListener("input", () => { dc.final_correction_steps = parseInt(fcsInp.value); });
        body.append(row("final correction steps", fcsInp));
        const snInp = numInput(dc.s_noise, 0, 0.5, 0.01);
        snInp.addEventListener("input", () => { dc.s_noise = parseFloat(snInp.value); });
        body.append(row("s_noise", snInp));
      } else if (cfg.type === "KSampler") {
        body.append(sectionTitle("KSampler settings"));
        const ksSelect = selectEl(KSAMPLER_NAMES, cfg.ksampler_name || "euler");
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
  `;
  document.head.append(style);
}

// ─── Extension ────────────────────────────────────────────────────────────────

app.registerExtension({
  name: "funpack.studio",
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
