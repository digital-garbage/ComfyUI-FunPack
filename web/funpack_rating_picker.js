import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PICKER_NODES = new Set(["FunPackStudio", "FunPackVideoRefinerV2"]);
const FORGET_LABEL = "-Just forget it-";

const CATEGORIES = [
  {
    id: "positive",
    label: "Positive",
    accent: "76,175,125",
    ratings: [
      { label: "Perfect",   reward: "+1.0",  hint: "Exact match — full prompt adherence reinforcement" },
      { label: "Nailed it", reward: "+0.75", hint: "Good adherence, weaker reinforcement" },
      { label: "Loved it",  reward: "+0.85", hint: "Great output — prompt adherence ignored" },
    ],
  },
  {
    id: "missing",
    label: "Missing",
    accent: "212,168,75",
    ratings: [
      { label: "Missing action",           reward: "+0.05", hint: "Action wasn't attempted" },
      { label: "Missing details",          reward: "+0.35", hint: "Details absent from output" },
      { label: "Missing quality",          reward: "−0.30", hint: "Visual quality degraded" },
      { label: "Missing action + quality", reward: "−0.55", hint: "Action missing, quality suffered" },
    ],
  },
  {
    id: "wrong",
    label: "Wrong",
    accent: "210,90,90",
    ratings: [
      { label: "Wrong action",            reward: "−0.10", hint: "Wrong action attempted — suppress" },
      { label: "Wrong action + quality",  reward: "−0.40", hint: "Wrong action caused quality/anatomy damage" },
      { label: "Wrong details",           reward: "+0.20", hint: "Details were wrong" },
      { label: "Wrong appearance",        reward: "0.0",   hint: "Appearance mismatch — suppresses auto-inject" },
    ],
  },
];

const NUCLEAR = [
  { label: "Awful",      reward: "−0.90", hint: "Everything wrong" },
  { label: FORGET_LABEL, reward: "none",  hint: "Skip learning — auto-set after each run" },
];

// ─── Styles ───────────────────────────────────────────────────────────────────

let stylesInjected = false;
function injectStyles() {
  if (stylesInjected) return;
  stylesInjected = true;
  const s = document.createElement("style");
  s.textContent = `
    .fp-picker {
      position: fixed; z-index: 10001;
      width: 340px;
      background: rgba(24,26,30,0.99);
      border: 1px solid rgba(180,190,200,0.3);
      border-radius: 8px;
      box-shadow: 0 16px 48px rgba(0,0,0,0.55);
      color: #ddd; font: 12px sans-serif;
      box-sizing: border-box; overflow: hidden;
    }
    .fp-picker * { box-sizing: border-box; }
    .fp-picker-header {
      display: flex; justify-content: space-between; align-items: center;
      padding: 9px 12px 8px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      font-weight: 700; font-size: 13px;
    }
    .fp-picker-close {
      background: none; border: none; color: #888; cursor: pointer;
      font-size: 16px; padding: 0 2px; line-height: 1;
    }
    .fp-picker-close:hover { color: #ddd; }
    .fp-picker-section { padding: 8px 10px 6px; border-bottom: 1px solid rgba(255,255,255,0.06); }
    .fp-picker-section-label {
      font-size: 9px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.8px; color: rgba(var(--accent), 0.9);
      margin-bottom: 6px;
    }
    .fp-picker-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 5px;
    }
    .fp-picker-positive-grid {
      display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 5px;
    }
    .fp-picker-opt {
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(var(--accent), 0.25);
      border-radius: 6px; padding: 7px 8px;
      cursor: pointer; text-align: left; color: #ccc;
      transition: background 0.1s, border-color 0.1s;
      display: flex; flex-direction: column; gap: 3px;
    }
    .fp-picker-opt:hover {
      background: rgba(var(--accent), 0.12);
      border-color: rgba(var(--accent), 0.55);
      color: #fff;
    }
    .fp-picker-opt.active {
      background: rgba(var(--accent), 0.18);
      border-color: rgba(var(--accent), 0.7);
      color: #fff;
    }
    .fp-picker-opt-label { font-size: 11px; font-weight: 600; line-height: 1.2; }
    .fp-picker-opt-reward { font-size: 10px; font-weight: 700; font-variant-numeric: tabular-nums; }
    .fp-picker-opt-hint { font-size: 9px; color: #888; line-height: 1.3; }
    .fp-picker-opt:hover .fp-picker-opt-hint { color: #aaa; }
    .fp-picker-nuclear { padding: 8px 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 5px; }
    .fp-picker-btn-label {
      text-align: left; width: 100%; padding: 4px 8px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 5px; color: #bbb; cursor: pointer;
      font: 11px sans-serif; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .fp-picker-btn-label:hover { background: rgba(255,255,255,0.09); color: #eee; }
  `;
  document.head.append(s);
}

// ─── Reward colour ────────────────────────────────────────────────────────────

function rewardColor(reward) {
  if (reward === "none") return "#888";
  const n = parseFloat(reward.replace("−", "-"));
  if (isNaN(n)) return "#888";
  if (n > 0) return "#6ecf8a";
  if (n < 0) return "#e07070";
  return "#888";
}

// ─── Picker popup ─────────────────────────────────────────────────────────────

let activePickerEl = null;

function closePicker() {
  activePickerEl?.remove();
  activePickerEl = null;
}

function makeOption(r, currentValue, accent, onPick) {
  const btn = document.createElement("button");
  btn.className = "fp-picker-opt" + (currentValue === r.label ? " active" : "");
  btn.style.setProperty("--accent", accent);
  btn.title = r.hint;

  const labelEl = document.createElement("div");
  labelEl.className = "fp-picker-opt-label";
  labelEl.textContent = r.label;

  const rewardEl = document.createElement("div");
  rewardEl.className = "fp-picker-opt-reward";
  rewardEl.style.color = rewardColor(r.reward);
  rewardEl.textContent = r.reward;

  const hintEl = document.createElement("div");
  hintEl.className = "fp-picker-opt-hint";
  hintEl.textContent = r.hint;

  btn.append(labelEl, rewardEl, hintEl);
  btn.addEventListener("click", () => onPick(r.label));
  return btn;
}

function openPicker(ratingWidget, buttonEl, onPick) {
  closePicker();
  injectStyles();

  const picker = document.createElement("div");
  picker.className = "fp-picker";
  activePickerEl = picker;

  // Header
  const header = document.createElement("div");
  header.className = "fp-picker-header";
  header.append(Object.assign(document.createElement("span"), { textContent: "Rate this generation" }));
  const closeBtn = document.createElement("button");
  closeBtn.className = "fp-picker-close";
  closeBtn.textContent = "×";
  closeBtn.addEventListener("click", closePicker);
  header.append(closeBtn);
  picker.append(header);

  const current = ratingWidget.value;

  // Category sections
  for (const cat of CATEGORIES) {
    const section = document.createElement("div");
    section.className = "fp-picker-section";
    section.style.setProperty("--accent", cat.accent);

    const sectionLabel = document.createElement("div");
    sectionLabel.className = "fp-picker-section-label";
    sectionLabel.textContent = cat.label;
    section.append(sectionLabel);

    const grid = document.createElement("div");
    grid.className = cat.id === "positive" ? "fp-picker-positive-grid" : "fp-picker-grid";
    for (const r of cat.ratings) grid.append(makeOption(r, current, cat.accent, onPick));
    section.append(grid);
    picker.append(section);
  }

  // Nuclear row
  const nuclearSection = document.createElement("div");
  nuclearSection.className = "fp-picker-nuclear";
  nuclearSection.style.setProperty("--accent", "140,100,100");
  for (const r of NUCLEAR) nuclearSection.append(makeOption(r, current, "140,100,100", onPick));
  picker.append(nuclearSection);

  document.body.append(picker);

  // Position: below the button, clamped to viewport
  const rect = buttonEl.getBoundingClientRect();
  requestAnimationFrame(() => {
    const ph = picker.offsetHeight, pw = picker.offsetWidth;
    let top = rect.bottom + 4, left = rect.left;
    if (top + ph > window.innerHeight - 8) top = rect.top - ph - 4;
    if (left + pw > window.innerWidth - 8) left = window.innerWidth - pw - 8;
    picker.style.top = Math.max(8, top) + "px";
    picker.style.left = Math.max(8, left) + "px";
  });

  // Close on outside click
  const onOutside = (e) => {
    if (!picker.contains(e.target) && e.target !== buttonEl) {
      closePicker();
      document.removeEventListener("mousedown", onOutside, true);
    }
  };
  setTimeout(() => document.addEventListener("mousedown", onOutside, true), 0);
}

// ─── Node wiring ──────────────────────────────────────────────────────────────

function widgetByName(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function hideWidget(widget) {
  if (!widget) return;
  widget.hidden = true;
  widget.options = widget.options || {};
  widget.options.hidden = true;
  widget.computeSize = () => [0, -4];
  widget.computedHeight = 0;
  widget.type = "hidden";
}

function setupRatingPicker(node) {
  // Avoid double-setup
  if (node.__fpPickerSetup) return;
  node.__fpPickerSetup = true;

  const ratingWidget = widgetByName(node, "rating");
  if (!ratingWidget) return;

  hideWidget(ratingWidget);

  // DOM button that lives outside the canvas
  let buttonEl = node.__fpPickerBtn;
  if (!buttonEl) {
    buttonEl = document.createElement("button");
    buttonEl.className = "fp-picker-btn-label";
    node.__fpPickerBtn = buttonEl;
  }

  const updateBtn = (label) => {
    const current = label || ratingWidget.value || FORGET_LABEL;
    const waiting = current === FORGET_LABEL;
    buttonEl.textContent = waiting ? "Waiting for rating..." : `Rating: ${current}`;
    buttonEl.style.opacity = waiting ? "0.5" : "1";
  };
  updateBtn(ratingWidget.value);

  buttonEl.addEventListener("click", () => {
    openPicker(ratingWidget, buttonEl, (label) => {
      ratingWidget.value = label;
      ratingWidget.callback?.(label);
      updateBtn(label);
      closePicker();
    });
  });

  // Attach button below the node using a custom DOM widget
  node.widgets = (node.widgets || []).filter((w) => w.__fpPickerWidget !== true);
  const domWidget = node.addDOMWidget("fp_rating_picker", "div", buttonEl, {
    getValue: () => ratingWidget.value,
    setValue: (v) => { ratingWidget.value = v; updateBtn(v); },
    serialize: false,
  });
  if (domWidget) domWidget.__fpPickerWidget = true;

  node.setDirtyCanvas?.(true, true);
}

function resetRating(node) {
  const ratingWidget = widgetByName(node, "rating");
  if (!ratingWidget) return;
  ratingWidget.value = FORGET_LABEL;
  ratingWidget.callback?.(FORGET_LABEL);
  if (node.__fpPickerBtn) {
    node.__fpPickerBtn.textContent = "Waiting for rating...";
    node.__fpPickerBtn.style.opacity = "0.5";
  }
}

// ─── Extension ────────────────────────────────────────────────────────────────

app.registerExtension({
  name: "funpack.rating_picker",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!PICKER_NODES.has(nodeData.name)) return;
    const orig = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      orig?.apply(this, arguments);
      setupRatingPicker(this);
    };
    const origCfg = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      origCfg?.apply(this, arguments);
      setupRatingPicker(this);
    };
  },
  setup() {
    api.addEventListener("executed", (event) => {
      const nodeId = parseInt(event.detail?.node);
      if (!nodeId) return;
      const node = app.graph?.getNodeById(nodeId);
      if (node && PICKER_NODES.has(node.type)) resetRating(node);
    });
  },
});
