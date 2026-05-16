import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "FunPackConditioningAdjust";
const HIDDEN_WIDGETS = new Set(["adjustments"]);
const CATEGORY_ORDER = ["action", "camera", "subject", "appearance", "environment", "style", "quality", "details"];

let activePanel = null;

function widgetByName(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function getAdjustments(node) {
  const widget = widgetByName(node, "adjustments");
  try {
    const parsed = JSON.parse(String(widget?.value || "[]"));
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function setAdjustments(node, items) {
  const widget = widgetByName(node, "adjustments");
  if (widget) widget.value = JSON.stringify(items);
  node.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
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
    if (el?.style) {
      el.style.display = "none";
      el.style.visibility = "hidden";
      el.style.pointerEvents = "none";
    }
    if (el) el.hidden = true;
  }
}

function linkedRefinementKey(node) {
  const input = (node.inputs || []).find((i) => i.name === "refinement_key_input");
  const linkId = Array.isArray(input?.link) ? input.link[0] : input?.link;
  if (linkId == null) return "";
  const link = app.graph?.links?.[linkId];
  const source = link ? app.graph?.getNodeById?.(link.origin_id) : null;
  if (!source) return "";
  const selected = String(widgetByName(source, "refinement_key")?.value || "").trim();
  const typed = String(widgetByName(source, "key_name")?.value || "").trim();
  const key = selected && selected !== "-None-" ? selected : typed;
  return key || "";
}

async function fetchPhraseMemory(key) {
  if (!key) return [];
  try {
    const res = await api.fetchApi(`/funpack/phrase_memory?key=${encodeURIComponent(key)}`, { cache: "no-store" });
    if (!res.ok) return [];
    const data = await res.json();
    return Array.isArray(data.phrases) ? data.phrases : [];
  } catch {
    return [];
  }
}

function closePanel() {
  activePanel?.remove();
  activePanel = null;
}

function panelButton(label, cls = "") {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = label;
  btn.className = `funpack-adj-button ${cls}`.trim();
  return btn;
}

function renderPanel(node) {
  closePanel();
  injectStyles();

  const root = document.createElement("div");
  root.className = "funpack-adj-panel";
  root.style.cssText = "position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);z-index:10000;";
  document.body.append(root);
  activePanel = root;

  const items = getAdjustments(node).map((item) => ({ ...item }));

  // Header
  const header = document.createElement("div");
  header.className = "funpack-adj-header";
  const title = document.createElement("span");
  title.textContent = "Conditioning Adjustments";
  const closeBtn = panelButton("Close");
  closeBtn.addEventListener("click", () => {
    setAdjustments(node, items.filter((i) => String(i.phrase || "").trim()));
    closePanel();
  });
  header.append(title, closeBtn);
  root.append(header);

  const hint = document.createElement("div");
  hint.className = "funpack-adj-hint";
  hint.textContent = "Each phrase is encoded by the connected CLIP. Positive strength pushes conditioning toward that phrase, negative away. Range: -1.0 to +1.0 (typical useful range: 0.05 - 0.30).";
  root.append(hint);

  // Row list
  const list = document.createElement("div");
  list.className = "funpack-adj-list";
  root.append(list);

  function renderRows() {
    list.replaceChildren();
    if (!items.length) {
      const empty = document.createElement("div");
      empty.className = "funpack-adj-empty";
      empty.textContent = "No adjustments yet. Add a phrase below or click a chip from the session bank.";
      list.append(empty);
      return;
    }
    for (let idx = 0; idx < items.length; idx++) {
      const item = items[idx];
      const row = document.createElement("div");
      row.className = "funpack-adj-row";

      const phraseInput = document.createElement("input");
      phraseInput.type = "text";
      phraseInput.value = String(item.phrase || "");
      phraseInput.placeholder = "phrase or word";
      phraseInput.className = "funpack-adj-phrase";
      phraseInput.addEventListener("input", () => { item.phrase = phraseInput.value; });

      const strengthInput = document.createElement("input");
      strengthInput.type = "number";
      strengthInput.value = String(Number.isFinite(item.strength) ? item.strength : 0.0);
      strengthInput.min = "-1";
      strengthInput.max = "1";
      strengthInput.step = "0.05";
      strengthInput.className = "funpack-adj-strength";
      strengthInput.addEventListener("input", () => {
        const v = parseFloat(strengthInput.value);
        item.strength = Number.isFinite(v) ? Math.max(-1, Math.min(1, v)) : 0.0;
      });

      const del = panelButton("×", "danger compact");
      del.title = "Remove";
      del.addEventListener("click", () => { items.splice(idx, 1); renderRows(); });

      row.append(phraseInput, strengthInput, del);
      list.append(row);
    }
  }
  renderRows();

  // Footer
  const footer = document.createElement("div");
  footer.className = "funpack-adj-footer";
  const addBtn = panelButton("+ Add phrase", "primary");
  addBtn.addEventListener("click", () => {
    items.push({ phrase: "", strength: 0.1 });
    renderRows();
    list.querySelectorAll(".funpack-adj-phrase")[items.length - 1]?.focus();
  });
  const clearBtn = panelButton("Clear all", "danger");
  clearBtn.addEventListener("click", () => { items.length = 0; renderRows(); });
  footer.append(addBtn, clearBtn);
  root.append(footer);

  // Session phrase bank
  const key = linkedRefinementKey(node);
  if (key) {
    const bankSection = document.createElement("div");
    bankSection.className = "funpack-adj-bank-section";
    const bankTitle = document.createElement("div");
    bankTitle.className = "funpack-adj-bank-title";
    bankTitle.textContent = `Session phrases  (${key})`;
    bankSection.append(bankTitle);

    const search = document.createElement("input");
    search.type = "search";
    search.placeholder = "Search learned phrases";
    search.className = "funpack-adj-search";
    bankSection.append(search);

    const bank = document.createElement("div");
    bank.className = "funpack-adj-bank";
    const loading = document.createElement("div");
    loading.className = "funpack-adj-empty";
    loading.textContent = "Loading...";
    bank.append(loading);
    bankSection.append(bank);
    root.append(bankSection);

    fetchPhraseMemory(key).then((phrases) => {
      const renderBank = () => {
        const q = search.value.toLowerCase().trim();
        const filtered = q ? phrases.filter((p) => p.text.toLowerCase().includes(q)) : phrases;
        bank.replaceChildren();
        if (!filtered.length) {
          const empty = document.createElement("div");
          empty.className = "funpack-adj-empty";
          empty.textContent = phrases.length ? "No matches." : "No learned phrases in this session yet.";
          bank.append(empty);
          return;
        }
        const byCategory = new Map();
        for (const p of filtered) {
          const cat = CATEGORY_ORDER.includes(p.category) ? p.category : "details";
          if (!byCategory.has(cat)) byCategory.set(cat, []);
          byCategory.get(cat).push(p);
        }
        for (const cat of CATEGORY_ORDER) {
          const group = byCategory.get(cat);
          if (!group?.length) continue;
          const label = document.createElement("div");
          label.className = "funpack-adj-cat-label";
          label.textContent = cat;
          bank.append(label);
          const chipRow = document.createElement("div");
          chipRow.className = "funpack-adj-chip-row";
          for (const p of group) {
            const chip = document.createElement("button");
            chip.type = "button";
            chip.className = "funpack-adj-chip";
            chip.textContent = p.text;
            chip.title = `${p.text}  (seen ${p.evidence}x)`;
            chip.addEventListener("click", () => {
              if (!items.find((i) => i.phrase.trim().toLowerCase() === p.text.toLowerCase())) {
                items.push({ phrase: p.text, strength: 0.1 });
                renderRows();
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

let stylesInjected = false;
function injectStyles() {
  if (stylesInjected) return;
  stylesInjected = true;
  const style = document.createElement("style");
  style.textContent = `
    .funpack-adj-panel {
      width: min(440px, calc(100vw - 24px));
      max-height: min(580px, calc(100vh - 24px));
      display: flex;
      flex-direction: column;
      padding: 10px;
      border: 1px solid rgba(180,190,200,0.35);
      border-radius: 8px;
      background: rgba(30,32,36,0.98);
      box-shadow: 0 18px 48px rgba(0,0,0,0.42);
      color: #ddd;
      font: 12px sans-serif;
      box-sizing: border-box;
      overflow: hidden;
    }
    .funpack-adj-panel * { box-sizing: border-box; }
    .funpack-adj-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-weight: 700;
      font-size: 14px;
      padding-bottom: 8px;
      margin-bottom: 4px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      flex-shrink: 0;
    }
    .funpack-adj-hint {
      color: #9da6b0;
      font-size: 11px;
      margin-bottom: 8px;
      line-height: 1.4;
      flex-shrink: 0;
    }
    .funpack-adj-list {
      overflow-y: auto;
      min-height: 40px;
      max-height: 200px;
      display: flex;
      flex-direction: column;
      gap: 5px;
      padding-right: 3px;
      flex-shrink: 0;
    }
    .funpack-adj-empty { color: #9da6b0; padding: 6px 0; }
    .funpack-adj-row {
      display: grid;
      grid-template-columns: minmax(0,1fr) 74px 28px;
      gap: 6px;
      align-items: center;
    }
    .funpack-adj-phrase,
    .funpack-adj-strength {
      min-height: 28px;
      padding: 5px 7px;
      border: 1px solid rgba(180,190,200,0.28);
      border-radius: 5px;
      background: #17191d;
      color: #f2f2f2;
      outline: none;
      width: 100%;
    }
    .funpack-adj-strength { text-align: right; }
    .funpack-adj-footer {
      display: flex;
      gap: 7px;
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px solid rgba(255,255,255,0.08);
      flex-shrink: 0;
    }
    .funpack-adj-button {
      min-height: 26px;
      padding: 4px 10px;
      border: 1px solid rgba(180,190,200,0.35);
      border-radius: 5px;
      background: #22252a;
      color: #eee;
      cursor: pointer;
      white-space: nowrap;
    }
    .funpack-adj-button:hover { background: #2b3037; }
    .funpack-adj-button.primary { border-color: rgba(100,210,140,0.6); background: #244832; }
    .funpack-adj-button.danger { border-color: rgba(255,130,130,0.45); background: #472626; }
    .funpack-adj-button.compact { min-height: 24px; padding: 2px 7px; }
    .funpack-adj-bank-section {
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px solid rgba(255,255,255,0.08);
      flex: 1;
      min-height: 0;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .funpack-adj-bank-title {
      font-weight: 700;
      color: #58a6d6;
      margin-bottom: 5px;
      font-size: 10px;
      text-transform: uppercase;
      flex-shrink: 0;
    }
    .funpack-adj-search {
      width: 100%;
      min-height: 28px;
      padding: 5px 7px;
      border: 1px solid rgba(180,190,200,0.28);
      border-radius: 5px;
      background: #17191d;
      color: #f2f2f2;
      outline: none;
      margin-bottom: 6px;
      flex-shrink: 0;
    }
    .funpack-adj-bank {
      overflow-y: auto;
      flex: 1;
      min-height: 0;
    }
    .funpack-adj-cat-label {
      color: #58a6d6;
      font-weight: 700;
      font-size: 10px;
      text-transform: uppercase;
      margin: 6px 0 3px;
    }
    .funpack-adj-chip-row { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 2px; }
    .funpack-adj-chip {
      padding: 3px 8px;
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 5px;
      background: rgba(255,255,255,0.07);
      color: #eee;
      cursor: pointer;
      font: 11px sans-serif;
    }
    .funpack-adj-chip:hover { background: rgba(100,210,140,0.25); border-color: rgba(100,210,140,0.5); }
  `;
  document.head.append(style);
}

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

  // Remove any previously added button to avoid duplicates on reconfigure
  node.widgets = (node.widgets || []).filter((w) => w.name !== "funpack_adj_open");

  node.addWidget("button", "Edit Adjustments", "funpack_adj_open", () => {
    renderPanel(node);
  }, { serialize: false });

  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "funpack.conditioningAdjust",
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
