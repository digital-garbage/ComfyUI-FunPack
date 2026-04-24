import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "FunPackApplyLoraWeights";
const LORA_TYPES = ["general", "concept", "style", "quality", "character"];
const ADD_BUTTON_NAME = "+ Add LoRA";
const HEADER_WIDGET_NAME = "funpack_lora_header";
const BASE_WIDGET_NAMES = new Set(["positive_prompt", "refinement_key", "mode", "per_block"]);

let cachedLoraValues = null;
let latestNodeData = null;
const trackedNodes = new Set();
let pendingRefresh = null;

function fitString(ctx, text, maxWidth) {
  text = String(text ?? "");
  if (ctx.measureText(text).width <= maxWidth) {
    return text;
  }

  const ellipsis = "...";
  let low = 0;
  let high = text.length;
  while (low < high) {
    const mid = Math.ceil((low + high) / 2);
    if (ctx.measureText(text.slice(0, mid) + ellipsis).width <= maxWidth) {
      low = mid;
    } else {
      high = mid - 1;
    }
  }
  return text.slice(0, low) + ellipsis;
}

function roundRect(ctx, x, y, width, height, radius = height / 2) {
  ctx.beginPath();
  ctx.roundRect(x, y, width, height, [radius]);
}

function drawRowBackground(ctx, x, y, width, height, active = true) {
  ctx.save();
  roundRect(ctx, x, y, width, height, height / 2);
  ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
  ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
  if (!active) {
    ctx.globalAlpha = app.canvas.editor_alpha * 0.65;
  }
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawToggle(ctx, x, y, height, value) {
  const width = height * 1.5;
  const radius = height * 0.34;
  ctx.save();
  ctx.globalAlpha = app.canvas.editor_alpha * 0.25;
  roundRect(ctx, x + 4, y + 4, width - 8, height - 8, height / 2);
  ctx.fillStyle = "rgba(255,255,255,0.45)";
  ctx.fill();
  ctx.globalAlpha = app.canvas.editor_alpha;
  ctx.fillStyle = value === true ? "#9ab" : "#888";
  const knobX = value === true ? x + height : value === false ? x + height * 0.5 : x + height * 0.75;
  ctx.beginPath();
  ctx.arc(knobX, y + height * 0.5, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  return [x, width];
}

function drawNumberControl(ctx, xRight, y, height, value) {
  const arrowWidth = 9;
  const inner = 3;
  const numberWidth = 36;
  const total = arrowWidth + inner + numberWidth + inner + arrowWidth;
  const x = xRight - total;
  const midY = y + height / 2;
  const arrowHeight = 10;

  ctx.save();
  ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
  ctx.fill(new Path2D(`M ${x} ${midY} l ${arrowWidth} ${arrowHeight / 2} l 0 -${arrowHeight} L ${x} ${midY} z`));
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(Number(value ?? 1).toFixed(2), x + arrowWidth + inner + numberWidth / 2, midY);
  const rx = x + arrowWidth + inner + numberWidth + inner;
  ctx.fill(new Path2D(`M ${rx} ${midY - arrowHeight / 2} l ${arrowWidth} ${arrowHeight / 2} l -${arrowWidth} ${arrowHeight / 2} v -${arrowHeight} z`));
  ctx.restore();

  return {
    dec: [x, arrowWidth],
    value: [x + arrowWidth + inner, numberWidth],
    inc: [rx, arrowWidth],
    any: [x, total],
    x,
    total,
  };
}

function inside(pos, bounds) {
  const x = bounds[0];
  const width = bounds.length > 2 ? bounds[2] : bounds[1];
  const hitX = pos[0] >= x && pos[0] <= x + width;
  if (bounds.length <= 2) {
    return hitX;
  }
  return hitX && pos[1] >= bounds[1] && pos[1] <= bounds[1] + bounds[3];
}

function getLoraValues(nodeData) {
  return cachedLoraValues || nodeData?.input?.optional?.lora_0?.[0] || ["None"];
}

function rememberLoraValues(nodeData, values) {
  cachedLoraValues = values;
  const loraInput = nodeData?.input?.optional?.lora_0;
  if (Array.isArray(loraInput)) {
    loraInput[0] = values;
  }
}

async function fetchLoraValues(nodeData) {
  try {
    const response = await api.fetchApi(`/funpack/loras?cache_bust=${Date.now()}`, {
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const values = await response.json();
    if (Array.isArray(values) && values.length) {
      rememberLoraValues(nodeData, values);
      return values;
    }
  } catch (error) {
    console.warn("FunPack: failed to refresh LoRA list", error);
  }
  rememberLoraValues(nodeData, getLoraValues(nodeData));
  return cachedLoraValues;
}

function valuesWithCurrent(values, current) {
  if (!current || values.includes(current)) {
    return values;
  }
  return [...values, current];
}

function normalizeRowValue(value = {}) {
  if (typeof value !== "object" || value === null) {
    return {
      on: true,
      lora: "None",
      type: "general",
      strength: 1.0,
    };
  }

  const strength = Number(value.strength ?? value.base_weight ?? value.base_model_weight ?? 1.0);
  return {
    on: value.on !== false,
    lora: value.lora || value.name || "None",
    type: LORA_TYPES.includes(value.type || value.lora_type) ? (value.type || value.lora_type) : "general",
    strength: Number.isFinite(strength) ? strength : 1.0,
  };
}

class FunPackBaseWidget {
  constructor(name) {
    this.name = name;
    this.type = "custom";
    this.options = { serialize: true };
    this.hitAreas = {};
    this.downHitAreas = [];
    this.clickHitAreas = [];
    this.mouseDown = false;
    this.last_y = 0;
  }

  serializeValue() {
    return this.value;
  }

  mouse(event, pos, node) {
    if (event.type === "pointerdown") {
      this.mouseDown = true;
      this.downHitAreas = [];
      this.clickHitAreas = [];
      let handled = false;
      for (const hitArea of Object.values(this.hitAreas)) {
        if (!inside(pos, hitArea.bounds)) {
          continue;
        }
        if (hitArea.onMove) {
          this.downHitAreas.push(hitArea);
          handled = true;
        }
        if (hitArea.onClick) {
          this.clickHitAreas.push(hitArea);
          handled = true;
        }
        if (hitArea.onDown) {
          handled = hitArea.onDown.call(this, event, pos, node, hitArea) === true || handled;
        }
      }
      return handled;
    }

    if (event.type === "pointermove") {
      if (!this.mouseDown) {
        return false;
      }
      for (const hitArea of this.downHitAreas) {
        hitArea.onMove.call(this, event, pos, node, hitArea);
      }
      return true;
    }

    if (event.type === "pointerup") {
      if (!this.mouseDown) {
        return false;
      }
      this.mouseDown = false;
      let handled = false;
      for (const hitArea of this.clickHitAreas) {
        if (inside(pos, hitArea.bounds)) {
          handled = hitArea.onClick.call(this, event, pos, node, hitArea) === true || handled;
        }
      }
      this.downHitAreas = [];
      this.clickHitAreas = [];
      if (!handled) {
        handled = this.onMouseClick?.(event, pos, node) === true;
      }
      return handled;
    }

    return false;
  }

  onMouseClick(_event, _pos, _node) {
    return false;
  }
}

class FunPackLoraHeaderWidget extends FunPackBaseWidget {
  constructor() {
    super(HEADER_WIDGET_NAME);
    this.options.serialize = false;
    this.value = {};
    this.hitAreas = {
      toggle: { bounds: [0, 0], onDown: this.onToggleAll },
    };
  }

  computeSize(width) {
    return [width, LiteGraph.NODE_WIDGET_HEIGHT];
  }

  draw(ctx, node, width, y, height) {
    if (!getLoraRows(node).length) {
      return;
    }
    const margin = 10;
    const midY = y + height / 2;
    let x = margin;
    this.hitAreas.toggle.bounds = drawToggle(ctx, x, y, height, allRowsState(node));
    x += this.hitAreas.toggle.bounds[1] + 4;

    ctx.save();
    ctx.globalAlpha = app.canvas.editor_alpha * 0.55;
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText("Toggle All", x, midY);
    ctx.textAlign = "center";
    ctx.fillText("Type", width - 106, midY);
    ctx.fillText("Strength", width - 38, midY);
    ctx.restore();
  }

  onToggleAll(event, pos, node) {
    toggleAllRows(node);
    node.setDirtyCanvas(true, true);
    return true;
  }
}

class FunPackLoraRowWidget extends FunPackBaseWidget {
  constructor(name, value, nodeData) {
    super(name);
    this.value = normalizeRowValue(value);
    this.nodeData = nodeData;
    this.draggedWeight = false;
    this.hitAreas = {
      toggle: { bounds: [0, 0], onDown: this.onToggle },
      lora: { bounds: [0, 0], onClick: this.onLoraClick },
      type: { bounds: [0, 0], onClick: this.onTypeClick },
      weightDec: { bounds: [0, 0], onClick: this.onWeightDec },
      weightValue: { bounds: [0, 0], onClick: this.onWeightValue },
      weightInc: { bounds: [0, 0], onClick: this.onWeightInc },
      weightAny: { bounds: [0, 0], onMove: this.onWeightMove },
    };
  }

  computeSize(width) {
    return [width, LiteGraph.NODE_WIDGET_HEIGHT];
  }

  serializeValue() {
    return {
      on: this.value.on !== false,
      lora: this.value.lora || "None",
      type: LORA_TYPES.includes(this.value.type) ? this.value.type : "general",
      strength: Number(this.value.strength ?? 1.0),
    };
  }

  updateLoraValues(values) {
    if (!values.includes(this.value.lora)) {
      this.value.lora = this.value.lora || "None";
    }
  }

  draw(ctx, node, width, y, height) {
    const margin = 10;
    const inner = 4;
    const rowX = margin;
    const rowWidth = width - margin * 2;
    const midY = y + height / 2;
    drawRowBackground(ctx, rowX, y, rowWidth, height, this.value.on);

    let x = rowX;
    this.hitAreas.toggle.bounds = drawToggle(ctx, x, y, height, this.value.on);
    x += this.hitAreas.toggle.bounds[1] + inner;

    ctx.save();
    if (!this.value.on) {
      ctx.globalAlpha = app.canvas.editor_alpha * 0.42;
    }
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.textBaseline = "middle";

    const number = drawNumberControl(ctx, width - margin - inner, y, height, this.value.strength);
    this.hitAreas.weightDec.bounds = number.dec;
    this.hitAreas.weightValue.bounds = number.value;
    this.hitAreas.weightInc.bounds = number.inc;
    this.hitAreas.weightAny.bounds = number.any;

    const typeWidth = 70;
    const typeX = number.x - inner - typeWidth;
    roundRect(ctx, typeX, y + 3, typeWidth, height - 6, 5);
    ctx.fillStyle = "rgba(255,255,255,0.08)";
    ctx.fill();
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.textAlign = "center";
    ctx.fillText(fitString(ctx, this.value.type || "general", typeWidth - 12), typeX + typeWidth / 2, midY);
    ctx.fillText("▾", typeX + typeWidth - 8, midY);
    this.hitAreas.type.bounds = [typeX, typeWidth];

    const loraWidth = Math.max(40, typeX - inner - x);
    ctx.textAlign = "left";
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.fillText(fitString(ctx, this.value.lora || "None", loraWidth), x, midY);
    this.hitAreas.lora.bounds = [x, loraWidth];
    ctx.restore();
  }

  onToggle(event, pos, node) {
    this.value.on = !this.value.on;
    node.setDirtyCanvas(true, true);
    return true;
  }

  onLoraClick(event, pos, node) {
    void fetchLoraValues(this.nodeData).then((freshValues) => {
      const values = valuesWithCurrent(freshValues, this.value.lora);
      new LiteGraph.ContextMenu(values, {
        event,
        title: "Choose LoRA",
        callback: (value) => {
          if (typeof value === "string") {
            this.value.lora = value;
            node.setDirtyCanvas(true, true);
          }
        },
      });
    });
    return true;
  }

  onTypeClick(event, pos, node) {
    new LiteGraph.ContextMenu(LORA_TYPES, {
      event,
      title: "LoRA Type",
      callback: (value) => {
        if (typeof value === "string") {
          this.value.type = value;
          node.setDirtyCanvas(true, true);
        }
      },
    });
    return true;
  }

  stepWeight(direction, node) {
    const current = Number(this.value.strength ?? 1.0);
    this.value.strength = Math.round(((Number.isFinite(current) ? current : 1.0) + direction * 0.05) * 100) / 100;
    node.setDirtyCanvas(true, true);
  }

  onWeightDec(event, pos, node) {
    this.stepWeight(-1, node);
    return true;
  }

  onWeightInc(event, pos, node) {
    this.stepWeight(1, node);
    return true;
  }

  onWeightMove(event, pos, node) {
    if (event.deltaX) {
      this.draggedWeight = true;
      const current = Number(this.value.strength ?? 1.0);
      this.value.strength = Math.round(((Number.isFinite(current) ? current : 1.0) + event.deltaX * 0.02) * 100) / 100;
      node.setDirtyCanvas(true, true);
    }
    return true;
  }

  onWeightValue(event, pos, node) {
    if (this.draggedWeight) {
      this.draggedWeight = false;
      return true;
    }
    app.canvas.prompt("Base weight", this.value.strength, (value) => {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) {
        this.value.strength = parsed;
        node.setDirtyCanvas(true, true);
      }
    }, event);
    return true;
  }
}

class FunPackButtonWidget extends FunPackBaseWidget {
  constructor(name, label, callback) {
    super(name);
    this.options.serialize = false;
    this.value = "";
    this.label = label;
    this.callback = callback;
    this.hitAreas = {
      button: { bounds: [0, 0], onClick: this.onButtonClick },
    };
  }

  computeSize(width) {
    return [width, LiteGraph.NODE_WIDGET_HEIGHT + 2];
  }

  draw(ctx, node, width, y, height) {
    const x = 20;
    const w = width - 40;
    this.hitAreas.button.bounds = [x, y + 1, w, height - 2];
    ctx.save();
    roundRect(ctx, x, y + 1, w, height - 2, 3);
    ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
    ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(this.label, width / 2, y + height / 2);
    ctx.restore();
  }

  onMouseClick(event, pos, node) {
    return this.callback(event, pos, node);
  }

  onButtonClick(event, pos, node) {
    return this.callback(event, pos, node);
  }
}

function isDefaultLoraWidget(widget) {
  return /^lora_\d+(?:_type|_base_weight)?$/.test(widget.name || "");
}

function isManagedWidget(widget) {
  return (
    isDefaultLoraWidget(widget) ||
    widget.name === HEADER_WIDGET_NAME ||
    widget.name === ADD_BUTTON_NAME ||
    widget.__funpackLoraRow === true
  );
}

function getLoraRows(node) {
  return (node.widgets || []).filter((widget) => widget.__funpackLoraRow === true);
}

function allRowsState(node) {
  const rows = getLoraRows(node);
  if (!rows.length) {
    return false;
  }
  const allOn = rows.every((row) => row.value.on !== false);
  const allOff = rows.every((row) => row.value.on === false);
  if (allOn) {
    return true;
  }
  if (allOff) {
    return false;
  }
  return null;
}

function toggleAllRows(node) {
  const rows = getLoraRows(node);
  const target = allRowsState(node) !== true;
  for (const row of rows) {
    row.value.on = target;
  }
}

function nextLoraIndex(node) {
  let next = 0;
  for (const widget of node.widgets || []) {
    const match = /^lora_(\d+)$/.exec(widget.name || "");
    if (match) {
      next = Math.max(next, Number(match[1]) + 1);
    }
  }
  return next;
}

function resizeNode(node) {
  const computed = node.computeSize();
  node.size[0] = Math.max(node.size[0], computed[0]);
  node.size[1] = computed[1];
}

function addLoraRow(node, nodeData, value = {}, markDirty = true) {
  const row = new FunPackLoraRowWidget(`lora_${nextLoraIndex(node)}`, value, nodeData);
  row.__funpackLoraRow = true;
  node.addCustomWidget(row);
  moveButtonToEnd(node);
  resizeNode(node);
  if (markDirty) {
    node.setDirtyCanvas(true, true);
  }
  return row;
}

function moveButtonToEnd(node) {
  const buttonIndex = (node.widgets || []).findIndex((widget) => widget.name === ADD_BUTTON_NAME);
  if (buttonIndex < 0 || buttonIndex === node.widgets.length - 1) {
    return;
  }
  const [button] = node.widgets.splice(buttonIndex, 1);
  node.widgets.push(button);
}

function removeManagedWidgets(node) {
  node.widgets = (node.widgets || []).filter((widget) => !isManagedWidget(widget));
}

function collectVisibleDefaultRows(node) {
  const rows = new Map();
  for (const widget of node.widgets || []) {
    const match = /^lora_(\d+)(?:_(type|base_weight))?$/.exec(widget.name || "");
    if (!match) {
      continue;
    }
    const index = Number(match[1]);
    const field = match[2] || "lora";
    const row = rows.get(index) || { on: true, lora: "None", type: "general", strength: 1.0 };
    if (field === "lora") {
      row.lora = widget.value || "None";
    } else if (field === "type") {
      row.type = LORA_TYPES.includes(widget.value) ? widget.value : "general";
    } else {
      row.strength = Number(widget.value ?? 1.0);
    }
    rows.set(index, row);
  }
  return [...rows.keys()].sort((a, b) => a - b).map((index) => rows.get(index));
}

function serializedBaseWidgetCount(node) {
  let count = 0;
  for (const widget of node.widgets || []) {
    if (widget.serialize === false || widget.options?.serialize === false || isDefaultLoraWidget(widget)) {
      continue;
    }
    if (BASE_WIDGET_NAMES.has(widget.name)) {
      count += 1;
    }
  }
  return count;
}

function parseSerializedRows(values, baseCount) {
  if (!Array.isArray(values)) {
    return [];
  }

  const rowValues = values.slice(baseCount);
  const objectRows = rowValues
    .filter((value) => value && typeof value === "object" && Object.prototype.hasOwnProperty.call(value, "lora"))
    .map(normalizeRowValue);
  if (objectRows.length) {
    return objectRows;
  }

  const rows = [];
  for (let offset = 0; offset + 2 < rowValues.length; offset += 3) {
    rows.push(normalizeRowValue({
      lora: rowValues[offset],
      type: rowValues[offset + 1],
      strength: rowValues[offset + 2],
    }));
  }
  return rows;
}

function ensureCompactLoraUi(node, nodeData, info = null) {
  trackNode(node);
  const defaultRows = collectVisibleDefaultRows(node);
  const baseCount = serializedBaseWidgetCount(node);
  const serializedRows = parseSerializedRows(info?.widgets_values, baseCount);
  const rows = serializedRows.length ? serializedRows : defaultRows;

  removeManagedWidgets(node);
  node.addCustomWidget(new FunPackLoraHeaderWidget());
  if (rows.length) {
    for (const row of rows) {
      addLoraRow(node, nodeData, row, false);
    }
  } else {
    addLoraRow(node, nodeData, {}, false);
  }
  node.addCustomWidget(new FunPackButtonWidget(ADD_BUTTON_NAME, "+ Add LoRA", (event, pos, currentNode) => {
    void fetchLoraValues(nodeData).then((values) => {
      new LiteGraph.ContextMenu(values, {
        event,
        title: "Choose LoRA",
        callback: (value) => {
          if (typeof value === "string" && value !== "None") {
            addLoraRow(currentNode, nodeData, { lora: value });
          } else {
            addLoraRow(currentNode, nodeData, {});
          }
        },
      });
    });
    return true;
  }));
  moveButtonToEnd(node);
  resizeNode(node);
}

function updateLoraWidgets(node, values) {
  for (const widget of getLoraRows(node)) {
    widget.updateLoraValues(values);
  }
  node.setDirtyCanvas(true, true);
}

async function refreshLoraWidgets(node, nodeData) {
  updateLoraWidgets(node, await fetchLoraValues(nodeData));
}

function trackNode(node) {
  trackedNodes.add(node);
}

async function refreshTrackedLoraWidgets(nodeData = latestNodeData) {
  if (!nodeData || pendingRefresh) {
    return pendingRefresh;
  }

  pendingRefresh = (async () => {
    const values = await fetchLoraValues(nodeData);
    for (const node of [...trackedNodes]) {
      if (!node?.graph) {
        trackedNodes.delete(node);
        continue;
      }
      updateLoraWidgets(node, values);
    }
  })().finally(() => {
    pendingRefresh = null;
  });

  return pendingRefresh;
}

function scheduleTrackedRefresh(nodeData = latestNodeData) {
  window.setTimeout(() => {
    void refreshTrackedLoraWidgets(nodeData);
  }, 100);
}

function wrapRefreshFunction(functionName) {
  const original = app[functionName];
  if (typeof original !== "function" || original.__funpackLoraWrapped) {
    return;
  }

  const wrapped = function () {
    const result = original.apply(this, arguments);
    Promise.resolve(result).finally(() => scheduleTrackedRefresh());
    return result;
  };
  wrapped.__funpackLoraWrapped = true;
  app[functionName] = wrapped;
}

function hookComfyRefreshControls() {
  for (const functionName of ["refreshComboInNodes", "refreshComboInNode", "refreshNodeDefs"]) {
    wrapRefreshFunction(functionName);
  }
}

function isRefreshControl(element) {
  for (let current = element; current && current !== document.body; current = current.parentElement) {
    const text = [
      current.title,
      current.ariaLabel,
      current.textContent,
    ].filter(Boolean).join(" ").toLowerCase();
    if (text.includes("refresh")) {
      return true;
    }
  }
  return false;
}

app.registerExtension({
  name: "funpack.loraWeights",
  setup() {
    hookComfyRefreshControls();
    window.setTimeout(hookComfyRefreshControls, 0);
    window.setTimeout(hookComfyRefreshControls, 1000);
    document.addEventListener("click", (event) => {
      if (isRefreshControl(event.target)) {
        scheduleTrackedRefresh();
      }
    }, true);
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) {
      return;
    }
    latestNodeData = nodeData;
    hookComfyRefreshControls();

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      originalOnNodeCreated?.apply(this, arguments);
      ensureCompactLoraUi(this, nodeData);
      void refreshLoraWidgets(this, nodeData);
    };

    const originalConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      originalConfigure?.apply(this, arguments);
      ensureCompactLoraUi(this, nodeData, info);
      void refreshLoraWidgets(this, nodeData);
    };
  },
});
