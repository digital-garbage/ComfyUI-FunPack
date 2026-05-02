import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "FunPackTemplateManager";
const NONE_TEMPLATE = "-None-";
const BUTTON_NAMES = new Set([
  "funpack_template_save",
  "funpack_template_update",
  "funpack_template_delete",
  "funpack_template_export",
  "funpack_template_import",
  "funpack_template_refresh",
]);

let cachedTemplates = [NONE_TEMPLATE];
let latestNodeData = null;
const trackedNodes = new Set();
let pendingRefresh = null;

function widgetByName(node, name) {
  return (node.widgets || []).find((widget) => widget.name === name);
}

function hideWidget(widget) {
  if (!widget || widget.__funpackHidden) {
    return;
  }
  widget.__funpackHidden = true;
  widget.computeSize = () => [0, -4];
  widget.type = "hidden";
}

function templateWidget(node) {
  return widgetByName(node, "template");
}

function actionWidget(node) {
  return widgetByName(node, "action");
}

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

function updateTemplateWidget(node, values = cachedTemplates) {
  const widget = templateWidget(node);
  if (!widget) {
    return;
  }
  const current = widget.value;
  const nextValues = current && !values.includes(current) ? [...values, current] : values;
  if (widget.options) {
    widget.options.values = nextValues;
  }
  if (!widget.value || !nextValues.includes(widget.value)) {
    widget.value = nextValues[0] || NONE_TEMPLATE;
  }
}

async function fetchTemplates() {
  try {
    const response = await api.fetchApi(`/funpack/templates?cache_bust=${Date.now()}`, {
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    const values = Array.isArray(data.templates) && data.templates.length ? data.templates : [NONE_TEMPLATE];
    cachedTemplates = values;
    return values;
  } catch (error) {
    console.warn("FunPack: failed to refresh template list", error);
    return cachedTemplates;
  }
}

async function refreshNode(node) {
  const values = await fetchTemplates();
  updateTemplateWidget(node, values);
  node.setDirtyCanvas(true, true);
}

async function refreshTrackedNodes() {
  if (pendingRefresh) {
    return pendingRefresh;
  }
  pendingRefresh = (async () => {
    const values = await fetchTemplates();
    for (const node of [...trackedNodes]) {
      if (!node?.graph) {
        trackedNodes.delete(node);
        continue;
      }
      updateTemplateWidget(node, values);
      node.setDirtyCanvas(true, true);
    }
  })().finally(() => {
    pendingRefresh = null;
  });
  return pendingRefresh;
}

function scheduleRefresh() {
  window.setTimeout(() => {
    void refreshTrackedNodes();
  }, 100);
}

function wrapRefreshFunction(functionName) {
  const original = app[functionName];
  if (typeof original !== "function" || original.__funpackTemplateWrapped) {
    return;
  }
  const wrapped = function () {
    const result = original.apply(this, arguments);
    Promise.resolve(result).finally(() => scheduleRefresh());
    return result;
  };
  wrapped.__funpackTemplateWrapped = true;
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

async function queueGraph() {
  if (typeof app.queuePrompt === "function") {
    return await app.queuePrompt(0);
  }
  const button = document.querySelector("#queue-button, .queue-button, button[title='Queue Prompt']");
  if (button) {
    button.click();
  }
}

async function runTemplateAction(node, action) {
  const widget = actionWidget(node);
  if (!widget) {
    return;
  }
  widget.value = action;
  node.setDirtyCanvas(true, true);
  try {
    await queueGraph();
  } finally {
    window.setTimeout(() => {
      widget.value = "load";
      node.setDirtyCanvas(true, true);
      scheduleRefresh();
    }, 250);
  }
}

async function exportTemplates() {
  const response = await api.fetchApi(`/funpack/templates/export?cache_bust=${Date.now()}`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`Export failed with HTTP ${response.status}`);
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "funpack_templates.json";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function importTemplates() {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".json,application/json";
  input.onchange = async () => {
    const file = input.files?.[0];
    if (!file) {
      return;
    }
    try {
      const data = JSON.parse(await file.text());
      const response = await api.fetchApi("/funpack/templates/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.error || `Import failed with HTTP ${response.status}`);
      }
      await refreshTrackedNodes();
    } catch (error) {
      console.warn("FunPack: template import failed", error);
      app.canvas?.prompt?.("Template import failed", error.message || String(error), () => {});
    }
  };
  input.click();
}

class FunPackTemplateButton {
  constructor(name, label, callback) {
    this.name = name;
    this.type = "custom";
    this.options = { serialize: false };
    this.value = "";
    this.label = label;
    this.callback = callback;
  }

  computeSize(width) {
    return [width, LiteGraph.NODE_WIDGET_HEIGHT + 2];
  }

  draw(ctx, _node, width, y, height) {
    const x = 20;
    const w = width - 40;
    const h = height - 2;
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(x, y + 1, w, h, [3]);
    ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
    ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(fitString(ctx, this.label, w - 12), width / 2, y + height / 2);
    ctx.restore();
  }

  mouse(event, _pos, node) {
    if (event.type === "pointerup") {
      this.callback(node);
      return true;
    }
    return event.type === "pointerdown";
  }
}

function removeTemplateButtons(node) {
  node.widgets = (node.widgets || []).filter((widget) => !BUTTON_NAMES.has(widget.name));
}

function addTemplateButtons(node) {
  removeTemplateButtons(node);
  const buttons = [
    new FunPackTemplateButton("funpack_template_save", "Save", (currentNode) => void runTemplateAction(currentNode, "save")),
    new FunPackTemplateButton("funpack_template_update", "Update", (currentNode) => void runTemplateAction(currentNode, "update")),
    new FunPackTemplateButton("funpack_template_delete", "Delete", (currentNode) => void runTemplateAction(currentNode, "delete")),
    new FunPackTemplateButton("funpack_template_export", "Export", () => void exportTemplates()),
    new FunPackTemplateButton("funpack_template_import", "Import", () => importTemplates()),
    new FunPackTemplateButton("funpack_template_refresh", "Refresh", (currentNode) => void refreshNode(currentNode)),
  ];
  for (const button of buttons) {
    node.addCustomWidget(button);
  }
}

function setupTemplateNode(node, nodeData) {
  latestNodeData = nodeData;
  trackedNodes.add(node);
  hideWidget(actionWidget(node));
  updateTemplateWidget(node);
  addTemplateButtons(node);
  void refreshNode(node);
}

app.registerExtension({
  name: "funpack.templateManager",
  setup() {
    hookComfyRefreshControls();
    window.setTimeout(hookComfyRefreshControls, 0);
    window.setTimeout(hookComfyRefreshControls, 1000);
    document.addEventListener("click", (event) => {
      if (isRefreshControl(event.target)) {
        scheduleRefresh();
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
      setupTemplateNode(this, nodeData);
    };

    const originalConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      originalConfigure?.apply(this, arguments);
      setupTemplateNode(this, nodeData, info);
    };
  },
});
