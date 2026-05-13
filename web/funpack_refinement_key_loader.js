import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "FunPackRefinementKeyLoader";
const NONE_KEY = "-None-";
const BUTTON_NAMES = new Set([
  "funpack_refinement_key_export",
  "funpack_refinement_key_import",
  "funpack_refinement_key_refresh",
]);

let cachedKeys = [NONE_KEY];
const trackedNodes = new Set();
let pendingRefresh = null;

function widgetByName(node, name) {
  return (node.widgets || []).find((widget) => widget.name === name);
}

function keyWidget(node) {
  return widgetByName(node, "refinement_key");
}

function keyNameWidget(node) {
  return widgetByName(node, "key_name");
}

function selectedKey(node) {
  const selected = String(keyWidget(node)?.value || "").trim();
  if (selected && selected !== NONE_KEY) {
    return selected;
  }
  return String(keyNameWidget(node)?.value || "").trim();
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

function updateKeyWidget(node, values = cachedKeys) {
  const widget = keyWidget(node);
  if (!widget) {
    return;
  }
  const current = widget.value;
  const nextValues = current && !values.includes(current) ? [...values, current] : values;
  if (widget.options) {
    widget.options.values = nextValues;
  }
  if (!widget.value || !nextValues.includes(widget.value)) {
    widget.value = nextValues[0] || NONE_KEY;
  }
}

async function fetchKeys() {
  try {
    const response = await api.fetchApi(`/funpack/refinement_keys?cache_bust=${Date.now()}`, {
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    const values = Array.isArray(data.keys) && data.keys.length ? data.keys : [NONE_KEY];
    cachedKeys = values;
    return values;
  } catch (error) {
    console.warn("FunPack: failed to refresh refinement keys", error);
    return cachedKeys;
  }
}

async function refreshNode(node) {
  const values = await fetchKeys();
  updateKeyWidget(node, values);
  node.setDirtyCanvas(true, true);
}

async function refreshTrackedNodes() {
  if (pendingRefresh) {
    return pendingRefresh;
  }
  pendingRefresh = (async () => {
    const values = await fetchKeys();
    for (const node of [...trackedNodes]) {
      if (!node?.graph) {
        trackedNodes.delete(node);
        continue;
      }
      updateKeyWidget(node, values);
      node.setDirtyCanvas(true, true);
    }
  })().finally(() => {
    pendingRefresh = null;
  });
  return pendingRefresh;
}

async function exportKey(node) {
  const key = selectedKey(node);
  if (!key) {
    app.canvas?.prompt?.("Refinement key export skipped", "Select or type a refinement key first.", () => {});
    return;
  }
  const response = await api.fetchApi(`/funpack/refinement_keys/export?key=${encodeURIComponent(key)}&cache_bust=${Date.now()}`, {
    cache: "no-store",
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error || `Export failed with HTTP ${response.status}`);
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `funpack_refinement_${key}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function importKey() {
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
      const response = await api.fetchApi("/funpack/refinement_keys/import", {
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
      console.warn("FunPack: refinement key import failed", error);
      app.canvas?.prompt?.("Refinement key import failed", error.message || String(error), () => {});
    }
  };
  input.click();
}

class FunPackRefinementKeyButton {
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

function removeButtons(node) {
  node.widgets = (node.widgets || []).filter((widget) => !BUTTON_NAMES.has(widget.name));
}

function addButtons(node) {
  removeButtons(node);
  const buttons = [
    new FunPackRefinementKeyButton("funpack_refinement_key_export", "Export", (currentNode) => void exportKey(currentNode)),
    new FunPackRefinementKeyButton("funpack_refinement_key_import", "Import", () => importKey()),
    new FunPackRefinementKeyButton("funpack_refinement_key_refresh", "Refresh", (currentNode) => void refreshNode(currentNode)),
  ];
  for (const button of buttons) {
    node.addCustomWidget(button);
  }
}

function setupNode(node) {
  trackedNodes.add(node);
  updateKeyWidget(node);
  addButtons(node);
  void refreshNode(node);
}

app.registerExtension({
  name: "funpack.refinementKeyLoader",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) {
      return;
    }

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      originalOnNodeCreated?.apply(this, arguments);
      setupNode(this);
    };

    const originalConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      originalConfigure?.apply(this, arguments);
      setupNode(this, info);
    };
  },
});
