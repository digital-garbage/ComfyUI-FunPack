import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "FunPackApplyLoraWeights";
const LORA_TYPES = ["general", "concept", "style", "quality", "character"];
const ADD_BUTTON_NAME = "+ Add LoRA";

let cachedLoraValues = null;

function getLoraValues(nodeData) {
  return cachedLoraValues || nodeData?.input?.optional?.lora_0?.[0] || ["None"];
}

async function fetchLoraValues(nodeData) {
  try {
    const response = await api.fetchApi("/funpack/loras");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const values = await response.json();
    if (Array.isArray(values) && values.length) {
      cachedLoraValues = values;
      return values;
    }
  } catch (error) {
    console.warn("FunPack: failed to refresh LoRA list", error);
  }
  cachedLoraValues = getLoraValues(nodeData);
  return cachedLoraValues;
}

function valuesWithCurrent(values, current) {
  if (!current || values.includes(current)) {
    return values;
  }
  return [...values, current];
}

function isLoraCombo(widget) {
  return /^lora_\d+$/.test(widget.name || "");
}

function updateLoraWidgets(node, values) {
  for (const widget of node.widgets || []) {
    if (!isLoraCombo(widget)) {
      continue;
    }
    widget.options = widget.options || {};
    widget.options.values = valuesWithCurrent(values, widget.value);
  }
  node.setDirtyCanvas(true, true);
}

async function refreshLoraWidgets(node, nodeData) {
  updateLoraWidgets(node, await fetchLoraValues(nodeData));
}

function getNextLoraIndex(node) {
  let next = 0;
  for (const widget of node.widgets || []) {
    const match = /^lora_(\d+)$/.exec(widget.name || "");
    if (match) {
      next = Math.max(next, Number(match[1]) + 1);
    }
  }
  return next;
}

function moveAddButtonToEnd(node) {
  const button = (node.widgets || []).find((widget) => widget.name === ADD_BUTTON_NAME);
  if (!button) {
    return;
  }
  const index = node.widgets.indexOf(button);
  if (index >= 0 && index !== node.widgets.length - 1) {
    node.widgets.splice(index, 1);
    node.widgets.push(button);
  }
}

function resizeNode(node) {
  const computed = node.computeSize();
  node.size[0] = Math.max(node.size[0], computed[0]);
  node.size[1] = computed[1];
}

function addLoraRow(node, nodeData, values, value = {}, markDirty = true) {
  const index = getNextLoraIndex(node);
  const loraWidget = node.addWidget(
    "combo",
    `lora_${index}`,
    value.lora || "None",
    undefined,
    { values: valuesWithCurrent(values || getLoraValues(nodeData), value.lora) },
  );
  const typeWidget = node.addWidget(
    "combo",
    `lora_${index}_type`,
    value.type || "general",
    undefined,
    { values: LORA_TYPES },
  );
  const weightWidget = node.addWidget(
    "number",
    `lora_${index}_base_weight`,
    value.base_weight ?? 1.0,
    undefined,
    { min: -10.0, max: 10.0, step: 0.01, precision: 3 },
  );

  loraWidget.serialize = true;
  typeWidget.serialize = true;
  weightWidget.serialize = true;
  moveAddButtonToEnd(node);
  resizeNode(node);
  if (markDirty) {
    node.setDirtyCanvas(true, true);
  }
}

function ensureAddButton(node, nodeData) {
  if ((node.widgets || []).some((widget) => widget.name === ADD_BUTTON_NAME)) {
    moveAddButtonToEnd(node);
    return;
  }

  const button = node.addWidget("button", ADD_BUTTON_NAME, null, async () => {
    addLoraRow(node, nodeData, await fetchLoraValues(nodeData));
  });
  button.serialize = false;
  moveAddButtonToEnd(node);
  resizeNode(node);
}

function restoreExtraRows(node, nodeData, info) {
  const values = info?.widgets_values;
  if (!Array.isArray(values)) {
    return;
  }

  node.widgets = (node.widgets || []).filter((widget) => {
    const match = /^lora_(\d+)(?:_type|_base_weight)?$/.exec(widget.name || "");
    return !match || Number(match[1]) === 0;
  });

  const knownWidgetCount = 6;
  for (let offset = knownWidgetCount; offset + 2 < values.length; offset += 3) {
    addLoraRow(
      node,
      nodeData,
      getLoraValues(nodeData),
      {
        lora: values[offset],
        type: values[offset + 1],
        base_weight: values[offset + 2],
      },
      false,
    );
  }
}

app.registerExtension({
  name: "funpack.loraWeights",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) {
      return;
    }

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      originalOnNodeCreated?.apply(this, arguments);
      ensureAddButton(this, nodeData);
      void refreshLoraWidgets(this, nodeData);
    };

    const originalConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      originalConfigure?.apply(this, arguments);
      ensureAddButton(this, nodeData);
      restoreExtraRows(this, nodeData, info);
      void refreshLoraWidgets(this, nodeData);
    };
  },
});
