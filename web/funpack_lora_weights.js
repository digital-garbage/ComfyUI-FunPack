import { app } from "../../scripts/app.js";

const NODE_NAME = "FunPackApplyLoraWeights";
const LORA_TYPES = ["general", "concept", "style", "quality", "character"];
const ADD_BUTTON_NAME = "+ Add LoRA";

function getLoraValues(nodeData) {
  return nodeData?.input?.optional?.lora_0?.[0] || ["None"];
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

function addLoraRow(node, nodeData, value = {}, markDirty = true) {
  const index = getNextLoraIndex(node);
  const loraWidget = node.addWidget(
    "combo",
    `lora_${index}`,
    value.lora || "None",
    undefined,
    { values: getLoraValues(nodeData) },
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

  const button = node.addWidget("button", ADD_BUTTON_NAME, null, () => {
    addLoraRow(node, nodeData);
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
    };

    const originalConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      originalConfigure?.apply(this, arguments);
      ensureAddButton(this, nodeData);
      restoreExtraRows(this, nodeData, info);
    };
  },
});
