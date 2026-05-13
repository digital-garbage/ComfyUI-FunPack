import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "FunPackSceneBuilder";
const NONE_SCENE = "-None-";
const PAYLOAD_WIDGET = "scene_payload";
const ACTION_WIDGET = "action";
const GROUP_ORDER = ["subject", "appearance", "action", "camera", "environment", "style", "quality", "details", "negative"];

let sceneData = null;
let activePicker = null;
const trackedNodes = new Set();

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

function getWidgetValue(node, name, fallback = "") {
  const widget = widgetByName(node, name);
  return widget ? widget.value : fallback;
}

function setWidgetValue(node, name, value) {
  const widget = widgetByName(node, name);
  if (widget) {
    widget.value = value;
  }
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

function roundRect(ctx, x, y, width, height, radius = 5) {
  ctx.beginPath();
  ctx.roundRect(x, y, width, height, [radius]);
}

function normalizePayload(payload) {
  if (typeof payload === "string") {
    try {
      payload = JSON.parse(payload);
    } catch (_error) {
      payload = {};
    }
  }
  if (!payload || typeof payload !== "object") {
    payload = {};
  }
  return {
    positive_phrases: Array.isArray(payload.positive_phrases) ? payload.positive_phrases.filter(Boolean) : [],
    negative_phrases: Array.isArray(payload.negative_phrases) ? payload.negative_phrases.filter(Boolean) : [],
  };
}

function payloadFromNode(node) {
  return normalizePayload(getWidgetValue(node, PAYLOAD_WIDGET, "{}"));
}

function writePayload(node, payload) {
  setWidgetValue(node, PAYLOAD_WIDGET, JSON.stringify(normalizePayload(payload)));
  node.setDirtyCanvas(true, true);
}

function sceneNames() {
  const names = sceneData?.scenes;
  return Array.isArray(names) && names.length ? names : [NONE_SCENE];
}

function updateSceneCombo(node) {
  const widget = widgetByName(node, "scene");
  if (!widget) {
    return;
  }
  const values = sceneNames();
  const current = widget.value;
  widget.options.values = current && !values.includes(current) ? [...values, current] : values;
  if (!widget.value || !widget.options.values.includes(widget.value)) {
    widget.value = widget.options.values[0] || NONE_SCENE;
  }
}

function currentRefinementKey(node) {
  return String(getWidgetValue(node, "refinement_key", "") || "").trim();
}

async function fetchScenes(refinementKey = "") {
  try {
    const params = new URLSearchParams({ cache_bust: String(Date.now()) });
    if (refinementKey) {
      params.set("key", refinementKey);
    }
    const response = await api.fetchApi(`/funpack/scenes?${params.toString()}`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    sceneData = await response.json();
  } catch (error) {
    console.warn("FunPack: failed to refresh scenes", error);
  }
  return sceneData;
}

async function refreshTracked() {
  for (const node of [...trackedNodes]) {
    if (!node?.graph) {
      trackedNodes.delete(node);
      continue;
    }
    await fetchScenes(currentRefinementKey(node));
    updateSceneCombo(node);
    node.setDirtyCanvas(true, true);
  }
}

function scheduleRefresh() {
  window.setTimeout(() => void refreshTracked(), 100);
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

async function runSceneAction(node, action) {
  setWidgetValue(node, ACTION_WIDGET, action);
  node.setDirtyCanvas(true, true);
  try {
    await queueGraph();
  } finally {
    window.setTimeout(() => {
      setWidgetValue(node, ACTION_WIDGET, "load");
      scheduleRefresh();
      node.setDirtyCanvas(true, true);
    }, 250);
  }
}

async function exportScenes(node) {
  const params = new URLSearchParams({ cache_bust: String(Date.now()) });
  const key = currentRefinementKey(node);
  if (key) {
    params.set("key", key);
  }
  const response = await api.fetchApi(`/funpack/scenes/export?${params.toString()}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Export failed with HTTP ${response.status}`);
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "funpack_scenes.json";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function importScenes(node) {
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
      const params = new URLSearchParams();
      const key = currentRefinementKey(node);
      if (key) {
        params.set("key", key);
      }
      const response = await api.fetchApi(`/funpack/scenes/import${params.toString() ? `?${params.toString()}` : ""}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.error || `Import failed with HTTP ${response.status}`);
      }
      await refreshTracked();
    } catch (error) {
      console.warn("FunPack: scene import failed", error);
      app.canvas?.prompt?.("Scene import failed", error.message || String(error), () => {});
    }
  };
  input.click();
}

function closePicker() {
  if (activePicker) {
    activePicker.remove();
    activePicker = null;
  }
}

function searchablePicker(event, values, currentValue, onSelect, title = "Search") {
  closePicker();
  const root = document.createElement("div");
  Object.assign(root.style, {
    position: "fixed",
    zIndex: 10000,
    minWidth: "320px",
    maxWidth: "520px",
    maxHeight: "420px",
    padding: "8px",
    border: "1px solid rgba(180, 190, 200, 0.35)",
    borderRadius: "8px",
    background: "rgba(30, 32, 36, 0.98)",
    boxShadow: "0 16px 40px rgba(0, 0, 0, 0.35)",
    color: "#ddd",
    font: "12px sans-serif",
  });
  const input = document.createElement("input");
  input.type = "search";
  input.placeholder = title;
  input.value = currentValue && currentValue !== "None" ? currentValue : "";
  Object.assign(input.style, {
    boxSizing: "border-box",
    width: "100%",
    margin: "0 0 6px 0",
    padding: "7px 8px",
    border: "1px solid rgba(180, 190, 200, 0.35)",
    borderRadius: "5px",
    background: "#15171a",
    color: "#fff",
    outline: "none",
  });
  const list = document.createElement("div");
  Object.assign(list.style, { maxHeight: "350px", overflowY: "auto" });
  const render = () => {
    const parts = input.value.toLowerCase().split(/\s+/).filter(Boolean);
    const filtered = values.filter((value) => {
      const haystack = String(value).toLowerCase().replace(/[_\-./\\]+/g, " ");
      return parts.every((part) => haystack.includes(part));
    }).slice(0, 160);
    list.replaceChildren();
    for (const value of filtered) {
      const row = document.createElement("button");
      row.type = "button";
      row.textContent = value;
      row.title = value;
      Object.assign(row.style, {
        display: "block",
        width: "100%",
        padding: "6px 8px",
        border: "0",
        borderRadius: "4px",
        background: value === currentValue ? "rgba(120, 150, 170, 0.32)" : "transparent",
        color: "#ddd",
        textAlign: "left",
        cursor: "pointer",
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
      });
      row.addEventListener("click", () => {
        onSelect(value);
        closePicker();
      });
      list.append(row);
    }
  };
  input.addEventListener("input", render);
  input.addEventListener("keydown", (keyEvent) => {
    if (keyEvent.key === "Escape") {
      closePicker();
      keyEvent.stopPropagation();
    } else if (keyEvent.key === "Enter") {
      const first = list.querySelector("button");
      if (first) {
        onSelect(first.textContent);
        closePicker();
      }
      keyEvent.preventDefault();
      keyEvent.stopPropagation();
    }
  });
  root.append(input, list);
  document.body.append(root);
  activePicker = root;
  const rect = root.getBoundingClientRect();
  root.style.left = `${Math.min(window.innerWidth - rect.width - 12, Math.max(12, event.clientX ?? 12))}px`;
  root.style.top = `${Math.min(window.innerHeight - rect.height - 12, Math.max(12, event.clientY ?? 12))}px`;
  window.setTimeout(() => {
    document.addEventListener("pointerdown", function outsideClick(clickEvent) {
      if (!root.contains(clickEvent.target)) {
        closePicker();
        document.removeEventListener("pointerdown", outsideClick, true);
      }
    }, true);
  }, 0);
  render();
  input.focus();
  input.select();
}

function scenePayloadFromStoredScene(scene) {
  return normalizePayload({
    positive_phrases: scene?.positive_phrases || [],
    negative_phrases: scene?.negative_phrases || [],
  });
}

function storedScene(node) {
  const name = getWidgetValue(node, "scene", NONE_SCENE);
  return sceneData?.data?.scenes?.[name] || null;
}

function loadStoredSceneIntoNode(node) {
  const name = getWidgetValue(node, "scene", NONE_SCENE);
  const scene = storedScene(node);
  if (!scene || name === NONE_SCENE) {
    return;
  }
  if (node.__funpackLoadedScene === name) {
    return;
  }
  node.__funpackLoadedScene = name;
  setWidgetValue(node, "scene_name", scene.name || name);
  setWidgetValue(node, "aliases", Array.isArray(scene.aliases) ? scene.aliases.join(", ") : "");
  setWidgetValue(node, "output_mode", scene.output_mode || "Manual");
  setWidgetValue(node, "refinement_key", scene.refinement_key || "");
  writePayload(node, scenePayloadFromStoredScene(scene));
}

function button(ctx, widget, key, label, x, y, width, height) {
  widget.hitAreas[key] = [x, y, width, height];
  roundRect(ctx, x, y, width, height, 4);
  ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
  ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(fitString(ctx, label, width - 8), x + width / 2, y + height / 2);
}

function drawChip(ctx, widget, key, text, x, y, selected, maxWidth) {
  const width = Math.min(maxWidth, Math.max(34, ctx.measureText(text).width + 18));
  widget.hitAreas[key] = [x, y, width, 19];
  roundRect(ctx, x, y, width, 19, 5);
  ctx.fillStyle = selected ? "rgba(72, 178, 112, 0.85)" : "rgba(255,255,255,0.065)";
  ctx.strokeStyle = selected ? "rgba(150, 230, 170, 0.75)" : "rgba(255,255,255,0.13)";
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = selected ? "#06140b" : LiteGraph.WIDGET_TEXT_COLOR;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(fitString(ctx, text, width - 10), x + width / 2, y + 9.5);
  return width;
}

function selectedSet(list) {
  return new Set((list || []).map((item) => String(item).toLowerCase()));
}

class SceneBuilderWidget {
  constructor(node) {
    this.node = node;
    this.name = "funpack_scene_builder";
    this.type = "custom";
    this.options = { serialize: false };
    this.value = "";
    this.hitAreas = {};
  }

  computeSize(width) {
    return [width, 430];
  }

  draw(ctx, node, width, y) {
    loadStoredSceneIntoNode(node);
    this.hitAreas = {};
    const payload = payloadFromNode(node);
    const memory = Array.isArray(sceneData?.memory) ? sceneData.memory : [];
    const margin = 10;
    let cy = y + 4;
    const innerWidth = width - margin * 2;
    const widgetHeight = this.computeSize(width)[1];
    const bottom = y + widgetHeight - 8;

    ctx.save();
    ctx.beginPath();
    ctx.rect(0, y, width, widgetHeight);
    ctx.clip();
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.font = "12px sans-serif";
    ctx.fillText(`Scene Builder: ${getWidgetValue(node, "output_mode", "Manual")}`, margin, cy + 9);
    button(ctx, this, "refresh", "Refresh", width - 244, cy, 56, 20);
    button(ctx, this, "save", "Save", width - 184, cy, 44, 20);
    button(ctx, this, "update", "Update", width - 136, cy, 56, 20);
    button(ctx, this, "delete", "Delete", width - 76, cy, 56, 20);
    cy += 27;
    button(ctx, this, "import", "Import", width - 136, cy, 56, 20);
    button(ctx, this, "export", "Export", width - 76, cy, 56, 20);
    ctx.globalAlpha = app.canvas.editor_alpha * 0.72;
    ctx.fillText("Prompt and intent text are collected from connected inputs.", margin, cy + 10);
    ctx.globalAlpha = app.canvas.editor_alpha;
    cy += 29;

    const preview = payload.positive_phrases.join(", ");
    ctx.fillStyle = "#58a6d6";
    ctx.fillText(`Preview: ${fitString(ctx, preview || "no positive scene phrases selected", innerWidth)}`, margin, cy + 8);
    cy += 25;

    cy = this.drawSelectedSection(ctx, payload, "positive_phrases", "Selected Positive", margin, cy, innerWidth, node, bottom);
    cy = this.drawSelectedSection(ctx, payload, "negative_phrases", "Selected Negative", margin, cy + 2, innerWidth, node, bottom);
    this.drawMemoryBank(ctx, memory, payload, margin, cy + 8, innerWidth, node, bottom);
    ctx.restore();
  }

  drawSelectedSection(ctx, payload, field, title, x, y, width, node, bottom) {
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.textAlign = "left";
    ctx.fillText(title, x, y + 8);
    const buttonWidth = field === "positive_phrases" ? 82 : 88;
    button(ctx, this, `${field}:add`, field === "positive_phrases" ? "+ Positive" : "+ Negative", x + width - buttonWidth, y, buttonWidth, 20);
    let cx = x;
    let cy = y + 18;
    const items = payload[field] || [];
    if (!items.length) {
      ctx.globalAlpha = app.canvas.editor_alpha * 0.55;
      ctx.fillText("none", x, cy + 9);
      ctx.globalAlpha = app.canvas.editor_alpha;
      return cy + 24;
    }
    let rows = 0;
    for (const item of items.slice(0, 40)) {
      const key = `${field}:remove:${item}`;
      const chipWidth = Math.min(width, Math.max(34, ctx.measureText(item).width + 18));
      if (cx + chipWidth > x + width) {
        cx = x;
        cy += 23;
        rows += 1;
      }
      if (cy + 21 > bottom || rows >= 2) {
        ctx.globalAlpha = app.canvas.editor_alpha * 0.55;
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.fillText(`+${items.length - items.indexOf(item)} more`, cx, cy + 9);
        ctx.globalAlpha = app.canvas.editor_alpha;
        break;
      }
      const used = drawChip(ctx, this, key, item, cx, cy, true, width);
      cx += used + 5;
    }
    return cy + 25;
  }

  drawMemoryBank(ctx, memory, payload, x, y, width, node, bottom) {
    if (y + 24 > bottom) {
      return;
    }
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.textAlign = "left";
    ctx.fillText("Universal Phrase Bank", x, y + 8);
    let cy = y + 23;
    const positive = selectedSet(payload.positive_phrases);
    const negative = selectedSet(payload.negative_phrases);
    const byGroup = new Map();
    for (const item of memory) {
      const group = item.category || (item.source === "negative" ? "negative" : "details");
      if (!byGroup.has(group)) {
        byGroup.set(group, []);
      }
      byGroup.get(group).push(item);
    }
    for (const group of GROUP_ORDER) {
      const items = (byGroup.get(group) || []).slice(0, 24);
      if (!items.length) {
        continue;
      }
      if (cy + 42 > bottom) {
        break;
      }
      ctx.fillStyle = "#58a6d6";
      ctx.fillText(group.toUpperCase(), x, cy + 8);
      cy += 17;
      let cx = x;
      for (const item of items) {
        const text = item.text || item.key;
        const selected = group === "negative" ? negative.has(text.toLowerCase()) : positive.has(text.toLowerCase());
        const key = `bank:${group}:${text}`;
        const chipWidth = Math.min(width, Math.max(34, ctx.measureText(text).width + 18));
        if (cx + chipWidth > x + width) {
          cx = x;
          cy += 23;
        }
        if (cy + 21 > bottom) {
          ctx.globalAlpha = app.canvas.editor_alpha * 0.55;
          ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
          ctx.fillText("Open + Positive or + Negative to pick more phrases.", x, Math.min(bottom - 6, cy + 9));
          ctx.globalAlpha = app.canvas.editor_alpha;
          return;
        }
        const used = drawChip(ctx, this, key, text, cx, cy, selected, width);
        cx += used + 5;
      }
      cy += 25;
    }
    if (!memory.length) {
      ctx.globalAlpha = app.canvas.editor_alpha * 0.55;
      ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
      ctx.fillText("Queue once with positive/negative prompts connected to collect phrases.", x, cy + 8);
      ctx.globalAlpha = app.canvas.editor_alpha;
    }
  }

  mouse(event, pos, node) {
    if (event.type !== "pointerup") {
      return event.type === "pointerdown" && this.hitAt(pos);
    }
    const hit = Object.entries(this.hitAreas).find(([, bounds]) => (
      pos[0] >= bounds[0] && pos[0] <= bounds[0] + bounds[2] &&
      pos[1] >= bounds[1] && pos[1] <= bounds[1] + bounds[3]
    ));
    if (!hit) {
      return false;
    }
    this.handleHit(hit[0], event, node);
    return true;
  }

  hitAt(pos) {
    return Object.values(this.hitAreas).some((bounds) => (
      pos[0] >= bounds[0] && pos[0] <= bounds[0] + bounds[2] &&
      pos[1] >= bounds[1] && pos[1] <= bounds[1] + bounds[3]
    ));
  }

  handleHit(key, event, node) {
    const payload = payloadFromNode(node);
    if (key === "refresh") {
      void refreshTracked();
      return;
    }
    if (["save", "update", "delete"].includes(key)) {
      void runSceneAction(node, key);
      return;
    }
    if (key === "import") {
      importScenes(node);
      return;
    }
    if (key === "export") {
      void exportScenes(node);
      return;
    }
    if (key.startsWith("positive_phrases:remove:")) {
      const text = key.slice("positive_phrases:remove:".length);
      payload.positive_phrases = payload.positive_phrases.filter((item) => item !== text);
      writePayload(node, payload);
      return;
    }
    if (key === "positive_phrases:add" || key === "negative_phrases:add") {
      const field = key === "positive_phrases:add" ? "positive_phrases" : "negative_phrases";
      const preferredSource = field === "negative_phrases" ? "negative" : "positive";
      const values = (sceneData?.memory || [])
        .filter((item) => field === "negative_phrases" ? item.source === "negative" || item.category === "negative" : item.source !== "negative" && item.category !== "negative")
        .map((item) => item.text || item.key)
        .filter(Boolean);
      searchablePicker(event, values.length ? values : (sceneData?.memory || []).map((item) => item.text || item.key).filter(Boolean), "", (value) => {
        if (!payload[field].some((item) => item.toLowerCase() === String(value).toLowerCase())) {
          payload[field].push(value);
          writePayload(node, payload);
        }
      }, preferredSource === "negative" ? "Search negative phrases" : "Search positive phrases");
      return;
    }
    if (key.startsWith("negative_phrases:remove:")) {
      const text = key.slice("negative_phrases:remove:".length);
      payload.negative_phrases = payload.negative_phrases.filter((item) => item !== text);
      writePayload(node, payload);
      return;
    }
    if (key.startsWith("bank:")) {
      const [, group, ...rest] = key.split(":");
      const text = rest.join(":");
      const field = group === "negative" ? "negative_phrases" : "positive_phrases";
      const lower = text.toLowerCase();
      if (payload[field].some((item) => item.toLowerCase() === lower)) {
        payload[field] = payload[field].filter((item) => item.toLowerCase() !== lower);
      } else {
        payload[field].push(text);
      }
      writePayload(node, payload);
      return;
    }
  }
}

function removeSceneBuilderWidget(node) {
  node.widgets = (node.widgets || []).filter((widget) => widget.name !== "funpack_scene_builder");
}

function setupSceneBuilderNode(node) {
  trackedNodes.add(node);
  hideWidget(widgetByName(node, ACTION_WIDGET));
  hideWidget(widgetByName(node, PAYLOAD_WIDGET));
  removeSceneBuilderWidget(node);
  node.addCustomWidget(new SceneBuilderWidget(node));
  updateSceneCombo(node);
  void fetchScenes(currentRefinementKey(node)).then(() => {
    updateSceneCombo(node);
    node.setDirtyCanvas(true, true);
  });
}

function wrapRefreshFunction(functionName) {
  const original = app[functionName];
  if (typeof original !== "function" || original.__funpackSceneBuilderWrapped) {
    return;
  }
  const wrapped = function () {
    const result = original.apply(this, arguments);
    Promise.resolve(result).finally(() => scheduleRefresh());
    return result;
  };
  wrapped.__funpackSceneBuilderWrapped = true;
  app[functionName] = wrapped;
}

function hookComfyRefreshControls() {
  for (const functionName of ["refreshComboInNodes", "refreshComboInNode", "refreshNodeDefs"]) {
    wrapRefreshFunction(functionName);
  }
}

app.registerExtension({
  name: "funpack.sceneBuilder",
  setup() {
    hookComfyRefreshControls();
    window.setTimeout(hookComfyRefreshControls, 0);
    window.setTimeout(hookComfyRefreshControls, 1000);
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) {
      return;
    }
    hookComfyRefreshControls();
    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      originalOnNodeCreated?.apply(this, arguments);
      setupSceneBuilderNode(this);
    };
    const originalConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      originalConfigure?.apply(this, arguments);
      setupSceneBuilderNode(this);
    };
  },
});
