import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "FunPackSceneBuilder";
const NONE_SCENE = "-None-";
const HIDDEN_WIDGETS = new Set([
  "scene_name",
  "mode",
  "scene",
  "aliases",
  "action",
  "scene_positive",
  "scene_negative",
  "refinement_key",
  "scene_payload",
]);
const GROUP_ORDER = ["subject", "appearance", "action", "camera", "environment", "style", "quality", "details", "negative"];

let sceneData = null;
let activePanel = null;
const trackedNodes = new Set();

function widgetByName(node, name) {
  return (node.widgets || []).find((widget) => widget.name === name);
}

function hideWidget(widget) {
  if (!widget) {
    return;
  }
  if (widget.linkedWidgets) {
    for (const linked of widget.linkedWidgets) {
      hideWidget(linked);
    }
  }
  widget.__funpackHidden = true;
  widget.hidden = true;
  widget.options = widget.options || {};
  widget.options.hidden = true;
  widget.computeSize = () => [0, -4];
  widget.computedHeight = 0;
  widget.type = "hidden";
  for (const key of ["element", "inputEl", "textElement", "parentEl"]) {
    const element = widget[key];
    if (element?.style) {
      element.style.display = "none";
      element.style.visibility = "hidden";
      element.style.pointerEvents = "none";
    }
    if (element) {
      element.hidden = true;
    }
  }
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

function setDirty(node) {
  node.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}

function normalizeRefinementKey(value) {
  const clean = String(value || "").trim();
  return clean && clean !== NONE_SCENE ? clean : "";
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

function button(ctx, widget, key, label, x, y, width, height) {
  widget.hitAreas[key] = [x, y, width, height];
  roundRect(ctx, x, y, width, height, 5);
  ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
  ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(fitString(ctx, label, width - 8), x + width / 2, y + height / 2);
}

function currentRefinementKey(node) {
  const linked = linkedRefinementKey(node);
  if (linked) {
    return linked;
  }
  return normalizeRefinementKey(getWidgetValue(node, "refinement_key", ""));
}

function linkedInputNode(node, inputName) {
  const input = (node.inputs || []).find((item) => item.name === inputName);
  const linkId = Array.isArray(input?.link) ? input.link[0] : input?.link;
  if (linkId == null) {
    return null;
  }
  const link = app.graph?.links?.[linkId];
  return link ? app.graph?.getNodeById?.(link.origin_id) : null;
}

function linkedRefinementKey(node) {
  const source = linkedInputNode(node, "refinement_key_input");
  if (!source) {
    return "";
  }
  const selected = normalizeRefinementKey(getWidgetValue(source, "refinement_key", ""));
  const typed = normalizeRefinementKey(getWidgetValue(source, "key_name", ""));
  return selected || typed;
}

function sceneNames() {
  const names = sceneData?.scenes;
  return Array.isArray(names) && names.length ? names : [NONE_SCENE];
}

function sceneByName(name) {
  return sceneData?.data?.scenes?.[name] || null;
}

function memoryItems() {
  return Array.isArray(sceneData?.memory) ? sceneData.memory : [];
}

function normalizeMemoryMap(items) {
  const output = {};
  for (const item of items || []) {
    const text = String(item.text || item.key || "").trim();
    if (!text) {
      continue;
    }
    const key = String(item.key || text).toLowerCase().replace(/[^\w'’]+/g, " ").replace(/\s+/g, " ").trim() || text.toLowerCase();
    output[key] = {
      text,
      key,
      source: item.source === "negative" || item.category === "negative" ? "negative" : "positive",
      category: GROUP_ORDER.includes(item.category) ? item.category : "details",
      tokens: Array.isArray(item.tokens) ? item.tokens : [],
      count: Number.isFinite(Number(item.count)) ? Number(item.count) : 0,
      created_at: item.created_at || new Date().toISOString(),
      updated_at: new Date().toISOString(),
      wildcard: Boolean(item.wildcard || String(item.wildcard_group || "").trim()),
    };
  }
  return output;
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

async function refreshNode(node) {
  await fetchScenes(currentRefinementKey(node));
  updateSceneCombo(node);
  setDirty(node);
}

async function refreshTracked() {
  for (const node of [...trackedNodes]) {
    if (!node?.graph) {
      trackedNodes.delete(node);
      continue;
    }
    await refreshNode(node);
  }
}

function scheduleRefresh() {
  window.setTimeout(() => void refreshTracked(), 100);
}

function loadSceneIntoNode(node, name) {
  const scene = sceneByName(name);
  if (!scene || name === NONE_SCENE) {
    return;
  }
  setWidgetValue(node, "scene", name);
  setWidgetValue(node, "scene_name", scene.name || name);
  setWidgetValue(node, "aliases", Array.isArray(scene.aliases) ? scene.aliases.join(", ") : "");
  setWidgetValue(node, "mode", scene.output_mode || "Manual");
  setWidgetValue(node, "refinement_key", scene.refinement_key || "");
  setWidgetValue(node, "scene_positive", scene.positive_text || (scene.positive_phrases || []).join(", "));
  setWidgetValue(node, "scene_negative", scene.negative_text || (scene.negative_phrases || []).join(", "));
  setDirty(node);
}

async function saveSceneFromNode(node) {
  const name = String(getWidgetValue(node, "scene_name", "") || "").trim();
  if (!name) {
    throw new Error("Scene name is required.");
  }
  const params = new URLSearchParams();
  const key = currentRefinementKey(node);
  if (key) {
    params.set("key", key);
  }
  const response = await api.fetchApi(`/funpack/scenes/scene${params.toString() ? `?${params.toString()}` : ""}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      action: "save",
      name,
      aliases: getWidgetValue(node, "aliases", ""),
      mode: getWidgetValue(node, "mode", "Manual"),
      positive_text: getWidgetValue(node, "scene_positive", ""),
      negative_text: getWidgetValue(node, "scene_negative", ""),
    }),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error || `Save failed with HTTP ${response.status}`);
  }
  await refreshNode(node);
  setWidgetValue(node, "scene", name);
}

async function deleteSceneFromNode(node) {
  const name = String(getWidgetValue(node, "scene_name", "") || "").trim();
  if (!name) {
    throw new Error("Scene name is required.");
  }
  const params = new URLSearchParams();
  const key = currentRefinementKey(node);
  if (key) {
    params.set("key", key);
  }
  const response = await api.fetchApi(`/funpack/scenes/scene${params.toString() ? `?${params.toString()}` : ""}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "delete", name }),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error || `Delete failed with HTTP ${response.status}`);
  }
  setWidgetValue(node, "scene", NONE_SCENE);
  setWidgetValue(node, "scene_name", "");
  setWidgetValue(node, "scene_positive", "");
  setWidgetValue(node, "scene_negative", "");
  await refreshNode(node);
}

async function saveDatabase(node, items) {
  const params = new URLSearchParams();
  const key = currentRefinementKey(node);
  if (key) {
    params.set("key", key);
  }
  const response = await api.fetchApi(`/funpack/scenes/database${params.toString() ? `?${params.toString()}` : ""}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ universal_memory: normalizeMemoryMap(items) }),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error || `Database save failed with HTTP ${response.status}`);
  }
  await refreshNode(node);
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
      await refreshNode(node);
      renderPanel(activePanel, node, "menu");
    } catch (error) {
      showPanelError(error);
    }
  };
  input.click();
}

function closePanel() {
  if (activePanel?.root) {
    activePanel.root.remove();
  }
  activePanel = null;
}

function showPanelError(error) {
  const target = activePanel?.root?.querySelector("[data-role='error']");
  if (target) {
    target.textContent = error.message || String(error);
  } else {
    console.warn("FunPack Scene Builder:", error);
  }
}

function panelButton(label, className = "") {
  const element = document.createElement("button");
  element.type = "button";
  element.textContent = label;
  element.className = `funpack-scene-button ${className}`.trim();
  return element;
}

function panelTextInput(value, placeholder) {
  const element = document.createElement("input");
  element.type = "text";
  element.value = String(value || "");
  element.placeholder = placeholder;
  return element;
}

function panelSelect(values, selected) {
  const element = document.createElement("select");
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    option.selected = value === selected;
    element.append(option);
  }
  return element;
}

function panelCheckbox(checked, label) {
  const wrapper = document.createElement("label");
  wrapper.className = "funpack-scene-checkbox";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = Boolean(checked);
  const span = document.createElement("span");
  span.textContent = label;
  wrapper.append(input, span);
  return { wrapper, input };
}

function editableTextLabel(value, onCommit) {
  const wrapper = document.createElement("div");
  wrapper.className = "funpack-scene-editable";
  const label = document.createElement("button");
  label.type = "button";
  label.textContent = value || "Unnamed";
  label.title = "Double-click to edit";
  wrapper.append(label);

  const beginEdit = () => {
    const editor = document.createElement("div");
    editor.className = "funpack-scene-inline-editor";
    const input = document.createElement("input");
    const original = label.textContent || "";
    input.type = "text";
    input.value = original;
    const ok = panelButton("OK", "compact primary");
    const cancel = panelButton("Cancel", "compact");
    let committed = false;
    const commit = () => {
      if (committed) {
        return;
      }
      committed = true;
      const next = input.value.trim();
      const value = next || original;
      onCommit(value);
      label.textContent = value;
      editor.replaceWith(label);
    };
    const cancelEdit = () => {
      if (committed) {
        return;
      }
      committed = true;
      editor.replaceWith(label);
    };
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        commit();
      } else if (event.key === "Escape") {
        event.preventDefault();
        cancelEdit();
      }
    });
    ok.addEventListener("click", commit);
    cancel.addEventListener("click", cancelEdit);
    editor.append(input, ok, cancel);
    label.replaceWith(editor);
    input.focus();
    input.select();
  };

  label.addEventListener("dblclick", beginEdit);
  label.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === "F2") {
      event.preventDefault();
      beginEdit();
    }
  });
  return wrapper;
}

function phraseButton(text, onInsert) {
  const element = document.createElement("button");
  element.type = "button";
  element.className = "funpack-scene-chip";
  element.textContent = text;
  element.title = text;
  element.draggable = true;
  element.addEventListener("click", () => onInsert(text));
  element.addEventListener("dragstart", (event) => {
    event.dataTransfer?.setData("text/plain", text);
  });
  return element;
}

function insertAtCursor(textarea, text) {
  const start = textarea.selectionStart ?? textarea.value.length;
  const end = textarea.selectionEnd ?? textarea.value.length;
  const prefix = textarea.value.slice(0, start);
  const suffix = textarea.value.slice(end);
  const needsComma = prefix.trim() && !/[,\s.;\n]$/.test(prefix);
  const insertion = `${needsComma ? ", " : ""}${text}`;
  textarea.value = `${prefix}${insertion}${suffix}`;
  const cursor = prefix.length + insertion.length;
  textarea.focus();
  textarea.setSelectionRange(cursor, cursor);
}

function shell(root, title, view) {
  root.replaceChildren();
  const header = document.createElement("div");
  header.className = "funpack-scene-panel-header";
  const h = document.createElement("div");
  h.textContent = title;
  const close = panelButton("Close");
  close.addEventListener("click", closePanel);
  header.append(h, close);

  const error = document.createElement("div");
  error.dataset.role = "error";
  error.className = "funpack-scene-error";

  const body = document.createElement("div");
  body.className = `funpack-scene-body funpack-scene-${view}`;
  root.append(header, error, body);
  return body;
}

function renderPanel(panel, node, view) {
  if (!panel?.root) {
    return;
  }
  if (view === "positive" || view === "negative") {
    renderPromptEditor(panel, node, view);
  } else if (view === "database") {
    renderDatabaseEditor(panel, node);
  } else {
    renderMenu(panel, node);
  }
}

function renderMenu(panel, node) {
  const body = shell(panel.root, "FunPack Scene Builder", "menu");

  const controls = document.createElement("div");
  controls.className = "funpack-scene-controls";
  const sceneSelect = panelSelect(sceneNames(), String(getWidgetValue(node, "scene", NONE_SCENE) || NONE_SCENE));
  sceneSelect.addEventListener("change", () => {
    const name = sceneSelect.value;
    if (name === NONE_SCENE) {
      setWidgetValue(node, "scene", NONE_SCENE);
      setWidgetValue(node, "scene_name", "");
      setWidgetValue(node, "aliases", "");
      setWidgetValue(node, "scene_positive", "");
      setWidgetValue(node, "scene_negative", "");
      setDirty(node);
      renderPanel(panel, node, "menu");
      return;
    }
    loadSceneIntoNode(node, name);
    renderPanel(panel, node, "menu");
  });
  const sceneName = panelTextInput(getWidgetValue(node, "scene_name", ""), "Scene name");
  sceneName.addEventListener("input", () => {
    setWidgetValue(node, "scene_name", sceneName.value);
    setDirty(node);
  });
  const mode = panelSelect(["Manual", "Auto", "Learning"], String(getWidgetValue(node, "mode", "Manual") || "Manual"));
  mode.addEventListener("change", () => {
    setWidgetValue(node, "mode", mode.value);
    setDirty(node);
  });
  const aliases = panelTextInput(getWidgetValue(node, "aliases", ""), "Aliases, comma separated");
  aliases.addEventListener("input", () => {
    setWidgetValue(node, "aliases", aliases.value);
    setDirty(node);
  });
  for (const [label, element] of [
    ["Saved scene", sceneSelect],
    ["Scene name", sceneName],
    ["Mode", mode],
    ["Aliases", aliases],
  ]) {
    const row = document.createElement("label");
    const span = document.createElement("span");
    span.textContent = label;
    row.append(span, element);
    controls.append(row);
  }
  body.append(controls);

  const current = document.createElement("div");
  current.className = "funpack-scene-current";
  const positive = String(getWidgetValue(node, "scene_positive", "") || "").trim();
  const negative = String(getWidgetValue(node, "scene_negative", "") || "").trim();
  for (const [label, value] of [
    ["Positive", positive || "empty"],
    ["Negative", negative || "empty"],
  ]) {
    const row = document.createElement("div");
    const b = document.createElement("b");
    b.textContent = label;
    const span = document.createElement("span");
    span.textContent = value;
    row.append(b, span);
    current.append(row);
  }
  body.append(current);

  const menu = document.createElement("div");
  menu.className = "funpack-scene-menu-buttons";
  for (const [label, target] of [["Positive prompt", "positive"], ["Negative prompt", "negative"], ["Database", "database"]]) {
    const item = panelButton(label, "large");
    item.addEventListener("click", () => renderPanel(panel, node, target));
    menu.append(item);
  }
  body.append(menu);

  const savedHeader = document.createElement("div");
  savedHeader.className = "funpack-scene-section-title";
  savedHeader.textContent = "Saved scenes";
  body.append(savedHeader);

  const saved = document.createElement("div");
  saved.className = "funpack-scene-saved";
  for (const name of sceneNames().filter((item) => item !== NONE_SCENE)) {
    const item = panelButton(name);
    item.title = name;
    item.addEventListener("click", () => {
      loadSceneIntoNode(node, name);
      renderPanel(panel, node, "menu");
    });
    saved.append(item);
  }
  if (!saved.children.length) {
    const empty = document.createElement("div");
    empty.className = "funpack-scene-muted";
    empty.textContent = "No saved scenes yet.";
    saved.append(empty);
  }
  body.append(saved);

  const footer = document.createElement("div");
  footer.className = "funpack-scene-footer";
  const refresh = panelButton("Refresh");
  refresh.addEventListener("click", async () => {
    await refreshNode(node);
    renderPanel(panel, node, "menu");
  });
  const save = panelButton("Save scene", "primary");
  save.addEventListener("click", async () => {
    try {
      await saveSceneFromNode(node);
      renderPanel(panel, node, "menu");
    } catch (error) {
      showPanelError(error);
    }
  });
  const del = panelButton("Delete scene", "danger");
  del.addEventListener("click", async () => {
    try {
      await deleteSceneFromNode(node);
      renderPanel(panel, node, "menu");
    } catch (error) {
      showPanelError(error);
    }
  });
  const imp = panelButton("Import");
  imp.addEventListener("click", () => importScenes(node));
  const exp = panelButton("Export");
  exp.addEventListener("click", async () => {
    try {
      await exportScenes(node);
    } catch (error) {
      showPanelError(error);
    }
  });
  footer.append(refresh, save, del, imp, exp);
  body.append(footer);
}

function renderPromptEditor(panel, node, kind) {
  const isNegative = kind === "negative";
  const widgetName = isNegative ? "scene_negative" : "scene_positive";
  const snapshotKey = `${kind}_snapshot`;
  if (!(snapshotKey in panel.snapshots)) {
    panel.snapshots[snapshotKey] = String(getWidgetValue(node, widgetName, "") || "");
  }

  const body = shell(panel.root, isNegative ? "Negative prompt" : "Positive prompt", "prompt");
  const textarea = document.createElement("textarea");
  textarea.className = "funpack-scene-textarea";
  textarea.spellcheck = false;
  textarea.value = String(getWidgetValue(node, widgetName, "") || "");
  textarea.placeholder = isNegative ? "Write negative prompt words and phrases here." : "Write positive prompt words and phrases here.";
  textarea.addEventListener("input", () => {
    setWidgetValue(node, widgetName, textarea.value);
    setDirty(node);
  });
  textarea.addEventListener("dragover", (event) => event.preventDefault());
  textarea.addEventListener("drop", (event) => {
    event.preventDefault();
    const text = event.dataTransfer?.getData("text/plain");
    if (text) {
      insertAtCursor(textarea, text);
      textarea.dispatchEvent(new Event("input"));
    }
  });
  body.append(textarea);

  const search = document.createElement("input");
  search.type = "search";
  search.placeholder = "Search phrase bank";
  search.className = "funpack-scene-search";
  body.append(search);

  const bank = document.createElement("div");
  bank.className = "funpack-scene-bank";
  body.append(bank);

  const renderBank = () => {
    const query = search.value.toLowerCase().trim();
    bank.replaceChildren();
    const items = memoryItems()
      .filter((item) => isNegative ? item.source === "negative" || item.category === "negative" : item.source !== "negative" && item.category !== "negative")
      .filter((item) => !query || String(item.text || "").toLowerCase().includes(query));
    const byGroup = new Map();
    for (const item of items) {
      const group = item.category || (item.source === "negative" ? "negative" : "details");
      if (!byGroup.has(group)) {
        byGroup.set(group, []);
      }
      byGroup.get(group).push(item);
    }
    for (const group of GROUP_ORDER) {
      const groupItems = byGroup.get(group) || [];
      if (!groupItems.length) {
        continue;
      }
      const title = document.createElement("div");
      title.className = "funpack-scene-category";
      title.textContent = group;
      bank.append(title);
      const chips = document.createElement("div");
      chips.className = "funpack-scene-chip-row";
      for (const item of groupItems) {
        chips.append(phraseButton(item.text || item.key, (text) => {
          insertAtCursor(textarea, text);
          textarea.dispatchEvent(new Event("input"));
        }));
      }
      bank.append(chips);
    }
  };
  search.addEventListener("input", renderBank);
  renderBank();

  const footer = document.createElement("div");
  footer.className = "funpack-scene-footer";
  const back = panelButton("Back");
  back.addEventListener("click", () => renderPanel(panel, node, "menu"));
  const cancel = panelButton("Cancel");
  cancel.addEventListener("click", () => {
    setWidgetValue(node, widgetName, panel.snapshots[snapshotKey]);
    setDirty(node);
    delete panel.snapshots[snapshotKey];
    renderPanel(panel, node, "menu");
  });
  const confirm = panelButton("Confirm", "primary");
  confirm.addEventListener("click", async () => {
    try {
      setWidgetValue(node, widgetName, textarea.value);
      await saveSceneFromNode(node);
      delete panel.snapshots[snapshotKey];
      renderPanel(panel, node, "menu");
    } catch (error) {
      showPanelError(error);
    }
  });
  footer.append(back, cancel, confirm);
  body.append(footer);
  textarea.focus();
}

function renderDatabaseEditor(panel, node) {
  if (!("database_snapshot" in panel.snapshots)) {
    panel.snapshots.database_snapshot = JSON.parse(JSON.stringify(memoryItems()));
    panel.databaseItems = JSON.parse(JSON.stringify(memoryItems()));
  }

  const body = shell(panel.root, "Database", "database");
  const tools = document.createElement("div");
  tools.className = "funpack-scene-db-tools";
  const addText = document.createElement("input");
  addText.placeholder = "Add word or phrase";
  const addCategory = document.createElement("select");
  for (const category of GROUP_ORDER) {
    const option = document.createElement("option");
    option.value = category;
    option.textContent = category;
    addCategory.append(option);
  }
  const addWildcard = panelCheckbox(false, "Wildcard");
  const add = panelButton("Add", "primary");
  add.addEventListener("click", () => {
    const text = addText.value.trim();
    if (!text) {
      return;
    }
    panel.databaseItems.unshift({
      text,
      key: text.toLowerCase(),
      source: addCategory.value === "negative" ? "negative" : "positive",
      category: addCategory.value,
      count: 0,
      wildcard: addWildcard.input.checked,
    });
    addText.value = "";
    addWildcard.input.checked = false;
    renderPanel(panel, node, "database");
  });
  tools.append(addText, addCategory, addWildcard.wrapper, add);
  body.append(tools);

  const search = document.createElement("input");
  search.type = "search";
  search.placeholder = "Search database";
  search.className = "funpack-scene-search";
  body.append(search);

  const list = document.createElement("div");
  list.className = "funpack-scene-db-list";
  body.append(list);

  const renderList = () => {
    const query = search.value.toLowerCase().trim();
    list.replaceChildren();
    const items = panel.databaseItems.filter((item) => !query || String(item.text || "").toLowerCase().includes(query));
    items.forEach((item) => {
      const row = document.createElement("div");
      row.className = "funpack-scene-db-row";
      const text = editableTextLabel(item.text || "", (value) => {
        item.text = value;
        item.key = value.toLowerCase();
      });
      const category = document.createElement("select");
      for (const categoryName of GROUP_ORDER) {
        const option = document.createElement("option");
        option.value = categoryName;
        option.textContent = categoryName;
        option.selected = categoryName === item.category;
        category.append(option);
      }
      category.addEventListener("change", () => {
        item.category = category.value;
        item.source = category.value === "negative" ? "negative" : "positive";
      });
      const wildcard = panelCheckbox(Boolean(item.wildcard || item.wildcard_group), "Wildcard");
      wildcard.input.addEventListener("change", () => {
        item.wildcard = wildcard.input.checked;
      });
      const del = panelButton("Delete", "danger");
      del.addEventListener("click", () => {
        panel.databaseItems = panel.databaseItems.filter((candidate) => candidate !== item);
        renderList();
      });
      row.append(text, category, wildcard.wrapper, del);
      list.append(row);
    });
  };
  search.addEventListener("input", renderList);
  renderList();

  const footer = document.createElement("div");
  footer.className = "funpack-scene-footer";
  const back = panelButton("Back");
  back.addEventListener("click", () => renderPanel(panel, node, "menu"));
  const cancel = panelButton("Cancel");
  cancel.addEventListener("click", () => {
    panel.databaseItems = JSON.parse(JSON.stringify(panel.snapshots.database_snapshot));
    delete panel.snapshots.database_snapshot;
    renderPanel(panel, node, "menu");
  });
  const confirm = panelButton("Confirm", "primary");
  confirm.addEventListener("click", async () => {
    try {
      await saveDatabase(node, panel.databaseItems);
      delete panel.snapshots.database_snapshot;
      renderPanel(panel, node, "menu");
    } catch (error) {
      showPanelError(error);
    }
  });
  footer.append(back, cancel, confirm);
  body.append(footer);
}

function openPanel(node, view = "menu") {
  closePanel();
  injectStyles();
  const root = document.createElement("div");
  root.className = "funpack-scene-panel";
  document.body.append(root);
  activePanel = { root, snapshots: {}, databaseItems: [] };
  const viewportWidth = Number(window.innerWidth);
  const viewportHeight = Number(window.innerHeight);
  if (Number.isFinite(viewportWidth) && Number.isFinite(viewportHeight) && viewportWidth > 0 && viewportHeight > 0) {
    root.style.left = "50%";
    root.style.top = "50%";
    root.style.transform = "translate(-50%, -50%)";
  } else {
    const canvasRect = app.canvas?.canvas?.getBoundingClientRect?.();
    const scale = app.canvas?.ds?.scale || 1;
    const offset = app.canvas?.ds?.offset || [0, 0];
    const screenX = (canvasRect?.left || 0) + (node.pos[0] + node.size[0]) * scale + offset[0];
    const screenY = (canvasRect?.top || 0) + node.pos[1] * scale + offset[1];
    const left = Math.min(Math.max(12, window.innerWidth - 460), Math.max(12, screenX + 16));
    const top = Math.min(Math.max(12, window.innerHeight - 620), Math.max(12, screenY));
    root.style.left = `${left}px`;
    root.style.top = `${top}px`;
    root.style.transform = "";
  }
  void refreshNode(node).catch((error) => {
    console.warn("FunPack: failed to refresh scenes before opening Scene Builder", error);
  }).finally(() => {
    if (activePanel?.root === root) {
      renderPanel(activePanel, node, view);
    }
  });
}

let stylesInjected = false;
function injectStyles() {
  if (stylesInjected) {
    return;
  }
  stylesInjected = true;
  const style = document.createElement("style");
  style.textContent = `
    .funpack-scene-panel {
      position: fixed;
      z-index: 10000;
      width: min(448px, calc(100vw - 24px));
      max-height: min(620px, calc(100vh - 24px));
      display: flex;
      flex-direction: column;
      padding: 10px;
      border: 1px solid rgba(180, 190, 200, 0.35);
      border-radius: 8px;
      background: rgba(30, 32, 36, 0.98);
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.42);
      color: #ddd;
      font: 12px sans-serif;
      box-sizing: border-box;
    }
    .funpack-scene-panel * { box-sizing: border-box; }
    .funpack-scene-panel-header,
    .funpack-scene-footer {
      display: flex;
      align-items: center;
      gap: 8px;
      justify-content: space-between;
      flex-wrap: wrap;
    }
    .funpack-scene-panel-header {
      font-weight: 700;
      font-size: 14px;
      padding-bottom: 8px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .funpack-scene-body {
      overflow: auto;
      padding-top: 8px;
      min-height: 0;
    }
    .funpack-scene-error {
      min-height: 16px;
      color: #ff9f9f;
      padding-top: 5px;
    }
    .funpack-scene-button {
      min-height: 26px;
      padding: 4px 10px;
      border: 1px solid rgba(180, 190, 200, 0.35);
      border-radius: 5px;
      background: #22252a;
      color: #eee;
      cursor: pointer;
      white-space: nowrap;
    }
    .funpack-scene-button:hover { background: #2b3037; }
    .funpack-scene-button.primary { border-color: rgba(100, 210, 140, 0.6); background: #244832; }
    .funpack-scene-button.danger { border-color: rgba(255, 130, 130, 0.45); background: #472626; }
    .funpack-scene-button.compact {
      min-height: 28px;
      padding: 4px 7px;
    }
    .funpack-scene-button.large {
      width: 100%;
      height: 42px;
      text-align: left;
      font-weight: 700;
    }
    .funpack-scene-current {
      display: grid;
      gap: 5px;
      margin-bottom: 10px;
    }
    .funpack-scene-controls {
      display: grid;
      gap: 7px;
      margin-bottom: 10px;
    }
    .funpack-scene-controls label {
      display: grid;
      grid-template-columns: 86px minmax(0, 1fr);
      gap: 8px;
      align-items: center;
    }
    .funpack-scene-controls span {
      color: #b8c0ca;
    }
    .funpack-scene-current div {
      display: grid;
      grid-template-columns: 78px minmax(0, 1fr);
      gap: 8px;
    }
    .funpack-scene-current span {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: #b8c0ca;
    }
    .funpack-scene-menu-buttons,
    .funpack-scene-saved {
      display: grid;
      gap: 7px;
    }
    .funpack-scene-section-title,
    .funpack-scene-category {
      margin: 12px 0 6px;
      color: #58a6d6;
      font-weight: 700;
      text-transform: uppercase;
    }
    .funpack-scene-textarea {
      width: 100%;
      min-height: 156px;
      resize: vertical;
      padding: 8px;
      border: 1px solid rgba(180, 190, 200, 0.35);
      border-radius: 6px;
      background: #17191d;
      color: #f4f4f4;
      line-height: 1.35;
      outline: none;
    }
    .funpack-scene-search,
    .funpack-scene-controls input,
    .funpack-scene-controls select,
    .funpack-scene-db-tools input,
    .funpack-scene-db-tools select,
    .funpack-scene-editable input,
    .funpack-scene-db-row input,
    .funpack-scene-db-row select {
      width: 100%;
      min-height: 28px;
      padding: 5px 7px;
      border: 1px solid rgba(180, 190, 200, 0.28);
      border-radius: 5px;
      background: #17191d;
      color: #f2f2f2;
      outline: none;
    }
    .funpack-scene-db-tools .funpack-scene-checkbox input,
    .funpack-scene-db-row .funpack-scene-checkbox input {
      width: auto;
      min-height: 0;
      padding: 0;
    }
    .funpack-scene-checkbox {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 5px;
      min-height: 28px;
      color: #d9dee5;
      white-space: nowrap;
    }
    .funpack-scene-editable {
      min-width: 0;
    }
    .funpack-scene-inline-editor {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto auto;
      gap: 5px;
      align-items: center;
    }
    .funpack-scene-editable button {
      width: 100%;
      min-height: 28px;
      padding: 5px 7px;
      border: 1px solid transparent;
      border-radius: 5px;
      background: transparent;
      color: #f2f2f2;
      text-align: left;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      cursor: default;
    }
    .funpack-scene-editable button:hover,
    .funpack-scene-editable button:focus {
      border-color: rgba(180, 190, 200, 0.28);
      background: #17191d;
      outline: none;
    }
    .funpack-scene-search { margin: 8px 0; }
    .funpack-scene-bank,
    .funpack-scene-db-list {
      max-height: 270px;
      overflow: auto;
      padding-right: 3px;
    }
    .funpack-scene-chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
    }
    .funpack-scene-chip {
      max-width: 100%;
      padding: 4px 8px;
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 6px;
      background: rgba(255,255,255,0.07);
      color: #eee;
      cursor: grab;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .funpack-scene-db-tools {
      display: grid;
      grid-template-columns: minmax(0, 1.6fr) 110px auto auto;
      gap: 6px;
      align-items: center;
    }
    .funpack-scene-db-row {
      display: grid;
      grid-template-columns: minmax(0, 1.7fr) 110px auto auto;
      gap: 6px;
      align-items: center;
      padding: 5px 0;
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .funpack-scene-footer {
      margin-top: 10px;
      padding-top: 8px;
      border-top: 1px solid rgba(255,255,255,0.08);
      justify-content: flex-end;
    }
    .funpack-scene-muted { color: #9da6b0; }
  `;
  document.head.append(style);
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
    return [width, 82];
  }

  draw(ctx, node, width, y) {
    this.hitAreas = {};
    const margin = 10;
    let cy = y + 4;
    const innerWidth = width - margin * 2;
    const mode = getWidgetValue(node, "mode", "Manual");
    const scene = String(getWidgetValue(node, "scene_name", "") || "Unnamed scene").trim();

    ctx.save();
    ctx.beginPath();
    ctx.rect(0, y, width, this.computeSize(width)[1]);
    ctx.clip();
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.font = "12px sans-serif";
    ctx.fillText(fitString(ctx, `Scene Builder: ${scene}`, width - 134), margin, cy + 9);
    button(ctx, this, "open:menu", "Open Editor", width - 112, cy, 92, 20);
    cy += 28;

    const third = Math.floor((innerWidth - 12) / 3);
    button(ctx, this, "open:positive", "Positive", margin, cy, third, 28);
    button(ctx, this, "open:negative", "Negative", margin + third + 6, cy, third, 28);
    button(ctx, this, "open:database", "Database", margin + (third + 6) * 2, cy, innerWidth - (third + 6) * 2, 28);
    ctx.globalAlpha = app.canvas.editor_alpha * 0.62;
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.fillText(`${mode}. ${memoryItems().length} database phrase(s). Connected prompts learn on queue.`, margin, cy + 42);
    ctx.globalAlpha = app.canvas.editor_alpha;
    ctx.restore();
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
    const [, view] = hit[0].split(":");
    openPanel(node, view || "menu");
    return true;
  }

  hitAt(pos) {
    return Object.values(this.hitAreas).some((bounds) => (
      pos[0] >= bounds[0] && pos[0] <= bounds[0] + bounds[2] &&
      pos[1] >= bounds[1] && pos[1] <= bounds[1] + bounds[3]
    ));
  }
}

function removeSceneBuilderWidget(node) {
  node.widgets = (node.widgets || []).filter((widget) => widget.name !== "funpack_scene_builder");
}

function setupSceneBuilderNode(node) {
  trackedNodes.add(node);
  const hideInternalWidgets = () => {
    for (const widget of node.widgets || []) {
      if (HIDDEN_WIDGETS.has(widget.name)) {
        hideWidget(widget);
      }
    }
    setDirty(node);
  };
  hideInternalWidgets();
  window.requestAnimationFrame?.(hideInternalWidgets);
  window.setTimeout(hideInternalWidgets, 0);
  window.setTimeout(hideInternalWidgets, 250);
  removeSceneBuilderWidget(node);
  node.addCustomWidget(new SceneBuilderWidget(node));
  updateSceneCombo(node);
  void fetchScenes(currentRefinementKey(node)).then(() => {
    updateSceneCombo(node);
    setDirty(node);
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
