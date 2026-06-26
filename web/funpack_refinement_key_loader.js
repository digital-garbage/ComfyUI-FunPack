import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "FunPackRefinementKeyLoader";
const NONE_KEY = "-None-";
const BUTTON_NAMES = new Set([
  "funpack_refinement_key_export",
  "funpack_refinement_key_import",
  "funpack_refinement_key_refresh",
  "funpack_refinement_key_delete",
  "funpack_refinement_key_clear_absolute",
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
  link.download = `${key}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

// Upload a refinement key. Streams large keys in chunks so reverse proxies on
// Vast.ai / Runpod can't reject them with HTTP 413, and surfaces a 409 name
// collision as a tagged error (err.exists / err.key) so the caller can confirm
// an overwrite and retry. Mirrors movie_editor/frontend/api.js.
async function importRefinementKeyData(data, overwrite) {
  const ow = overwrite ? "&overwrite=true" : "";
  const bytes = new TextEncoder().encode(JSON.stringify(data));
  const CHUNK = 256 * 1024;
  const finish = async (res) => {
    if (res.status === 409) {
      const p = await res.json().catch(() => ({}));
      const err = new Error(p.error || "Refinement key already exists.");
      err.exists = true; err.key = p.key || "";
      throw err;
    }
    if (!res.ok) {
      const p = await res.json().catch(() => ({}));
      throw new Error(p.error || `Import failed with HTTP ${res.status}`);
    }
    return res.json().catch(() => ({}));
  };
  if (bytes.length <= CHUNK) {
    const res = await api.fetchApi(`/funpack/refinement_keys/import?_=${Date.now()}${ow}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (res.status !== 413) return finish(res);
  }
  const uploadId = `${Date.now()}_${Math.random().toString(36).slice(2)}`;
  const postChunk = async (path, body) => {
    const res = await api.fetchApi(path, {
      method: "POST", headers: { "Content-Type": "application/octet-stream" }, body,
    });
    if (!res.ok) {
      const p = await res.json().catch(() => ({}));
      throw new Error(p.error || `Upload failed with HTTP ${res.status}`);
    }
  };
  let index = 0;
  for (let off = 0; off < bytes.length; off += CHUNK, index++) {
    await postChunk(
      `/funpack/refinement_keys/import_chunk?upload_id=${encodeURIComponent(uploadId)}&index=${index}`,
      bytes.slice(off, off + CHUNK),
    );
  }
  return finish(await api.fetchApi(
    `/funpack/refinement_keys/import_finalize?upload_id=${encodeURIComponent(uploadId)}${ow}`,
    { method: "POST" },
  ));
}

// Atomically delete a refinement key AND all its sidecars (value function, blessed
// attention maps / K/V banks, creativity latent, velocity store). Deleting only the
// visible <key>.json by hand orphans those sidecars, which keep steering future runs
// and survive a restart — the backend sweep fixes that.
async function deleteKey(node) {
  const key = selectedKey(node);
  if (!key) {
    app.canvas?.prompt?.("Delete refinement key skipped", "Select or type a refinement key first.", () => {});
    return;
  }
  if (!confirm(`Delete refinement key "${key}"?\n\nThis removes its learned state AND all sidecars (value function, blessed attention/K-V banks, creativity latent, velocity memory). This cannot be undone.`)) {
    return;
  }
  try {
    const res = await api.fetchApi(`/funpack/refinement_keys/delete?key=${encodeURIComponent(key)}&_=${Date.now()}`, {
      method: "POST", cache: "no-store",
    });
    if (!res.ok) {
      const p = await res.json().catch(() => ({}));
      throw new Error(p.error || `Delete failed with HTTP ${res.status}`);
    }
    const kw = keyWidget(node);
    if (kw) kw.value = NONE_KEY;
    await refreshTrackedNodes();
  } catch (error) {
    console.warn("FunPack: refinement key delete failed", error);
    app.canvas?.prompt?.("Refinement key delete failed", error.message || String(error), () => {});
  }
}

// Clear the keyless Absolute "global taste" store. It learns from EVERY rated
// generation across all prompts and is invisible in the key list (dunder name), so it
// can silently bias output (in absolute/both steer mode) and only Session Reset wipes
// it otherwise. Surfaced here so it can be inspected + cleared directly.
async function clearAbsoluteStore() {
  let info = { total_iterations: 0, liked_count: 0, bad_count: 0, exists: false };
  try {
    const r = await api.fetchApi(`/funpack/refinement_keys/absolute?_=${Date.now()}`, { cache: "no-store" });
    if (r.ok) info = await r.json();
  } catch (_) { /* fall through with defaults */ }
  if (!info.exists) {
    app.canvas?.prompt?.("Global taste store", "The Absolute global-taste store is already empty.", () => {});
    return;
  }
  if (!confirm(`Clear the Absolute global-taste store?\n\nIt has pooled ${info.total_iterations} rated generation(s) (${info.liked_count} liked / ${info.bad_count} disliked directions) across all prompts. This is applied only in absolute/both steer mode. This cannot be undone.`)) {
    return;
  }
  try {
    const res = await api.fetchApi(`/funpack/refinement_keys/clear_absolute?_=${Date.now()}`, {
      method: "POST", cache: "no-store",
    });
    if (!res.ok) {
      const p = await res.json().catch(() => ({}));
      throw new Error(p.error || `Clear failed with HTTP ${res.status}`);
    }
  } catch (error) {
    console.warn("FunPack: clear absolute store failed", error);
    app.canvas?.prompt?.("Clear global taste failed", error.message || String(error), () => {});
  }
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
      try {
        await importRefinementKeyData(data, false);
      } catch (e) {
        if (!e.exists) throw e;
        if (!confirm(`A refinement key named "${e.key}" already exists.\n\nOverwrite it with the imported file?`)) {
          return;
        }
        await importRefinementKeyData(data, true);
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
    new FunPackRefinementKeyButton("funpack_refinement_key_delete", "Delete Key", (currentNode) => void deleteKey(currentNode)),
    new FunPackRefinementKeyButton("funpack_refinement_key_clear_absolute", "Clear Global Taste", () => void clearAbsoluteStore()),
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
