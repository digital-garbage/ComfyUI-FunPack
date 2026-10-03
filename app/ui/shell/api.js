// The server's routes, one line each. An error carries the sentence the server gave, not a bare status.
const BASE = "/funpack";

function why(res, payload) {
  const p = payload && typeof payload === "object" ? payload : {};
  const detail = Array.isArray(p.detail) ? p.detail.map((d) => (d && d.msg) || JSON.stringify(d)).join("; ") : p.detail;
  const list = (p.problems && p.problems.length ? p.problems : p.refused) || [];
  return [detail, p.why, p.error, list.join("; ")].find((s) => typeof s === "string" && s.trim()) || `HTTP ${res.status}`;
}

export async function call(method, path, body) {
  const res = await fetch(BASE + path, {
    method, headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const payload = await res.json().catch(() => null);
  if (!res.ok) throw new Error(why(res, payload));
  return payload;
}

export const api = {
  pipeline: () => call("GET", "/api/pipeline"),
  editPipeline: (body) => call("POST", "/api/pipeline", body),
  modules: (traits) => call("GET", `/api/modules${traits ? `?traits=${encodeURIComponent(traits)}` : ""}`),
  releaseModule: (id) => call("POST", "/api/control/release", { id }),
  describeNodes: (classes) => call("GET", `/api/nodes?classes=${encodeURIComponent(classes.join(","))}`),
  system: () => call("GET", "/api/system"),
  storyMarkers: () => call("GET", "/api/story/markers"),
  saveStoryMarkers: (markers) => call("POST", "/api/story/markers", { markers }),
  storySplit: (text) => call("POST", "/api/story/split", { text }),
  shortcuts: () => call("GET", "/api/shortcuts"),
  saveShortcut: (shortcut, originalName) => call("POST", "/api/shortcuts", { ...shortcut, original_name: originalName }),
  deleteShortcut: (name) => call("DELETE", `/api/shortcuts/${encodeURIComponent(name)}`),
  revolver: () => call("GET", "/api/shortcuts/revolver"),
  setRevolver: (enabled, random) => call("POST", "/api/shortcuts/revolver", { enabled, random }),
  enhancerDefaults: () => call("GET", "/api/m/conditioning_prompt_enhancer/defaults"),
  enhancerRuns: () => call("GET", "/api/m/conditioning_prompt_enhancer/runs"),
  rateTaste: (promptId, rating, axis) => call("POST", "/api/m/taste/rate", { prompt_id: promptId, rating, axis: axis || null }),
  newTasteGeneration: () => call("POST", "/api/m/taste/generation", {}),
  tasteKeys: () => call("GET", "/api/m/taste/keys"),
  deleteTasteKey: (name) => call("DELETE", `/api/m/taste/keys/${encodeURIComponent(name)}`),
  tasteKeyUrl: (name) => `${BASE}/api/m/taste/keys/${encodeURIComponent(name)}/export`,
  /** The zip as the raw body; `exists: true` comes back (not thrown) when the name is taken and `overwrite` was not asked. */
  async importTasteKey(name, file, overwrite) {
    const res = await fetch(`${BASE}/api/m/taste/keys/import?name=${encodeURIComponent(name)}${overwrite ? "&overwrite=1" : ""}`, { method: "POST", body: file });
    const payload = await res.json().catch(() => null);
    if (res.status === 409) return { exists: true };
    if (!res.ok) throw new Error((payload && payload.why) || `HTTP ${res.status}`);
    return payload;
  },
  upscaleModels: () => call("GET", "/api/m/render/upscale_models"),
  /** The upscale is an ordinary ComfyUI job: queued on /prompt, read back from /history. -> the prompt id. */
  async queueUpscale(media, model) {
    const graph = { 1: { class_type: "FunPackUpscaleVideo", inputs: { filename: media.filename, subfolder: media.subfolder || "", type: media.type === "temp" ? "temp" : "output", upscale_model: model } } };
    const res = await fetch("/prompt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: graph }) });
    const body = await res.json().catch(() => ({}));
    if (!res.ok || !body.prompt_id) {
      const nodes = Object.values(body.node_errors || {}).flatMap((n) => (n.errors || []).map((x) => x.message));
      throw new Error([(body.error || {}).message, ...nodes].filter(Boolean).join("; ") || `ComfyUI refused the job (HTTP ${res.status})`);
    }
    return body.prompt_id;
  },
  /** null while it is queued or running, else {videos} | {error} | {gone} | {retry} (a dropped reply is not a failed job). */
  async upscaleResult(promptId) {
    let entry = null;
    try { const res = await fetch(`/history/${encodeURIComponent(promptId)}`); entry = res.ok ? (await res.json())[promptId] : null; } catch { return { retry: true }; }
    if (!entry) {
      try {
        const q = await (await fetch("/queue")).json();
        return [...(q.queue_running || []), ...(q.queue_pending || [])].some((r) => r[1] === promptId) ? null : { gone: true };
      } catch { return { retry: true }; }
    }
    const st = entry.status || {};
    if (st.status_str === "error") { const m = (st.messages || []).find((x) => x[0] === "execution_error"); return { error: (m && m[1] && m[1].exception_message) || "failed inside ComfyUI" }; }
    if (!st.completed) return null;
    const videos = Object.values(entry.outputs || {}).flatMap((o) => o.videos || []);
    return videos.length ? { videos } : { error: "no video came out, check the ComfyUI terminal" };
  },
  suggestionStats: () => call("GET", "/api/shortcuts/suggestion_stats"),
  expandPrompt: (body) => call("POST", "/api/prompt/expand", { ...body, seed: 1 }),
  packs: () => call("GET", "/api/packs"),
  pack: (action, body) => call("POST", `/api/packs/${action}`, body || {}),
  git: (action, body) => call("POST", `/api/git/${action}`, body || {}),
  health: () => call("GET", "/api/health"),
  gitFull: () => call("GET", "/api/git/status"),
  gitStatus: () => call("GET", "/api/git/status?remote=0"),
  media: () => call("GET", "/api/media"),
  renameMedia: (id, name) => call("PATCH", `/api/media/${encodeURIComponent(id)}`, { name }),
  deleteMedia: (id) => call("DELETE", `/api/media/${encodeURIComponent(id)}`),
  /** Upload files as media, one request each so one refusal (too big, wrong type) does not sink the rest.
   *  Resolves {media, problems}. Not JSON, so not through call(). */
  async uploadMedia(files) {
    const out = { media: [], problems: [] };
    for (const file of files) {
      const form = new FormData();
      form.append("file", file, file.name);
      try {
        const res = await fetch(`${BASE}/api/media`, { method: "POST", body: form });
        const payload = await res.json().catch(() => null);
        if (!res.ok) throw new Error(why(res, payload));
        out.media.push(...(payload.media || [])); out.problems.push(...(payload.problems || []));
      } catch (err) { out.problems.push(`${file.name}: ${err.message}`); }
    }
    return out;
  },
  probeFamily: (file) => call("GET", `/api/probe?file=${encodeURIComponent(file)}`),
};
