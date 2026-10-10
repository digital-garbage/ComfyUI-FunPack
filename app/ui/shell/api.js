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
  if (!res.ok) throw Object.assign(new Error(why(res, payload)), { status: res.status });      // status: a refusal (4xx) is not an outage
  return payload;
}

export const api = {
  pipeline: () => call("GET", "/api/pipeline"),
  editPipeline: (body) => call("POST", "/api/pipeline", body),
  modules: (traits) => call("GET", `/api/modules${traits != null ? `?traits=${encodeURIComponent(traits)}` : ""}`),
  releaseModule: (id) => call("POST", "/api/control/release", { id }),
  searchNodes: (q, limit = 40) => call("GET", `/api/nodes/search?q=${encodeURIComponent(q || "")}&limit=${limit}`),
  pipelinePresets: () => call("GET", "/api/pipeline/presets"),
  setMediaSubject: (id, subject) => call("PATCH", `/api/media/${encodeURIComponent(id)}`, { subject }),
  readiness: () => call("GET", "/api/readiness"),
  importWorkflow: (workflow) => call("POST", "/api/pipeline/import", { workflow }),
  /** The pipeline as a PNG (loaders, typed-in values, host torch/CUDA). -> a Blob. */
  async settingsCard(slots, projectName, theme) {
    const res = await fetch(`${BASE}/api/settings-card`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ slots, project_name: projectName || null, theme: theme || "dark" }) });
    if (!res.ok) throw new Error(why(res, await res.json().catch(() => null)));
    return res.blob();
  },
  describeNodes: (classes) => call("GET", `/api/nodes?classes=${encodeURIComponent(classes.join(","))}`),
  system: () => call("GET", "/api/system"),
  storyMarkers: () => call("GET", "/api/story/markers"),
  saveStoryMarkers: (markers) => call("POST", "/api/story/markers", { markers }),
  storySplit: (text) => call("POST", "/api/story/split", { text }),
  shortcuts: () => call("GET", "/api/shortcuts"),
  saveShortcut: (shortcut, originalName) => call("POST", "/api/shortcuts", { ...shortcut, original_name: originalName }),
  clearShortcuts: () => call("POST", "/api/shortcuts/clear", {}),
  importShortcuts: (data, mode) => call("POST", "/api/shortcuts/import", { data, mode }),
  addShortcutCategory: (category, sub_category) => call("POST", "/api/shortcuts/category", { category, sub_category }),
  deleteShortcut: (name) => call("DELETE", `/api/shortcuts/${encodeURIComponent(name)}`),
  revolver: () => call("GET", "/api/shortcuts/revolver"),
  setRevolver: (enabled, random) => call("POST", "/api/shortcuts/revolver", { enabled, random }),
  enhancerDefaults: () => call("GET", "/api/m/conditioning_prompt_enhancer/defaults"),
  enhancerRuns: () => call("GET", "/api/m/conditioning_prompt_enhancer/runs"),
  rateTaste: (promptId, rating, axis) => call("POST", "/api/m/taste/rate", { prompt_id: promptId, rating, axis: axis || null }),
  newTasteGeneration: (promptId) => call("POST", "/api/m/taste/generation", promptId ? { prompt_id: promptId } : {}),
  shotMemory: () => call("GET", "/api/m/conditioning_shot_camera/memory"),
  forgetShotMemory: (kind, name) => call("POST", "/api/m/conditioning_shot_camera/forget", { kind, name }),
  tasteKeys: () => call("GET", "/api/m/taste/keys"),
  blockInfluence: (key) => call("GET", `/api/m/block_influence/status?key=${encodeURIComponent(key || "default")}`),
  blockInfluenceGroups: (key) => call("GET", `/api/m/block_influence/groups?key=${encodeURIComponent(key || "default")}`),
  setBlockInfluence: (on) => call("POST", `/api/m/block_influence/enabled?key=default`, { enabled: !!on }),
  libraryFiles: () => call("GET", "/api/files"),
  deleteLibraryFile: (name) => call("DELETE", `/api/files/${encodeURIComponent(name)}`),
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
  /** Whether ComfyUI's queue holds a run of this project (queued from any tab). Unreadable: false, the queue decides. */
  async projectQueued(projectId) {
    try {
      const q = await (await fetch("/queue", { signal: AbortSignal.timeout(5000) })).json();      // a stalled server: not a hang
      return [...(q.queue_running || []), ...(q.queue_pending || [])].some((r) => (r[3] || {}).funpack_project_id === projectId);
    } catch { return false; }
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
  renderLibrary: () => call("GET", "/api/m/render/library"),
  lastFrame: (pid, body) => call("POST", `/api/m/render/projects/${encodeURIComponent(pid)}/last-frame`, body),
  suggestionStats: () => call("GET", "/api/shortcuts/suggestion_stats"),
  expandPrompt: (body) => call("POST", "/api/prompt/expand", { ...body, seed: 1 }),
  packs: () => call("GET", "/api/packs"),
  packProviders: (classes) => call("GET", `/api/packs/providers?classes=${encodeURIComponent(classes.join(","))}`),
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
  async uploadMedia(files, onProgress) {       // onProgress(done, total, name) before each file
    const out = { media: [], problems: [] };
    let done = 0;
    for (const file of files) {
      if (onProgress) onProgress(done++, files.length, file.name);
      const form = new FormData();
      form.append("file", file, file.name);
      try {
        const res = await fetch(`${BASE}/api/media`, { method: "POST", body: form });
        const payload = await res.json().catch(() => null);
        if (!res.ok) throw Object.assign(new Error(why(res, payload)), { status: res.status });      // status: a refusal (4xx) is not an outage
        out.media.push(...(payload.media || [])); out.problems.push(...(payload.problems || []));
      } catch (err) { out.problems.push(`${file.name}: ${err.message}`); }
    }
    return out;
  },
  probeFamily: (file) => call("GET", `/api/probe?file=${encodeURIComponent(file)}`),
};
