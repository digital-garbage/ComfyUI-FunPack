// Thin backend client. Served by ComfyUI, so all URLs are same-origin (relative):
// the browser automatically uses whatever host:port ComfyUI runs on.
(function () {
  // Base = the directory this app is served from, e.g. /funpack/movie
  const BASE = window.location.pathname.replace(/\/+$/, "").replace(/\/index\.html$/, "");
  const API = (p) => `${BASE}/api${p}`;

  function readApiError(res, payload) {
    if (payload && typeof payload === "object") {
      const detail = payload.detail;
      if (typeof detail === "string" && detail.trim()) return detail.trim();
      if (Array.isArray(detail)) {
        const parts = detail.map((item) => {
          if (item && typeof item === "object" && item.msg) return String(item.msg);
          return JSON.stringify(item);
        }).filter(Boolean);
        if (parts.length) return parts.join("; ");
      }
      if (payload.error) return String(payload.error);
      if (payload.parse_errors && typeof payload.parse_errors === "object") {
        return Object.entries(payload.parse_errors).map(([k, v]) => `${k}: ${v}`).join("; ");
      }
    }
    const status = res && res.status ? `HTTP ${res.status}` : "";
    const statusText = (res && res.statusText ? res.statusText : "").trim();
    return statusText || status || "Request failed";
  }

  async function j(method, url, body) {
    const opts = { method, headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(url, opts);
    if (!res.ok) {
      let payload = null;
      try { payload = await res.json(); } catch (_) {}
      throw new Error(readApiError(res, payload));
    }
    return res.status === 204 ? null : res.json();
  }

  const ClientAPI = {
    health: () => j("GET", API("/health")),

    // projects
    listProjects: () => j("GET", API("/projects")),
    createProject: (name) => j("POST", API("/projects"), { name }),
    getProject: (id) => j("GET", API(`/projects/${id}`)),
    saveProject: (id, data) => j("PUT", API(`/projects/${id}`), data),
    deleteProject: (id) => j("DELETE", API(`/projects/${id}`)),
    downloadProjectUrl: (id) => API(`/projects/${id}/download`),
    importProject: (data) => j("POST", API("/projects/import"), data),

    // timeline preview
    preview: (id, includeExcluded, forGeneration = true) =>
      j("GET", API(`/projects/${id}/preview?include_excluded=${includeExcluded ? "true" : "false"}&for_generation=${forGeneration ? "true" : "false"}`)),
    parsePrompt: (id, prompt) => j("POST", API(`/projects/${id}/parse`), { prompt }),

    // libraries
    transitions: () => j("GET", API("/library/transitions")),
    saveTransition: (item) => j("POST", API("/library/transitions"), item),
    deleteTransition: (name) => j("DELETE", API(`/library/transitions/${encodeURIComponent(name)}`)),
    exportTransitionsUrl: () => API("/library/transitions/export"),
    importTransitions: (data) => j("POST", API("/library/transitions/import"), data),
    shortcuts: () => j("GET", API("/library/shortcuts")),
    saveShortcut: (item) => j("POST", API("/library/shortcuts"), item),
    deleteShortcut: (name) => j("DELETE", API(`/library/shortcuts/${encodeURIComponent(name)}`)),
    exportShortcutsUrl: () => API("/library/shortcuts/export"),
    importShortcuts: (data) => j("POST", API("/library/shortcuts/import"), data),
    characters: () => j("GET", API("/library/characters")),
    saveCharacter: (item) => j("POST", API("/library/characters"), item),
    deleteCharacter: (id) => j("DELETE", API(`/library/characters/${encodeURIComponent(id)}`)),
    nleLibrary: () => j("GET", API("/library/nle")),

    // media bin
    listMedia: () => j("GET", API("/media")),
    mediaUrl: (id) => API(`/media/${encodeURIComponent(id)}`),
    deleteMedia: (id) => j("DELETE", API(`/media/${encodeURIComponent(id)}`)),
    async uploadMedia(file) {
      const fd = new FormData(); fd.append("file", file, file.name);
      const res = await fetch(API("/media"), { method: "POST", body: fd });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
      return res.json();
    },

    // models / node slots
    nodeRoles: () => j("GET", API("/node-roles")),
    nodeCandidates: (role, refresh) => j("GET", API(`/node-candidates/${role}${refresh ? "?refresh=true" : ""}`)),
    allNodes: () => j("GET", API("/all-nodes")),
    nodeSpec: (cls) => j("GET", API(`/node/${encodeURIComponent(cls)}`)),
    pipelinePorts: () => j("GET", API("/pipeline-ports")),
    imageTargets: (pid) => j("GET", API("/image-targets" + (pid ? `?pid=${encodeURIComponent(pid)}` : ""))),
    // Models config is per-project; pass the project id. Falls back to the global
    // default route (used as the seed/template) when no project is given.
    coreGraph: (pid) => j("GET", API("/core-graph" + (pid ? `?pid=${encodeURIComponent(pid)}` : ""))),
    getModels: (pid) => j("GET", API(pid ? `/projects/${pid}/models` : "/models")),
    saveModels: (pid, data) => j("PUT", API(pid ? `/projects/${pid}/models` : "/models"), data),
    refreshModels: () => j("POST", API("/models/refresh")),
    restart: () => j("POST", API("/restart")),

    // generate (a single scene, or an explicit run of scene ids = one chain request)
    generate: (id, onlyScene, sceneIds, resetSession) =>
      j("POST", API(`/projects/${id}/generate`), { only_scene: onlyScene || null, scene_ids: sceneIds || null, reset_session: !!resetSession }),
    status: (id, promptId) => j("GET", API(`/projects/${id}/status/${promptId}`)),
    progress: () => j("GET", API("/progress")),
    ratingLabels: () => j("GET", API("/rating-labels")),
    log: (limit) => j("GET", API("/log" + (limit ? `?limit=${limit}` : ""))),
    interrupt: () => j("POST", API("/interrupt")),
    renderFinal: (id, clips) => j("POST", API(`/projects/${id}/render`), { clips }),
    exportClip: (id, clip) => j("POST", API(`/projects/${id}/export-clip`), { clip }),
    resultUrl: (id, m) =>
      API(`/projects/${id}/result?filename=${encodeURIComponent(m.filename)}`) +
      `&subfolder=${encodeURIComponent(m.subfolder || "")}&type=${encodeURIComponent(m.type || "output")}`,
  };

  window.MovieEditorAPI = ClientAPI;
})();
