// Thin backend client. Served by ComfyUI, so all URLs are same-origin (relative):
// the browser automatically uses whatever host:port ComfyUI runs on.
(function () {
  // Base = the directory this app is served from, e.g. /funpack/movie
  const BASE = window.location.pathname.replace(/\/+$/, "").replace(/\/index\.html$/, "");
  const API = (p) => `${BASE}/api${p}`;
  async function j(method, url, body) {
    const opts = { method, headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(url, opts);
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (_) {}
      throw new Error(detail);
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

    // timeline preview
    preview: (id, includeExcluded) =>
      j("GET", API(`/projects/${id}/preview?include_excluded=${includeExcluded ? "true" : "false"}`)),

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
    imageTargets: () => j("GET", API("/image-targets")),
    getModels: () => j("GET", API("/models")),
    saveModels: (data) => j("PUT", API("/models"), data),
    refreshModels: () => j("POST", API("/models/refresh")),
    restart: () => j("POST", API("/restart")),

    // generate
    generate: (id, onlyScene) => j("POST", API(`/projects/${id}/generate`), { only_scene: onlyScene || null }),
    status: (id, promptId) => j("GET", API(`/projects/${id}/status/${promptId}`)),
    resultUrl: (id, m) =>
      API(`/projects/${id}/result?filename=${encodeURIComponent(m.filename)}`) +
      `&subfolder=${encodeURIComponent(m.subfolder || "")}&type=${encodeURIComponent(m.type || "output")}`,
  };

  window.MovieEditorAPI = ClientAPI;
})();
