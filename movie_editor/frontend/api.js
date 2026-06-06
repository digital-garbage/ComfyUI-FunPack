// Thin backend client for the Movie Editor sidecar.
(function () {
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

  const API = {
    health: () => j("GET", "/api/health"),

    // projects
    listProjects: () => j("GET", "/api/projects"),
    createProject: (name) => j("POST", "/api/projects", { name }),
    getProject: (id) => j("GET", `/api/projects/${id}`),
    saveProject: (id, data) => j("PUT", `/api/projects/${id}`, data),
    deleteProject: (id) => j("DELETE", `/api/projects/${id}`),

    // timeline preview
    preview: (id, includeExcluded) =>
      j("GET", `/api/projects/${id}/preview?include_excluded=${includeExcluded ? "true" : "false"}`),

    // libraries
    transitions: () => j("GET", "/api/library/transitions"),

    // generate
    generate: (id, onlyScene) => j("POST", `/api/projects/${id}/generate`, { only_scene: onlyScene || null }),
    status: (id, promptId) => j("GET", `/api/projects/${id}/status/${promptId}`),
    resultUrl: (id, m) =>
      `/api/projects/${id}/result?filename=${encodeURIComponent(m.filename)}` +
      `&subfolder=${encodeURIComponent(m.subfolder || "")}&type=${encodeURIComponent(m.type || "output")}`,
  };

  window.MovieEditorAPI = API;
})();
