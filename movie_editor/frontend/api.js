// Thin backend client. Served by ComfyUI, so all URLs are same-origin (relative):
// the browser automatically uses whatever host:port ComfyUI runs on.
(function () {
  // Base = the directory this app is served from, i.e. /funpack
  const BASE = window.location.pathname.replace(/\/+$/, "").replace(/\/index\.html$/, "");
  const API = (p) => `${BASE}/api${p}`;
  const FUNPACK = "/funpack";

  async function funpackFetch(method, path, body) {
    const opts = { method, headers: {}, cache: "no-store" };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(path, opts);
    if (!res.ok) {
      let payload = null;
      try { payload = await res.json(); } catch (_) {}
      throw new Error(readApiError(res, payload));
    }
    return res;
  }

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
    importTransitions: (data, mode) => j("POST", API(`/library/transitions/import?mode=${mode || "merge"}`), data),
    clearTransitions: () => j("POST", API("/library/transitions/clear"), {}),
    shortcuts: () => j("GET", API("/library/shortcuts")),
    suggestionStats: () => j("GET", API("/library/suggestion_stats")),
    saveCategory: (payload) => j("POST", API("/library/categories"), payload),
    saveShortcut: (item) => j("POST", API("/library/shortcuts"), item),
    deleteShortcut: (name) => j("DELETE", API(`/library/shortcuts/${encodeURIComponent(name)}`)),
    exportShortcutsUrl: () => API("/library/shortcuts/export"),
    importShortcuts: (data, mode) => j("POST", API(`/library/shortcuts/import?mode=${mode || "merge"}`), data),
    clearShortcuts: () => j("POST", API("/library/shortcuts/clear"), {}),
    revolverSettings: () => j("GET", API("/library/revolver")),
    setRevolverSettings: (payload) => j("POST", API("/library/revolver"), payload),

    // FunPack file manager (Composer ▸ Files)
    listFiles: () => j("GET", API("/files")),
    deleteFile: (group, name) => j("DELETE", API(`/files/${encodeURIComponent(group)}/${encodeURIComponent(name)}`)),
    clearFiles: (group) => j("POST", API(`/files/${encodeURIComponent(group)}/clear`), {}),
    nleLibrary: () => j("GET", API("/library/nle")),
    customNodes: () => j("GET", API("/custom-nodes")),
    customNodesCheck: () => j("POST", API("/custom-nodes/check"), {}),
    customNodeInstall: (url) => j("POST", API("/custom-nodes/install"), { url }),
    customNodeUpdate: (name) => j("POST", API("/custom-nodes/update"), { name }),
    customNodeRemove: (name) => j("POST", API("/custom-nodes/remove"), { name }),

    // media bin
    listMedia: () => j("GET", API("/media")),
    mediaUrl: (id) => API(`/media/${encodeURIComponent(id)}`),
    deleteMedia: (id) => j("DELETE", API(`/media/${encodeURIComponent(id)}`)),
    renameMedia: (id, name) => j("PATCH", API(`/media/${encodeURIComponent(id)}`), { name }),
    importClipToMediaBin: (clip, name) => j("POST", API("/media/import-clip"), { clip, name: name || null }),
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
    // pid is optional but should be passed whenever a project is open: without it the answer
    // describes the GLOBAL default, which is not necessarily this project's family.
    pipelinePorts: (pid) => j("GET", API("/pipeline-ports" + (pid ? "?pid=" + encodeURIComponent(pid) : ""))),
    pipelineDeps: (pid) => j("GET", API("/pipeline-deps" + (pid ? "?pid=" + encodeURIComponent(pid) : ""))),
    pipelineDepsInstall: (packIds) => j("POST", API("/pipeline-deps/install"), { pack_ids: packIds }),
    pipelineDepsInstallManager: () => j("POST", API("/pipeline-deps/install"), { install_manager: true }),
    pipelineDepsInstallStatus: (jobId) => j("GET", API(`/pipeline-deps/install/${encodeURIComponent(jobId)}`)),
    pipelineDepsInstallCancel: (jobId) => j("POST", API(`/pipeline-deps/install/${encodeURIComponent(jobId)}/cancel`), {}),
    imageTargets: (pid) => j("GET", API("/image-targets" + (pid ? `?pid=${encodeURIComponent(pid)}` : ""))),
    // Models config is per-project; pass the project id. Falls back to the global
    // default route (used as the seed/template) when no project is given.
    coreGraph: (pid) => j("GET", API("/core-graph" + (pid ? `?pid=${encodeURIComponent(pid)}` : ""))),
    getModels: (pid) => j("GET", API(pid ? `/projects/${pid}/models` : "/models")),
    // The settings card is a PNG, not JSON — fetched as a blob so the modal can show it,
    // download it and put it on the clipboard from the one response.
    settingsCard: async (pid, theme) => {
      const q = new URLSearchParams();
      if (pid) q.set("pid", pid);
      if (theme) q.set("theme", theme);
      const r = await fetch(API("/settings-card") + (q.toString() ? `?${q}` : ""));
      if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
      return r.blob();
    },
    saveModels: (pid, data) => j("PUT", API(pid ? `/projects/${pid}/models` : "/models"), data),
    refreshModels: () => j("POST", API("/models/refresh")),
    parseWorkflow: (workflow) => j("POST", API("/workflow/parse"), { workflow }),
    applyWorkflow: (pid, workflow, bindings) =>
      j("POST", API(`/projects/${pid}/workflow/apply`), { workflow, bindings }),
    restart: () => j("POST", API("/restart")),
    systemInfo: () => j("GET", API("/system/info")),
    gitStatus: () => j("GET", API("/git/status")),
    gitUpdate: (branch) => j("POST", API("/git/update"), branch ? { branch } : {}),
    gitCheckout: (branch) => j("POST", API("/git/checkout"), { branch }),

    // generate (a single scene, or an explicit run of scene ids = one chain request)
    // `simple` strips the enhancements for THIS run only (see pipeline_caps.apply_simple_mode).
    generate: (id, onlyScene, sceneIds, resetSession, nodeOverrides) =>
      j("POST", API(`/projects/${id}/generate`), { only_scene: onlyScene || null, scene_ids: sceneIds || null, reset_session: !!resetSession, node_overrides: nodeOverrides || null, simple: !!window.FunPackMode?.isSimple() }),
    status: (id, promptId) => j("GET", API(`/projects/${id}/status/${promptId}`)),
    progress: () => j("GET", API("/progress")),
    // The editor's own in-flight generation, recovered from ComfyUI's queue (survives a UI reload).
    active: () => j("GET", API("/active")),
    ratingLabels: () => j("GET", API("/rating-labels")),
    log: (limit) => j("GET", API("/log" + (limit ? `?limit=${limit}` : ""))),
    interrupt: () => j("POST", API("/interrupt")),

    // ComfyUI temp folder browser (scene previews & other transient outputs, wiped on restart).
    listTemp: () => j("GET", API("/temp")),

    // trajectory probe (Settings ▸ Learning)
    probeStatus: () => j("GET", API("/trajectory_probe")),
    probeSetEnabled: (enabled) => j("POST", API("/trajectory_probe"), { enabled: !!enabled }),
    probeClear: () => j("POST", API("/trajectory_probe/clear"), {}),
    probeAnalyse: (trials) => j("POST", API("/trajectory_probe/analyse"), { trials: trials || 2000 }),

    // The measurement has to outlive the box it was taken on: a rental gets
    // replaced, refinements/ is not in git, and the reading needs ~16 rated runs.
    async probeExport() {
      const res = await fetch(API("/trajectory_probe/export") + `?t=${Date.now()}`, { cache: "no-store" });
      if (!res.ok) throw new Error(readApiError(res, null));
      const url = URL.createObjectURL(await res.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = "trajectory_probe.pt";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    },
    async probeImport(file) {
      const res = await fetch(API("/trajectory_probe/import"), { method: "POST", body: file });
      let payload = null;
      try { payload = await res.json(); } catch (_) {}
      if (!res.ok) throw new Error(readApiError(res, payload));
      return payload;
    },

    // H3 representation steering (Settings ▸ Learning) — per-KEY, unlike the probe above.
    reinsStatus: (key) => j("GET", API("/h3_repr_steering") + `?key=${encodeURIComponent(key || "default")}`),
    reinsClear: (key) => j("POST", API("/h3_repr_steering/clear"), { key: key || "default" }),
    async reinsExport(key) {
      const k = key || "default";
      const res = await fetch(API("/h3_repr_steering/export") + `?key=${encodeURIComponent(k)}&t=${Date.now()}`, { cache: "no-store" });
      if (!res.ok) throw new Error(readApiError(res, null));
      const url = URL.createObjectURL(await res.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = `${k}.repr_steer.pt`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    },
    // Served by ComfyUI's own same-origin /view endpoint — no editor route needed.
    tempFileUrl: (f) => "/view?" + new URLSearchParams({
      filename: f.filename, subfolder: f.subfolder || "", type: "temp", t: String(f.mtime || Date.now()),
    }).toString(),
    async downloadTempFile(f) {
      const res = await fetch(this.tempFileUrl(f), { cache: "no-store" });
      if (!res.ok) throw new Error(`Could not fetch ${f.filename} (HTTP ${res.status})`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = f.filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    },
    renderFinal: (id, clips) => j("POST", API(`/projects/${id}/render`), { clips }),
    renderFinalStatus: (id, jobId) => j("GET", API(`/projects/${id}/render/${encodeURIComponent(jobId)}`)),
    exportClip: (id, clip) => j("POST", API(`/projects/${id}/export-clip`), { clip }),
    exportClipsCombined: (id, clips) => j("POST", API(`/projects/${id}/export-clips`), { clips }),
    exportClipsStatus: (id, jobId) => j("GET", API(`/projects/${id}/export-clips/${encodeURIComponent(jobId)}`)),
    resultUrl: (id, m) =>
      API(`/projects/${id}/result?filename=${encodeURIComponent(m.filename)}`) +
      `&subfolder=${encodeURIComponent(m.subfolder || "")}&type=${encodeURIComponent(m.type || "output")}`,
    previewSegmentUrl: (id, sceneId, spec) => {
      let u = API(`/projects/${id}/preview-segment/${encodeURIComponent(sceneId)}`);
      const m = spec?.media;
      if (m?.filename) {
        const q = new URLSearchParams({
          filename: m.filename,
          subfolder: m.subfolder || "",
          type: m.type || "output",
          render_in: String(spec.renderIn != null ? spec.renderIn : 0),
        });
        // dur is in the URL for two reasons: ghosts have no scene server-side (the trim
        // window must travel in the query), and it versions the browser cache — segments
        // are cached for an hour, so a timeline trim must produce a different URL.
        if (spec.dur != null) q.set("dur", String(spec.dur));
        // Reverse changes the bytes, so it must change the URL — segments are cached for
        // an hour and would otherwise replay the forward encode.
        if (spec.reverse) q.set("rev", "1");
        u += "?" + q.toString();
      }
      return u;
    },

    // FunPack refinement keys (ComfyUI root routes, same as Studio / Refinement Key Loader)
    refinementKeys: async () => {
      const res = await funpackFetch("GET", `${FUNPACK}/refinement_keys?cache_bust=${Date.now()}`);
      return res.json();
    },
    importRefinementKey: async (data, opts = {}) => {
      const ow = opts.overwrite ? "&overwrite=true" : "";
      const json = JSON.stringify(data);
      const bytes = new TextEncoder().encode(json);
      // A 409 means a key of that name already exists — surface it as a tagged
      // error so the caller can ask the user to overwrite (then retry).
      const finish = async (res) => {
        if (res.status === 409) {
          let p = null; try { p = await res.json(); } catch (_) {}
          const err = new Error((p && p.error) || "Refinement key already exists.");
          err.exists = true; err.key = (p && p.key) || "";
          throw err;
        }
        if (!res.ok) {
          let p = null; try { p = await res.json(); } catch (_) {}
          throw new Error(readApiError(res, p));
        }
        return res.json();
      };
      // Reverse proxies on Vast.ai / Runpod cap the request body well below
      // ComfyUI's 100 MB, so a one-shot POST of a large key gets HTTP 413 at the
      // proxy. Keep a fast single POST for small keys; stream big ones in chunks
      // (each far under any proxy limit) so they can never be rejected.
      const CHUNK = 256 * 1024;
      if (bytes.length <= CHUNK) {
        const res = await fetch(`${FUNPACK}/refinement_keys/import?_=${Date.now()}${ow}`, {
          method: "POST", cache: "no-store",
          headers: { "Content-Type": "application/json" }, body: json,
        });
        if (res.status !== 413) return finish(res); // 413 → fall through to chunked
      }
      const uploadId = `${Date.now()}_${Math.random().toString(36).slice(2)}`;
      const postChunk = async (path, body) => {
        const res = await fetch(path, {
          method: "POST", cache: "no-store",
          headers: { "Content-Type": "application/octet-stream" }, body,
        });
        if (!res.ok) {
          let payload = null;
          try { payload = await res.json(); } catch (_) {}
          throw new Error(readApiError(res, payload));
        }
        return res;
      };
      let index = 0;
      for (let off = 0; off < bytes.length; off += CHUNK, index++) {
        // Slice by bytes (not string chars) so a multi-byte char is never split
        // across the seam where the two halves get re-encoded independently.
        await postChunk(
          `${FUNPACK}/refinement_keys/import_chunk?upload_id=${encodeURIComponent(uploadId)}&index=${index}`,
          bytes.slice(off, off + CHUNK),
        );
      }
      return finish(await fetch(
        `${FUNPACK}/refinement_keys/import_finalize?upload_id=${encodeURIComponent(uploadId)}${ow}`,
        { method: "POST", cache: "no-store" },
      ));
    },
    async exportRefinementKeyFile(key) {
      const res = await fetch(
        `${FUNPACK}/refinement_keys/export?key=${encodeURIComponent(key)}&cache_bust=${Date.now()}`,
        { cache: "no-store" },
      );
      if (!res.ok) {
        let payload = null;
        try { payload = await res.json(); } catch (_) {}
        throw new Error(readApiError(res, payload));
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${key}.json`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    },

    // Atomically delete a refinement key and ALL its sidecars (value function,
    // blessed attention / K-V banks, creativity latent, velocity store). Deleting
    // only the visible <key>.json orphans those, and they keep steering future runs
    // across restarts — the backend sweep is what makes "key deleted" mean it.
    async deleteRefinementKey(key) {
      const res = await funpackFetch(
        "POST", `${FUNPACK}/refinement_keys/delete?key=${encodeURIComponent(key)}&_=${Date.now()}`,
      );
      if (!res.ok) {
        let payload = null;
        try { payload = await res.json(); } catch (_) {}
        throw new Error(readApiError(res, payload));
      }
      return res.json();
    },

    // The keyless Absolute "global taste" store — learns from every rated generation
    // across all prompts, invisible in the key list, applied only in absolute/both
    // steer mode, and otherwise only wiped by Session Reset.
    async absoluteStoreInfo() {
      const res = await funpackFetch("GET", `${FUNPACK}/refinement_keys/absolute?_=${Date.now()}`);
      return res.json();
    },
    async clearAbsoluteStore() {
      const res = await funpackFetch("POST", `${FUNPACK}/refinement_keys/clear_absolute?_=${Date.now()}`);
      if (!res.ok) {
        let payload = null;
        try { payload = await res.json(); } catch (_) {}
        throw new Error(readApiError(res, payload));
      }
      return res.json();
    },
  };

  window.MovieEditorAPI = ClientAPI;
})();
