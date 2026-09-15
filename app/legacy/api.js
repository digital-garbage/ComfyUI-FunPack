// Thin backend client, rewired against v5's real routes (core/routes.py).
//
// Three kinds of function live here, per the port plan:
//   (a) compatible, same shape        -- just a different URL.
//   (b) compatible route, different shape -- a small adapter in the body.
//   (c) no v5 equivalent yet          -- a stub. Calling one REJECTS with a
//       plain Error naming why, same as any other failed request. v4's ~60
//       caller files were already written defensively for exactly this --
//       every call site that matters is already inside a try/catch or a
//       .catch() with a sensible fallback, because a real request can always
//       fail on a rental. A stub that instead *resolved* to a sentinel value
//       would skip all of that existing handling and hand callers a value
//       shaped nothing like what they expect, which breaks quietly further
//       downstream instead of failing where the call happened -- worse, not
//       better, for "one unbuilt module can't take the rest down with it."
(function () {
  const BASE = window.location.pathname.replace(/\/+$/, "").replace(/\/index\.html$/, "");
  const FUNPACK = "/funpack";
  const API = (p) => `${FUNPACK}${p}`; // v5 mounts everything under core/config.py's UI_PREFIX ("/funpack")

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
      if (Array.isArray(payload.problems) && payload.problems.length) return payload.problems.join("; ");
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

  // A module v5 has not built yet. See the header: this rejects, on purpose,
  // so the try/catch every real caller already has around a fallible request
  // is what handles it -- not a new contract those ~60 files never learned.
  const unsupported = (reason) => Promise.reject(new Error(reason));
  // A plain (non-async) URL getter has no promise to reject -- a placeholder
  // anchor is the closest equivalent; whatever it's set as `href` on simply
  // goes nowhere when clicked, which is a control staying inert, not broken.
  const unsupportedUrl = () => "#";

  const ClientAPI = {
    health: () => j("GET", API("/api/health")),

    // --- projects (a) ------------------------------------------------------
    listProjects: () => j("GET", API("/api/projects")), // {projects:[...]} -- callers unwrap .projects themselves
    createProject: (name) => j("POST", API("/api/projects"), { name }),
    getProject: (id) => j("GET", API(`/api/projects/${id}`)),
    saveProject: (id, data) => j("PUT", API(`/api/projects/${id}`), data),
    deleteProject: (id) => j("DELETE", API(`/api/projects/${id}`)),
    downloadProjectUrl: (id) => API(`/api/projects/${id}/download`),
    importProject: (data) => j("POST", API("/api/projects/import"), data),

    // --- prompt preview (b) -------------------------------------------------
    // v4's `preview()` rendered a whole scene-by-scene montage plan server
    // side; v5 has no render/stitch stage yet (HIGH #2, not built). Stub.
    preview: () => unsupported("no render/stitch stage yet"),
    // `parsePrompt` becomes an adapter onto /api/prompt/expand: fetch the
    // project for its anchor/postfix/variables, then expand THIS text the
    // same way generation would.
    async parsePrompt(id, prompt) {
      const project = await ClientAPI.getProject(id).catch(() => null);
      return j("POST", API("/api/prompt/expand"), {
        text: prompt,
        anchor: project?.anchor || "",
        postfix: project?.postfix || "",
        postfix_enabled: project?.postfix_enabled !== false,
        variables: project?.variables || [],
      });
    },

    // --- transitions library (c) -- no equivalent in v5 --------------------
    transitions: () => unsupported("no transitions library in v5 yet"),
    saveTransition: () => unsupported("no transitions library in v5 yet"),
    deleteTransition: () => unsupported("no transitions library in v5 yet"),
    exportTransitionsUrl: unsupportedUrl,
    importTransitions: () => unsupported("no transitions library in v5 yet"),
    clearTransitions: () => unsupported("no transitions library in v5 yet"),

    // --- prompt shortcuts (a/b) ---------------------------------------------
    shortcuts: () => j("GET", API("/api/shortcuts")), // {shortcuts:[...]} -- callers unwrap .shortcuts themselves
    suggestionStats: () => unsupported("no revolver in v5 yet"),
    saveCategory: () => unsupported("shortcut categories are per-shortcut fields in v5, not a separate CRUD"),
    saveShortcut: (item) => j("POST", API("/api/shortcuts"), item),
    deleteShortcut: (name) => j("DELETE", API(`/api/shortcuts/${encodeURIComponent(name)}`)),
    exportShortcutsUrl: unsupportedUrl,
    importShortcuts: () => unsupported("no import route in v5 yet"),
    clearShortcuts: () => j("POST", API("/api/shortcuts/clear"), {}),
    revolverSettings: () => unsupported("no revolver in v5 yet"),
    setRevolverSettings: () => unsupported("no revolver in v5 yet"),

    // --- Composer file manager / NLE library (c) ----------------------------
    listFiles: () => unsupported("the Composer file manager was retired with the composer shell"),
    deleteFile: () => unsupported("the Composer file manager was retired with the composer shell"),
    clearFiles: () => unsupported("the Composer file manager was retired with the composer shell"),
    nleLibrary: () => unsupported("no render/stitch stage yet"),

    // --- node packs (a) ------------------------------------------------------
    customNodes: async () => (await j("GET", API("/api/packs"))),
    customNodesCheck: () => j("POST", API("/api/packs/check"), {}),
    customNodeInstall: (url) => j("POST", API("/api/packs/install"), { url }),
    customNodeUpdate: (name) => j("POST", API("/api/packs/update"), { name }),
    customNodeRemove: (name) => j("POST", API("/api/packs/remove"), { name }),

    // --- media bin (a) ---------------------------------------------------------
    listMedia: () => j("GET", API("/api/media")), // {media:[...]} -- callers unwrap .media themselves
    mediaUrl: (id) => API(`/api/media/${encodeURIComponent(id)}/file`),
    deleteMedia: (id) => j("DELETE", API(`/api/media/${encodeURIComponent(id)}`)),
    renameMedia: () => unsupported("media has no name field in v5 -- it is addressed by id"),
    importClipToMediaBin: () => unsupported("no render/stitch stage yet, so there is no clip to import"),
    async uploadMedia(file) {
      const fd = new FormData(); fd.append("file", file, file.name);
      const res = await fetch(API("/api/media"), { method: "POST", body: fd });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
      const body = await res.json();
      return body.media && body.media[0] ? body.media[0] : body;
    },

    // --- models / node slots (b/c) -------------------------------------------
    // v4's Studio-shaped "roles" and "candidates per role" have no v5
    // equivalent -- v5's pipeline is a flat list of swappable slots, not a
    // fixed set of named roles. Real replacements below; the rest stub until
    // the Models & Pipeline panel itself is rewritten against this shape
    // (tracked separately -- passing mismatched data through here would make
    // that panel fail in a way that looks like a bug in the panel, not an
    // honest "not built yet").
    nodeRoles: () => unsupported("v5 has slots, not fixed roles -- see pipelinePorts()"),
    nodeCandidates: () => unsupported("v5 has slots, not fixed roles -- see nodeSpec()/pipelinePorts()"),
    allNodes: () => unsupported("no whole-catalogue route in v5 -- nothing scans every installed node at once"),
    nodeSpec: async (cls) => {
      const body = await j("GET", API(`/api/nodes?classes=${encodeURIComponent(cls)}`));
      return body.nodes ? body.nodes[cls] : null;
    },
    // v5's /api/pipeline is real, but its shape ({slots, refused, incomplete,
    // queueable}) is nothing like v4's ({ports, core_producers, requirements,
    // wiring, default_slots}) -- returning it under this name would answer
    // with real JSON that quietly means nothing to the one caller (models.js)
    // that reads it, which is worse than an honest stub: it never surfaces as
    // "not built", just as permanently-empty ports/wiring. Stub until the
    // Models & Pipeline panel is rewritten to read the real shape directly.
    pipelinePorts: () => unsupported("v5's pipeline shape does not match v4's ports/wiring model yet"),
    pipelineDeps: () => unsupported("v5 offers manual pack install only -- no auto-detect for the loaded pipeline"),
    pipelineDepsInstall: () => unsupported("v5 offers manual pack install only -- no auto-detect for the loaded pipeline"),
    pipelineDepsInstallManager: () => unsupported("v5 offers manual pack install only -- no auto-detect for the loaded pipeline"),
    pipelineDepsInstallStatus: () => unsupported("v5 offers manual pack install only -- no auto-detect for the loaded pipeline"),
    pipelineDepsInstallCancel: () => unsupported("v5 offers manual pack install only -- no auto-detect for the loaded pipeline"),
    imageTargets: () => unsupported("not built in v5 yet"),
    // Same mismatch as pipelinePorts above -- v4's caller reads `.nodes`,
    // which /api/pipeline's real response does not have.
    coreGraph: () => unsupported("v5's pipeline shape does not match v4's core-graph model yet"),
    getModels: () => unsupported("the Models panel needs its own rewrite against v5's pipeline-slot shape"),
    saveModels: () => unsupported("the Models panel needs its own rewrite against v5's pipeline-slot shape"),
    settingsCard: async () => { throw new Error("settings card export is not built in v5 yet"); },
    refreshModels: () => unsupported("v5 has no model-file cache to refresh -- it reads the folders live"),
    parseWorkflow: () => unsupported("workflow import is LOW priority, not built yet"),
    applyWorkflow: () => unsupported("workflow import is LOW priority, not built yet"),

    // --- updates / system (a) ------------------------------------------------
    restart: () => j("POST", API("/api/git/restart"), {}),
    systemInfo: () => j("GET", API("/api/system")),
    gitStatus: () => j("GET", API("/api/git/status")),
    gitUpdate: (branch) => j("POST", API("/api/git/update"), branch ? { branch } : {}),
    gitCheckout: (branch) => j("POST", API("/api/git/checkout"), { branch }),
    gitRollback: () => j("POST", API("/api/git/rollback"), {}),

    // --- model-family detection (a) -- new in v5, no v4 equivalent ---------
    probeFamily: (file) => j("GET", API(`/api/probe?file=${encodeURIComponent(file)}`)),

    // --- generate / run (c) --------------------------------------------------
    // v5 deliberately has no server-side queue/status route (run.js's own
    // header comment: "two things tracking what is running is two things
    // that can disagree"). Real generation goes through app/shell/session.js
    // + run.js talking to ComfyUI's own /prompt and websocket directly --
    // that is a UI-flow rewrite (the Generate button's click handler), not
    // an api.js body swap, and is tracked as its own step in the port plan.
    generate: () => unsupported("Generate is wired directly through session.js/run.js, not this seam -- pending the button rewrite"),
    status: () => unsupported("no server-side run status in v5 -- see run.js"),
    progress: () => unsupported("no server-side run status in v5 -- see run.js"),
    active: () => unsupported("no server-side run status in v5 -- see run.js"),
    ratingLabels: () => unsupported("v5's ratings are the fixed pair liked/disliked -- nothing to label"),
    log: (limit) => j("GET", API("/api/log" + (limit ? `?limit=${limit}` : ""))),
    interrupt: () => j("POST", "/interrupt"), // ComfyUI's own native route, not FunPack's

    // --- temp files (a) -------------------------------------------------------
    listTemp: () => j("GET", API("/api/temp")),

    // --- trajectory probe / REINS / block-influence / detail probe (c) ------
    // All LOW priority research tooling per the roadmap; none built in v5.
    probeStatus: () => unsupported("trajectory probe is LOW priority, not built yet"),
    probeSetEnabled: () => unsupported("trajectory probe is LOW priority, not built yet"),
    probeClear: () => unsupported("trajectory probe is LOW priority, not built yet"),
    probeAnalyse: () => unsupported("trajectory probe is LOW priority, not built yet"),
    probeExport: async () => { throw new Error("trajectory probe is LOW priority, not built yet"); },
    probeImport: () => unsupported("trajectory probe is LOW priority, not built yet"),
    reinsStatus: () => unsupported("REINS is LOW priority, not built yet"),
    reinsSweep: () => unsupported("REINS is LOW priority, not built yet"),
    reinsClear: () => unsupported("REINS is LOW priority, not built yet"),
    reinsRateSlot: () => unsupported("REINS is LOW priority, not built yet"),
    reinsDiscardSlot: () => unsupported("REINS is LOW priority, not built yet"),
    reinsExport: async () => { throw new Error("REINS is LOW priority, not built yet"); },
    blockInfluenceStatus: () => unsupported("block influence probe is LOW priority, not built yet"),
    blockInfluenceSetEnabled: () => unsupported("block influence probe is LOW priority, not built yet"),
    blockInfluenceClear: () => unsupported("block influence probe is LOW priority, not built yet"),
    blockInfluenceExport: async () => { throw new Error("block influence probe is LOW priority, not built yet"); },
    detailProbeStatus: () => unsupported("detail probe is LOW priority, not built yet"),
    detailProbeSetEnabled: () => unsupported("detail probe is LOW priority, not built yet"),
    detailProbeClear: () => unsupported("detail probe is LOW priority, not built yet"),

    // --- ComfyUI's own temp-output viewer (a) -- unrelated to FunPack's prefix
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

    // --- render / stitch / export (c) -- HIGH #2, not built ------------------
    renderFinal: () => unsupported("no render/stitch stage yet"),
    renderFinalStatus: () => unsupported("no render/stitch stage yet"),
    exportClip: () => unsupported("no render/stitch stage yet"),
    exportClipsCombined: () => unsupported("no render/stitch stage yet"),
    exportClipsStatus: () => unsupported("no render/stitch stage yet"),
    // (b): v5 has no per-project /result route, but a finished output is just
    // a file ComfyUI itself saved -- its own /view endpoint serves it exactly
    // the way it serves a temp preview.
    resultUrl: (_id, m) => "/view?" + new URLSearchParams({
      filename: m.filename, subfolder: m.subfolder || "", type: m.type || "output",
    }).toString(),
    previewSegmentUrl: unsupportedUrl,

    // --- refinement keys / absolute taste store (c) -- not built in v5 -------
    refinementKeys: () => unsupported("refinement keys are not built in v5 yet"),
    importRefinementKey: () => unsupported("refinement keys are not built in v5 yet"),
    exportRefinementKeyFile: async () => { throw new Error("refinement keys are not built in v5 yet"); },
    deleteRefinementKey: () => unsupported("refinement keys are not built in v5 yet"),
    absoluteStoreInfo: () => unsupported("refinement keys are not built in v5 yet"),
    clearAbsoluteStore: () => unsupported("refinement keys are not built in v5 yet"),
  };

  window.MovieEditorAPI = ClientAPI;
})();
