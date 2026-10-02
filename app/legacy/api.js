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
      if (payload.why) return String(payload.why);
      if (payload.error) return String(payload.error);
      if (Array.isArray(payload.problems) && payload.problems.length) return payload.problems.join("; ");
      // /api/pipeline's malformed-request 400 (core/routes.py's shape_problems
      // check) uses "refused" rather than "problems" -- its own sibling 400s
      // in the same handler use "problems", but this one doesn't, so both are
      // checked rather than trusting the route to be internally consistent.
      if (Array.isArray(payload.refused) && payload.refused.length) return payload.refused.join("; ");
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

    // --- module manifest / live pipeline (new in v5, no v4 equivalent) -----
    // What the Engine Settings panel renders from: every module that loaded,
    // its category, and its typed settings schema. `traits`, when given, is
    // a comma-joined list -- modules the current model can't use are excluded
    // server-side rather than shown disabled (core/traits.py's split()).
    modules: (traits) => j("GET", API("/api/modules" + (traits ? `?traits=${encodeURIComponent(traits)}` : ""))),
    // The live, editable pipeline -- {slots, refused, incomplete, queueable}.
    // Stateless on the server (see routes.py): the caller holds `slots` in
    // memory across edits and re-sends the full list each time.
    pipeline: () => j("GET", API("/api/pipeline")),
    editPipeline: (body) => j("POST", API("/api/pipeline"), body),

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
    // The Story box's text, cut into scenes by the marker words (core/story.py).
    // The split is the server's -- one implementation -- and the anchor is NOT
    // part of a story, so it is never reported back: store.js leaves it alone.
    async parsePrompt(_id, prompt) {
      const { scenes } = await j("POST", API("/api/story/split"), { text: prompt });
      return { parsed_verbatim: { anchor: "", scenes: scenes.map((text) => ({ text })), transitions: [] } };
    },
    storyMarkers: () => j("GET", API("/api/story/markers")),            // {markers:[...]}
    saveStoryMarkers: (markers) => j("POST", API("/api/story/markers"), { markers }),

    // --- transitions library (c) -- no equivalent in v5 --------------------
    transitions: () => unsupported("no transitions library in v5 yet"),
    saveTransition: () => unsupported("no transitions library in v5 yet"),
    deleteTransition: () => unsupported("no transitions library in v5 yet"),
    exportTransitionsUrl: unsupportedUrl,
    importTransitions: () => unsupported("no transitions library in v5 yet"),
    clearTransitions: () => unsupported("no transitions library in v5 yet"),

    // --- prompt shortcuts (a/b) ---------------------------------------------
    shortcuts: () => j("GET", API("/api/shortcuts")), // {shortcuts:[...]} -- callers unwrap .shortcuts themselves
    suggestionStats: () => j("GET", API("/api/shortcuts/suggestion_stats")),
    saveCategory: (item) => j("POST", API("/api/shortcuts/category"), item),
    saveShortcut: (item) => j("POST", API("/api/shortcuts"), item),
    deleteShortcut: (name) => j("DELETE", API(`/api/shortcuts/${encodeURIComponent(name)}`)),
    exportShortcutsUrl: () => API("/api/shortcuts/export"),
    importShortcuts: (data, mode) => j("POST", API("/api/shortcuts/import"), { data, mode }),
    clearShortcuts: () => j("POST", API("/api/shortcuts/clear"), {}),
    revolverSettings: () => j("GET", API("/api/shortcuts/revolver")),
    setRevolverSettings: (payload) => j("POST", API("/api/shortcuts/revolver"), payload),

    // --- Composer file manager / NLE library (c) ----------------------------
    listFiles: () => unsupported("the Composer file manager was retired with the composer shell"),
    deleteFile: () => unsupported("the Composer file manager was retired with the composer shell"),
    clearFiles: () => unsupported("the Composer file manager was retired with the composer shell"),
    nleLibrary: () => j("GET", API("/api/m/render/library")),

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
    renameMedia: (id, name) => j("PATCH", API(`/api/media/${encodeURIComponent(id)}`), { name }),
    // A small server-cached JPEG (made on first request) instead of the full original: for grid
    // thumbnails only, never for playback, preview or export.
    mediaThumbUrl: (id) => API(`/api/media/${encodeURIComponent(id)}/thumb`),
    importClipToMediaBin: (clip, name) => j("POST", API("/api/m/render/import-clip"), { clip, name: name || null }),
    // XMLHttpRequest, not fetch: fetch has no cross-browser upload-progress event, and a multi-MB
    // video over a slow connection is exactly where the caller most wants something moving.
    // onProgress(loadedBytes, totalBytes) fires repeatedly; totalBytes is 0 if the browser cannot say.
    uploadMedia(file, onProgress) {
      return new Promise((resolve, reject) => {
        const fd = new FormData(); fd.append("file", file, file.name);
        const xhr = new XMLHttpRequest();
        xhr.open("POST", API("/api/media"));
        if (onProgress) xhr.upload.onprogress = (e) => onProgress(e.loaded, e.lengthComputable ? e.total : 0);
        xhr.onload = () => {
          let body = {};
          try { body = JSON.parse(xhr.responseText || "{}"); } catch (_) { /* not JSON */ }
          if (xhr.status >= 200 && xhr.status < 300) resolve(body.media && body.media[0] ? body.media[0] : body);
          else reject(new Error((body.problems && body.problems.join("; ")) || body.detail || xhr.statusText || `HTTP ${xhr.status}`));
        };
        xhr.onerror = () => reject(new Error("Network error during upload"));
        xhr.send(fd);
      });
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
    // Several at once -- {nodes: {className: spec|null}} -- for a pipeline
    // view describing every slot's node in one request instead of one per row.
    describeNodes: (classes) => j("GET", API(`/api/nodes?classes=${encodeURIComponent((classes || []).join(","))}`)),
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
    // The settings card is a PNG, not JSON -- fetched as a blob so the modal
    // can show it, download it and put it on the clipboard from the one
    // response. `slots` is the CALLER's live pipeline (v5's pipeline has no
    // server-side copy to fall back on, see /api/pipeline's own docs) --
    // window.PipelineState.slots() is what every other pipeline-editing
    // caller already sends the same way.
    settingsCard: async (slots, projectName, theme) => {
      const res = await fetch(API("/api/settings-card"), {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slots: slots || [], project_name: projectName || null, theme: theme || "dark" }),
      });
      if (!res.ok) {
        let payload = null;
        try { payload = await res.json(); } catch (_) {}
        throw new Error(readApiError(res, payload));
      }
      return res.blob();
    },
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
    upscaleModels: () => j("GET", API("/api/m/render/upscale_models")),
    // The upscale is an ordinary ComfyUI job: queued on /prompt, read back from /history.
    queueUpscale: async (media, model) => {
      const graph = { 1: { class_type: "FunPackUpscaleVideo", inputs: {
        filename: media.filename, subfolder: media.subfolder || "",
        type: media.type === "temp" ? "temp" : "output", upscale_model: model } } };
      const res = await fetch("/prompt", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: graph }) });
      const body = await res.json().catch(() => ({}));
      if (!res.ok || !body.prompt_id) {
        const e = body.error || {};
        const nodes = Object.values(body.node_errors || {}).flatMap((n) => (n.errors || []).map((x) => x.message));
        throw new Error([e.message, ...nodes].filter(Boolean).join("; ") || `ComfyUI refused the job (HTTP ${res.status})`);
      }
      return body.prompt_id;
    },
    // null while it is still queued or running, else {error} or {videos}.
    upscaleResult: async (promptId) => {
      const res = await fetch(`/history/${encodeURIComponent(promptId)}`);
      const entry = res.ok ? (await res.json())[promptId] : null;
      if (!entry) return null;
      const st = entry.status || {};
      if (st.status_str === "error") {
        const msg = (st.messages || []).find((m) => m[0] === "execution_error");
        return { error: (msg && msg[1] && msg[1].exception_message) || "failed inside ComfyUI" };
      }
      if (!st.completed) return null;
      const videos = Object.values(entry.outputs || {}).flatMap((o) => o.videos || []);
      return videos.length ? { videos } : { error: "no video came out, check the ComfyUI terminal" };
    },
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
    phraseProbeStatus: () => j("GET", API("/api/m/phrase_probe/status")),
    phraseProbeSetEnabled: (enabled) => j("POST", API("/api/m/phrase_probe/enabled"), { enabled: !!enabled }),
    phraseProbeClear: () => j("POST", API("/api/m/phrase_probe/clear"), {}),
    phraseProbeExportUrl: () => API("/api/m/phrase_probe/export"),
    blockInfluenceStatus: (key) => j("GET", API("/api/m/block_influence/status") + `?key=${encodeURIComponent(key || "default")}`),
    blockInfluenceSetEnabled: (key, enabled) => j("POST", API("/api/m/block_influence/enabled"), { key: key || "default", enabled: !!enabled }),
    blockInfluenceClear: (key) => j("POST", API("/api/m/block_influence/clear"), { key: key || "default" }),
    async blockInfluenceExport(key) {
      const k = key || "default";
      const res = await fetch(API("/api/m/block_influence/export") + `?key=${encodeURIComponent(k)}&t=${Date.now()}`, { cache: "no-store" });
      if (!res.ok) {
        let payload = null;
        try { payload = await res.json(); } catch (_) {}
        throw new Error(readApiError(res, payload));
      }
      const url = URL.createObjectURL(await res.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = `${k}.block_influence.pt`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    },
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

    // --- render / stitch / export: modules/output/render --------------------
    renderFinal: (id, clips) => j("POST", API(`/api/m/render/projects/${id}/render`), { clips }),
    renderFinalStatus: (id, jobId) => j("GET", API(`/api/m/render/projects/${id}/render/${encodeURIComponent(jobId)}`)),
    exportClip: (id, clip) => j("POST", API(`/api/m/render/projects/${id}/export-clip`), { clip }),
    exportClipsCombined: (id, clips) => j("POST", API(`/api/m/render/projects/${id}/export-clips`), { clips }),
    exportClipsStatus: (id, jobId) => j("GET", API(`/api/m/render/projects/${id}/export-clips/${encodeURIComponent(jobId)}`)),
    // Served by FunPack, not ComfyUI's /view: a render whose index is at the END of the file
    // (what ComfyUI's own saver writes) cannot be seeked in a browser until it is remuxed.
    resultUrl: (_id, m) => API("/api/m/render/result?" + new URLSearchParams({
      filename: m.filename, subfolder: m.subfolder || "", type: m.type || "output",
    }).toString()),
    previewSegmentUrl: (id, sceneId, spec) => {
      let u = API(`/api/m/render/projects/${id}/preview-segment/${encodeURIComponent(sceneId)}`);
      const m = spec?.media;
      if (m?.filename) {
        const q = new URLSearchParams({
          filename: m.filename, subfolder: m.subfolder || "", type: m.type || "output",
          render_in: String(spec.renderIn != null ? spec.renderIn : 0),
        });
        // dur is in the URL for two reasons: a removed scene (a ghost) has no scene server-side, so
        // its trim window must travel in the query; and segments are cached for an hour, so a
        // timeline trim must produce a different URL. Anything else that changes the bytes
        // (reverse) must be in the URL too, for the same reason.
        if (spec.dur != null) q.set("dur", String(spec.dur));
        if (spec.reverse) q.set("rev", "1");
        u += "?" + q.toString();
      }
      return u;
    },

    // --- taste keys: modules/system/taste --------------------------------------
    rateTaste: (prompt_id, rating, axis) => j("POST", API("/api/m/taste/rate"), { prompt_id, rating, axis: axis || null }),
    tasteKeys: () => j("GET", API("/api/m/taste/keys")),

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
