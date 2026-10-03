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
  storySplit: (text) => call("POST", "/api/story/split", { text }),
  shortcuts: () => call("GET", "/api/shortcuts"),
  saveShortcut: (shortcut, originalName) => call("POST", "/api/shortcuts", { ...shortcut, original_name: originalName }),
  deleteShortcut: (name) => call("DELETE", `/api/shortcuts/${encodeURIComponent(name)}`),
  revolver: () => call("GET", "/api/shortcuts/revolver"),
  setRevolver: (enabled, random) => call("POST", "/api/shortcuts/revolver", { enabled, random }),
  packs: () => call("GET", "/api/packs"),
  pack: (action, body) => call("POST", `/api/packs/${action}`, body || {}),
  git: (action, body) => call("POST", `/api/git/${action}`, body || {}),
  health: () => call("GET", "/api/health"),
  gitFull: () => call("GET", "/api/git/status"),
  gitStatus: () => call("GET", "/api/git/status?remote=0"),
  media: () => call("GET", "/api/media"),
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
