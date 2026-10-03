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
  media: () => call("GET", "/api/media"),
  deleteMedia: (id) => call("DELETE", `/api/media/${encodeURIComponent(id)}`),
  /** Upload files as media; resolves {media, problems}. Not JSON, so not through call(). */
  async uploadMedia(files) {
    const form = new FormData();
    files.forEach((f) => form.append("file", f, f.name));
    const res = await fetch(`${BASE}/api/media`, { method: "POST", body: form });
    const payload = await res.json().catch(() => null);
    if (!res.ok) throw new Error(why(res, payload));
    return payload;
  },
  probeFamily: (file) => call("GET", `/api/probe?file=${encodeURIComponent(file)}`),
};
