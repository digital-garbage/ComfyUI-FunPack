// One upscale job from queue to result, and the swap of the new file under every render that plays the old one. Pure of the page.
export const keyOf = (m) => `${m.type || "output"}/${m.subfolder || ""}/${m.filename}`;
export const UPSCALED = "funpack_upscaled";
export const RTX = "NVIDIA RTX Super Resolution";           // sent as the model name for that engine; the same string as modules/output/render/rtx.py NAME
export const RTX_DEFAULTS = { quality: "ULTRA", resize: "scale", scale: 2, width: 1920, height: 1080, strength: 1, tenbit: false };
const LEVEL = { LOW: "Low", MEDIUM: "Medium", HIGH: "High", ULTRA: "Ultra" };
/** "DENOISE_HIGH" -> "Denoise · High": what the SDK calls a quality level, said to a person. */
export const qualityLabel = (q) => {
  if (q === "BICUBIC") return "Bicubic (no AI)";
  const i = q.lastIndexOf("_"), level = LEVEL[q.slice(i + 1)] || q.slice(i + 1);
  if (i < 0) return `Upscale · ${LEVEL[q] || q}`;
  const kind = { DENOISE: "Denoise", DEBLUR: "Deblur", HIGHBITRATE: "High bitrate", STREAMING: "Streaming" }[q.slice(0, i)] || q.slice(0, i);
  return `${kind} · ${level}`;
};
/** Denoise and deblur clean a video at its own size: scale and target size do not apply. */
export const keepsSize = (q) => q.startsWith("DENOISE_") || q.startsWith("DEBLUR_");

/** -> the upscaled media, or throws a sentence. `sleep` is injectable so a test need not wait. */
export async function upscale(api, media, model, sleep = (ms) => new Promise((r) => setTimeout(r, ms)), rtx) {
  const id = await api.queueUpscale(media, model, rtx);
  let lost = 0;
  for (;;) {
    await sleep(2000);
    const r = await api.upscaleResult(id);
    if (!r) { lost = 0; continue; }
    if (r.retry) { if (++lost < 30) continue; throw new Error("lost contact with ComfyUI"); }
    if (r.gone) { if (++lost < 3) continue; throw new Error("the job left ComfyUI's queue without finishing (interrupted or ComfyUI restarted)"); }
    if (r.error) throw new Error(r.error);
    return r.videos[0];
  }
}

/** One file backs a whole chain run, so every scene render and ghost that plays `from` gets `to`. -> how many were swapped. */
export function swap(p, from, to) {
  const same = (m) => !!m && keyOf(m) === keyOf(from);
  let n = 0;
  for (const [id, r] of Object.entries(p.scene_renders || {})) if (r && same(r.media)) { p.scene_renders[id] = { ...r, media: to }; n += 1; }
  for (const list of Object.values(p.scene_variants || {})) list.forEach((t) => { if (same(t.media)) { t.media = to; n += 1; } });       // a take stepped back to must be the upscaled file too
  p.scene_ghosts = (p.scene_ghosts || []).map((g) => (same(g.media) ? (n += 1, { ...g, media: to }) : g));
  return n;
}
