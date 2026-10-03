// One upscale job from queue to result, and the swap of the new file under every render that plays the old one. Pure of the page.
export const keyOf = (m) => `${m.type || "output"}/${m.subfolder || ""}/${m.filename}`;
export const UPSCALED = "funpack_upscaled";

/** -> the upscaled media, or throws a sentence. `sleep` is injectable so a test need not wait. */
export async function upscale(api, media, model, sleep = (ms) => new Promise((r) => setTimeout(r, ms))) {
  const id = await api.queueUpscale(media, model);
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
  p.scene_ghosts = (p.scene_ghosts || []).map((g) => (same(g.media) ? (n += 1, { ...g, media: to }) : g));
  return n;
}
