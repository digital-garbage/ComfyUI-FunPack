// Shot camera memory: the words and views the camera has learned from your prompts and ratings, to read and forget.
import { composer as c } from "../../composer/composer.js";

export const shotMemory = (app) => function mount() {
  const page = c.region.stack({ gap: "md" });
  let data = null, error = "";

  async function refresh() {
    try { data = await app.api.shotMemory(); error = ""; } catch (err) { data = null; error = err.message; }
    draw();
  }
  async function forget(kind, name) {
    const what = kind === "all" ? "everything the camera has learned" : `“${name}”`;
    if (!(await c.modal.dialogue({ title: "Forget", message: `Forget ${what}? This cannot be undone.`, tone: "danger", confirmLabel: "Forget" }).result)) return;
    try { await app.api.forgetShotMemory(kind, name); } catch (err) { c.toast.warn({ text: err.message }); }
    refresh();
  }
  const row = (label, hint, kind, name) => c.settingsRow.default({ label, hint, control: c.button.sm({ label: "Forget", tone: "danger", onClick: () => forget(kind, name) }) });
  function draw() {
    if (error) return page.set([c.banner.warn({ text: `Could not read the camera's memory: ${error}` })]);
    if (!data) return page.set([c.hint.default({ text: "Loading…" })]);
    const words = data.words || [], views = data.views || [];
    if (!words.length && !views.length && !(data.arms || []).length) return page.set([c.emptyState.default({ icon: "◎", title: "Nothing learned yet", hint: "Turn on camera moves or shot views, then rate renders: the camera remembers which words and views you liked." })]);
    page.set([c.hint.default({ text: `${data.prompts} prompts seen · ${data.shots} shots reviewed. Forget anything that taught it the wrong thing.` }),
      ...(views.length ? [c.label.section({ text: "Views" }), ...views.map((v) => row(v.view, `liked ${v.good.toFixed(1)} · disliked ${v.bad.toFixed(1)}`, "view", v.view))] : []),
      ...(words.length ? [c.label.section({ text: "Words the camera aims at" }), ...words.map((w) => row(w.word, `picked ${w.picks} · replaced ${w.rejects} · seen ${w.seen}`, "word", w.word)),
        data.more ? c.hint.default({ text: `and ${data.more} more` }) : null] : []),
      ...((data.arms || []).length ? [c.label.section({ text: "Camera choices" }), ...data.arms.map((a) => row(a.arm.replace(":", " · "), `liked ${a.good.toFixed(1)} · disliked ${a.bad.toFixed(1)}`, "arm", a.arm))] : []),
      c.button.sm({ label: "Forget everything", tone: "danger", onClick: () => forget("all") })].filter(Boolean));
  }
  draw(); refresh();
  return { node: page.node, destroy: () => page.node.remove() };
};
