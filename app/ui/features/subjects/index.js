// Subject text: say once what a picture MEANS as a reference ("<Subject 1> is the woman in <Picture 1>…"); every scene that uses it
// as a reference then starts its prompt with that line, before the anchor. The text is derived from the reference list at run time,
// never written into the prompt box, and leaves with the reference.
import { composer as c } from "../../composer/composer.js";
import { subjects } from "../../shell/bin.js";

export default {
  id: "subjects",
  mount: "assets.actions",
  needs: ["api", "mediaMenu", "promptPrefix"],
  setup({ app }) {
    const menu = (it, item) => (item && item.kind !== "audio" ? [{ id: "subject", label: subjects.has(it.id) ? "Edit subject text…" : "Subject text…", run: async () => {
      const text = await c.modal.prompt({ title: "What is this picture, as a reference?", label: "Subject text", value: subjects.get(it.id) || "", placeholder: "<Subject 1> is the woman in <Picture 1>: her face, hair and build only.", confirmLabel: "Save" }).result;
      if (text === undefined || text === null) return;
      try { await app.api.setMediaSubject(it.id, text); if (text.trim()) subjects.set(it.id, text.trim()); else subjects.delete(it.id); app.say("bin"); } catch (err) { c.toast.warn({ text: `Could not save: ${err.message}` }); }
    } }] : []);
    // Lines in the order the references were picked; a picture with no text adds nothing.
    const prefix = (scene) => (scene.references || []).map((id) => subjects.get(id)).filter(Boolean);
    app.mediaMenu.push(menu);
    app.promptPrefix.push(prefix);
    return () => { for (const [list, x] of [[app.mediaMenu, menu], [app.promptPrefix, prefix]]) { const i = list.indexOf(x); if (i >= 0) list.splice(i, 1); } };
  },
};
