// Cuts: the words that cut a story into scenes. The first is the one written between scenes.
import { composer as c } from "../../composer/composer.js";

export function cuts(app) {
  const page = c.region.stack({ gap: "sm" });
  let words = "";
  const save = async () => {
    try {
      const r = await app.api.saveStoryMarkers(words.split("\n").map((w) => w.trim()).filter(Boolean));
      words = r.markers.join("\n");
      c.toast.good({ text: "Cut words saved." });
      draw();
    } catch (err) { c.toast.warn({ text: err.message }); }
  };
  const draw = () => page.set([
    c.label.section({ text: "Cut words" }),
    c.textarea.md({ label: "Cut words", rows: 5, value: words, onInput: (v) => { words = v; } }),
    c.hint.default({ text: "One per line. A whole word typed in the Story starts a new scene and is never part of a scene's text. The first is the one written between scenes." }),
    c.button.sm({ label: "Save", tone: "primary", onClick: save }),
  ]);
  app.api.storyMarkers().then((r) => { words = (r.markers || []).join("\n"); draw(); }).catch((err) => page.set([c.banner.warn({ text: `Could not load the cut words: ${err.message}` })]));
  return page;
}
