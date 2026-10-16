// Separate audio / Remove audio: pull the picked clip's own sound onto a lane of its own, or give it back.
import { composer as c } from "../../composer/composer.js";
import { inPlace } from "../../shell/place.js";
import { hasEmbeddedAudio, removeTrack, separate, trackFor } from "../../shell/audio.js";

export default {
  id: "audio",
  mount: "timeline.toolbar",
  needs: ["project", "selection"],
  setup({ host, app }) {
    const p = app.project, sel = app.selection;
    const put = inPlace(host);
    const draw = () => {
      const open = p.project, sc = open && sel.focus ? open.scenes.find((s) => s.id === sel.focus) : null;
      const lane = open && sc ? trackFor(open, sc.id) : null;
      put(c.button.sm({ label: "⊟ Separate audio", tone: "ghost", disabled: !(open && sc && hasEmbeddedAudio(sc, open)), title: "Detach this clip's audio onto its own track (the clip keeps picture only)",
          onClick: () => p.edit((pr) => separate(pr, sel.focus)) }).node,
        c.button.sm({ label: "Remove audio", tone: "danger", disabled: !lane, title: "Remove this clip's separated audio track",
          onClick: () => p.edit((pr) => removeTrack(pr, lane.id)) }).node);
    };
    draw();
    return app.on(draw);
  },
};
