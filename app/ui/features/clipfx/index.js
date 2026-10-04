// Clip effects and seam transitions: "＋ Effect" on the timeline toolbar, the tags on a clip saying what is on it, and the
// original-sound volume under the scene's fields. The render reads scene.effects / video_transition / audio_volume.
import { composer as c } from "../../composer/composer.js";
import { applyEffect, applyTransition, clearTransition, tags } from "../../shell/clipfx.js";

export default {
  id: "clipfx",
  mount: "timeline.toolbar",
  needs: ["project", "api", "selection"],
  setup({ host, app }) {
    const p = app.project;
    let library = null;
    const load = async () => (library ||= await app.api.renderLibrary());

    async function ask(kind) {       // kind: "effects" | "video_transitions"
      const sc = p.selected;
      if (!sc) return c.toast.warn({ text: "Pick a clip first." });
      let items;
      try { items = (await load())[kind] || []; } catch (err) { return c.toast.warn({ text: `The effects list could not be loaded: ${err.message}` }); }
      if (!items.length) return c.toast.warn({ text: "Nothing to offer: the list is empty." });
      const isFx = kind === "effects";
      let item = items[0], value = item.param ? item.param.default : null;
      const slot = c.region.stack({ gap: "sm" });
      const draw = () => slot.set([
        c.field.default({ label: isFx ? "Effect" : "Transition", hint: item.description, control: c.select.md({ label: isFx ? "Effect" : "Transition", value: item.id, options: items.map((i) => ({ value: i.id, label: i.name || i.id })),
          onChange: (v) => { item = items.find((i) => i.id === v); value = item.param ? item.param.default : null; draw(); } }) }),
        item.param ? c.field.default({ label: item.param.label, control: c.number.md({ label: item.param.label, value, min: item.param.min, max: item.param.max, step: item.param.step, onChange: (v) => { value = v; } }) }) : null,
      ].filter(Boolean));
      draw();
      const win = c.modal.generic({ title: isFx ? "Add effect" : "Add transition", subtitle: isFx ? "On the picked clip." : "A blend on the picked clip's outgoing edge.", size: "sm", body: slot });
      win.setFooter({ actions: [c.button.sm({ label: "Cancel", tone: "ghost", onClick: () => win.close("cancel") }),
        c.button.sm({ label: "Apply", tone: "primary", onClick: () => {
          const ok = p.edit((pr) => { const s = pr.scenes.find((x) => x.id === sc.id); return s && (isFx ? applyEffect(s, item.id, value) : applyTransition(s, item, value)); });
          if (!ok) return c.toast.warn({ text: "Could not apply that." });
          win.close("done");
        } })] });
    }

    const menu = c.button.menu({ label: "＋ Effect", tone: "ghost", onClick: () => c.menu.dropdown({ anchor: menu, items: [
      { id: "fx", label: "Effect…", disabled: !p.selected }, { id: "tr", label: "Transition…", disabled: !p.selected },
      { separator: true }, { id: "off", label: "Remove effects and transition", danger: true, disabled: !p.selected || !tags(p.selected).length }],
      onPick: (id) => {
        if (id === "fx") ask("effects"); else if (id === "tr") ask("video_transitions");
        else p.edit((pr) => { const s = pr.scenes.find((x) => x.id === p.selectedId); if (!s) return false; applyEffect(s, "reset"); return clearTransition(s); });
      } }) });
    host.append(menu.node);

    app.clipTags.push(tags);
    const section = { key: (sc) => `${sc.audio_volume}|${sc.audio_separated}|${sc.gap_after_sec}`, rows: (sc) => [
      c.field.default({ label: "Pause after this clip (s)", hint: "Black and silence before the next clip.", control: c.number.md({ label: "Pause after this clip", value: sc.gap_after_sec || 0, min: 0, max: 60, step: 0.1, onChange: (v) => p.setScene(sc.id, "gap_after_sec", v) }) }),
      c.label.section({ text: "Sound" }),
      c.field.default({ label: "Original sound volume", hint: sc.audio_separated ? "This clip's sound is on its own audio track: set the volume there." : "How loud this clip's own sound is in the render.",
        control: c.slider.readout({ label: "Original sound volume", value: sc.audio_volume != null ? sc.audio_volume : 1, min: 0, max: 1, step: 0.05, precision: 2, disabled: Boolean(sc.audio_separated), onCommit: (v) => p.setScene(sc.id, "audio_volume", v) }) })] };
    app.sceneSections.push(section);
    return () => {
      for (const [list, x] of [[app.clipTags, tags], [app.sceneSections, section]]) { const i = list.indexOf(x); if (i >= 0) list.splice(i, 1); }
      menu.node.remove();
    };
  },
};
