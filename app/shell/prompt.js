// The inputs a pipeline puts on the app's own surface.
//
// The prompt is a slot input like any other -- `positive.text` on a
// CLIPTextEncode -- and until this existed the only way to type one was to open
// the pipeline window, find the node and edit it there. That is a graph editor,
// not an app. The same mechanism now carries the size a project generates at
// and the sampler dials, because they are slot inputs too.
//
// Nothing here knows what a prompt IS. A slot says "this input of mine belongs
// at generation.prompt, call it Prompt", the shell either offers a region with
// that name or does not, and a role naming a place nobody offers is simply not
// shown -- the same rule a module's panel lives by.
//
// The CONTROL comes from the node's own description, through the same
// settingFor/rendererFor pair the pipeline window uses. That is what makes a
// combo a dropdown of the node's real choices and a number obey the node's real
// bounds, without this file knowing that samplers or schedulers exist.

import { composer } from "../composer/composer.js";
import { rendererFor, rendererNameFor, SELF_LABELLING } from "../composer/panel.js";
import { settingFor } from "./widgets.js";
import { hostFor } from "./mounts.js";

const isLink = (value) => Array.isArray(value);

/**
 * createPrompts(slots, { describe }) -> { overrides, sync, fields, at, destroy }
 *
 * `overrides` is what the app is holding, addressed by slot and input, ready to
 * be sent with a run. It is not a copy of the pipeline: the pipeline window
 * owns the structure, and these own one value each inside it.
 */
export async function createPrompts(slots = [], { describe, onChange } = {}) {
  let fields = [];

  const wanted = (from) => (from || []).flatMap((slot) =>
    (slot.roles || []).map((role) => ({ slot, role })).filter(({ role }) => hostFor(role.at)));

  async function build(from) {
    // The FIELD, not the control inside it: destroying the control leaves the
    // label and the box it sat in behind, which is a labelled empty space.
    for (const field of fields) field.field.destroy();
    fields = [];

    const asked = wanted(from);
    if (!asked.length) return;
    const described = describe
      ? await describe([...new Set(asked.map(({ slot }) => slot.node))])
      : {};

    for (const { slot, role } of asked) {
      const current = (slot.inputs || {})[role.input];
      // A wired input is fed by another node, and a box over one would offer to
      // write a value the server refuses. Nothing is said: an input that cannot
      // be typed into has no control, which is the same answer the pipeline
      // window gives for a socket.
      if (isLink(current)) continue;

      const node = described[slot.node];
      const widget = ((node && node.widgets) || []).find((w) => w.name === role.input);
      // No description, or nothing sane to draw for it: absent, not a guess.
      const setting = widget && settingFor(widget);
      if (!setting) continue;

      // What this input is when nothing has an opinion on it. The PIPELINE's
      // own value, not the raw node's widget-schema default -- a module can
      // and does bake its own sensible number in (this app's latent slot
      // declares 512, the underlying node's own schema default is 768) and
      // the pipeline's is the one Generate would actually use.
      const atRest = current === undefined ? setting.default : current;
      const entry = { slot: slot.id, input: role.input, at: role.at,
                      value: atRest, default: atRest };
      const told = (next) => {
        entry.value = next;
        if (onChange) onChange(entry);
      };
      entry.control = rendererFor(setting)({ ...setting, label: role.label || setting.label },
                                           entry.value, told);
      // A checkbox row draws its own label and hint; wrapping it in a field
      // would print both of them twice, one above the other.
      entry.field = SELF_LABELLING.has(rendererNameFor(setting))
        ? entry.control
        : composer.field.default({ label: role.label || setting.label,
                                   hint: setting.hint, control: entry.control });
      hostFor(role.at).appendChild(entry.field.node);
      fields.push(entry);
    }
  }

  /** What a caller holds: the value, and a way to set it that both halves see. */
  const handleFor = (entry) => ({
    get value() { return entry.value; },
    // What this control shows when nothing has an opinion -- the pipeline's own
    // declared default, not whatever a PREVIOUS project last set it to. Needed
    // by anyone syncing this control from per-project state (project.video):
    // without it, a project that never touched this input has no way to say
    // "go back to normal" and the last project's value is left standing.
    get default() { return entry.default; },
    setValue(next) {
      entry.value = next;
      if (entry.control.setValue) entry.control.setValue(next);
    },
  });

  await build(slots);

  return {
    /** {slotId: {input: value}} -- only what a person can actually have set. */
    overrides() {
      const out = {};
      for (const { slot, input, value } of fields) {
        out[slot] = out[slot] || {};
        out[slot][input] = value;
      }
      return out;
    },

    /**
     * The pipeline changed underneath: rebuild against it.
     *
     * A slot the pipeline window removed takes its control with it, and a value
     * saved in that window is what the control then shows. Keeping the old value
     * would make the two windows disagree about one input, and the last one sent
     * would win silently.
     */
    sync(next) { return build(next); },

    get fields() { return fields.map((f) => ({ slot: f.slot, input: f.input, at: f.at })); },

    /**
     * The control at a place, or null. What makes a scene's text editable: the
     * timeline owns the text and this is where it is typed, so the two are bound
     * rather than each holding their own copy of a prompt.
     */
    at(point) {
      const found = fields.find((f) => f.at === point);
      return found ? handleFor(found) : null;
    },

    /** Every control at a place, by the input it writes. */
    controlsAt(point) {
      return fields.filter((f) => f.at === point)
        .map((f) => ({ input: f.input, control: handleFor(f) }));
    },

    destroy() { return build([]); },
  };
}
