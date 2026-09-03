// The inputs a pipeline puts on the app's own surface.
//
// The prompt is a slot input like any other -- `positive.text` on a
// CLIPTextEncode -- and until this existed the only way to type one was to open
// the pipeline window, find the node and edit it there. That is a graph editor,
// not an app.
//
// Nothing here knows what a prompt IS. A slot says "this input of mine belongs
// at generation.prompt, call it Prompt", the shell either offers a region with
// that name or does not, and a role naming a place nobody offers is simply not
// shown -- the same rule a module's panel lives by.

import { composer } from "../composer/composer.js";
import { hostFor } from "./mounts.js";

const isLink = (value) => Array.isArray(value);

/**
 * createPrompts(slots) -> { overrides, sync, fields, destroy }
 *
 * `overrides` is what the app is holding, addressed by slot and input, ready to
 * be sent with a run. It is not a copy of the pipeline: the pipeline window
 * owns the structure, and these own one value each inside it.
 */
export function createPrompts(slots = [], { onChange } = {}) {
  let fields = [];

  function build(from) {
    // The FIELD, not the control inside it: destroying the control leaves the
    // label and the box it sat in behind, which is a labelled empty space.
    for (const field of fields) field.field.destroy();
    fields = [];

    for (const slot of from || []) {
      for (const role of slot.roles || []) {
        const host = hostFor(role.at);
        if (!host) continue;                       // nobody offers that place

        const current = (slot.inputs || {})[role.input];
        // A wired input is fed by another node, and a box over one would offer
        // to write a value the server refuses. Nothing is said: an input that
        // cannot be typed into has no control, which is the same answer the
        // pipeline window gives for a socket.
        if (isLink(current)) continue;
        if (current !== undefined && typeof current !== "string") continue;

        const control = composer.textarea.md({
          // No label here: the field around it draws one and points at it, and
          // an aria-label as well would win over the visible words.
          value: current || "",
          rows: 2,
          autoGrow: true,
          placeholder: role.label || role.input,
          onInput: () => { if (onChange) onChange(); },
        });
        const field = composer.field.default({
          label: role.label || role.input,
          control,
        });
        host.appendChild(field.node);
        fields.push({ slot: slot.id, input: role.input, at: role.at, control, field });
      }
    }
  }

  build(slots);

  return {
    /** {slotId: {input: value}} -- only what a person can actually have typed. */
    overrides() {
      const out = {};
      for (const { slot, input, control } of fields) {
        out[slot] = out[slot] || {};
        out[slot][input] = control.value;
      }
      return out;
    },

    /**
     * The pipeline changed underneath: rebuild against it.
     *
     * A slot the pipeline window removed takes its box with it, and a value
     * saved in that window is what the box then shows. Keeping the old value
     * would make the two windows disagree about one input, and the last one
     * sent would win silently.
     */
    sync(next) { build(next); },

    get fields() { return fields.map((f) => ({ slot: f.slot, input: f.input, at: f.at })); },

    /**
     * The control mounted at a place, or null.
     *
     * What makes a scene's text editable: the timeline owns the text and this is
     * the box it is typed in, so the two are bound rather than each holding
     * their own copy of a prompt.
     */
    at(point) {
      const found = fields.find((f) => f.at === point);
      return found ? found.control : null;
    },
    destroy() { build([]); },
  };
}
