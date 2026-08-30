// A node's input, as something a person can edit.
//
// The server describes an input the way ComfyUI declares it -- a type name, a
// list of choices, bounds, a default. The panel renders a SETTING, which is the
// shape a module declares. They are the same idea said twice, so this is the one
// place that translates between them, and every control in the app comes from
// the same renderer table either way. A second table for node inputs would be a
// second answer to "how is a number edited", and the two would drift.
//
// Nothing here names a node.

const HUMAN = {
  cfg: "CFG",
  fp16: "FP16",
  fp32: "FP32",
  bf16: "BF16",
  fps: "FPS",
  vae: "VAE",
  clip: "CLIP",
  lora: "LoRA",
  ckpt: "Checkpoint",
  id: "ID",
};

/**
 * `sampler_name` -> "Sampler name". An identifier is what a node author writes
 * for code; a label is what someone reads while choosing a value, and printing
 * the identifier is how a form ends up looking like a debug dump.
 */
export function label(name) {
  const words = String(name).split(/[_\s]+/).filter(Boolean);
  if (!words.length) return String(name);
  return words
    .map((word, i) => {
      const known = HUMAN[word.toLowerCase()];
      if (known) return known;
      if (i > 0) return word.toLowerCase();
      return word.charAt(0).toUpperCase() + word.slice(1);
    })
    .join(" ");
}

/**
 * settingFor(widget) -> a declaration `panel.js` can render, or null.
 *
 * Null when there is nothing sane to draw -- a combo whose list is empty, a
 * type this app has no control for. Not a disabled box and not a guess: an
 * input the app cannot edit has to look different from one it can, or the value
 * behind it is a mystery nobody notices.
 */
export function settingFor(widget) {
  if (!widget || typeof widget !== "object") return null;
  const common = {
    label: label(widget.name),
    hint: widget.tooltip || undefined,
  };

  switch (widget.type) {
    case "COMBO": {
      // Several choices at once. There is no control here that holds a list, and
      // a single-select saves a string where the node wants an array.
      if (widget.multiselect) return null;
      const choices = Array.isArray(widget.choices) ? widget.choices : [];
      // An empty list is a real state -- a file picker on a machine with no
      // models -- and a select holding nothing offers no way to tell that from
      // a control that failed to load.
      if (!choices.length) return null;
      const value = choices.includes(widget.default) ? widget.default : choices[0];
      return {
        ...common,
        type: "enum",
        default: value,
        options: choices.map((c) => ({ value: c, label: String(c) })),
        // Each choice on a dynamic combo can bring inputs of its own, and this
        // window does not draw those yet. Said rather than dropped: offering an
        // incomplete node as a complete one is how a run fails at the point it
        // is queued, over a field nobody was shown.
        hint: widget.reveals_more
          ? [common.hint, "This choice brings further settings that are not shown here yet."]
              .filter(Boolean).join(" ")
          : common.hint,
      };
    }
    case "BOOLEAN":
      return { ...common, type: "bool", default: Boolean(widget.default) };
    case "INT":
      return {
        ...common, type: "int", default: numberOr(widget.default, 0),
        min: widget.min, max: widget.max, step: widget.step ?? 1,
      };
    case "FLOAT":
      return {
        ...common, type: "float", default: numberOr(widget.default, 0),
        min: widget.min, max: widget.max, step: widget.step ?? 0.01,
        precision: widget.precision,
      };
    case "STRING":
      return {
        ...common,
        type: widget.multiline ? "multiline" : "text",
        default: widget.default == null ? "" : String(widget.default),
        placeholder: widget.placeholder,
      };
    default:
      return null;
  }
}

// A node may declare a default of the wrong type -- ComfyUI does not check --
// and a number control handed a string reports NaN the moment it is read.
function numberOr(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/** Why an input has no control, in words that fit under the input's name. */
export function whyNotEditable(widget) {
  if (widget.type === "COMBO") {
    if (widget.multiselect) return "several choices at once, which has no control here yet";
    return "nothing to choose from — no files of this kind were found";
  }
  return `${widget.type} is filled by a wire, not typed`;
}
