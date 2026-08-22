# Roadmap

Planned work, in no particular order. Items move to the CHANGELOG when they ship.

## Advisor multitool

Turn the prompt advisor into a general-purpose assistant surface rather than a
prompt-repair step: a picker wheel of operations that act on whatever the user points it at.

Intended operations:
- **Describe image** — look at the selected image, return text the user can copy or paste
  straight into any prompt field.
- **Enhance prompt** — rewrite or extend the current prompt.
- **Write** — longer-form story text for multi-scene timelines.
- **Change settings** — act on Engine Settings by request rather than by navigation.

The shape is an assistant that reaches into FunPack's own state, not a text box that returns a
string.

## Family-aware settings

Detect the model family when the loader state is saved, and hide controls the loaded model
cannot use, instead of reporting per run that they do not apply. Owning the loaders makes the
family known before generation starts.

Replaces the "this model has no cross-attention, feature X is skipped" class of log line: what
is not wired does not appear.
