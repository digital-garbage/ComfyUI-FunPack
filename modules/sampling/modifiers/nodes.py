"""Modifiers: what the user chose, and putting it on the model.

Two nodes, because the two jobs are genuinely different. One carries the values a
person picked; the other reads what is installed and applies whatever announced
itself. Neither names a modifier, and no sampler is involved in either.

**Modifiers attach to the MODEL, not to a sampler.** They install through
ComfyUI's own `patcher_extension` wrappers on a CLONE of the patcher, which has
two consequences worth stating:

* Wire the resulting model into ComfyUI's own SamplerCustomAdvanced and every
  per-step modifier still runs. The sampler's cooperation is not required, so no
  sampler can own them -- which is the whole architectural claim, made physical.
* `ModelPatcher.clone()` copies the wrapper lists per key, so installing on the
  clone cannot touch the model everyone else is holding. Accumulation across runs
  -- the fault that once degraded output for a hundred generations and survived
  every reset short of a restart -- is structurally impossible here rather than
  something to remember to clean up.
"""

import json

from comfy_api.latest import io

from ..._core import log, registry as registry_mod, schema as schema_mod

CAPABILITY = "modifier"

# Namespaced so removing ours never touches a wrapper somebody else installed.
KEY_PREFIX = "funpack."


class FunPackModifierSettings(io.ComfyNode):
    """The values a person picked, on their way to the graph.

    One widget, not one per setting. A socket per setting would mean this node's
    shape changing every time any module gained an option, and every saved
    workflow referring to sockets that had moved.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackModifierSettings",
            display_name="FunPack Modifier Settings",
            category="FunPack/Sampling",
            description="Settings for every FunPack modifier, as one payload.",
            inputs=[
                io.String.Input(
                    "settings", multiline=True, default="{}",
                    tooltip='{"module_id": {"setting": value}}. The app writes this; '
                            'editing it by hand is fine, and mistakes are reported.'),
            ],
            outputs=[
                io.Custom("FUNPACK_SETTINGS").Output(display_name="settings"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, settings: str) -> io.NodeOutput:
        text = (settings or "").strip() or "{}"
        try:
            raw = json.loads(text)
        except ValueError as exc:
            raise RuntimeError(f"FunPack Modifier Settings: this is not valid JSON -- {exc}")
        if not isinstance(raw, dict):
            raise RuntimeError(
                f"FunPack Modifier Settings: expected an object keyed by module id, "
                f"got {type(raw).__name__}.")

        specs = registry_mod.current().specs
        problems, checked = [], {}
        for module_id, values in raw.items():
            spec = specs.get(module_id)
            if spec is None:
                # Silently ignoring it is how v4 accumulated settings that meant
                # nothing and a list of dead keys to remember.
                problems.append(f"no module named {module_id!r} is installed.")
                continue
            clean, said = schema_mod.check_values(spec, values)
            checked[module_id] = clean
            problems.extend(said)

        if problems:
            raise RuntimeError("FunPack Modifier Settings:\n  " + "\n  ".join(problems))

        named = ", ".join(sorted(checked)) or "nothing set"
        return io.NodeOutput(checked, f"settings for {named}")


class FunPackLoadModifiers(io.ComfyNode):
    """Whatever announced itself, filtered by this model, installed in order."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackLoadModifiers",
            display_name="FunPack Load Modifiers",
            category="FunPack/Sampling",
            description="Apply the enabled modifiers to a model. Works with any sampler.",
            inputs=[
                io.Model.Input("model"),
                io.Custom("FUNPACK_SETTINGS").Input("settings", optional=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, model, settings=None) -> io.NodeOutput:
        from ..._core import traits as traits_mod
        from core.relations import order

        registry = registry_mod.current()
        specs = list(registry.specs.values())

        # What this model actually is, read off the model itself.
        available = traits_mod.traits_of(model, specs)

        offering = [spec for spec in specs if spec.provides.get(CAPABILITY)]
        compatible, incompatible = traits_mod.split(offering, available)
        ordered, rejected = order(compatible)

        patched = model.clone()
        applied, notes = [], []

        for spec in ordered:
            values = (settings or {}).get(spec.id) or spec.defaults()
            install = spec.provides[CAPABILITY]
            try:
                note = install(patched, values, key=KEY_PREFIX + spec.id)
            except Exception as exc:             # noqa: BLE001
                # Absent, and said out loud. A modifier that half-installed and
                # carried on is how a run silently stops meaning what it says.
                log.failed(f"{spec.id}.{CAPABILITY}", exc)
                notes.append(f"{spec.id}: failed to install -- {type(exc).__name__}: {exc}")
                continue
            if note is None:
                continue                          # the module decided it is off
            applied.append(spec.id)
            notes.append(f"{spec.id}: {note}")

        for spec in incompatible:
            notes.append(f"{spec.id}: needs {', '.join(traits_mod.missing_for(spec, available))}")
        for spec, why in rejected:
            notes.append(f"{spec.id}: {why}")

        headline = f"{len(applied)} modifier(s) applied" + (f": {', '.join(applied)}" if applied else "")
        return io.NodeOutput(patched, "\n".join([headline, f"model traits: {', '.join(available) or 'none read'}", *notes]))
