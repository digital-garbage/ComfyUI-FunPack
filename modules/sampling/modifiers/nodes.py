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
  clone cannot touch the model everyone else is holding.
* A module never gets the real patcher. It gets one that guards every hook it
  installs, so a modifier that raises mid-sampling is dropped for the rest of
  that run instead of ending it. Catching a failure to INSTALL was never the
  hard part: the one that costs a rental is the hook that works at step 0 and
  raises at step 3, after the GPU time is spent.

That second point protects the ORIGINAL, and on its own it does not stop this
node's own output being fed back in. `add_wrapper_with_key` APPENDS to a list
under the key, and a clone carries that list, so running a model through here
twice -- two passes, a flattened subgraph, anything that chains -- installed the
same wrapper twice and ran it twice per step at double strength, reporting
"1 modifier applied" both times. So every install is preceded by stripping our
own keys: tag, then strip, which is the discipline the leaked-hook fault taught
this project the first time. Foreign keys are never touched.
"""

import json

from comfy_api.latest import io

from ..._core import (log, patching, registry as registry_mod,
                      relations as relations_mod, schema as schema_mod)

CAPABILITY = "modifier"

# Anything answering this decides whether the foundation's guards step aside.
GUARDS_OFF = "guards_off"

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
        order = relations_mod.order

        if settings is not None and not isinstance(settings, dict):
            raise RuntimeError(
                f"FunPack Load Modifiers: settings must be an object keyed by module "
                f"id, got {type(settings).__name__}. Wire FunPack Modifier Settings "
                f"into this input.")

        registry = registry_mod.current()
        specs = list(registry.specs.values())

        # What this model actually is, read off the model itself.
        available = traits_mod.traits_of(model, specs)

        offering = [spec for spec in specs if spec.provides.get(CAPABILITY)]
        compatible, incompatible = traits_mod.split(offering, available)
        ordered, rejected = order(compatible)

        patched = model.clone()
        stripped = patching.strip(patched, KEY_PREFIX)
        applied, notes = [], []

        # One per run, so a long session reports every generation that went
        # inert rather than only the first one after a restart.
        dropped = patching.Dropped()
        patched.funpack_dropped = dropped

        # Asked for by capability, so nothing here names the module that answers
        # and a build without it installed has nobody saying yes -- which reads
        # as guarded, the safe direction.
        full_control = any(answer(settings)
                           for _spec, answer in registry.providers(GUARDS_OFF))
        if full_control:
            notes.append("full control is ON: modifiers are not guarded, and a "
                         "failing one will end the run. Any consequences of "
                         "these settings are yours.")
        if stripped:
            notes.append(f"replaced {stripped} modifier(s) already on this model")

        # Checked HERE, not only where the payload was made. A FUNPACK_SETTINGS
        # socket is typed by a string tag, so a dict can reach this node without
        # ever passing through FunPackModifierSettings -- a raw API prompt, or a
        # second producer of that type. Validating only at the producer is a
        # guard on one of two doors, and the one left open let NaN through:
        # `sigma > nan` is False at every step, so a modifier became a permanent
        # no-op while reporting that it had been applied.
        problems = []
        chosen = {}
        for spec in ordered:
            values, said = schema_mod.check_values(spec, (settings or {}).get(spec.id))
            chosen[spec.id] = values
            problems.extend(said)
        if problems:
            raise RuntimeError("FunPack Load Modifiers:\n  " + "\n  ".join(problems))

        for spec in ordered:
            values = chosen[spec.id]
            install = spec.provides[CAPABILITY]
            key = KEY_PREFIX + spec.id
            guarded = patching.GuardedPatcher(patched, key, dropped, guarding=not full_control)
            try:
                note = install(guarded, values, key=key)
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
            for name in guarded.unguarded:
                # Named rather than assumed safe: this hook shape has no known
                # neutral result, so a failure inside it can still end the run.
                notes.append(f"{spec.id}: {name} could not be guarded")

        for spec in incompatible:
            notes.append(f"{spec.id}: needs {', '.join(traits_mod.missing_for(spec, available))}")
        for spec, why in rejected:
            notes.append(f"{spec.id}: {why}")

        headline = f"{len(applied)} modifier(s) applied" + (f": {', '.join(applied)}" if applied else "")
        return io.NodeOutput(patched, "\n".join([headline, f"model traits: {', '.join(available) or 'none read'}", *notes]))
