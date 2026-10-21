"""Every modifier, every setting, on a real tiny model: nothing is shipped that was never switched on.

A module's own tests exercise the settings its author thought of. This one switches each module on and walks every
setting through each of its choices (every enum option, a bool both ways, a number at its smallest, largest and
default) on the real tiny H3 and LTX models, then runs a forward pass. A choice that was offered in Settings and
raises, installs nothing, or makes the output non-finite is a path the app promised and does not keep.

Weights are random and the model is four blocks wide, so this says nothing about quality: only that the path runs.
"""

import math

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def _variants(spec):
    """(label, values) for the module on, then one setting changed at a time to each of its other choices."""
    base = {"enabled": True} if "enabled" in spec.settings else {}
    yield "default", base
    for key, s in spec.settings.items():
        if key == "enabled":
            continue
        kind = s["type"]
        if kind == "bool":
            choices = [not s["default"]]
        elif kind == "enum":
            choices = [o["value"] for o in s["options"] if o["value"] != s["default"]]
        elif kind in ("int", "float"):
            lo, hi = s.get("min"), s.get("max")
            choices = [v for v in (lo, hi) if v is not None and v != s["default"] and math.isfinite(v)]
            if kind == "int":
                choices = [int(v) for v in choices]
        else:
            choices = []                                    # free text, paths and colours have no edges to walk
        for value in choices:
            yield f"{key}={value!r}", {**base, key: value}


def _cases():
    reg = _registry()
    for spec in sorted(reg.specs.values(), key=lambda s: s.id):
        if spec.provides.get("modifier") and spec.settings:
            for label, values in _variants(spec):
                yield pytest.param(spec.id, values, id=f"{spec.id}:{label}")


def _registry():
    from core import registry
    return registry.current()


def _apply(model, module_id, values):
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    settings = {"taste": {"key": "sweep"}, module_id: values}
    return FunPackLoadModifiers.execute(model.patcher, settings).result


def _finite(out):
    parts = out if isinstance(out, (list, tuple)) else [out]
    return all(torch.isfinite(p).all() for p in parts if torch.is_tensor(p))


#: Failures that come from the stand-in model, not from the module: a four-block model has no blocks 31-40, and the
#: LTX stand-in is not a model_base class with the windowing hook (the module's own tests patch that in).
TINY_MODEL = ("no usable blocks", "map_context_window_to_modalities")


def _install_failures(status):
    return [line for line in status.splitlines()
            if ("failed to install" in line or "could not be guarded" in line) and not any(t in line for t in TINY_MODEL)]


def _stand_in_limit(status):
    return any("failed to install" in line and any(t in line for t in TINY_MODEL) for line in status.splitlines())


@pytest.mark.parametrize("module_id,values", list(_cases()))
def test_h3_every_choice_installs_and_runs(tiny_h3, module_id, values):
    spec = _registry().specs[module_id]
    patched, status = _apply(tiny_h3, module_id, values)
    assert not _install_failures(status), status
    if _stand_in_limit(status):
        pytest.skip("the four-block stand-in has no such blocks")
    if spec.requires and not set(spec.requires) <= _traits(patched, spec):
        pytest.skip(f"{module_id} is not offered for H3")
    assert _finite(tiny_h3.sample(patched)), f"{module_id} {values}: non-finite output"


@pytest.mark.parametrize("module_id,values", list(_cases()))
def test_ltx_every_choice_installs_and_runs(tiny_ltx, module_id, values):
    spec = _registry().specs[module_id]
    patched, status = _apply(tiny_ltx, module_id, values)
    assert not _install_failures(status), status
    if _stand_in_limit(status):
        pytest.skip("the stand-in model lacks what this module asks of a real one")
    if spec.requires and not set(spec.requires) <= _traits(patched, spec):
        pytest.skip(f"{module_id} is not offered for LTX")
    assert _finite(tiny_ltx.run(patched)), f"{module_id} {values}: non-finite output"


def _traits(patched, spec):
    from core import traits as traits_mod
    return set(traits_mod.traits_of(patched, list(_registry().specs.values())))
