"""Every place that reports "it loaded and then broke", checked where it is said.

The wording is not decoration: "did not load" sends a reader to an import that
never happened, and this project wrote it in six places before noticing. Testing
`log.broke` on its own does not stop that -- what recurs is a call SITE reaching
for the wrong function, and a unit test of the right one passes either way. So
each site is exercised through the code that logs, and asserts on the record.

Three of the six had no such test when this file was written; reverting them to
`log.failed` left the whole suite green.
"""

import pytest

from core import log
from core.contract import ModuleSpec


def _said():
    return [r["message"] for r in log.history()]


def test_a_sampler_modifier_that_breaks_while_starting_up(monkeypatch):
    from core import chain, patching

    def explode(_values, _hooks):
        raise RuntimeError("the schedule is upside down")

    spec = ModuleSpec(id="breaks", title="B", mount="",
                      provides={"sampler_modifier": explode})

    log._reset()
    built, notes = chain.build([spec], {}, ("latent",), patching.Dropped())

    assert any("failed while starting up" in m for m in _said()), _said()
    assert not any("did not load" in m for m in _said()), _said()
    assert any("breaks" in note for note in notes), notes
    assert built.ids == [], "a modifier that could not start is running anyway"


def test_a_node_that_breaks_while_describing_itself():
    from core import nodes as nodes_mod

    class Explodes:
        @classmethod
        def GET_SCHEMA(cls):
            raise RuntimeError("the schema is upside down")

    spec = ModuleSpec(id="breaks", title="B", mount="", nodes=[Explodes])

    class Registry:
        specs = {"breaks": spec}

    log._reset()
    collected, rejected = nodes_mod.collect(Registry())

    assert collected == [], collected
    assert rejected, "a node that could not describe itself was accepted"
    assert any("failed while describing itself" in m for m in _said()), _said()
    assert not any("did not load" in m for m in _said()), _said()


def test_a_model_module_that_breaks_while_decoding(monkeypatch, comfyui):
    import torch
    from core import registry as registry_mod
    from modules.output.decode import nodes as decode_mod

    class Spec:
        id = "pretend_model"

    def explode(_latent, **_kw):
        raise RuntimeError("the audio branch is upside down")

    class Registry:
        specs = {}

        def providers(self, _capability):
            return [(Spec(), explode)]

    monkeypatch.setattr(registry_mod, "current", Registry)
    log._reset()

    class Vae:
        def decode(self, latent):
            return torch.zeros(1, 8, 8, 3)

    with pytest.raises(RuntimeError, match="upside down"):
        decode_mod.FunPackDecode.execute({"samples": torch.zeros(1, 4, 8, 8)}, Vae())

    assert any("failed while decoding" in m for m in _said()), _said()
    assert not any("did not load" in m for m in _said()), _said()
