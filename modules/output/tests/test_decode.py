"""Decoding: the ordinary path, and the one a model has to claim."""

import pytest


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


class _FakeOf:
    """An object reporting another class's identity, so no weights are needed."""

    def __init__(self, cls):
        self.__class__ = type(cls.__name__, (object,), {"__module__": cls.__module__})


def _h3_model():
    """A model tree carrying H3's own classes, so the probe fires for the reason
    it would fire on a real one."""
    import comfy.ldm.minimax.model as h3

    class Root:
        def modules(self):
            return [_FakeOf(h3.MiniMaxH3Model), _FakeOf(h3.AdalnProj)]

    class Inner:
        diffusion_model = Root()
        model_config = None

    class Model:
        model = Inner()

    return Model()


class _NotH3Model:
    """A model tree with none of H3's classes in it."""

    class model:
        class diffusion_model:
            @staticmethod
            def modules():
                return []
        model_config = None


class _Vae:
    def __init__(self, out=None):
        self.seen = []
        self._out = out

    def decode(self, latent):
        import torch
        self.seen.append(latent)
        return self._out if self._out is not None else torch.zeros(4, 8, 8, 3)


def test_a_single_tensor_latent_decodes_normally():
    import torch
    from modules.output.decode.nodes import FunPackDecode

    vae = _Vae()
    out = FunPackDecode.execute({"samples": torch.zeros(1, 4, 8, 8)}, vae)
    images, audio, status = out.result

    assert audio is None and "single latent" in status
    assert vae.seen and images.shape[-1] == 3


def test_a_batch_of_clips_becomes_one_strip_of_frames():
    """A 5-D decode is [B, T, H, W, C]; downstream wants frames."""
    import torch
    from modules.output.decode.nodes import FunPackDecode

    vae = _Vae(out=torch.zeros(2, 5, 8, 8, 3))
    images, _audio, _status = FunPackDecode.execute({"samples": torch.zeros(1, 4, 8, 8)}, vae).result
    assert images.shape == (10, 8, 8, 3)


def test_a_nested_latent_nobody_claims_is_refused_not_decoded_as_one_tensor(monkeypatch):
    """vae.decode on a NestedTensor would either raise somewhere confusing or
    quietly decode one branch as if it were the picture.

    H3 is installed and claims any nested latent, so proving this path needs its
    claim taken away -- which is what a machine without that module would look
    like.
    """
    import torch
    from comfy.nested_tensor import NestedTensor
    from core import registry as registry_mod
    from modules.output.decode.nodes import FunPackDecode

    for spec in registry_mod.current().specs.values():
        if "decode" in spec.provides:
            monkeypatch.delitem(spec.provides, "decode")

    nested = NestedTensor((torch.zeros(1, 24, 2, 8, 8), torch.zeros(1, 32, 2, 16)))
    with pytest.raises(RuntimeError, match="more than one part"):
        FunPackDecode.execute({"samples": nested}, _Vae(), model=_h3_model())


def test_h3_claims_a_nested_latent_and_uses_both_vaes(monkeypatch):
    import torch
    from comfy.nested_tensor import NestedTensor
    from modules.models import minimax_h3

    video_vae = _Vae(out=torch.zeros(1, 3, 8, 8, 3))
    audio_vae = _Vae()

    monkeypatch.setattr("comfy_extras.nodes_audio.vae_decode_audio",
                        lambda vae, samples: {"waveform": "decoded", "sample_rate": 32000})

    nested = NestedTensor((torch.zeros(1, 24, 2, 8, 8), torch.zeros(1, 32, 2, 16)))
    images, audio = minimax_h3.decode(nested, model=_h3_model(),
                                      vae=video_vae, audio_vae=audio_vae)

    assert images.shape == (3, 8, 8, 3)
    assert audio["waveform"] == "decoded"


def test_h3_does_not_speak_for_a_plain_latent():
    import torch
    from modules.models import minimax_h3
    assert minimax_h3.decode(torch.zeros(1, 4, 8, 8), model=_h3_model(), vae=_Vae()) is None


def test_h3_does_not_claim_another_models_two_part_latent():
    """The reason the node takes a model at all. "Two parts" describes plenty of
    models that are not this one, and H3's branch order is its own -- reading
    someone else's by shape decodes noise as a picture rather than failing."""
    import torch
    from comfy.nested_tensor import NestedTensor
    from modules.models import minimax_h3

    nested = NestedTensor((torch.zeros(1, 24, 2, 8, 8), torch.zeros(1, 32, 2, 16)))
    assert minimax_h3.decode(nested, model=_NotH3Model(), vae=_Vae()) is None


def test_h3_declines_when_the_model_is_not_wired_rather_than_guessing():
    import torch
    from comfy.nested_tensor import NestedTensor
    from modules.models import minimax_h3

    nested = NestedTensor((torch.zeros(1, 24, 2, 8, 8), torch.zeros(1, 32, 2, 16)))
    assert minimax_h3.decode(nested, model=None, vae=_Vae()) is None


def test_an_unrecognised_nested_latent_says_the_model_was_not_wired():
    import torch
    from comfy.nested_tensor import NestedTensor
    from modules.output.decode.nodes import FunPackDecode

    nested = NestedTensor((torch.zeros(1, 24, 2, 8, 8), torch.zeros(1, 32, 2, 16)))
    with pytest.raises(RuntimeError, match="model input is not wired"):
        FunPackDecode.execute({"samples": nested}, _Vae())


def test_a_missing_audio_vae_is_named_rather_than_silently_silent():
    """Returning no audio would look exactly like a model that makes none."""
    import torch
    from comfy.nested_tensor import NestedTensor
    from modules.models import minimax_h3

    nested = NestedTensor((torch.zeros(1, 24, 2, 8, 8), torch.zeros(1, 32, 2, 16)))
    with pytest.raises(RuntimeError, match="audio VAE"):
        minimax_h3.decode(nested, model=_h3_model(),
                          vae=_Vae(out=torch.zeros(1, 3, 8, 8, 3)), audio_vae=None)


def test_a_claiming_module_that_breaks_stops_the_decode(monkeypatch):
    import torch
    from comfy.nested_tensor import NestedTensor
    from core import registry as registry_mod
    from modules.output.decode.nodes import FunPackDecode

    def broken(latent, model=None, vae=None, audio_vae=None):
        if not getattr(latent, "is_nested", False):
            return None
        raise RuntimeError("branch order changed")

    spec = registry_mod.current().specs["model_minimax_h3"]
    monkeypatch.setitem(spec.provides, "decode", broken)

    nested = NestedTensor((torch.zeros(1, 24, 2, 8, 8), torch.zeros(1, 32, 2, 16)))
    with pytest.raises(RuntimeError, match="Refusing to decode"):
        FunPackDecode.execute({"samples": nested}, _Vae())
