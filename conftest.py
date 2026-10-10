"""Shared test setup for the whole tree.

Kept at the root so `modules/**/tests` get the same fixtures as `core/tests`
without each one re-deriving the path to ComfyUI.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _comfyui_root():
    candidates = [os.environ.get("COMFYUI_ROOT"), str(Path.home() / "Documents" / "ComfyUI")]
    for candidate in candidates:
        if candidate and (Path(candidate) / "comfy" / "supported_models.py").is_file():
            if candidate not in sys.path:
                sys.path.append(candidate)
            return candidate
    return None


# Put ComfyUI on the path at IMPORT time, not only when a fixture asks. A module
# that ships a modifier imports comfy at module level -- as it must, since that is
# how it runs inside ComfyUI -- so collection touches comfy before any fixture has
# had a chance to run.
COMFYUI = _comfyui_root()


@pytest.fixture(scope="session")
def comfyui():
    """ComfyUI's source tree on sys.path, or skip.

    Everything that touches a node schema needs it, because a schema is only
    meaningful in terms of comfy_api's own types.
    """
    if COMFYUI is None:
        pytest.skip("no ComfyUI source on this machine")
    return COMFYUI


class TinyH3:
    """A REAL MiniMax H3 -- ComfyUI's own class, four blocks wide as a thumb.

    Random weights, CPU, milliseconds per forward. Every H3 modifier is tested
    against this rather than a mock, because what breaks these features is the
    real forward's shapes (joint sequence without a batch axis, tensor
    containers around q/k/v, mod_segments) and a mock only has the shapes its
    author remembered.
    """

    def __init__(self, blocks=4, seed=0):
        import torch
        import comfy.ops
        from comfy.latent_formats import MiniMaxH3AV
        from comfy.ldm.minimax.model import MiniMaxH3Model
        from comfy.model_patcher import ModelPatcher

        torch.manual_seed(seed)
        dm = MiniMaxH3Model(
            hidden_size=64, num_layers=blocks, token_refiner_num_layers=1,
            num_attention_heads=2, attention_head_dim=32, ffn_hidden_size=64,
            text_dim=48, timestep_input_dim=32, time_embed_hidden_size=64,
            time_embed_dim=32, rope_inv_freq_len=4, dtype=torch.float32,
            operations=comfy.ops.manual_cast)
        for p in dm.parameters():
            torch.nn.init.normal_(p, std=0.05)
            p.requires_grad_(False)       # the in-place rope kernel refuses autograd
        dm.rope.inv_freq.copy_(torch.rand(4))

        class Config:
            latent_format = MiniMaxH3AV

        class Base(torch.nn.Module):
            pass

        base = Base()
        base.diffusion_model = dm
        base.model_config = Config()
        self.patcher = ModelPatcher(base, load_device=torch.device("cpu"),
                                    offload_device=torch.device("cpu"))
        self.video = torch.randn(1, 24, 2, 4, 4)
        self.audio = torch.randn(1, 32, 2, 3)
        self.context = torch.randn(1, 5, 48)

    def run(self, patcher=None, sigma=0.5, sigmas=None, denoise_mask=None, options=None):
        """One denoise call through `patcher`'s transformer_options, as the
        sampler would make it. Returns (video, audio)."""
        import torch
        patcher = patcher or self.patcher
        import comfy.patcher_extension as ext
        to = dict(patcher.model_options.get("transformer_options", {}))
        # What prepare_model_patcher does before sampling: the patcher's wrappers
        # join transformer_options, where H3's forward looks for them.
        to["wrappers"] = ext.copy_nested_dicts(to.get("wrappers", {}))
        ext.merge_nested_dicts(to["wrappers"], patcher.wrappers, copy_dict1=False)
        sched = sigmas if sigmas is not None else torch.tensor([1.0, sigma, 0.0])
        to.setdefault("sample_sigmas", sched)
        to["sigmas"] = torch.tensor([sigma])
        to.update(options or {})
        dm = patcher.model.diffusion_model
        with torch.inference_mode():
            return dm([self.video, self.audio], torch.tensor([sigma * 1000.0]),
                      self.context, transformer_options=to, denoise_mask=denoise_mask)

    def sample(self, patcher=None, guider=None, **run):
        """`run`, inside the patcher's OUTER_SAMPLE wrappers -- how a whole
        sampling call wraps the forward. `guider` is the executor's class_obj."""
        from comfy.patcher_extension import WrappersMP
        patcher = patcher or self.patcher
        result = {}

        class _Executor:
            class_obj = guider

            def __init__(self, inner):
                self.inner = inner

            def __call__(self, *a, **k):
                return self.inner()

        def call():
            result["out"] = self.run(patcher, **run)
            return result["out"]

        for ws in patcher.wrappers.get(WrappersMP.OUTER_SAMPLE, {}).values():
            for w in ws:
                call = (lambda w, inner: (lambda: w(_Executor(inner))))(w, call)
        call()
        return result["out"]

    def pinned_first_frame(self):
        """An i2v-style mask: latent frame 0 kept, the rest generated. Gives the
        video rows their own per-token mod rows, the shape v4 choked on."""
        import torch
        mask = torch.ones(1, 1, *self.video.shape[2:])
        mask[:, :, 0] = 0.0
        return mask


@pytest.fixture
def tiny_h3(comfyui):
    return TinyH3()


def packed_av(video, audio):
    """[video, audio] exactly as H3's model call hands them back: comfy's
    `_apply_model` packs the pair flat with `pack_latents`, and the shapes that
    undo it travel with the call as `latent_shapes`. -> (x, latent_shapes)."""
    import comfy.utils
    return comfy.utils.pack_latents([video, audio])


def unpacked(x, latent_shapes):
    import comfy.utils
    return comfy.utils.unpack_latents(x, latent_shapes)


@pytest.fixture(autouse=True)
def _quarantine_in_tmp(tmp_path, monkeypatch):
    """A test that makes a module fail must never write the real quarantine file."""
    from core import config
    monkeypatch.setattr(config, "QUARANTINE_FILE", tmp_path / "quarantine.json")


@pytest.fixture(autouse=True)
def _block_influence_off_in_tmp(tmp_path, monkeypatch):
    """Recording is on by default, so a test that rates a clip would also pick up its block-influence
    capture. Off here; a test that wants it on turns it on itself."""
    from modules.sampling.block_influence import measure
    switch = tmp_path / "block_influence.enabled"
    switch.write_text("0")
    monkeypatch.setattr(measure, "SWITCH", switch)


class TinyLTX:
    """A REAL LTX audio+video model -- ComfyUI's own LTXAVModel, four blocks wide as a thumb.

    Random weights, CPU, milliseconds per forward: every LTX modifier is tested against its real
    forward (the (video, audio) pair every block takes, the packed context) rather than a mock.
    """

    def __init__(self, blocks=4, seed=0):
        import torch
        import comfy.ops
        from comfy.latent_formats import LTXAV
        from comfy.ldm.lightricks.av_model import LTXAVModel
        from comfy.model_patcher import ModelPatcher

        torch.manual_seed(seed)
        dm = LTXAVModel(
            in_channels=128, audio_in_channels=128, cross_attention_dim=32, audio_cross_attention_dim=16,
            attention_head_dim=16, audio_attention_head_dim=8, num_attention_heads=2,
            audio_num_attention_heads=2, caption_channels=48, num_layers=blocks,
            caption_proj_before_connector=True, dtype=torch.float32, operations=comfy.ops.manual_cast)
        for p in dm.parameters():
            torch.nn.init.normal_(p, std=0.05)
            p.requires_grad_(False)

        class Config:
            latent_format = LTXAV

        class Base(torch.nn.Module):
            pass

        base = Base()
        base.diffusion_model = dm
        base.model_config = Config()
        self.patcher = ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
        self.video = torch.randn(1, 128, 2, 2, 2)
        self.audio = torch.randn(1, 8, 3, 16)
        self.context = torch.randn(1, 5, 48)

    def run(self, patcher=None, sigma=0.5, sigmas=None, options=None, video=None, audio=None):
        """One denoise call through `patcher`'s transformer_options. -> [video, audio]."""
        import torch
        import comfy.patcher_extension as ext
        patcher = patcher or self.patcher
        to = dict(patcher.model_options.get("transformer_options", {}))
        to["wrappers"] = ext.copy_nested_dicts(to.get("wrappers", {}))
        ext.merge_nested_dicts(to["wrappers"], patcher.wrappers, copy_dict1=False)
        sched = sigmas if sigmas is not None else torch.tensor([1.0, sigma, 0.0])
        to.setdefault("sample_sigmas", sched)
        to["sigmas"] = torch.tensor([sigma])
        to.update(options or {})
        dm = patcher.model.diffusion_model
        with torch.inference_mode():
            return dm([self.video if video is None else video, self.audio if audio is None else audio],
                      torch.tensor([sigma * 1000.0]), self.context, frame_rate=25, transformer_options=to)


_TINY_LTX = []


@pytest.fixture
def tiny_ltx(comfyui):
    """One model for the whole run (building it costs seconds): tests clone its patcher and never
    leave anything installed on the model itself."""
    if not _TINY_LTX:
        _TINY_LTX.append(TinyLTX())
    return _TINY_LTX[0]
