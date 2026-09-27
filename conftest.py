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

    def run(self, patcher=None, sigma=0.5, sigmas=None, negative=None):
        """One denoise call through `patcher`'s transformer_options, as the
        sampler would make it. Returns (video, audio)."""
        import torch
        patcher = patcher or self.patcher
        to = dict(patcher.model_options.get("transformer_options", {}))
        sched = sigmas if sigmas is not None else torch.tensor([1.0, sigma, 0.0])
        to.setdefault("sample_sigmas", sched)
        to["sigmas"] = torch.tensor([sigma])
        dm = patcher.model.diffusion_model
        with torch.inference_mode():
            return dm([self.video, self.audio], torch.tensor([sigma * 1000.0]),
                      self.context, transformer_options=to)


@pytest.fixture
def tiny_h3(comfyui):
    return TinyH3()
