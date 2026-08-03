"""Studio's half of the MiniMax H3 split, and the LTX path it must not disturb.

Studio owns the CLIP; the Chain Sampler owns the VAE. Two things therefore have to travel
between them on the conditioning, and both are silent when they go missing:

1. The i2v anchor image. H3 has no latent i2v path — the anchor is a frame-0 keyframe pin,
   which needs the VAE — so Studio presents the image to Qwen and hands the PIXELS on. On
   LTX nothing of the sort happens: the anchor is written into the latent by the graph.
2. The resolved reference list, with every field that shaped the presentation. A video's
   soundtrack especially: Studio already emitted its "<Audio j>" label, so a sampler that
   never hears about the track packs no audio rows and every later audio ordinal shifts.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("folder_paths", types.SimpleNamespace(models_dir=""))

from conditioning import FunPackVideoRefinerV2  # noqa: E402


class H3Clip:
    """What comfy builds for H3: an SD1Tokenizer named after the Qwen3-VL TE."""

    def __init__(self):
        self.tokenizer = types.SimpleNamespace(qwen3vl_32b=object())
        self.tokenize_kwargs = []

    def tokenize(self, text, **kwargs):
        self.tokenize_kwargs.append(kwargs)
        return text

    def encode_from_tokens_scheduled(self, tokens):
        return [(torch.ones(1, 12, 5120), {"pooled_output": None})]


class LTXClip:
    def __init__(self):
        self.tokenizer = types.SimpleNamespace(gemma3=object())

    def tokenize(self, text, **kwargs):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        return [(torch.ones(1, 12, 4096), {"pooled_output": None})]


IMAGE = torch.zeros(1, 64, 64, 3)


@pytest.fixture
def h3_media(monkeypatch):
    import minimax_h3 as h3mod
    monkeypatch.setattr(h3mod, "load_input_image", lambda f: torch.zeros(1, 64, 64, 3))
    monkeypatch.setattr(h3mod, "load_input_audio",
                        lambda f: {"waveform": torch.zeros(1, 2, 800), "sample_rate": 32000})
    monkeypatch.setattr(h3mod, "load_input_video", lambda f, **k: torch.zeros(39, 64, 64, 3))


def test_the_anchor_image_is_handed_to_the_sampler_as_pixels():
    node = FunPackVideoRefinerV2()
    clip = H3Clip()
    _cond, meta, _status = node._v2_encode_prompt(clip, "a shot", reference_image=IMAGE)

    # presented to Qwen as <Picture 1> ...
    assert clip.tokenize_kwargs[0]["images"] == [IMAGE]
    # ... and handed on, because pinning it needs the VAE the sampler owns
    assert meta["funpack_h3_anchor"]["image"] is IMAGE


def test_an_ltx_anchor_is_not_carried_on_the_conditioning():
    """LTX writes the anchor into the latent in the graph. Carrying it here too would put a
    second, competing anchor mechanism on a family that already has one."""
    node = FunPackVideoRefinerV2()
    node._gemma3_has_vision = lambda clip: True
    _cond, meta, _status = node._v2_encode_prompt(LTXClip(), "a shot", reference_image=IMAGE)
    assert "funpack_h3_anchor" not in meta


def test_references_take_over_from_the_anchor_and_say_so(h3_media, capsys):
    """ref2va and fl2va are separate conditioning modes — an anchor image cannot ride along
    with reference media, and the user has to be told which one won."""
    node = FunPackVideoRefinerV2()
    _cond, meta, _status = node._v2_encode_prompt(
        H3Clip(), "a shot", reference_image=IMAGE,
        h3_references=[{"kind": "image", "filename": "face.png"}])

    assert "funpack_h3_anchor" not in meta
    assert [r["filename"] for r in meta["funpack_h3_refs"]] == ["face.png"]
    assert "source_image is NOT" in capsys.readouterr().out


def test_a_video_references_soundtrack_reaches_the_sampler(h3_media):
    """Studio emits "<Audio 1>" for the track it resolved. If the sampler never learns the
    track exists it packs no audio rows for it, and every later <Audio j> points one
    reference earlier than the prompt says."""
    node = FunPackVideoRefinerV2()
    _cond, meta, _status = node._v2_encode_prompt(
        H3Clip(), "a shot",
        h3_references=[{"kind": "video", "filename": "clip.mp4", "audio": "clip.wav"}])

    assert meta["funpack_h3_refs"] == [{"kind": "video", "filename": "clip.mp4",
                                        "audio": "clip.wav"}]


def test_the_reference_size_mode_reaches_the_sampler(h3_media):
    node = FunPackVideoRefinerV2()
    _cond, meta, _status = node._v2_encode_prompt(
        H3Clip(), "a shot",
        h3_references=[{"kind": "image", "filename": "face.png", "size": "max"}])
    assert meta["funpack_h3_refs"][0]["size"] == "max"
