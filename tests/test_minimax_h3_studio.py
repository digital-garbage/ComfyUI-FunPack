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
    node._encoder_has_vision = lambda clip: True
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


# ── the multi-scene split must not drop the visual conditioning ──────────────
# The Movie Editor always runs split_by_transitions, and the per-scene conditionings REPLACE
# the single entry Studio built. Anything that lived on that entry — the anchor, the resolved
# references — has to be re-established per scene or it never reaches the sampler at all.

def test_the_anchor_survives_the_multi_scene_split_on_the_opening_scene():
    node = FunPackVideoRefinerV2()
    clip = H3Clip()
    scenes = node._v2_transition_scene_conditionings(
        clip, ["a room", "a street", "a roof"], reference_image=IMAGE)

    assert [m["funpack_scene_index"] for _c, m in scenes] == [0, 1, 2]
    # scene 0 carries the pin AND was encoded with the image in front of Qwen
    assert scenes[0][1]["funpack_h3_anchor"]["image"] is IMAGE
    assert clip.tokenize_kwargs[0]["images"] == [IMAGE]
    # later scenes continue from the previous scene's output, so no anchor and no vision
    assert all("funpack_h3_anchor" not in m for _c, m in scenes[1:])
    assert all(not kw.get("images") for kw in clip.tokenize_kwargs[1:])


def test_references_survive_the_split_on_every_scene(h3_media):
    """A reference identity holds across the whole chain, and <Picture 1> only resolves in a
    scene whose own encode presented it."""
    node = FunPackVideoRefinerV2()
    clip = H3Clip()
    scenes = node._v2_transition_scene_conditionings(
        clip, ["a room", "a street"],
        h3_references=[{"kind": "image", "filename": "face.png"}])

    assert all([r["filename"] for r in m["funpack_h3_refs"]] == ["face.png"] for _c, m in scenes)
    assert all(kw.get("minimax_ref_items") for kw in clip.tokenize_kwargs)


def test_an_ltx_split_is_left_alone():
    """On LTX the anchor reaches the model through the latent. Adding Gemma3 vision to scene 0
    here would change what every existing multi-scene LTX project generates."""
    node = FunPackVideoRefinerV2()
    node._encoder_has_vision = lambda clip: True
    scenes = node._v2_transition_scene_conditionings(LTXClip(), ["a room", "a street"],
                                                     reference_image=IMAGE)
    assert all("funpack_h3_anchor" not in m for _c, m in scenes)


# --- which input actually owns the conditioning -----------------------------------
# A wired positive CONDITIONING owns the prompt; CLIP keeps the negative, the references and
# phrase classification. CLIP used to win, silently, so a graph feeding Studio from an i2v or
# r2v node had its conditioning replaced by a text-only re-encode Studio could not make
# equivalent — it has no way to reach the reference image the node had.

class _StubClip:
    pass


def _source(node, clip, wired, monkeypatch, clip_owns_prompt=False):
    monkeypatch.setattr(
        node, "_v2_encode_prompt",
        lambda *a, **k: (torch.ones(1, 4, 8), {"pooled_output": None}, "encoded"),
        raising=False)
    monkeypatch.setattr(
        node, "_v2_extract_conditioning",
        lambda c: (torch.zeros(1, 4, 8), {"pooled_output": None}), raising=False)
    monkeypatch.setattr(node, "_v2_text_tokenizer_status", lambda: "tokenizer ok", raising=False)
    return node._v2_conditioning_source(clip, "a prompt", wired,
                                        clip_owns_prompt=clip_owns_prompt)


def test_a_wired_conditioning_owns_the_prompt_even_with_clip_connected(monkeypatch, capsys):
    """CLIP used to win, so Studio re-encoded from text and the node's own conditioning was
    discarded. The node may have seen a reference image Studio has no way to reach, making
    the re-encode a different tensor rather than an approximation of the same one."""
    node = FunPackVideoRefinerV2()
    wired = [[torch.zeros(1, 4, 8), {}]]
    cond, _meta, status, owner = _source(node, _StubClip(), wired, monkeypatch)
    assert owner == "CONDITIONING-owned"
    assert torch.equal(cond, torch.zeros(1, 4, 8))     # the wired tensor, not a re-encode
    assert "owns the prompt" in status
    assert "CLIP still encodes the negative" in status


def test_clip_still_owns_the_prompt_when_asked_to(monkeypatch):
    """The old precedence, on request rather than by default."""
    node = FunPackVideoRefinerV2()
    wired = [[torch.zeros(1, 4, 8), {}]]
    cond, _meta, _status, owner = _source(node, _StubClip(), wired, monkeypatch,
                                          clip_owns_prompt=True)
    assert owner == "CLIP-owned"
    assert torch.equal(cond, torch.ones(1, 4, 8))


def test_the_ownership_note_is_said_once_not_once_per_scene(monkeypatch, capsys):
    node = FunPackVideoRefinerV2()
    wired = [[torch.zeros(1, 4, 8), {}]]
    for _ in range(3):
        _source(node, _StubClip(), wired, monkeypatch)
    assert capsys.readouterr().out.count("owns the prompt") == 1


def test_a_wired_conditioning_is_used_when_clip_is_absent(monkeypatch):
    node = FunPackVideoRefinerV2()
    wired = [[torch.zeros(1, 4, 8), {}]]
    cond, _meta, status, owner = _source(node, None, wired, monkeypatch)
    assert owner == "CONDITIONING-owned"
    assert torch.equal(cond, torch.zeros(1, 4, 8))
    assert "CLIP still encodes" not in status


def test_clip_alone_says_nothing_extra(monkeypatch, capsys):
    node = FunPackVideoRefinerV2()
    _cond, _meta, status, owner = _source(node, _StubClip(), None, monkeypatch)
    assert owner == "CLIP-owned"
    assert "IGNORED" not in status
    assert capsys.readouterr().out == ""
