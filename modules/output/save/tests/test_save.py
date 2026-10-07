import av
import numpy as np
import pytest

from modules.output.save import nodes


def _clip(n=30):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (n, 64, 96, 3), dtype=np.uint8)


def test_cpu_encode_writes_a_playable_h264_mp4_with_sound_and_metadata(tmp_path):
    import torch
    path = str(tmp_path / "a.mp4")
    audio = {"waveform": torch.zeros(1, 2, 44100 * 2), "sample_rate": 44100}
    used = nodes.encode(path, _clip(), 24.0, audio, {"prompt": {"1": "x"}}, encoder="cpu")
    assert "libx264" in used
    with av.open(path) as f:
        v = f.streams.video[0]
        assert v.codec_context.name == "h264" and v.frames == 30 and f.streams.audio
        assert "prompt" in f.metadata
    head = open(path, "rb").read(4096)
    assert head.find(b"moov") != -1 and (head.find(b"mdat") == -1 or head.find(b"moov") < head.find(b"mdat")), "index first: seekable while streaming"


def test_auto_falls_back_to_the_cpu_when_there_is_no_gpu_encoder(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "NVENC", {"h265": ("no_such_encoder_x", {})})
    monkeypatch.setattr(av, "codecs_available", av.codecs_available | {"no_such_encoder_x"})
    path = str(tmp_path / "b.mp4")
    assert "libx264" in nodes.encode(path, _clip(10), 24.0, encoder="auto")


def test_gpu_demanded_without_one_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "NVENC", {"h265": ("no_such_encoder_x", {})})
    with pytest.raises(RuntimeError, match="NVENC"):
        nodes.encode(str(tmp_path / "c.mp4"), _clip(5), 24.0, encoder="gpu")


def test_h265_is_tagged_hvc1_so_a_mac_browser_plays_it(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "NVENC", {"h265": ("libx265", {"crf": "23"})})       # the software HEVC encoder stands in for NVENC here
    if "libx265" not in av.codecs_available:
        pytest.skip("no libx265 in this PyAV build")
    path = str(tmp_path / "d.mp4")
    assert "H265" in nodes.encode(path, _clip(10), 24.0, encoder="gpu")
    with av.open(path) as f:
        assert f.streams.video[0].codec_context.name == "hevc" and f.streams.video[0].codec_tag == "hvc1"
