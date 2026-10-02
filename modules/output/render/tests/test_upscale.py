import pytest

from modules.output.render.tests.clips import comfy, make_clip, needs_ffmpeg  # noqa: F401
from modules.output.render import files, upscale

torch = pytest.importorskip("torch")


@needs_ffmpeg
def test_upscale_doubles_size_keeps_fps_frames_and_sound(comfy):
    src, dst = str(comfy.output / "a.mp4"), str(comfy.output / "up.mp4")
    make_clip(src, seconds=1.0, size="64x48", rate=10, audio=True)
    seen = []

    def twice(x):
        seen.append(x.shape[0])
        return x.repeat_interleave(2, 1).repeat_interleave(2, 2)

    assert upscale.upscale_file(src, dst, twice) == 10
    assert seen == [8, 2]                                   # streamed in chunks, not all at once
    w, h, fps, sound = upscale.probe(dst)
    assert (w, h, fps, sound) == (128, 96, "10/1", True)
    assert files.moov_position(dst) == "front"


@needs_ffmpeg
def test_interrupt_stops_and_leaves_no_file(comfy):
    src, dst = str(comfy.output / "a.mp4"), str(comfy.output / "up.mp4")
    make_clip(src, seconds=1.0, size="64x48", rate=10, audio=False)
    calls = []

    def stop():
        calls.append(1)
        if len(calls) > 1:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        upscale.upscale_file(src, dst, lambda x: x, stop)
    import os
    assert not os.path.exists(dst)


def test_free_name_never_overwrites(tmp_path):
    (tmp_path / "a_m.mp4").write_bytes(b"x")
    assert upscale.free_name(str(tmp_path), "a.mp4", "sub/m.safetensors") == "a_m_1.mp4"
