"""FunPackLoadMedia: the bridge from a media library id to an IMAGE tensor."""

import io as pyio

import pytest

from core import config, media


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy_api."""


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEDIA_DIR", tmp_path / "media")
    return tmp_path / "media"


def _png_bytes():
    from PIL import Image
    buf = pyio.BytesIO()
    Image.new("RGB", (4, 6), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_a_picked_image_loads_as_an_image_tensor(store):
    from modules.media.load.nodes import FunPackLoadMedia
    entry = media.save_upload("ref.png", _png_bytes())

    images, frames, _audio = FunPackLoadMedia.execute(entry["id"]).result
    assert images.shape[0] == 1 and frames.shape[0] == 1
    assert images.shape[1:3] == (6, 4)   # H, W


def test_no_media_id_is_refused_loudly(store):
    from modules.media.load.nodes import FunPackLoadMedia
    with pytest.raises(RuntimeError, match="nothing is picked"):
        FunPackLoadMedia.execute("").result


def test_an_unknown_id_is_refused_loudly(store):
    from modules.media.load.nodes import FunPackLoadMedia
    with pytest.raises(RuntimeError, match="not in the library"):
        FunPackLoadMedia.execute("deadbeef0000").result


def test_a_file_that_cannot_be_read_is_refused_in_plain_words(store):
    from modules.media.load.nodes import FunPackLoadMedia
    for name in ("clip.mp4", "voice.wav"):
        entry = media.save_upload(name, b"not real media but has an ext")
        with pytest.raises(RuntimeError, match="could not read"):
            FunPackLoadMedia.execute(entry["id"]).result


def test_deleted_media_is_refused_even_with_a_well_formed_id(store):
    from modules.media.load.nodes import FunPackLoadMedia
    entry = media.save_upload("ref.png", _png_bytes())
    media.delete(entry["id"])
    with pytest.raises(RuntimeError, match="not in the library"):
        FunPackLoadMedia.execute(entry["id"]).result


def test_a_blank_size_stands_in_for_nothing_picked_and_a_picked_image_still_wins(store):
    from modules.media.load.nodes import FunPackLoadMedia
    images, *_ = FunPackLoadMedia.execute("", blank_width=64, blank_height=32).result
    assert tuple(images.shape) == (1, 32, 64, 3)
    entry = media.save_upload("ref.png", _png_bytes())
    images, *_ = FunPackLoadMedia.execute(entry["id"], blank_width=64, blank_height=32).result
    assert images.shape[1:3] == (6, 4)
    with pytest.raises(RuntimeError, match="nothing is picked"):
        FunPackLoadMedia.execute("", blank_width=64, blank_height=0).result


def _clip(store, seconds=1):
    import shutil, subprocess
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg needed to make a clip")
    clip = store.parent / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=size=64x48:rate=8:duration={seconds}",
                    "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-shortest", "-pix_fmt", "yuv420p", str(clip)], check=True)
    return media.save_upload("clip.mp4", clip.read_bytes())


def test_a_video_gives_its_first_frame_by_default(store):
    from modules.media.load.nodes import FunPackLoadMedia
    entry = _clip(store)
    image, frames, _ = FunPackLoadMedia.execute(entry["id"]).result
    assert image.shape[0] == 1 and frames.shape[0] == 1
    assert image.shape[1:3] == (48, 64)


def test_a_full_video_gives_every_frame_at_24_fps_and_its_sound(store):
    from modules.media.load.nodes import FunPackLoadMedia
    entry = _clip(store, seconds=2)
    image, frames, audio = FunPackLoadMedia.execute(entry["id"], full=True).result
    assert image.shape[0] == 1
    assert frames.shape[0] == 48            # 2 s at 24 fps
    assert audio["waveform"].shape[-1] > 1000


def test_audio_loads_as_audio(store):
    from modules.media.load.nodes import FunPackLoadMedia
    import struct, wave
    path = store.parent / "a.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(struct.pack("<800h", *([1000] * 800)))
    entry = media.save_upload("a.wav", path.read_bytes())
    _, _, audio = FunPackLoadMedia.execute(entry["id"], full=True).result
    assert audio["sample_rate"] == 8000 and audio["waveform"].shape == (1, 1, 800)


def test_slots_of_the_retired_video_and_audio_nodes_are_dropped_with_their_wires():
    from modules.media.load import upgrade
    slots = [{"id": "r2v", "node": "X", "inputs": {"a": ["v1", 1], "keep": 3, "b": ["ref", 0]}},
             {"id": "v1", "node": "FunPackLoadMediaVideo", "inputs": {}},
             {"id": "ref", "node": "FunPackLoadMedia", "inputs": {}}]
    out = upgrade(slots, None)
    assert [s["id"] for s in out] == ["r2v", "ref"]
    assert out[0]["inputs"] == {"keep": 3, "b": ["ref", 0]}
