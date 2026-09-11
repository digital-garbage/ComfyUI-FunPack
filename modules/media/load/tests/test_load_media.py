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

    images, = FunPackLoadMedia.execute(entry["id"]).result
    assert images.shape[0] == 1
    assert images.shape[1:3] == (6, 4)   # H, W


def test_no_media_id_is_refused_loudly(store):
    from modules.media.load.nodes import FunPackLoadMedia
    with pytest.raises(RuntimeError, match="nothing is picked"):
        FunPackLoadMedia.execute("").result


def test_an_unknown_id_is_refused_loudly(store):
    from modules.media.load.nodes import FunPackLoadMedia
    with pytest.raises(RuntimeError, match="not in the library"):
        FunPackLoadMedia.execute("deadbeef0000").result


def test_a_non_image_media_kind_is_refused(store):
    from modules.media.load.nodes import FunPackLoadMedia
    entry = media.save_upload("clip.mp4", b"not a real video but has an ext")
    with pytest.raises(RuntimeError, match="not an image"):
        FunPackLoadMedia.execute(entry["id"]).result


def test_deleted_media_is_refused_even_with_a_well_formed_id(store):
    from modules.media.load.nodes import FunPackLoadMedia
    entry = media.save_upload("ref.png", _png_bytes())
    media.delete(entry["id"])
    with pytest.raises(RuntimeError, match="not in the library"):
        FunPackLoadMedia.execute(entry["id"]).result
