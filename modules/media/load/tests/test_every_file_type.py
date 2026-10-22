"""Every file type the Media bin accepts, made for real, uploaded, thumbnailed and loaded.

The bin lists the types it takes (core/media.py); a type on that list that cannot be read is a path offered and
broken. Pictures are written with Pillow, videos and sounds with ffmpeg (a type whose encoder this machine's ffmpeg
lacks is skipped, not passed). Each is then loaded the way a reference is: a picture, a video (frames at 24 fps and
its soundtrack), a sound.
"""

import shutil
import subprocess

import pytest

from core import config, media

FFMPEG = shutil.which("ffmpeg")


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy_api."""


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEDIA_DIR", tmp_path / "media")
    return tmp_path


def _image(path):
    from PIL import Image
    Image.new("RGB", (64, 48), (200, 30, 30)).save(path)


# extension -> ffmpeg codec arguments for a 1 second clip
VIDEOS = {".mp4": ["-c:v", "mpeg4", "-c:a", "aac"], ".webm": ["-c:v", "libvpx", "-c:a", "libvorbis"], ".mov": ["-c:v", "mpeg4", "-c:a", "aac"],
          ".mkv": ["-c:v", "mpeg4", "-c:a", "aac"], ".avi": ["-c:v", "mpeg4", "-c:a", "mp3"]}
AUDIO = {".wav": [], ".mp3": ["-c:a", "libmp3lame"], ".m4a": ["-c:a", "aac"], ".aac": ["-c:a", "aac"], ".ogg": ["-c:a", "libvorbis"],
         ".flac": ["-c:a", "flac"], ".opus": ["-c:a", "libopus"], ".weba": ["-c:a", "libvorbis", "-f", "webm"]}


def _ffmpeg(args, out):
    if not FFMPEG:
        pytest.skip("no ffmpeg on this machine")
    run = subprocess.run([FFMPEG, "-y", "-loglevel", "error", *args, str(out)], capture_output=True, text=True)
    if run.returncode:
        pytest.skip(f"this ffmpeg cannot make {out.suffix}: {run.stderr.strip().splitlines()[-1:]}")


def _video(path, sound, fps=25):
    inputs = ["-f", "lavfi", "-i", f"testsrc=duration=1:size=64x48:rate={fps}"]
    if sound:
        inputs += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1"]
    _ffmpeg([*inputs, *VIDEOS[path.suffix], "-shortest"], path)


def _sound(path):
    _ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=1", *AUDIO[path.suffix]], path)


def _load(entry_id, **kw):
    from modules.media.load.nodes import FunPackLoadMedia
    return FunPackLoadMedia.execute(entry_id, **kw).result


def test_the_bin_lists_exactly_the_types_these_tests_cover():
    assert media.ALLOWED_EXT == {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"} | set(VIDEOS) | set(AUDIO)


@pytest.mark.parametrize("ext", sorted([".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"]))
def test_a_picture_of_every_type_is_listed_thumbnailed_and_loaded(store, ext):
    path = store / f"p{ext}"
    _image(path)
    entry = media.save_upload(path.name, path.read_bytes())
    assert entry["kind"] == "image"
    assert media.thumb_path_for(entry["id"]) is not None
    image, frames, audio = _load(entry["id"])
    assert tuple(image.shape) == (1, 48, 64, 3) and frames.shape[0] == 1


@pytest.mark.parametrize("ext", sorted(VIDEOS))
@pytest.mark.parametrize("sound", [True, False], ids=["with sound", "silent"])
def test_a_video_of_every_type_says_whether_it_has_sound_and_loads_its_frames_and_soundtrack(store, ext, sound):
    path = store / f"v{ext}"
    _video(path, sound)
    entry = media.save_upload(path.name, path.read_bytes())
    assert entry["kind"] == "video" and entry["sound"] is sound
    assert media.thumb_path_for(entry["id"]) is not None
    image, frames, audio = _load(entry["id"], full=True)
    assert image.shape[0] == 1
    assert 22 <= frames.shape[0] <= 25, "a 1 second clip at 25 fps is about 24 frames at the model's rate"
    assert (audio["waveform"].shape[-1] > 1000) is sound, "the soundtrack comes through, and silence stays one silent sample"
    first_only = _load(entry["id"])
    assert first_only[1].shape[0] == 1                                    # as a picture input: the first frame only


@pytest.mark.parametrize("ext", sorted(AUDIO))
def test_a_sound_of_every_type_loads_as_a_waveform(store, ext):
    path = store / f"a{ext}"
    _sound(path)
    entry = media.save_upload(path.name, path.read_bytes())
    assert entry["kind"] == "audio"
    assert media.thumb_path_for(entry["id"]) is None                    # a sound has no picture: the grid draws a placeholder
    _image_out, _frames, audio = _load(entry["id"])
    assert audio["waveform"].dim() == 3 and audio["waveform"].shape[-1] > 1000 and audio["sample_rate"] > 0


def test_a_video_at_another_rate_is_resampled_to_the_models_24(store):
    path = store / "slow.mp4"
    _video(path, False, fps=12)
    entry = media.save_upload(path.name, path.read_bytes())
    _image_out, frames, _audio = _load(entry["id"], full=True)
    assert 22 <= frames.shape[0] <= 25


def _load_image_file(store, name, make):
    path = store / name
    make(path)
    entry = media.save_upload(path.name, path.read_bytes())
    return entry, _load(entry["id"])


def test_a_phone_photo_stored_sideways_is_loaded_upright_like_the_thumbnail(store):
    from PIL import Image

    def make(path):
        img = Image.new("RGB", (64, 48), (10, 200, 10))
        exif = Image.Exif()
        exif[0x0112] = 6                                                # rotate 90 clockwise to view
        img.save(path, exif=exif)

    entry, (image, _frames, _audio) = _load_image_file(store, "phone.jpg", make)
    thumb = Image.open(media.thumb_path_for(entry["id"]))
    assert (thumb.width, thumb.height) == (48, 64), "the thumbnail shows it upright"
    assert tuple(image.shape[1:3]) == (64, 48), "the model must see the same upright picture the bin shows"


@pytest.mark.parametrize("mode", ["RGBA", "L", "P", "LA", "CMYK", "I;16"])
def test_pictures_of_other_colour_modes_load_as_three_channel_pictures(store, mode):
    from PIL import Image

    def make(path):
        img = Image.new(mode, (64, 48))
        img.save(path.with_suffix(".tiff") if mode in ("CMYK", "I;16") else path)

    ext = ".tiff" if mode in ("CMYK", "I;16") else ".png"
    path = store / f"m{ext}"
    make(path)
    entry = media.save_upload(path.name, path.read_bytes())
    image, _frames, _audio = _load(entry["id"])
    assert image.shape[-1] == 3 and image.shape[1:3] == (48, 64)
