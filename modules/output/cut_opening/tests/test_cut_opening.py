import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def _cut(frames, n=48, fps=24.0, with_audio=True):
    from modules.output.cut_opening.nodes import FunPackCutOpening
    images = torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1).expand(n, 2, 2, 3)
    audio = {"waveform": torch.arange(2 * 44100, dtype=torch.float32).view(1, 1, -1),
             "sample_rate": 44100} if with_audio else None
    return FunPackCutOpening.execute(images, frames, fps, audio).result


def test_n_frames_off_the_front_and_the_same_time_off_the_sound():
    images, audio, status = _cut(12)
    assert images.shape[0] == 36 and float(images[0, 0, 0, 0]) == 12.0
    assert audio["waveform"].shape[-1] == 2 * 44100 - 22050      # 12 frames at 24 fps = 0.5 s
    assert float(audio["waveform"][0, 0, 0]) == 22050.0
    assert "cut 12 of 48" in status


def test_zero_is_untouched():
    images, audio, status = _cut(0)
    assert images.shape[0] == 48 and status == "nothing cut"


def test_cutting_everything_leaves_one_frame_and_says_so():
    images, _audio, status = _cut(500)
    assert images.shape[0] == 1 and "one frame has to stay" in status


def test_no_audio_is_fine():
    images, audio, _ = _cut(6, with_audio=False)
    assert images.shape[0] == 42 and audio is None
