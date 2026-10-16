"""The fixture itself: a real LTX model runs, and a block replacement is honoured."""

import torch

from core import dit_hooks


def test_a_real_ltx_forward_runs(tiny_ltx):
    video, audio = tiny_ltx.run()
    assert video.shape == tiny_ltx.video.shape and audio.shape == tiny_ltx.audio.shape


def test_a_block_replacement_sees_the_video_and_audio_pair(tiny_ltx):
    seen = []

    def hook(args, extra):
        seen.append(type(args["img"]).__name__ + str(len(args["img"])))
        return extra["original_block"](args)

    dit_hooks.add_block_hook(tiny_ltx.patcher, "funpack.test", 1, hook)
    tiny_ltx.run()
    assert seen == ["tuple2"]
