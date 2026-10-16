import torch

from core import patching


def _install(tiny, scale):
    from modules.sampling.audio_coupling import install
    patched = tiny.patcher.clone()
    note = install(patching.GuardedPatcher(patched, "funpack.audio_coupling", patching.Dropped()),
                   {"scale": scale}, key="funpack.audio_coupling")
    return patched, note


def _forward(tiny, patched):
    # object patches are applied when the model is loaded for sampling
    patched.patch_model(device_to=torch.device("cpu"), load_weights=False)
    try:
        return tiny.run(patched)
    finally:
        patched.unpatch_model(unpatch_weights=False)


def test_at_one_nothing_is_installed(tiny_ltx):
    patched, note = _install(tiny_ltx, 1.0)
    assert note is None and not patched.object_patches


def test_scaling_the_link_changes_the_sound_and_zero_changes_it_more(tiny_ltx):
    base_v, base_a = tiny_ltx.run()
    patched, note = _install(tiny_ltx, 3.0)
    assert "3x across 4 blocks" in note
    v, a = _forward(tiny_ltx, patched)
    assert not torch.allclose(a, base_a)
    muted, _ = _install(tiny_ltx, 0.0)
    _, a0 = _forward(tiny_ltx, muted)
    assert (a0 - base_a).abs().mean() > 0


def test_it_comes_off_again(tiny_ltx):
    patched, _ = _install(tiny_ltx, 2.0)
    assert patched.object_patches
    patching.strip(patched, "funpack.audio_coupling")
    assert not patched.object_patches
    _, a = _forward(tiny_ltx, patched)
    assert torch.allclose(a, tiny_ltx.run()[1])
