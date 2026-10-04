import torch

from modules.conditioning.negative_erase import erase


def _cond(rows=6, dim=16, seed=0, tags=None):
    g = torch.Generator().manual_seed(seed)
    meta = {} if tags is None else {"minimax_token_tags": torch.tensor(tags)}
    return [[torch.randn(1, rows, dim, generator=g), meta]]


def test_projection_leaves_nothing_along_the_negative_at_full_strength():
    pos, neg = _cond(seed=1), _cond(rows=3, seed=2)
    unit = erase.direction(neg[0][0])
    out, note = erase.apply(pos, neg, 1.0, "project", renorm=False)
    assert float((out[0][0][0] @ unit).abs().max()) < 1e-5 and "project" in note


def test_a_word_at_right_angles_is_untouched():
    unit = torch.zeros(16); unit[0] = 1.0
    t = torch.zeros(1, 2, 16); t[0, :, 1] = 3.0
    assert torch.allclose(erase.erase(t, unit, 1.0, renorm=False), t)


def test_image_rows_are_not_touched_when_the_layout_says_which_are_text():
    pos = _cond(rows=4, tags=[0, 0, 1, 1])
    out, _ = erase.apply(pos, _cond(rows=3, seed=5), 1.0)
    assert torch.equal(out[0][0][0, :2], pos[0][0][0, :2]) and not torch.equal(out[0][0][0, 2:], pos[0][0][0, 2:])


def test_off_and_empty_say_so_and_change_nothing():
    pos = _cond()
    assert erase.apply(pos, _cond(), 0.0) == (pos, "off")
    out, note = erase.apply(pos, [[torch.zeros(1, 3, 16), {}]], 1.0)
    assert out is pos and "no usable direction" in note
    out, note = erase.apply(pos, [[torch.randn(1, 3, 8), {}]], 1.0)       # width mismatch
    assert out is pos and "could not be modified" in note


def test_a_word_that_was_the_negative_is_not_blown_up_by_the_size_restore():
    neg = _cond(rows=1, seed=3)
    unit = erase.direction(neg[0][0])
    t = (unit * 5.0 + 1e-4 * torch.randn(16)).reshape(1, 1, 16)
    out = erase.erase(t, unit, 1.0)
    assert float(out.norm()) <= float(t.norm()) * erase.MAX_GAIN + 1e-6
