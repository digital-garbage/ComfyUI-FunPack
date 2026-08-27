"""A keyframe pin can cover PART of a frame instead of all of it.

A whole-frame pin says "frame k looks exactly like this". A region lock says "this part of
frame k looks like this, and the rest is yours to invent" — which is what an i2v anchor with
a transparent background actually means.

The mechanism is row DELETION, not blanking or noising: a blanked row still asserts a colour
and a noised one still occupies a position the picture attends to, while a row that is not in
the sequence says nothing at all.

Two patches have to agree exactly on which rows survive — the layout drops them from the
packed sequence, ``_cond_video_rows`` drops them from the condition latents — so most of what
is tested here is that they cannot diverge.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import minimax_h3 as h3

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
ROWS = 6                        # one latent frame's patch rows in the fake grid


def _spans(n):
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)]


class _FakePackedLayout:
    """Upstream's structure: text, cond rows, audio, video — with every derived field."""

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t,
                 keyframes=None, refs=None, frame_count=None):
        segments, pos = [("text", text_len)], [torch.zeros(text_len, 3, dtype=torch.float64)]
        img_pos, img_update, audio_pos, audio_update = [], [], [], []
        row = text_len
        for index, kf in enumerate(keyframes or ()):
            pixel_index = kf["resolved_frame_index"]
            if pixel_index == 0:
                cond_t = float(text_len)
            elif frame_count is not None and pixel_index == frame_count - 1:
                cond_t = float(text_len) + sum(_spans(latent_t)) - FRAME_RESCALE
            else:
                raise ValueError("only first/last keyframe anchors are supported")
            g = torch.zeros(ROWS, 3, dtype=torch.float64)
            g[:, 0] = cond_t
            g[:, 1] = torch.arange(ROWS, dtype=torch.float64)   # identifies each row
            g[:, 2] = float(index)
            segments.append(("cond", ROWS))
            pos.append(g)
            img_pos.append(torch.arange(row, row + ROWS))
            img_update.append(torch.zeros(ROWS, dtype=torch.bool))
            row += ROWS
        segments.append(("audio", audio_t * 2))
        pos.append(torch.full((audio_t * 2, 3), 5.0, dtype=torch.float64))
        audio_pos.append(torch.arange(row, row + audio_t * 2))
        audio_update.append(torch.ones(audio_t * 2, dtype=torch.bool))
        row += audio_t * 2
        n_video = latent_t * ROWS
        segments.append(("video", n_video))
        pos.append(torch.full((n_video, 3), 9.0, dtype=torch.float64))
        img_pos.append(torch.arange(row, row + n_video))
        img_update.append(torch.ones(n_video, dtype=torch.bool))
        row += n_video

        self.seq_len = row
        self.position_ids = torch.cat(pos)
        self.img_pos = torch.cat(img_pos)
        self.img_update = torch.cat(img_update)
        self.audio_pos = torch.cat(audio_pos)
        self.audio_update = torch.cat(audio_update)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        seg_abs, off = [], 0
        for kind, n in segments:
            seg_abs.append((off, off + n, kind))
            off += n
        self.segments = seg_abs


class _FakeModel:
    """Upstream's ``_cond_video_rows``: patchify each cond latent, concatenate, in order."""

    def _cond_video_rows(self, payload, device):
        rows = []
        for z in payload.get("cond_video_latents", []):
            b, c, t, hh, ww = z.shape
            n = b * t * (hh // 2) * (ww // 2)
            rows.append(z.reshape(-1)[:n].reshape(n, 1).repeat(1, 3))
        return torch.cat(rows, dim=0) if rows else None


@pytest.fixture
def upstream(monkeypatch):
    mod = types.ModuleType("comfy.ldm.minimax.model")
    mod.PackedLayout = type("PackedLayout", (_FakePackedLayout,), {})
    mod.MiniMaxH3Model = type("MiniMaxH3Model", (_FakeModel,), {})
    mod.FRAME_PER_TOKEN = FRAME_PER_TOKEN
    mod.FRAME_RESCALE = FRAME_RESCALE
    mod._video_t_spans = _spans
    for name in ("comfy.ldm.minimax", "comfy.ldm.minimax.model"):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setitem(h3._INTERIOR_PINS, "state", None)
    monkeypatch.setitem(h3._REGION_LOCKS, "state", None)
    monkeypatch.setitem(h3._LAYOUT_PATCH, "state", None)
    return mod


def _mask(*keep):
    return torch.tensor([bool(k) for k in keep])


def _pin(index=0, region=None):
    pin = {"resolved_frame_index": index, "latent": None}
    if region is not None:
        pin[h3.REGION_META] = region
    return pin


def _build(mod, keyframes, text_len=4, latent_t=2, audio_t=2):
    return mod.PackedLayout(text_len, latent_t, 4, 4, audio_t, keyframes=keyframes,
                            frame_count=sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t)))


def _rows(layout, kind):
    return sum(b - a for a, b, k in layout.segments if k == kind)


# ── the keep-mask decision, shared by both patches ───────────────────────────

def test_no_region_keeps_every_row():
    assert h3._region_keep(None, ROWS) is None


def test_a_full_mask_is_not_a_restriction():
    """All-True must be indistinguishable from having no region at all."""
    assert h3._region_keep(_mask(1, 1, 1, 1, 1, 1), ROWS) is None


def test_an_empty_mask_is_refused():
    """A pin with nothing left is a pin that should not have been made."""
    assert h3._region_keep(_mask(0, 0, 0, 0, 0, 0), ROWS) is None


def test_a_wrong_length_mask_is_refused():
    """The two patches must never act on a mask only one of them can apply."""
    assert h3._region_keep(_mask(1, 0, 1), ROWS) is None


def test_a_partial_mask_is_taken_as_given():
    kept = h3._region_keep(_mask(1, 0, 1, 0, 0, 1), ROWS)
    assert kept.tolist() == [True, False, True, False, False, True]


# ── alpha as the region source ───────────────────────────────────────────────

def _rgba(alpha):
    image = torch.ones(1, 4, 4, 4)
    image[..., 3] = torch.tensor(alpha, dtype=torch.float32)
    return image


def test_a_fully_opaque_image_declares_no_region():
    """The overwhelmingly common case must behave exactly as it did before."""
    assert h3.region_rows_from_alpha(_rgba([[1.0] * 4] * 4), 2, 2) is None


def test_a_fully_transparent_image_declares_no_region():
    assert h3.region_rows_from_alpha(_rgba([[0.0] * 4] * 4), 2, 2) is None


def test_a_three_channel_image_declares_no_region():
    assert h3.region_rows_from_alpha(torch.ones(1, 4, 4, 3), 2, 2) is None


def test_alpha_pools_to_the_patch_grid():
    alpha = [[1.0, 1.0, 0.0, 0.0],
             [1.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0]]
    assert h3.region_rows_from_alpha(_rgba(alpha), 2, 2).tolist() == [True, False, False, False]


def test_the_mask_is_raster_ordered_like_the_patch_rows():
    """Row-major over the patch grid — the same order patchify_video emits."""
    alpha = [[0.0, 0.0, 1.0, 1.0],
             [0.0, 0.0, 1.0, 1.0],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0]]
    assert h3.region_rows_from_alpha(_rgba(alpha), 2, 2).tolist() == [False, True, False, False]


# ── the layout patch ─────────────────────────────────────────────────────────

def test_a_pin_without_a_region_is_untouched(upstream):
    assert h3.install_region_locks() is True
    layout = _build(upstream, [_pin(0)])
    assert _rows(layout, "cond") == ROWS
    assert layout.seq_len == 4 + ROWS + 4 + 2 * ROWS


def test_a_region_drops_exactly_the_excluded_rows(upstream):
    assert h3.install_region_locks() is True
    layout = _build(upstream, [_pin(0, _mask(1, 0, 1, 0, 0, 1))])
    assert _rows(layout, "cond") == 3
    assert layout.position_ids[4:7, 1].tolist() == [0.0, 2.0, 5.0]


def test_the_sequence_shortens_by_what_was_dropped(upstream):
    assert h3.install_region_locks() is True
    full = _build(upstream, [_pin(0)])
    partial = _build(upstream, [_pin(0, _mask(1, 0, 1, 0, 0, 1))])
    assert partial.seq_len == full.seq_len - 3


def test_the_other_segments_keep_all_their_rows(upstream):
    """Only condition rows are ever removed. Losing a video row would lose output."""
    assert h3.install_region_locks() is True
    layout = _build(upstream, [_pin(0, _mask(1, 0, 0, 0, 0, 0))])
    assert _rows(layout, "video") == 2 * ROWS
    assert _rows(layout, "audio") == 4
    assert _rows(layout, "text") == 4


def test_the_segment_table_stays_contiguous(upstream):
    assert h3.install_region_locks() is True
    layout = _build(upstream, [_pin(0, _mask(1, 1, 0, 0, 0, 0))])
    offset = 0
    for a, b, _kind in layout.segments:
        assert a == offset
        offset = b
    assert offset == layout.seq_len == layout.position_ids.shape[0]


def test_the_update_masks_follow_the_surviving_rows(upstream):
    """img_update drives which rows get condition content and which get the noised latent —
    a stale one would write the video's rows into the condition's slots."""
    assert h3.install_region_locks() is True
    layout = _build(upstream, [_pin(0, _mask(1, 0, 1, 0, 0, 1))])
    assert layout.img_update.shape[0] == 3 + 2 * ROWS
    assert layout.img_update[:3].tolist() == [False, False, False]
    assert layout.img_update[3:].all()


def test_the_row_indices_are_renumbered_into_the_new_sequence(upstream):
    assert h3.install_region_locks() is True
    layout = _build(upstream, [_pin(0, _mask(1, 0, 1, 0, 0, 1))])
    assert layout.img_pos.max().item() < layout.seq_len
    assert layout.audio_pos.max().item() < layout.seq_len
    assert layout.img_pos.tolist() == sorted(layout.img_pos.tolist())


def test_two_pins_carry_independent_regions(upstream):
    assert h3.install_region_locks() is True
    layout = _build(upstream, [_pin(0, _mask(1, 1, 0, 0, 0, 0)), _pin(4, _mask(0, 0, 0, 0, 1, 1))])
    spans = [(a, b) for a, b, k in layout.segments if k == "cond"]
    assert [b - a for a, b in spans] == [2, 2]
    assert layout.position_ids[spans[0][0]:spans[0][1], 1].tolist() == [0.0, 1.0]
    assert layout.position_ids[spans[1][0]:spans[1][1], 1].tolist() == [4.0, 5.0]


def test_a_region_survives_alongside_an_interior_pin(upstream):
    """The two rewrites share one wrapper and must both land."""
    assert h3.install_interior_keyframes() is True
    assert h3.install_region_locks() is True
    layout = _build(upstream, [_pin(3, _mask(1, 0, 1, 0, 0, 0))], latent_t=6)
    start = next(a for a, _b, k in layout.segments if k == "cond")
    assert _rows(layout, "cond") == 2
    assert float(layout.position_ids[start, 0]) == pytest.approx(h3.keyframe_cond_t(4, 3))


def test_regions_stay_off_until_installed(upstream):
    """Region locks must not ride in on the interior-pin patch."""
    assert h3.install_interior_keyframes() is True
    layout = _build(upstream, [_pin(0, _mask(1, 0, 0, 0, 0, 0))])
    assert _rows(layout, "cond") == ROWS


def test_a_comfy_without_h3_declines_quietly(monkeypatch, capsys):
    monkeypatch.setitem(h3._REGION_LOCKS, "state", None)
    monkeypatch.setitem(h3._LAYOUT_PATCH, "state", None)
    monkeypatch.setitem(sys.modules, "comfy.ldm.minimax.model", None)
    assert h3.install_region_locks() is False
    assert "unavailable" in capsys.readouterr().out


# ── the condition-rows patch ─────────────────────────────────────────────────

def _payload(latents, pins, refs=None):
    payload = {"cond_video_latents": latents, "keyframes": pins}
    if refs is not None:
        payload["refs"] = refs
    return payload


def _latent(value):
    return torch.full((1, 1, 1, 4, 6), float(value))   # -> 2*3 = 6 patch rows


def test_the_condition_rows_drop_the_same_rows_as_the_layout(upstream):
    assert h3.install_region_locks() is True
    model = upstream.MiniMaxH3Model()
    mask = _mask(1, 0, 1, 0, 0, 1)
    rows = model._cond_video_rows(_payload([_latent(1)], [_pin(0, mask)]), "cpu")
    layout = _build(upstream, [_pin(0, mask)])
    assert rows.shape[0] == _rows(layout, "cond") == 3


def test_the_condition_rows_are_untouched_without_a_region(upstream):
    assert h3.install_region_locks() is True
    model = upstream.MiniMaxH3Model()
    rows = model._cond_video_rows(_payload([_latent(1)], [_pin(0)]), "cpu")
    assert rows.shape[0] == ROWS


def test_each_pin_is_sliced_with_its_own_mask(upstream):
    """Two pins, two different masks: the second pin's rows must not be cut by the first."""
    assert h3.install_region_locks() is True
    model = upstream.MiniMaxH3Model()
    pins = [_pin(0, _mask(1, 1, 0, 0, 0, 0)), _pin(4, _mask(0, 0, 0, 1, 1, 1))]
    rows = model._cond_video_rows(_payload([_latent(1), _latent(2)], pins), "cpu")
    assert rows.shape[0] == 5
    assert rows[:, 0].tolist() == [1.0, 1.0, 2.0, 2.0, 2.0]


def test_references_disable_pin_regions_on_the_rows(upstream):
    """With refs present upstream rebuilds cond_video_latents from the REFS, so the pins no
    longer index the rows and slicing by them would cut the wrong content."""
    assert h3.install_region_locks() is True
    model = upstream.MiniMaxH3Model()
    payload = _payload([_latent(1)], [_pin(0, _mask(1, 0, 0, 0, 0, 0))], refs=[{"kind": "image"}])
    assert model._cond_video_rows(payload, "cpu").shape[0] == ROWS


def test_a_pin_count_mismatch_changes_nothing(upstream):
    assert h3.install_region_locks() is True
    model = upstream.MiniMaxH3Model()
    payload = _payload([_latent(1), _latent(2)], [_pin(0, _mask(1, 0, 0, 0, 0, 0))])
    assert model._cond_video_rows(payload, "cpu").shape[0] == 2 * ROWS


def test_an_empty_payload_stays_empty(upstream):
    assert h3.install_region_locks() is True
    model = upstream.MiniMaxH3Model()
    assert model._cond_video_rows(_payload([], []), "cpu") is None


def test_installing_twice_does_not_stack_patches(upstream):
    assert h3.install_region_locks() is True
    h3._REGION_LOCKS["state"] = None
    assert h3.install_region_locks() is True
    model = upstream.MiniMaxH3Model()
    rows = model._cond_video_rows(_payload([_latent(1)], [_pin(0, _mask(1, 0, 1, 0, 0, 1))]), "cpu")
    assert rows.shape[0] == 3
