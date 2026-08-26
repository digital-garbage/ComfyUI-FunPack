"""Prompt timestep: the prompt is told how finished it is, independently of the picture.

H3 tells every row of its packed sequence what noise level it is at, and it gives the text
rows the video's. Early in a run that means the prompt is treated as unfinished too. This
gives the text rows a level of their own by putting ONE extra row through each block's
AdaLN projection and copying its tag-1 modulation over the text rows — the model's own row
bookkeeping never sees it, because the extra row is dropped before the result comes back.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import minimax_h3 as h3

MOD = 3
EXPAND = 6
HIDDEN = 4


class RowProj(torch.nn.Module):
    """AdalnProj stand-in whose every output row reports (timestep, modality).

    Row value is `t * 100 + tag`, so which timestep a modality was modulated at is readable
    straight off the tensor.
    """

    def __init__(self, modalities=MOD, expand=EXPAND, hidden=HIDDEN):
        super().__init__()
        self.modalities = modalities
        self.expand = expand
        self.hidden = hidden

    def forward(self, t_emb):
        rows = int(t_emb.shape[0])
        base = t_emb[:, 0].repeat_interleave(self.modalities) * 100.0
        tags = torch.arange(self.modalities, dtype=torch.float32).repeat(rows)
        out = (base + tags).unsqueeze(1).expand(
            rows * self.modalities, self.expand * self.hidden).contiguous()
        return out.chunk(self.expand, dim=-1)


def t_emb(*values):
    return torch.tensor(values, dtype=torch.float32).unsqueeze(1)


def row_values(chunk):
    return chunk[:, 0].tolist()


def constant_row(value):
    return lambda like: torch.tensor([[value]], dtype=like.dtype, device=like.device)


# ── the edit itself ─────────────────────────────────────────────────────────

def test_the_prompt_is_modulated_at_its_own_timestep():
    inner = RowProj()
    wrapped = h3.AdalnEdit(inner, None, prompt_row=constant_row(0.95))
    out = wrapped(t_emb(0.2, 0.7))
    # rows are t_row*3 + tag: video 0/3, text 1/4, audio 2/5. The fake reports t*100+tag,
    # so 96 proves the copy came from the extra timestep's TEXT row, not its video row.
    assert row_values(out[0]) == [20.0, 96.0, 22.0, 70.0, 96.0, 72.0]


def test_every_other_modality_keeps_the_timestep_it_had():
    inner = RowProj()
    plain = row_values(inner(t_emb(0.2, 0.7))[0])
    out = row_values(h3.AdalnEdit(inner, None, prompt_row=constant_row(0.95))(t_emb(0.2, 0.7))[0])
    for row in (0, 2, 3, 5):
        assert out[row] == plain[row]


def test_the_extra_row_never_reaches_the_caller():
    """The model indexes modulation as t_row*3+tag off its OWN row count. An extra row left
    on the end would not break those indices, but a changed row COUNT would be a lie about
    how many timesteps are in play."""
    out = h3.AdalnEdit(RowProj(), None, prompt_row=constant_row(0.9))(t_emb(0.2, 0.7))
    assert all(chunk.shape[0] == 2 * MOD for chunk in out)


def test_all_six_modulation_vectors_move_together():
    """shift/scale/gate for attention and for the MLP all belong to one timestep — moving
    some and not others would describe a state the model was never trained on."""
    out = h3.AdalnEdit(RowProj(), None, prompt_row=constant_row(0.95))(t_emb(0.2))
    assert all(row_values(chunk)[1] == 96.0 for chunk in out)


def test_off_by_default_leaves_the_projection_alone():
    inner = RowProj()
    out = h3.AdalnEdit(inner, None)(t_emb(0.2, 0.7))
    assert all(torch.equal(a, b) for a, b in zip(out, inner(t_emb(0.2, 0.7))))


def test_a_single_modality_projection_is_left_alone():
    """FinalLayer shares AdalnProj with modalities=1, where tag 1 means nothing."""
    inner = RowProj(modalities=1, expand=2)
    wrapped = h3.AdalnEdit(inner, None, prompt_row=constant_row(0.9))
    assert all(torch.equal(a, b) for a, b in zip(wrapped(t_emb(0.2)), inner(t_emb(0.2))))


def test_a_failed_embedding_renders_the_prompt_as_trained():
    """Never a hard failure: the run continues with the model's own behaviour."""
    def boom(_like):
        raise RuntimeError("no time basis")
    inner = RowProj()
    out = h3.AdalnEdit(inner, None, prompt_row=boom)(t_emb(0.2))
    assert all(torch.equal(a, b) for a, b in zip(out, inner(t_emb(0.2))))


def test_it_composes_with_the_modality_gains():
    """Both edits ride one wrapper — installing two on the same key would nest, and
    unwrapped_forward deliberately strips a FunPack wrapper rather than stacking on it."""
    wrapped = h3.AdalnEdit(RowProj(), {h3.MODALITY_TAGS["text"]: 2.0},
                           prompt_row=constant_row(0.95))
    gate = row_values(wrapped(t_emb(0.2))[2])
    assert gate[1] == 96.0 * 2.0     # re-timed AND gained
    assert gate[0] == 20.0           # video untouched by both


# ── the timestep row ────────────────────────────────────────────────────────

def test_the_curve_table_is_read_the_way_the_model_reads_it():
    table = torch.tensor([[0.0, 0.0], [10.0, 20.0], [20.0, 40.0]], dtype=torch.float32)
    row = h3.PromptTimestepRow(0.5, table=table)(torch.zeros(1, 2))
    assert torch.allclose(row, torch.tensor([[10.0, 20.0]]))    # t=0.5 -> grid row 1


def test_a_fractional_position_interpolates():
    table = torch.tensor([[0.0, 0.0], [10.0, 20.0]], dtype=torch.float32)
    row = h3.PromptTimestepRow(0.25, table=table)(torch.zeros(1, 2))
    assert torch.allclose(row, torch.tensor([[2.5, 5.0]]))


def test_the_top_of_the_range_stays_on_the_last_interval():
    """t=1.0 must read the final grid row, not one past the end of the table."""
    table = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
    assert torch.allclose(h3.PromptTimestepRow(1.0, table=table)(torch.zeros(1, 1)),
                          torch.tensor([[2.0]]))


def test_a_checkpoint_without_curves_uses_its_time_embedder():
    class Embedder:
        def __call__(self, value):
            return torch.stack([value, value * 2.0], dim=-1)
    row = h3.PromptTimestepRow(0.4, embedder=Embedder())(torch.zeros(1, 2))
    assert torch.allclose(row, torch.tensor([[0.4, 0.8]]))


def test_the_row_is_cast_to_whatever_the_live_embedding_is_using():
    table = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    row = h3.PromptTimestepRow(0.0, table=table)(torch.zeros(1, 1, dtype=torch.bfloat16))
    assert row.dtype == torch.bfloat16


def test_the_row_is_computed_once_and_reused():
    """All 50 blocks ask for the same row every step."""
    calls = []

    class Counting:
        def __call__(self, value):
            calls.append(1)
            return value.unsqueeze(-1)
    provider = h3.PromptTimestepRow(0.5, embedder=Counting())
    like = torch.zeros(1, 1)
    for _ in range(5):
        provider(like)
    assert len(calls) == 1


# ── installing it ───────────────────────────────────────────────────────────

class FakeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adaln_proj = RowProj()


class FakePatcher:
    def __init__(self, n=3, table=torch.tensor([[0.0], [1.0]])):
        self.objects = {"diffusion_model.blocks": [FakeBlock() for _ in range(n)]}
        for i in range(n):
            self.objects[f"diffusion_model.blocks.{i}.adaln_proj"] = \
                self.objects["diffusion_model.blocks"][i].adaln_proj
        if table is not None:
            self.objects["diffusion_model.adaln_t_table"] = table
        self.patched = {}

    def get_model_object(self, name):
        if name in self.patched:
            return self.patched[name]
        return self.objects[name]

    def clone(self):
        other = FakePatcher.__new__(FakePatcher)
        other.objects = self.objects
        other.patched = dict(self.patched)
        return other

    def add_object_patch(self, name, obj):
        self.patched[name] = obj


def test_zero_changes_nothing_and_does_not_clone():
    model = FakePatcher()
    out, note = h3.apply_adaln_edits(model, None, prompt_timestep=0.0)
    assert out is model and note is None and model.patched == {}


def test_a_timestep_alone_is_enough_to_install():
    model = FakePatcher(n=3)
    out, note = h3.apply_adaln_edits(model, None, prompt_timestep=0.9)
    assert len(out.patched) == 3
    assert all(v.prompt_row is not None for v in out.patched.values())
    assert "0.9" in note


def test_every_block_shares_one_row_provider():
    out, _ = h3.apply_adaln_edits(FakePatcher(n=3), None, prompt_timestep=0.9)
    providers = {id(v.prompt_row) for v in out.patched.values()}
    assert len(providers) == 1


def test_a_checkpoint_with_no_time_basis_declines_with_a_reason():
    model = FakePatcher(table=None)
    out, note = h3.apply_adaln_edits(model, None, prompt_timestep=0.9)
    assert out is model
    assert "no time embedding" in note


def test_the_gains_still_apply_when_the_timestep_cannot_be_read():
    """A missing time basis must not silently cancel the edit the user also asked for."""
    model = FakePatcher(table=None)
    out, note = h3.apply_adaln_edits(model, {"video": 0.8}, prompt_timestep=0.9)
    assert out is not model and len(out.patched) == 3
    assert "Prompt timestep skipped" in note
