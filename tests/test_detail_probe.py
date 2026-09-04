"""Detail probe: the three numbers that separate "sharper" from "grainier" from "different",
and the block-spec parser feeding the repeat toggle they exist to judge."""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _name in (
    "comfy", "comfy.k_diffusion", "comfy.k_diffusion.sampling",
    "comfy.model_sampling", "comfy.nested_tensor", "comfy.sample",
    "comfy.samplers", "comfy.utils",
):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["comfy.nested_tensor"].NestedTensor = object

import detail_probe as dp  # noqa: E402
import samplers  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(dp, "state_path", lambda key: str(tmp_path / f"{key}.pt"))


def _edged(h=32, w=32, seed=0):
    """A plane with real structure: a few hard edges, flat elsewhere."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(1, 1, h, w)
    x[:, :, :, w // 3:] = 1.0          # vertical edge
    x[:, :, h // 2:, :] += 0.5         # horizontal edge
    x += torch.rand(1, 1, h, w, generator=g) * 0.01
    return x


# --- the three numbers ----------------------------------------------------------

def test_identical_inputs_are_a_perfect_null():
    a = _edged()
    r = dp.compare(a, a.clone())
    assert r["detail"] == pytest.approx(1.0)
    assert r["structure"] == pytest.approx(1.0)


def test_sharpening_existing_edges_reads_as_edge_aligned_detail():
    """An unsharp-mask style boost: more high frequency, concentrated where edges already
    were, low frequencies untouched. This is the outcome the probe exists to recognise."""
    a = _edged()
    hf = dp._hf(a)
    b = a + 0.6 * torch.nn.functional.conv2d(
        a, dp._LAPLACIAN.view(1, 1, 3, 3), padding=1) * -1.0
    r = dp.compare(a, b)
    assert r["detail"] > 1.1               # finer structure appeared
    assert r["structure"] > 0.9            # same picture underneath
    assert r["edge_aligned"] > 0.5         # and it landed on the edges
    assert float(hf.mean()) > 0            # sanity: the baseline had edges at all


def test_uniform_noise_is_detail_without_edge_alignment():
    """Grain raises `detail` exactly like sharpening does -- `edge_aligned` is the only
    number that separates them, which is the reason it exists."""
    a = _edged()
    g = torch.Generator().manual_seed(1)
    b = a + torch.randn(a.shape, generator=g) * 0.05
    r = dp.compare(a, b)
    assert r["detail"] > 1.1               # looks like "more detail"
    assert r["edge_aligned"] < 0.35        # but it is spread everywhere


def test_a_different_picture_shows_up_as_lost_structure():
    a = _edged(seed=0)
    b = _edged(seed=1) * 0.3 + 0.7         # same kind of content, different low frequencies
    r = dp.compare(a, b)
    assert r["structure"] < 0.85


def test_blurring_reads_as_detail_lost():
    a = _edged()
    b = dp._lf(a)
    assert dp.compare(a, b)["detail"] < 1.0


def test_shape_mismatch_is_not_compared():
    assert dp.compare(_edged(32, 32), _edged(16, 16)) is None


def test_a_flat_baseline_is_refused_rather_than_dividing_by_zero():
    flat = torch.zeros(1, 1, 16, 16)
    assert dp.compare(flat, flat + 1.0) is None


def test_planes_pool_leading_dims_whatever_they_mean():
    """C/T ordering differs between model families; this needs spatial structure only, so it
    must not care which leading axis is which."""
    x = torch.randn(4, 7, 16, 16)
    assert dp._planes(x).shape == (28, 1, 16, 16)
    assert dp._planes(torch.randn(2, 3, 5, 16, 16)).shape == (30, 1, 16, 16)


def test_too_small_to_filter_is_none():
    assert dp._planes(torch.randn(2, 2)) is None


# --- pairing consecutive runs ---------------------------------------------------

def test_the_first_run_has_nothing_to_compare_against():
    assert dp.record("k", _edged(), label="no repeat") is None
    assert dp.rows("k") == []


def test_the_second_run_is_scored_against_the_first():
    dp.record("k", _edged(), label="no repeat")
    row = dp.record("k", _edged(), label="repeat 40")
    assert row is not None
    assert row["label_before"] == "no repeat" and row["label_after"] == "repeat 40"
    assert len(dp.rows("k")) == 1


def test_each_run_becomes_the_baseline_for_the_next():
    for i in range(3):
        dp.record("k", _edged(seed=i), label=f"run{i}")
    rows = dp.rows("k")
    assert [r["label_before"] for r in rows] == ["run0", "run1"]
    assert [r["label_after"] for r in rows] == ["run1", "run2"]


def test_a_resolution_change_is_not_an_ab():
    dp.record("k", _edged(32, 32), label="a")
    assert dp.record("k", _edged(16, 16), label="b") is None
    assert dp.rows("k") == []
    # ...but it still becomes the baseline, so the NEXT matching run pairs with it.
    assert dp.record("k", _edged(16, 16), label="c") is not None


def test_rows_are_capped():
    for i in range(dp.MAX_ROWS + 5):
        dp.record("k", _edged(seed=i % 3), label=f"r{i}")
    assert len(dp.rows("k")) == dp.MAX_ROWS


def test_clear_removes_everything():
    dp.record("k", _edged(), label="a")
    dp.record("k", _edged(), label="b")
    dp.clear_all("k")
    assert dp.rows("k") == []


def test_a_non_tensor_is_a_safe_no_op():
    assert dp.record("k", None) is None
    assert dp.record("", _edged()) is None


def test_corrupt_state_degrades_to_empty(tmp_path, monkeypatch):
    path = tmp_path / "junk.pt"
    path.write_bytes(b"not a torch file")
    monkeypatch.setattr(dp, "state_path", lambda key: str(path))
    assert dp.rows("k") == []


# --- the block-spec parser the repeat toggle reads ------------------------------

@pytest.mark.parametrize("spec,expected", [
    ("", set()),
    ("40", {40}),
    ("38-42", {38, 39, 40, 41, 42}),
    ("10,40,44", {10, 40, 44}),
    ("10, 40 , 44", {10, 40, 44}),          # whitespace
    ("42-38", {38, 39, 40, 41, 42}),        # reversed range
    ("48-99", {48, 49}),                    # clamped to the stack
    ("-5", set()),                          # negative: no leading-dash range
    ("99", set()),                          # out of range
    ("40,junk,41", {40, 41}),               # a typo costs that entry, not the run
    ("40-", set()),
])
def test_block_spec_parsing(spec, expected):
    assert samplers.FunPackLTXAVSceneChainSampler._parse_block_spec(spec, 50) == expected


# --- the repeat hook actually repeats -------------------------------------------

class _Model:
    def __init__(self):
        self.model_options = {}

    def clone(self):
        m = _Model()
        m.model_options = {k: dict(v) if isinstance(v, dict) else v
                           for k, v in self.model_options.items()}
        return m


def _repeat_hook(blocks, times):
    s = samplers.FunPackLTXAVSceneChainSampler()
    patched = s._install_block_repeat(_Model(), blocks, times)
    return patched.model_options["transformer_options"]["patches_replace"]["dit"]


def test_the_block_runs_once_more_per_extra_pass():
    calls = []

    def original(args):
        calls.append(args["img"].clone())
        return {"img": args["img"] + 1.0}

    dit = _repeat_hook({0}, 1)
    out = dit[("double_block", 0)]({"img": torch.zeros(2, 2)}, {"original_block": original})
    assert len(calls) == 2                       # ran twice
    assert torch.equal(out["img"], torch.full((2, 2), 2.0))
    assert torch.equal(calls[1], torch.ones(2, 2))  # 2nd pass saw the 1st pass's output


def test_extra_passes_scale():
    def original(args):
        return {"img": args["img"] + 1.0}

    dit = _repeat_hook({0}, 3)
    out = dit[("double_block", 0)]({"img": torch.zeros(1, 1)}, {"original_block": original})
    assert float(out["img"]) == 4.0              # 1 normal pass + 3 extra


def test_only_the_named_blocks_are_patched():
    dit = _repeat_hook({0, 3}, 1)
    assert set(b for _tag, b in dit) == {0, 3}


def test_no_blocks_is_an_untouched_model():
    s = samplers.FunPackLTXAVSceneChainSampler()
    m = _Model()
    assert s._install_block_repeat(m, set(), 1) is m


def test_the_callers_args_dict_is_not_mutated():
    """The block reads "img" out of args; leaking the intermediate back to the caller would
    hand whatever reads args after this hook a half-processed stream."""
    def original(args):
        return {"img": args["img"] + 1.0}

    dit = _repeat_hook({0}, 2)
    args = {"img": torch.zeros(1, 1), "t_emb": "sentinel"}
    dit[("double_block", 0)](args, {"original_block": original})
    assert float(args["img"]) == 0.0
    assert args["t_emb"] == "sentinel"


def test_an_existing_patch_at_the_same_block_is_repeated_not_dropped():
    calls = []

    def inner(args, extra):
        calls.append(True)
        return {"img": args["img"] + 1.0}

    s = samplers.FunPackLTXAVSceneChainSampler()
    m = _Model()
    m.model_options["transformer_options"] = {
        "patches_replace": {"dit": {("double_block", 0): inner}}}
    patched = s._install_block_repeat(m, {0}, 1)
    hook = patched.model_options["transformer_options"]["patches_replace"]["dit"][
        ("double_block", 0)]
    out = hook({"img": torch.zeros(1, 1)}, {"original_block": lambda a: {"img": a["img"]}})
    assert len(calls) == 2                       # the existing patch is what gets repeated
    assert float(out["img"]) == 2.0


# --- the seed is what separates "invalid comparison" from "real result" -----------
#
# A low `structure` means the low-frequency picture moved. That is EITHER two different
# generations (different seed -- a reference pins the subject, not the sample, so R2V on a
# new seed is a new shot) OR the same seed whose shot was moved by the change itself. The
# three numbers cannot tell those apart, so the seed is recorded rather than guessed at.

def test_same_seed_is_flagged_on_the_row():
    dp.record("k", _edged(), label="off", seed=1234)
    row = dp.record("k", _edged(), label="on", seed=1234)
    assert row["same_seed"] is True
    assert row["seed_before"] == 1234 and row["seed_after"] == 1234


def test_a_changed_seed_is_flagged_as_not_an_ab():
    dp.record("k", _edged(seed=0), label="off", seed=1)
    row = dp.record("k", _edged(seed=1), label="on", seed=2)
    assert row["same_seed"] is False


def test_a_missing_seed_is_unknown_not_a_match():
    """Old rows and callers that pass no seed must not be reported as same-seed -- that would
    silently upgrade an uncheckable comparison into a claimed A/B."""
    dp.record("k", _edged(), label="off")
    row = dp.record("k", _edged(), label="on")
    assert row["same_seed"] is False
    assert row["seed_before"] is None and row["seed_after"] is None


def test_the_seed_rides_along_to_the_next_pairing():
    dp.record("k", _edged(), label="a", seed=7)
    dp.record("k", _edged(), label="b", seed=7)
    row = dp.record("k", _edged(), label="c", seed=9)
    assert row["seed_before"] == 7 and row["seed_after"] == 9 and row["same_seed"] is False


# --- what the A/B actually WAS ----------------------------------------------------
#
# The probe cannot know which knob the user considers the variable, and a hand-picked label
# only ever tracked one of them (every row read "no repeat -> no repeat" while the real
# difference was elsewhere, or nowhere). So it diffs the run's scalar settings instead and
# the row names the difference itself.

def test_the_row_names_what_changed_between_the_two_runs():
    dp.record("k", _edged(), seed=1, settings={"seed": 1, "h3_block_repeat": "", "cfg": 1.0})
    row = dp.record("k", _edged(), seed=1, settings={"seed": 1, "h3_block_repeat": "40", "cfg": 1.0})
    assert row["changed"] == {"h3_block_repeat": ["", "40"]}


def test_identical_settings_report_no_change_which_is_the_noise_floor():
    """Two identical runs are not a wasted comparison -- they measure how much the generation
    varies on its own, and every other row is uninterpretable without that number."""
    cfg = {"seed": 5, "h3_block_repeat": ""}
    dp.record("k", _edged(), seed=5, settings=cfg)
    row = dp.record("k", _edged(), seed=5, settings=dict(cfg))
    assert row["changed"] == {}
    assert row["same_seed"] is True


def test_a_new_or_removed_setting_counts_as_changed():
    dp.record("k", _edged(), settings={"a": 1})
    row = dp.record("k", _edged(), settings={"a": 1, "b": 2})
    assert row["changed"] == {"b": [None, 2]}


def test_several_changes_are_all_reported():
    dp.record("k", _edged(), settings={"a": 1, "b": 1, "c": 1})
    row = dp.record("k", _edged(), settings={"a": 2, "b": 1, "c": 3})
    assert row["changed"] == {"a": [1, 2], "c": [1, 3]}


def test_no_settings_at_all_is_an_empty_diff_not_a_crash():
    dp.record("k", _edged())
    row = dp.record("k", _edged())
    assert row["changed"] == {}


def test_a_prompt_change_is_reported_and_not_treated_as_a_repeatability_check():
    """The prompt is not a scalar -- it arrives as conditioning -- so scalars alone called two
    runs of different text "nothing changed", and the resulting structure drop read as
    "generation is not reproducible" rather than "you changed the prompt"."""
    base = {"seed": 1, "cfg": 1.0, "conditioning": "(1,9,4096):0.01:0.5"}
    dp.record("k", _edged(seed=0), seed=1, settings=base)
    row = dp.record("k", _edged(seed=1), seed=1,
                    settings={**base, "conditioning": "(1,12,4096):0.03:0.7"})
    assert "conditioning" in row["changed"]
    assert row["same_seed"] is True          # same seed, but NOT the same generation


# --- conditioning: drift vs a genuinely different prompt --------------------------

def test_float_wobble_in_the_encoder_is_not_a_prompt_change():
    """The first version compared a formatted string, so any float noise read as "prompt
    changed" — on runs where the prompt was demonstrably identical."""
    a = {"conditioning": [[4096.0, 0.010000, 0.500000]]}
    b = {"conditioning": [[4096.0, 0.010000001, 0.500000002]]}
    dp.record("k", _edged(seed=0), settings=a)
    row = dp.record("k", _edged(seed=0), settings=b)
    assert "conditioning" not in row["changed"]
    assert row["cond_shift"] < dp.COND_TOL


def test_a_real_drift_is_reported_with_its_size():
    a = {"conditioning": [[4096.0, 0.100, 0.500]]}
    b = {"conditioning": [[4096.0, 0.102, 0.500]]}
    dp.record("k", _edged(seed=0), settings=a)
    row = dp.record("k", _edged(seed=0), settings=b)
    assert "conditioning" in row["changed"]
    # Relative to the LARGER magnitude: |0.102-0.100| / 0.102. The point is that the row
    # carries a size at all, so drift can be told from a prompt swap.
    assert row["cond_shift"] == pytest.approx(0.002 / 0.102, rel=1e-6)


def test_a_different_token_count_is_a_different_prompt_not_drift():
    a = {"conditioning": [[4096.0, 0.1, 0.5]]}
    b = {"conditioning": [[8192.0, 0.1, 0.5]]}
    dp.record("k", _edged(seed=0), settings=a)
    row = dp.record("k", _edged(seed=0), settings=b)
    assert row["cond_shift"] == 1.0


def test_a_different_number_of_scenes_is_structural():
    a = {"conditioning": [[4096.0, 0.1, 0.5]]}
    b = {"conditioning": [[4096.0, 0.1, 0.5], [4096.0, 0.2, 0.6]]}
    dp.record("k", _edged(seed=0), settings=a)
    row = dp.record("k", _edged(seed=0), settings=b)
    assert row["cond_shift"] == 1.0


def test_identical_conditioning_shifts_nothing():
    cond = {"conditioning": [[4096.0, 0.1, 0.5], [2048.0, -0.3, 0.9]]}
    dp.record("k", _edged(seed=0), settings=cond)
    row = dp.record("k", _edged(seed=0), settings=dict(cond))
    assert row["cond_shift"] == 0.0
    assert row["changed"] == {}


def test_the_detail_probe_switch_is_its_own(monkeypatch, tmp_path):
    """Deliberately NOT shared with the block probe. They cost wildly different amounts —
    this copies one latent per run, that gathers every block on every step — and answering
    "is the measurement moving the picture?" requires running this one while that one is off.
    A shared toggle made that experiment impossible."""
    import block_influence as bi
    monkeypatch.delenv(dp._ENV_SWITCH, raising=False)
    monkeypatch.delenv(bi._ENV_SWITCH, raising=False)
    monkeypatch.setattr(dp, "_switch_path", lambda: str(tmp_path / "dp" / "enabled"))
    monkeypatch.setattr(bi, "_switch_dir", lambda: str(tmp_path / "bi"))

    bi.set_collection_enabled(True)
    assert dp.collection_enabled() is False       # not dragged along by the other one
    dp.set_collection_enabled(True)
    bi.set_collection_enabled(False)
    assert dp.collection_enabled() is True        # and not switched off by it either


def test_the_detail_switch_persists_and_env_wins(monkeypatch, tmp_path):
    monkeypatch.delenv(dp._ENV_SWITCH, raising=False)
    monkeypatch.setattr(dp, "_switch_path", lambda: str(tmp_path / "dp" / "enabled"))
    assert dp.collection_enabled() is False
    dp.set_collection_enabled(True)
    monkeypatch.delenv(dp._ENV_SWITCH, raising=False)
    assert dp.collection_enabled() is True        # from disk, after a restart
    monkeypatch.setenv(dp._ENV_SWITCH, "0")
    assert dp.collection_enabled() is False       # env overrides
