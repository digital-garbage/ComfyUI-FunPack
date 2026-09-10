"""REINS steer_block went from a single combo choice to free text ("31-40", "4,5,6"),
each named block steering with its OWN learned direction. Exercised at the install level:
does the resulting hook actually inject at every named block and NOWHERE else."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, ".")
import _comfy_stubs  # noqa: E402, F401
from samplers import FunPackLTXAVSceneChainSampler as S  # noqa: E402
import h3_repr_steering as rs  # noqa: E402


class _FakeModel:
    def __init__(self):
        self.model_options = {}

    def clone(self):
        m = _FakeModel()
        m.model_options = dict(self.model_options)
        return m


def _run_block(patched, block, seq_len=4):
    """Invoke the installed hook for `block` and return the resulting "img" tensor."""
    dit = patched.model_options["transformer_options"]["patches_replace"]["dit"]
    hook = dit[("double_block", block)]
    img = torch.ones(seq_len, 3)
    args = {"img": img, "mod_segments": [(0, seq_len, 0)]}  # all rows tagged video
    extra = {"original_block": lambda a: {"img": a["img"]}}
    return hook(args, extra)["img"]


def test_multiple_named_blocks_each_steer_with_their_own_direction(monkeypatch):
    directions = {5: torch.tensor([1.0, 0.0, 0.0]), 10: torch.tensor([0.0, 1.0, 0.0])}

    def fake_direction(_key, block=None):
        d = directions.get(block)
        return (d, 5, 5) if d is not None else (None, 0, 0)

    monkeypatch.setattr(rs, "direction", fake_direction)
    node = S()
    patched = node._install_h3_repr_steering(
        _FakeModel(), "key", strength=1.0, capture_holder=[{}], steer_block="5,10")

    out5 = _run_block(patched, 5)
    out10 = _run_block(patched, 10)
    out6 = _run_block(patched, 6)  # captured, never steered

    assert not torch.allclose(out5, torch.ones(4, 3)), "block 5 should have been steered"
    assert not torch.allclose(out10, torch.ones(4, 3)), "block 10 should have been steered"
    assert torch.allclose(out6, torch.ones(4, 3)), "an unnamed block must stay untouched"
    # Different learned directions -> different injected deltas.
    assert not torch.allclose(out5, out10)


def test_a_range_steers_every_block_in_it(monkeypatch):
    monkeypatch.setattr(rs, "direction", lambda _k, block=None: (torch.tensor([1.0, 0.0, 0.0]), 5, 5))
    node = S()
    patched = node._install_h3_repr_steering(
        _FakeModel(), "key", strength=1.0, capture_holder=[{}], steer_block="7-9")

    for b in (7, 8, 9):
        assert not torch.allclose(_run_block(patched, b), torch.ones(4, 3)), b
    assert torch.allclose(_run_block(patched, 6), torch.ones(4, 3))
    assert torch.allclose(_run_block(patched, 10), torch.ones(4, 3))


def test_a_block_without_enough_rated_data_only_captures(monkeypatch):
    monkeypatch.setattr(rs, "direction", lambda _k, block=None: (None, 1, 0))
    node = S()
    capture_holder = [{}]
    patched = node._install_h3_repr_steering(
        _FakeModel(), "key", strength=1.0, capture_holder=capture_holder, steer_block="5")

    out = _run_block(patched, 5)
    assert torch.allclose(out, torch.ones(4, 3)), "no direction yet -- must not inject"
    assert 5 in capture_holder[0], "capture should still happen regardless of steering"


def test_empty_steer_block_falls_back_to_default_block():
    node = S()
    patched = node._install_h3_repr_steering(
        _FakeModel(), "", strength=1.0, capture_holder=[{}], steer_block="")
    dit = patched.model_options["transformer_options"]["patches_replace"]["dit"]
    assert ("double_block", rs.DEFAULT_BLOCK) in dit


def test_strength_zero_does_not_claim_to_be_applying(monkeypatch, capsys):
    """passive_capture (see _sample_chunk) calls this with strength forced to 0.0 while a
    real learned direction may already exist -- the console must not say "applying" when
    nothing was actually injected (the same trap already fixed for h3_q_steering's version
    of this print)."""
    monkeypatch.setattr(rs, "direction", lambda _k, block=None: (torch.tensor([1.0, 0.0, 0.0]), 5, 5))
    node = S()
    patched = node._install_h3_repr_steering(
        _FakeModel(), "key", strength=0.0, capture_holder=[{}], steer_block="5")

    out = _run_block(patched, 5)
    assert torch.allclose(out, torch.ones(4, 3)), "strength 0 -> no injection despite a real direction"
    stdout = capsys.readouterr().out
    assert "applying learned direction" not in stdout
    assert "strength is 0" in stdout


def test_passive_capture_installs_and_captures_with_reins_toggle_off(monkeypatch):
    """The _sample_chunk-level gate: `if refinement_key and (h3_repr_steering or
    h3_repr_steering_passive_capture))` -- with the REINS checkbox off, passive_capture
    alone must still get _install_h3_repr_steering called, with strength forced to 0.0."""
    monkeypatch.setattr(rs, "direction", lambda _k, block=None: (None, 0, 0))
    node = S()
    h3_repr_steering = False
    h3_repr_steering_passive_capture = True
    refinement_key = "key"
    capture_holder = [{}]
    if refinement_key and (h3_repr_steering or h3_repr_steering_passive_capture):
        patched = node._install_h3_repr_steering(
            _FakeModel(), refinement_key,
            0.05 if h3_repr_steering else 0.0,
            capture_holder, steer_block="5")
    out = _run_block(patched, 5)
    assert torch.allclose(out, torch.ones(4, 3)), "REINS off -> never injects, even passively"
    assert 5 in capture_holder[0], "passive capture must still record a descriptor"


def test_passive_capture_default_matches_input_types_default():
    """The exact bug class caught live in h3_q_steer_block (sample()'s own default disagreed
    with INPUT_TYPES, so every pre-existing workflow -- missing this new optional widget --
    silently turned the feature on): both must default to False here too."""
    import inspect
    sig_default = inspect.signature(S.sample).parameters[
        "h3_repr_steering_passive_capture"].default
    input_types_default = S.INPUT_TYPES()["optional"][
        "h3_repr_steering_passive_capture"][1]["default"]
    assert sig_default is False and input_types_default is False


def test_strip_dit_patches_unwinds_a_leaked_reins_chain(monkeypatch):
    """An interrupt mid-run can leave REINS' dit hooks on the shared model (same class as
    the scene-wrapper leak this project already fixed once -- see
    project_scene_wrapper_leak / _strip_funpack_scene_wrappers). Simulate that: install
    REINS on a model (the "leaked" state a killed run would have left behind), then
    install it AGAIN on the clone (what the next run's _install_h3_repr_steering does
    when it sees the leak and chains onto it as its own "inner") -- and confirm
    _strip_funpack_dit_patches unwinds back to the pristine, unpatched state instead of
    leaving two layers of hooks stacked."""
    import samplers as sm

    monkeypatch.setattr(rs, "direction", lambda _k, block=None: (None, 0, 0))
    node = S()
    leaked = node._install_h3_repr_steering(
        _FakeModel(), "key", strength=0.0, capture_holder=[{}], steer_block="5")
    # A second, un-cleaned-up install on top -- what the next run would produce if the
    # strip never ran.
    doubly_leaked = node._install_h3_repr_steering(
        leaked, "key", strength=0.0, capture_holder=[{}], steer_block="5")
    dit = doubly_leaked.model_options["transformer_options"]["patches_replace"]["dit"]
    assert getattr(dit[("double_block", 5)], sm._FUNPACK_DIT_HOOK_TAG, False)

    # REINS installs a hook at every CANDIDATE_BLOCK (50), not just the steered one, so
    # two stacked installs leave 2 layers at each of the 50 -- 100 total.
    stripped = sm._strip_funpack_dit_patches(doubly_leaked)
    assert stripped == 2 * len(rs.CANDIDATE_BLOCKS), "both leaked layers at every block should unwind"
    dit_after = doubly_leaked.model_options.get("transformer_options", {}) \
        .get("patches_replace", {}).get("dit", {})
    assert ("double_block", 5) not in dit_after, "nothing left to restore -- key removed"
    # Idempotent: stripping an already-clean model finds nothing to do.
    assert sm._strip_funpack_dit_patches(doubly_leaked) == 0


def test_strip_dit_patches_preserves_a_foreign_untagged_hook():
    """The strip must only remove FunPack's own tagged hooks -- an untagged (third-party,
    or hand-installed-by-a-test) entry at a DIFFERENT block must survive untouched, same
    guarantee _strip_funpack_scene_wrappers gives for a foreign wrapper."""
    import samplers as sm

    model = _FakeModel()

    def _foreign(args, extra):
        return {"img": args["img"]}

    model.model_options = {"transformer_options": {"patches_replace": {
        "dit": {("double_block", 3): _foreign}}}}
    stripped = sm._strip_funpack_dit_patches(model)
    assert stripped == 0
    dit = model.model_options["transformer_options"]["patches_replace"]["dit"]
    assert dit[("double_block", 3)] is _foreign


if __name__ == "__main__":
    test_empty_steer_block_falls_back_to_default_block()
    print("ok (run via pytest for the monkeypatch-dependent cases)")
