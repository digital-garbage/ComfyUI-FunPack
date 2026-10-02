"""Block influence on a real H3 forward, paired with ratings through a real taste key."""

import asyncio

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    from modules.sampling.block_influence import measure
    from modules.system.taste import store
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")
    monkeypatch.setattr(store, "current_prompt_id", lambda: "run-1")
    monkeypatch.setattr(measure, "SWITCH", tmp_path / "switch")


def _load(tiny, key="fox"):
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    settings = {"taste": {"key": key}} if key else {}
    return FunPackLoadModifiers.execute(tiny.patcher, settings).result


def test_off_by_default_it_stores_nothing_changes_nothing_and_says_so(tiny_h3):
    from core import log
    from modules.system.taste import store
    base = tiny_h3.run()
    patched, _ = _load(tiny_h3)
    log.new_run()
    out = tiny_h3.sample(patched)
    assert torch.equal(out[0], base[0]) and torch.equal(out[1], base[1])
    assert not (store.ROOT / "fox" / "block_influence.pending.pt").exists()
    assert any("Inactive | recording is off" in e["message"] for e in log.history())


def test_no_taste_key_installs_nothing(tiny_h3):
    patched, status = _load(tiny_h3, key=None)
    assert "block_influence" not in status.splitlines()[0]


def test_recording_measures_every_block_without_changing_the_clip(tiny_h3):
    from modules.sampling.block_influence import measure
    from modules.system.taste import store
    base = tiny_h3.run()
    patched, _ = _load(tiny_h3)
    measure.set_enabled(True)                       # toggled AFTER the node ran: the next run sees it
    out = tiny_h3.sample(patched)
    assert torch.equal(out[0], base[0]) and torch.equal(out[1], base[1])     # measurement only
    assert store.rate("run-1", "liked")["recorded"] == ["block_influence"]
    row = store.load("fox", "block_influence")["rows"][0]["rows"]
    n = len(tiny_h3.patcher.model.diffusion_model.blocks)
    assert all(row[k].shape == (n,) for k in ("ratio", "raw", "novelty"))
    assert torch.isfinite(row["raw"]).all() and (row["raw"] > 0).all()       # an in-place block still moves
    assert torch.isnan(row["novelty"][0]) and torch.isfinite(row["novelty"][1:]).all()


def test_the_measured_push_is_the_blocks_own_size():
    from modules.sampling.block_influence.measure import Tally
    from core import dit_hooks
    x = torch.randn(10, 4)

    def block(args):                                # in place, as real blocks write their residual
        args["img"] += 1.0
        return {"img": args["img"]}

    t = Tally(2)
    import unittest.mock as mock
    with mock.patch.object(dit_hooks, "target_rows", lambda *a: torch.ones(10, dtype=torch.bool)):
        before = x.clone()
        t.measure(1, {"img": x}, block)
    rows = t.rows()
    assert float(rows["raw"][1]) == pytest.approx(float((torch.ones(10, 4)).norm()))
    assert float(rows["ratio"][1]) == pytest.approx(float(torch.ones(10, 4).norm() / before.norm()), rel=1e-5)
    assert torch.isnan(rows["raw"][0])


def test_a_run_that_measured_nothing_says_so_and_stores_nothing(tiny_h3, monkeypatch):
    from core import dit_hooks, log
    from modules.sampling.block_influence import measure
    from modules.system.taste import store
    patched, _ = _load(tiny_h3)
    measure.set_enabled(True)
    monkeypatch.setattr(dit_hooks, "target_rows", lambda *a: None)
    log.new_run()
    tiny_h3.sample(patched)
    assert any("nothing was measured" in e["message"] for e in log.history(level=log.WARNING))
    assert not (store.ROOT / "fox" / "block_influence.pending.pt").exists()


def test_a_discarded_candidate_is_not_measured(tiny_h3):
    from modules.sampling.block_influence import measure
    from modules.system.taste import store
    patched, _ = _load(tiny_h3)
    measure.set_enabled(True)
    patched.model_options.setdefault("transformer_options", {})["funpack_probe"] = True
    tiny_h3.sample(patched)
    assert not (store.ROOT / "fox" / "block_influence.pending.pt").exists()


def _rows(profiles):
    return [{"reward": w, "rows": {"ratio": torch.tensor(p), "raw": torch.tensor(p),
                                   "novelty": torch.tensor([float("nan"), 0.5, 0.5])}}
            for p, w in profiles]


def test_a_flat_profile_reads_flat_and_a_difference_needs_two_of_each():
    from modules.sampling.block_influence.measure import profile
    flat = profile(_rows([([1.0, 1.0, 1.0], 1.0)] * 2 + [([1.0, 1.0, 1.0], -1.0)] * 2))
    assert flat["flatness"] == pytest.approx(0.0) and flat["n_liked"] == 2
    assert flat["difference"] == {"0": 0.0, "1": 0.0, "2": 0.0}
    assert flat["mean_novelty"] == pytest.approx(0.5) and "0" not in flat["novelty"]
    one_each = profile(_rows([([1.0, 2.0, 3.0], 1.0), ([1.0, 2.0, 3.0], -1.0)]))
    assert one_each["difference"] is None and one_each["flatness"] > 0.3
    assert sum(one_each["share"].values()) == pytest.approx(1.0)


def test_liked_minus_disliked_points_at_the_block_that_runs_hotter_on_liked():
    from modules.sampling.block_influence.measure import profile
    p = profile(_rows([([2.0, 1.0, 1.0], 1.0)] * 2 + [([1.0, 1.0, 1.0], -1.0)] * 2))
    assert p["difference"]["0"] > 0 and p["difference"]["1"] == 0.0


def test_no_rows_is_an_empty_profile_not_a_flat_one():
    from modules.sampling.block_influence.measure import profile
    p = profile([])
    assert p["flatness"] is None and p["overall"] == {} and p["difference"] is None


def test_the_panel_routes_read_toggle_clear_and_export(tiny_h3):
    from aiohttp import web
    from modules.sampling import block_influence as mod
    from modules.sampling.block_influence import measure
    from modules.system.taste import store
    handlers = {}

    class Table:
        def get(self, path):
            return lambda fn: handlers.setdefault(("GET", path), fn)

        def post(self, path):
            return lambda fn: handlers.setdefault(("POST", path), fn)

    mod.routes(Table(), "/b", web)

    class Req:
        def __init__(self, body=None, key="fox"):
            self._body = body
            self.rel_url = type("U", (), {"query": {"key": key}})()

        async def json(self):
            return self._body

    def call(method, path, **kw):
        out = asyncio.run(handlers[(method, path)](Req(**kw)))
        return out.status, out

    import json
    patched, _ = _load(tiny_h3)
    status, out = call("POST", "/b/enabled", body={"enabled": True, "key": "fox"})
    assert json.loads(out.text)["enabled"] is True and measure.enabled()
    tiny_h3.sample(patched)
    store.rate("run-1", "liked")
    status, out = call("GET", "/b/status")
    assert json.loads(out.text)["runs"] == 1
    status, out = call("GET", "/b/export")
    assert status == 200 and out.body
    status, out = call("POST", "/b/clear", body={"key": "fox"})
    assert json.loads(out.text)["runs"] == 0
    assert call("GET", "/b/export")[0] == 404
    assert call("GET", "/b/status", key="no/slash")[0] == 400


def test_a_bad_key_is_refused_before_the_switch_flips_and_a_placeholder_finds_the_real_key(tiny_h3):
    from aiohttp import web
    from modules.sampling import block_influence as mod
    from modules.sampling.block_influence import measure
    from modules.system.taste import store
    import json
    handlers = {}

    class Table:
        def get(self, path):
            return lambda fn: handlers.setdefault(("GET", path), fn)

        def post(self, path):
            return lambda fn: handlers.setdefault(("POST", path), fn)

    mod.routes(Table(), "/b", web)

    class Req:
        def __init__(self, body=None, key="default"):
            self._body, self.rel_url = body, type("U", (), {"query": {"key": key}})()

        async def json(self):
            return self._body

    out = asyncio.run(handlers[("POST", "/b/enabled")](Req({"enabled": True, "key": "no/slash"})))
    assert out.status == 400 and not measure.enabled()
    patched, _ = _load(tiny_h3, key="mystyle")
    measure.set_enabled(True)
    tiny_h3.sample(patched)
    store.rate("run-1", "liked")
    state = json.loads(asyncio.run(handlers[("GET", "/b/status")](Req())).text)
    assert state["key"] == "mystyle" and state["runs"] == 1       # "default" had nothing: the real key


def test_the_sliders_perturbed_passes_are_not_measured(tiny_h3):
    from modules.sampling.block_influence import measure
    from core import dit_hooks
    patched, _ = _load(tiny_h3)
    measure.set_enabled(True)
    patched.model_options.setdefault("transformer_options", {})[dit_hooks.PROBE] = True
    tiny_h3.sample(patched)                           # same flag the slider sets on its +/- passes
    from modules.system.taste import store
    assert not (store.ROOT / "fox" / "block_influence.pending.pt").exists()


def test_each_run_starts_a_fresh_tally_and_the_toggle_off_stores_nothing(tiny_h3):
    from modules.sampling.block_influence import measure
    from modules.system.taste import store
    patched, _ = _load(tiny_h3)
    measure.set_enabled(True)
    tiny_h3.sample(patched)
    first = torch.load(store.ROOT / "fox" / "block_influence.pending.pt")["rows"]["raw"]
    tiny_h3.sample(patched)
    second = torch.load(store.ROOT / "fox" / "block_influence.pending.pt")["rows"]["raw"]
    assert torch.allclose(first, second, rtol=1e-4)     # one run's mean, not two runs summed or merged


def test_the_novelty_pairing_never_crosses_a_step_boundary():
    from core import dit_hooks
    from modules.sampling.block_influence.measure import Tally
    import unittest.mock as mock
    t = Tally(3)

    def block(args):
        args["img"] = args["img"] + torch.randn_like(args["img"])
        return {"img": args["img"]}

    with mock.patch.object(dit_hooks, "target_rows", lambda *a: torch.ones(6, dtype=torch.bool)):
        for b in (0, 1, 2, 0, 1, 2):                    # two steps
            t.measure(b, {"img": torch.randn(6, 4)}, block)
    assert t._count["novelty"] == [0, 2, 2]             # block 0 of step 2 has no predecessor


def test_clear_and_export_act_on_exactly_the_key_named_never_a_fallback(tiny_h3, monkeypatch):
    from modules.sampling.block_influence import measure
    from modules.system.taste import store
    for key in ("a", "b"):
        monkeypatch.setattr(store, "current_prompt_id", lambda key=key: f"run-{key}")
        patched, _ = _load(tiny_h3, key=key)
        measure.set_enabled(True)
        tiny_h3.sample(patched)
        store.rate(f"run-{key}", "liked")
    assert measure.state("default")["key"] == "b"       # the panel's placeholder resolves to the latest
    measure.clear("default")                            # ...but a clear of "default" touches neither
    assert measure.path_of("a").exists() and measure.path_of("b").exists()
    measure.clear("b")
    assert measure.path_of("a").exists() and not measure.path_of("b").exists()


def test_every_run_builds_its_own_tally_and_turning_recording_off_stops_it(tiny_h3, monkeypatch):
    from modules.sampling.block_influence import measure
    from modules.system.taste import store
    made = []
    real = measure.Tally
    monkeypatch.setattr(measure, "Tally", lambda n: made.append(1) or real(n))
    patched, _ = _load(tiny_h3)
    made.clear()
    measure.set_enabled(True)
    monkeypatch.setattr(store, "current_prompt_id", lambda: "run-1")
    tiny_h3.sample(patched)
    monkeypatch.setattr(store, "current_prompt_id", lambda: "run-2")
    tiny_h3.sample(patched)
    assert len(made) == 2                               # a fresh tally per run
    assert torch.load(store.ROOT / "fox" / "block_influence.pending.pt")["prompt_id"] == "run-2"
    measure.set_enabled(False)
    monkeypatch.setattr(store, "current_prompt_id", lambda: "run-3")
    tiny_h3.sample(patched)
    assert torch.load(store.ROOT / "fox" / "block_influence.pending.pt")["prompt_id"] == "run-2"


def test_a_run_that_measured_nothing_says_so_in_the_panel_state_too(tiny_h3, monkeypatch):
    from core import dit_hooks
    from modules.sampling.block_influence import measure
    patched, _ = _load(tiny_h3)
    measure.set_enabled(True)
    monkeypatch.setattr(dit_hooks, "target_rows", lambda *a: None)
    tiny_h3.sample(patched)
    assert "measured nothing" in measure.state("fox")["problem"]
    monkeypatch.undo()
    measure.problem = None


def test_clearing_one_key_never_answers_with_another_keys_state(tiny_h3, monkeypatch):
    from modules.sampling.block_influence import measure
    from modules.system.taste import store
    for key in ("a", "b"):
        monkeypatch.setattr(store, "current_prompt_id", lambda key=key: f"run-{key}")
        patched, _ = _load(tiny_h3, key=key)
        measure.set_enabled(True)
        tiny_h3.sample(patched)
        store.rate(f"run-{key}", "liked")
    measure.clear("a")
    assert measure.state("a", fallback=False)["runs"] == 0
    assert measure.state("a")["key"] == "b"             # the placeholder lookup still resolves
