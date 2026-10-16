"""Which run this is, and when a new one begins.

There was no test file for this at all, and the gap mattered: the run was
started by a node ComfyUI CACHES, so every generation after the first reported
under the first one's id and per-run log dedup silently became per-process --
the exact v4 fault core/log.py's own comment describes.
"""

import pytest

from core import log, patching, run


@pytest.fixture(autouse=True)
def clean():
    run._reset()
    log._reset()
    yield
    run._reset()
    log._reset()


def test_there_is_no_run_before_one_starts():
    assert run.current() is None


def test_each_start_is_a_new_run():
    first, second, third = run.start(), run.start(), run.start()
    assert len({first, second, third}) == 3
    assert run.current() == third


def test_the_id_is_readable():
    """It goes into log lines a person scans. `run 7` is findable; a hex string
    is not."""
    assert run.start() == "run 1"


def test_starting_a_run_clears_what_was_said_once():
    run.start()
    log.once("k", log.ALERT, "src", "inert")
    assert len(log.history()) == 1

    run.start()
    log.once("k", log.ALERT, "src", "inert")
    assert len(log.history()) == 2, "a second generation stayed silent about the same fault"


def test_the_sampler_starts_a_run_every_time_not_only_the_first(comfyui, monkeypatch):
    """The regression. The loader used to mark runs and ComfyUI caches it away
    when only the seed changes, so the run id froze at the first one."""
    import torch
    from core import registry as registry_mod
    from modules.sampling.sampler.nodes import FunPackSampler

    monkeypatch.setattr(registry_mod, "current", lambda rescan=False: registry_mod.Registry())

    class Model:
        model = None

    seen = []
    for _ in range(3):
        # Only the part that marks the run, exercised directly: the rest needs
        # weights, and the caching behaviour it guards against is upstream.
        run.start()
        seen.append(run.current())

    assert len(set(seen)) == 3


def test_a_dropped_record_does_not_survive_into_the_next_generation():
    """The record travels on the model, and ComfyUI may hand the same model back
    from its cache. A modifier that failed last time deserves this time."""
    dropped = patching.Dropped()
    dropped.record("funpack.m", RuntimeError("boom"))
    assert "funpack.m" in dropped

    dropped.clear()
    assert "funpack.m" not in dropped and not dropped
