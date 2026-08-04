"""The KSampler pass builds its own schedule from steps + scheduler.

A SAMPLER object is only the step function — it carries no schedule — so before this the
KSampler pass handed the chain sampler no SIGMAS at all unless the user typed a sigma list
by hand. Studio now builds one the way ComfyUI's BasicScheduler does, from the loaded model's
model_sampling.

The schedule dropdown is the switch between the two sources, and 'use_user_sigmas' — the
default — is what keeps the old behaviour reachable: it runs the typed field untouched, so a
hand-written schedule can be parked there and switched back to without retyping it. Any other
entry computes, and the typed field is ignored for that pass.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("folder_paths", types.SimpleNamespace(models_dir=""))

import comfy.samplers as _cs  # noqa: E402  (the stub tree; behaviour installed per test)

from conditioning import FunPackStudio  # noqa: E402


class FakeModel:
    """A ModelPatcher's one relevant surface."""

    def __init__(self):
        self.asked = []

    def get_model_object(self, name):
        self.asked.append(name)
        return f"model_sampling::{name}"


@pytest.fixture
def comfy_sampler_stub(monkeypatch):
    calls = types.SimpleNamespace(sigmas=[], objects=[])

    def calculate_sigmas(model_sampling, scheduler_name, steps):
        calls.sigmas.append((model_sampling, scheduler_name, steps))
        return torch.linspace(1.0, 0.0, steps + 1)

    def sampler_object(name):
        calls.objects.append(name)
        return f"SAMPLER::{name}"

    monkeypatch.setattr(_cs, "calculate_sigmas", calculate_sigmas, raising=False)
    monkeypatch.setattr(_cs, "sampler_object", sampler_object, raising=False)
    return calls


def test_ksampler_builds_schedule_from_steps_and_scheduler(comfy_sampler_stub):
    model = FakeModel()
    sampler, sigmas = FunPackStudio._build_one_sampler(
        {"type": "KSampler", "ksampler_name": "res_multistep",
         "ksampler_steps": 8, "ksampler_scheduler": "karras"},
        model=model,
    )

    assert sampler == "SAMPLER::res_multistep"
    assert comfy_sampler_stub.sigmas == [("model_sampling::model_sampling", "karras", 8)]
    assert sigmas is not None and len(sigmas) == 9
    assert float(sigmas[-1]) == 0.0


def test_use_user_sigmas_runs_the_typed_field_and_computes_nothing(comfy_sampler_stub):
    _, sigmas = FunPackStudio._build_one_sampler(
        {"type": "KSampler", "sigmas": "0.9, 0.5, 0.0",
         "ksampler_steps": 8, "ksampler_scheduler": "use_user_sigmas"},
        model=FakeModel(),
    )

    assert comfy_sampler_stub.sigmas == []          # never consulted
    assert [round(float(v), 3) for v in sigmas] == [0.9, 0.5, 0.0]


def test_a_computed_schedule_wins_over_a_typed_field_left_in_place(comfy_sampler_stub):
    """The whole point of the switch: the field keeps its text, the schedule ignores it."""
    _, sigmas = FunPackStudio._build_one_sampler(
        {"type": "KSampler", "sigmas": "0.9, 0.5, 0.0",
         "ksampler_steps": 8, "ksampler_scheduler": "karras"},
        model=FakeModel(),
    )

    assert comfy_sampler_stub.sigmas == [("model_sampling::model_sampling", "karras", 8)]
    assert len(sigmas) == 9


def test_the_default_is_the_typed_field(comfy_sampler_stub):
    """A config saved before steps/schedule existed must behave exactly as it used to."""
    _, sigmas = FunPackStudio._build_one_sampler(
        {"type": "KSampler", "sigmas": "0.9, 0.5, 0.0"}, model=FakeModel(),
    )

    assert comfy_sampler_stub.sigmas == []
    assert [round(float(v), 3) for v in sigmas] == [0.9, 0.5, 0.0]


def test_a_computed_schedule_without_steps_falls_back_to_the_field(comfy_sampler_stub):
    _, sigmas = FunPackStudio._build_one_sampler(
        {"type": "KSampler", "sigmas": "0.9, 0.0",
         "ksampler_steps": 0, "ksampler_scheduler": "karras"},
        model=FakeModel(),
    )

    assert comfy_sampler_stub.sigmas == []
    assert [round(float(v), 3) for v in sigmas] == [0.9, 0.0]


def test_a_bad_scheduler_costs_the_schedule_not_the_sampler(monkeypatch):
    def boom(*_a, **_k):
        raise ValueError("error invalid scheduler nonsense")

    monkeypatch.setattr(_cs, "calculate_sigmas", boom, raising=False)
    monkeypatch.setattr(_cs, "sampler_object", lambda n: f"SAMPLER::{n}", raising=False)

    sampler, sigmas = FunPackStudio._build_one_sampler(
        {"type": "KSampler", "ksampler_steps": 8, "ksampler_scheduler": "nonsense"},
        model=FakeModel(),
    )

    assert sampler == "SAMPLER::euler"
    assert sigmas is None


def test_missing_model_is_survivable(comfy_sampler_stub):
    """No model to read model_sampling from — build the sampler, skip the schedule."""
    sampler, sigmas = FunPackStudio._build_one_sampler(
        {"type": "KSampler", "ksampler_steps": 8, "ksampler_scheduler": "karras"}, model=None,
    )

    assert sampler == "SAMPLER::euler"
    assert sigmas is None
