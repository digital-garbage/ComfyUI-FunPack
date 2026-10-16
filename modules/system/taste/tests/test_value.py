"""The judge: needs enough ratings, learns their order, and its push helps."""

import torch

from modules.system.taste import value


def _rated(n=12):
    torch.manual_seed(0)
    liked_dir = torch.randn(value.DIM)
    descs, rewards = [], []
    for i in range(n):
        sign = 1.0 if i % 2 else -1.0
        descs.append(sign * liked_dir + 0.3 * torch.randn(value.DIM))
        rewards.append(sign)
    return descs, rewards, liked_dir


def test_too_few_or_all_alike_is_no_judge():
    descs, rewards, _ = _rated(9)
    assert value.Judge.train_on(descs, rewards) is None
    descs, _r, _ = _rated(12)
    assert value.Judge.train_on(descs, [1.0] * 12) is None


def test_its_push_raises_the_score_and_points_at_liked():
    descs, rewards, liked_dir = _rated()
    judge = value.Judge.train_on(descs, rewards)
    video = torch.randn(1, 4, 2, 8, 8)
    moved = judge.nudge(video, 0.05)
    score = lambda v: float(sum(net(value.describe(v).unsqueeze(0)) for net in judge.nets))
    assert score(moved) > score(video)
    shift = value.describe(moved) - value.describe(video)
    assert torch.nn.functional.cosine_similarity(shift, liked_dir, dim=0) > 0.3
    assert torch.allclose((moved - video).norm(), 0.05 * video.norm(), rtol=1e-3)


def test_the_judge_trains_and_steers_under_comfyuis_inference_mode():
    # ComfyUI runs every prompt under torch.inference_mode(): weights made there cannot be trained.
    import torch
    from modules.system.taste import value as v
    with torch.inference_mode():
        judge = v.Judge.train_on([torch.randn(v.DIM) for _ in range(v.MIN_SAMPLES + 2)], [i % 2 for i in range(v.MIN_SAMPLES + 2)])
        assert judge is not None
        assert judge.gradient(torch.randn(1, 4, 2, 8, 8)).shape == (1, 4, 2, 8, 8)
