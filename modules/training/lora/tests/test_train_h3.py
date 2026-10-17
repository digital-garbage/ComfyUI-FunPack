"""A whole training run through the module's own entry point, on a tiny real H3 on CPU.

The model is ComfyUI's own MiniMaxH3 class, a few hundred thousand parameters wide, with a stand-in text
encoder and VAEs that return latents of the right shape. What this proves is OUR side: the set, the
captions, the latents packed the way H3 wants, the two H3 repairs, the loss stream, and the saved file.
It does not prove anything about real weights, a real VAE or real hardware.
"""

import pytest


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Needs ComfyUI's source for the trainer."""


def tiny_h3_patcher():
    import torch
    import comfy.supported_models as sm
    from comfy.model_patcher import ModelPatcher
    unet = dict(image_model="minimax_h3", hidden_size=64, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2,
                attention_head_dim=32, ffn_hidden_size=64, text_dim=48, timestep_input_dim=32, time_embed_hidden_size=64,
                time_embed_dim=32, rope_inv_freq_len=4)
    cfg = sm.MiniMaxH3(dict(unet))
    cfg.set_inference_dtype(torch.float32, None)

    def make(disable_dynamic=False):
        base = cfg.get_model({}, "", device=torch.device("cpu"))
        torch.manual_seed(0)
        for p in base.parameters():
            torch.nn.init.normal_(p, std=0.05)
        base.diffusion_model.rope.inv_freq.copy_(torch.rand(4))
        return ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    patcher = make()
    patcher.cached_patcher_init = (make, ())
    return patcher


class FakeClip:
    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        import torch
        return [[torch.randn(1, 6, 48), {}]]


class FakeVae:
    def encode(self, pixels):
        import torch
        from comfy_extras.nodes_minimax_h3 import video_latent_t
        frames, h, w = pixels.shape[0], pixels.shape[1], pixels.shape[2]
        return torch.randn(1, 24, video_latent_t(frames), h // 16, w // 16)


class FakeAudioVae:
    audio_sample_rate = 44100

    def encode(self, wave):
        import torch
        seconds = wave.shape[1] / 44100
        return torch.randn(1, 32, 2, round(seconds * 40))


@pytest.fixture
def run_env(tmp_path, monkeypatch):
    import folder_paths
    from PIL import Image
    from modules.training.lora import jobs, store, train
    from core import config
    monkeypatch.setattr(config, "TRAINING_DIR", tmp_path / "training")
    monkeypatch.setattr(train, "roots", lambda: {"input": str(tmp_path)})
    loras = tmp_path / "loras"
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda kind: [str(loras)])
    from modules.models.minimax_h3 import training as h3
    monkeypatch.setattr(train, "registry", None, raising=False)
    import modules.training.lora.train as t
    class Reg:
        def ask(self, capability, *a, **k):
            return h3.adapt(*a, **k) if capability == "training_adapt" else None
    from modules import _core
    monkeypatch.setattr(_core.registry, "current", lambda: Reg())
    names = []
    for i in range(2):
        Image.new("RGB", (80, 80), (40 * i, 90, 160)).save(tmp_path / f"p{i}.png")
        names.append(f"p{i}.png")
    store.create("tiny")
    store.add("tiny", [{"src": {"kind": "output", "type": "input", "subfolder": "", "filename": n}, "media_kind": "image",
                        "label": "own", "caption": "a blue square, flat"} for n in names])
    return tmp_path, jobs


def settings(**over):
    base = {"name": "tiny_lora", "use": "normal", "steps": 6, "learning_rate": 3e-3, "rank": 4, "algorithm": "LoRA", "optimizer": "AdamW",
            "loss": "MSE", "batch_size": 4, "grad_accumulation": 1, "resolution": 64, "aspect_buckets": True, "frames": 5, "fps": 0.0,
            "start": "start", "trigger": "zq", "seed": 0, "training_dtype": "bf16", "lora_dtype": "fp32",
            "gradient_checkpointing": False, "offloading": False, "bypass": False}
    base.update(over)
    return base


def test_a_tiny_h3_trains_and_saves_a_lora(run_env):
    import safetensors.torch
    from modules.training.lora import train
    tmp_path, jobs = run_env
    file = train.run(tiny_h3_patcher(), FakeClip(), FakeVae(), FakeAudioVae(), "tiny", settings())
    saved = safetensors.torch.load_file(str(tmp_path / "loras" / file))
    assert file == "tiny_lora.safetensors" and len(saved) > 20
    snap = jobs.snapshot()
    assert snap["state"] == "done" and snap["step"] == 6 and len(snap["loss"]) == 6
    assert all(x == x for x in snap["loss"]), "a NaN loss"


def test_a_second_run_never_overwrites_the_first(run_env):
    from modules.training.lora import train
    tmp_path, _ = run_env
    first = train.run(tiny_h3_patcher(), FakeClip(), FakeVae(), FakeAudioVae(), "tiny", settings(steps=2))
    second = train.run(tiny_h3_patcher(), FakeClip(), FakeVae(), FakeAudioVae(), "tiny", settings(steps=2))
    assert first != second and (tmp_path / "loras" / first).is_file()


def test_h3_without_its_audio_vae_says_what_to_add(run_env):
    from modules.training.lora import train
    with pytest.raises(ValueError, match="audio VAE"):
        train.run(tiny_h3_patcher(), FakeClip(), FakeVae(), None, "tiny", settings())


def test_stop_ends_the_run_and_records_it(run_env):
    from modules.training.lora import train
    _, jobs = run_env
    real = jobs.step

    def stop_after_two(loss):
        real(loss)
        if jobs.snapshot()["step"] >= 2:
            jobs.request_stop()
    jobs.step = stop_after_two
    try:
        with pytest.raises(train.Stopped):
            train.run(tiny_h3_patcher(), FakeClip(), FakeVae(), FakeAudioVae(), "tiny", settings(steps=50))
    finally:
        jobs.step = real
    snap = jobs.snapshot()
    assert snap["state"] == "stopped" and snap["step"] < 50


def test_a_set_with_nothing_for_the_chosen_use_is_refused(run_env):
    from modules.training.lora import train
    with pytest.raises(ValueError, match="nothing to train on"):
        train.run(tiny_h3_patcher(), FakeClip(), FakeVae(), FakeAudioVae(), "tiny", settings(use="avoid"))


# --- a picture model: no adapter, plain latent tensors ------------------------------------------------

def tiny_sd_patcher():
    import torch
    import comfy.supported_models as sm
    from comfy.model_patcher import ModelPatcher
    unet = {"context_dim": 32, "model_channels": 32, "use_linear_in_transformer": False, "adm_in_channels": None, "use_temporal_attention": False,
            "in_channels": 4, "out_channels": 4, "image_size": 32, "num_res_blocks": [1, 1], "transformer_depth": [1, 1], "channel_mult": [1, 2],
            "transformer_depth_output": [1, 1, 1, 1], "transformer_depth_middle": 1, "use_spatial_transformer": True, "num_head_channels": 8}
    cfg = sm.SD15(dict(unet))
    cfg.set_inference_dtype(torch.float32, None)

    def make(disable_dynamic=False):
        base = cfg.get_model({}, "", device=torch.device("cpu"))
        torch.manual_seed(0)
        for p in base.parameters():
            torch.nn.init.normal_(p, std=0.05)
        return ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    patcher = make()
    patcher.cached_patcher_init = (make, ())
    return patcher


class FakeClip32(FakeClip):
    def encode_from_tokens_scheduled(self, tokens):
        import torch
        return [[torch.randn(1, 6, 32), {}]]


class FakeVae4:
    def encode(self, pixels):
        import torch
        return torch.randn(pixels.shape[0], 4, pixels.shape[1] // 8, pixels.shape[2] // 8)


@pytest.fixture
def plain_env(run_env, monkeypatch):
    from modules import _core
    class NoAdapter:
        def ask(self, capability, *a, **k):
            return None
    monkeypatch.setattr(_core.registry, "current", lambda: NoAdapter())
    return run_env


def test_a_picture_model_trains_from_a_set_of_one_shape(plain_env):
    from modules.training.lora import train
    _, jobs = plain_env
    file = train.run(tiny_sd_patcher(), FakeClip32(), FakeVae4(), None, "tiny", settings(aspect_buckets=False, batch_size=2, steps=4))
    snap = jobs.snapshot()
    assert file.endswith(".safetensors") and snap["state"] == "done" and len(snap["loss"]) == 4
    assert all(x == x for x in snap["loss"])


def test_a_picture_model_trains_from_pictures_of_different_shapes(plain_env):
    from PIL import Image
    from modules.training.lora import store, train
    tmp_path, jobs = plain_env
    Image.new("RGB", (160, 64), (200, 30, 30)).save(tmp_path / "wide.png")
    store.add("tiny", [{"src": {"kind": "output", "type": "input", "subfolder": "", "filename": "wide.png"}, "media_kind": "image", "label": "own", "caption": "wide"}])
    train.run(tiny_sd_patcher(), FakeClip32(), FakeVae4(), None, "tiny", settings(aspect_buckets=True, batch_size=1, steps=3))
    assert jobs.snapshot()["state"] == "done"


def test_a_dynamic_model_is_copied_as_a_plain_one_for_the_trainer():
    from modules.training.lora import train

    class Dynamic:
        calls = []

        def is_dynamic(self):
            return True

        def clone(self, disable_dynamic=False, force_deepcopy=False):
            self.calls.append((disable_dynamic, force_deepcopy))
    m = Dynamic()
    train._train_on_a_plain_copy(m)
    m.clone(force_deepcopy=True)
    assert m.calls == [(True, True)]

    class Plain(Dynamic):
        def is_dynamic(self):
            return False
    p = Plain()
    train._train_on_a_plain_copy(p)
    assert "clone" not in vars(p)
