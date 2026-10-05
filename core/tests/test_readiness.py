from core import readiness as r


def levels(rows):
    return [x["level"] for x in rows]


def test_no_model_files_is_a_failure_and_empty_folders_are_named():
    rows = r.models({"diffusion_models": 0, "checkpoints": 0, "vae": 2})
    assert rows[0]["level"] == "fail" and "No model files" in rows[0]["text"]
    assert any("No diffusion models" in x["text"] for x in rows) and any(x["text"] == "2 VAEs" for x in rows)
    assert "fail" not in levels(r.models({"diffusion_models": 1}))


def test_a_preset_naming_a_node_that_is_not_installed_says_which_and_where_to_get_it():
    presets = [{"id": "a", "title": "H3", "slots": [{"node": "Have"}, {"node": "ImageTransformKJ"}]}, {"id": "b", "slots": [{"node": "Have"}]}]
    rows = r.pipelines(presets, lambda cls: cls == "Have")
    assert len(rows) == 1 and "ImageTransformKJ" in rows[0]["text"] and "H3" in rows[0]["text"]
    assert r.pipelines(presets[1:], lambda cls: True)[0]["level"] == "ok"


def test_modules_that_failed_or_are_switched_off_are_listed():
    rows = r.modules([("modules.x", "ImportError: nope")], {"reins": {"reason": "boom"}})
    assert len(rows) == 2 and "nope" in rows[0]["text"] and "reins" in rows[1]["text"]
    assert r.modules([], {})[0]["level"] == "ok"


def test_the_machine_checks_never_raise_and_low_disk_is_a_warning(monkeypatch):
    monkeypatch.setattr(r.shutil, "which", lambda name: None)
    rows = r.machine(disk_free_gb=3)
    assert any(x["level"] == "fail" and "ffmpeg" in x["text"] for x in rows)
    assert any(x["level"] == "warn" and "3 GB free" in x["text"] for x in rows)


def test_the_row_names_the_device_comfyui_generates_on(monkeypatch):
    monkeypatch.setattr(r, "device", lambda: "mps")
    rows = r.machine()
    assert any(x["level"] == "ok" and "MPS" in x["text"] for x in rows)
    assert not any("sageattention" in x["text"] for x in rows)           # a CUDA-only library
    monkeypatch.setattr(r, "device", lambda: "cpu")                       # ComfyUI started with --cpu, GPU or not
    assert any(x["level"] == "warn" and "on the CPU" in x["text"] for x in r.machine())
    monkeypatch.setattr(r, "device", lambda: "xpu")                       # an Intel GPU is not the CPU
    rows = r.machine()
    assert any("'xpu'" in x["text"] for x in rows) and not any("on the CPU" in x["text"] for x in rows)
