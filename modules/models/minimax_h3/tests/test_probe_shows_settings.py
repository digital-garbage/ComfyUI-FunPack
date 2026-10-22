"""A picked H3 file must keep Shadow negative and Video detail on screen.

Fake .safetensors header (names only, no weights) -> the probe -> the manifest the
browser gets, over the real module tree.
"""

import json
import struct

from core import probe, routes


def _fake_h3_file(tmp_path):
    names = ["video_patch_proj.weight", "audio_patch_proj.weight"]
    header = {n: {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]} for n in names}
    raw = json.dumps(header).encode()
    path = tmp_path / "h3_fixture.safetensors"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


def test_a_picked_h3_file_shows_shadow_negative_and_video_detail(tmp_path):
    found = probe.detect(_fake_h3_file(tmp_path))
    assert found["detected"] and found["module"] == "model_minimax_h3"

    shown = {m["id"] for m in routes.manifest(found["traits"])["modules"]}
    assert {"shadow_negative", "video_detail"} <= shown
