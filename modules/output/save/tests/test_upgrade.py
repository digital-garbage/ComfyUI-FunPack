from modules.output.save import upgrade

OUT = {"VAEDecode": ["IMAGE"], "Cut": ["IMAGE", "AUDIO"], "CreateVideo": ["VIDEO"], "SaveVideo": []}.get


def test_old_pair_becomes_one_save():
    role = {"at": "project.video", "input": "fps", "drives": "fps"}
    slots = [{"id": "cut", "node": "Cut", "inputs": {}},
             {"id": "video", "node": "CreateVideo", "roles": [role], "inputs": {"images": ["cut", 0], "audio": ["cut", 1], "fps": 25.0}},
             {"id": "save", "node": "SaveVideo", "inputs": {"video": ["video", 0], "filename_prefix": "X"}}]
    got = upgrade(slots, lambda n: OUT(n) or [])
    assert [s["id"] for s in got] == ["cut", "save"]
    assert got[1] == {"id": "save", "node": "FunPackSaveVideo", "roles": [role],
                      "inputs": {"filename_prefix": "X", "images": ["cut", 0], "audio": ["cut", 1], "fps": 25.0}}
    assert slots[2]["node"] == "SaveVideo"                    # the caller's slots are left alone


def test_unfed_save_takes_the_loose_picture_source():
    slots = [{"id": "cut", "node": "Cut", "inputs": {}}, {"id": "save", "node": "SaveVideo", "inputs": {}}]
    got = upgrade(slots, lambda n: OUT(n) or [])
    assert got[1]["inputs"] == {"filename_prefix": "FunPack", "images": ["cut", 0], "audio": ["cut", 1]}


def test_unfed_save_with_two_candidates_is_left_alone():
    slots = [{"id": "a", "node": "Cut", "inputs": {}}, {"id": "b", "node": "VAEDecode", "inputs": {}},
             {"id": "save", "node": "SaveVideo", "inputs": {}}]
    assert upgrade(slots, lambda n: OUT(n) or [])[2]["node"] == "SaveVideo"
