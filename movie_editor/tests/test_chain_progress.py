from movie_editor.backend.chain_progress import begin, finish, read_for_prompt, update


def test_chain_progress_partial_read():
    begin("pid1", "run-a", 3)
    update("run-a", 1, {"kind": "videos", "filename": "partial_s1.mp4", "subfolder": "funpack_chain_progress/run-a", "type": "temp"})
    snap = read_for_prompt("pid1")
    assert snap is not None
    assert snap["partial"] is True
    assert snap["completed_scenes"] == 1
    assert snap["scene_count"] == 3
    assert snap["media"][0]["filename"] == "partial_s1.mp4"
    update("run-a", 3, {"kind": "videos", "filename": "partial_s3.mp4", "subfolder": "x", "type": "temp"})
    assert read_for_prompt("pid1") is None
    finish("pid1")
    assert read_for_prompt("pid1") is None


def test_chain_progress_single_scene_skipped():
    begin("pid2", "run-b", 1)
    assert read_for_prompt("pid2") is None
    finish("pid2")
