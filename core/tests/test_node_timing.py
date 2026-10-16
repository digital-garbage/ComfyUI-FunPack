from core import log, node_timing


def test_a_run_is_split_by_node_and_logged_once_at_the_end():
    log._reset()
    seen = []
    class Server:
        def send_sync(self, event, data, sid=None):
            seen.append(event)
    s = Server()
    node_timing.install(s)
    node_timing.install(s)                                   # twice: still wrapped once
    obs = node_timing.observe
    obs("execution_start", {"prompt_id": "p"}, now=0.0)
    obs("executing", {"prompt_id": "p", "node": "te"}, now=1.0)
    obs("executing", {"prompt_id": "p", "node": "sampler"}, now=21.0)
    obs("executing", {"prompt_id": "p", "node": "save"}, now=51.0)
    obs("execution_success", {"prompt_id": "p"}, now=70.0)
    line = [r["message"] for r in log.history() if r["source"] == "FunPack Timing"]
    assert line == ["70.0s in nodes: sampler 30.0s · te 20.0s · save 19.0s · (before the first node) 1.0s"]
    s.send_sync("status", {})
    assert seen == ["status"], "messages still go out"


def test_a_sampler_reports_its_first_steps_apart_from_the_rest():
    log._reset()
    obs = node_timing.observe
    obs("execution_start", {"prompt_id": "p"}, now=0.0)
    obs("executing", {"prompt_id": "p", "node": "sampler"}, now=1.0)
    for i, at in enumerate((31.0, 33.0, 35.0, 37.0)):
        obs("progress", {"prompt_id": "p", "value": i + 1, "max": 4}, now=at)
    obs("execution_success", {"prompt_id": "p"}, now=38.0)
    lines = [r["message"] for r in log.history() if r["source"] == "FunPack Timing"]
    assert lines[0] == "sampler: step 1 done 30.0s after the node started, step 2 took 2.0s, the 2 after that average 2.0s"
    assert len(lines) == 2, "the per-node total still follows"
