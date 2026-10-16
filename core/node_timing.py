"""Where a run's time went, node by node: ComfyUI only logs the total ("Prompt executed in 70s").

Read off the messages ComfyUI already sends its client: `executing` as each node starts, `execution_success`
(or error/interrupted) at the end. The time between two starts belongs to the first node, its model loading
included. Logged once per run; runs without a client id (no messages) are not timed.
"""

import time

from . import log

_runs = {}
_END = ("execution_success", "execution_error", "execution_interrupted")
STEPS = 3          # a node that reports at least this many steps gets a per-step line
TAG = "_funpack_node_timing"


def observe(event, data, now=None):
    if not isinstance(data, dict) or not data.get("prompt_id"):
        return
    pid, now = data["prompt_id"], time.perf_counter() if now is None else now
    if event == "execution_start":
        _runs[pid] = {"start": now, "node": "(before the first node)", "at": now, "spent": {}}
        return
    run = _runs.get(pid)
    if run is None:
        return
    if event == "progress":
        run.setdefault("marks", []).append(now)
        return
    if event != "executing" and event not in _END:
        return
    _steps(run, now)
    run["spent"][run["node"]] = run["spent"].get(run["node"], 0.0) + now - run["at"]
    if event == "executing" and data.get("node") is not None:
        run["node"], run["at"] = str(data.get("display_node") or data["node"]), now
        return
    if event in _END:
        _runs.pop(pid, None)
        parts = sorted(run["spent"].items(), key=lambda kv: -kv[1])
        shown = " · ".join(f"{name} {secs:.1f}s" for name, secs in parts if secs >= 0.3)
        log.info("FunPack Timing", f"{now - run['start']:.1f}s in nodes: {shown or 'nothing over 0.3s'}")


def _steps(run, now):
    """The sampler's own step times, once its node is over: how long until step 1 was done (loading and
    first-call setup included), how long step 2 took, and what the later steps average."""
    marks, run["marks"] = run.get("marks") or [], []
    if len(marks) < STEPS:
        return
    first, second = marks[0] - run["at"], marks[1] - marks[0]
    rest = (marks[-1] - marks[1]) / (len(marks) - 2)
    log.info("FunPack Timing", f"{run['node']}: step 1 done {first:.1f}s after the node started, step 2 took "
             f"{second:.1f}s, the {len(marks) - 2} after that average {rest:.1f}s")


def install(server):
    """Listen to the server's outgoing messages; once, and never in the way of sending them."""
    send = getattr(server, "send_sync", None)
    if send is None or getattr(send, TAG, False):
        return

    def send_sync(event, data, sid=None):
        try:
            observe(event, data)
        except Exception:                                      # noqa: BLE001 -- a report must never stop a message
            pass
        return send(event, data, sid)

    setattr(send_sync, TAG, True)
    server.send_sync = send_sync
