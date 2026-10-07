"""Where a run's time went, node by node: ComfyUI only logs the total ("Prompt executed in 70s").

Read off the messages ComfyUI already sends its client: `executing` as each node starts, `execution_success`
(or error/interrupted) at the end. The time between two starts belongs to the first node, its model loading
included. Logged once per run; runs without a client id (no messages) are not timed.
"""

import time

from . import log

_runs = {}
_END = ("execution_success", "execution_error", "execution_interrupted")
TAG = "_funpack_node_timing"


def observe(event, data, now=None):
    if not isinstance(data, dict) or not data.get("prompt_id"):
        return
    pid, now = data["prompt_id"], time.perf_counter() if now is None else now
    if event == "execution_start":
        _runs[pid] = {"start": now, "node": "(before the first node)", "at": now, "spent": {}}
        return
    run = _runs.get(pid)
    if run is None or (event != "executing" and event not in _END):
        return
    run["spent"][run["node"]] = run["spent"].get(run["node"], 0.0) + now - run["at"]
    if event == "executing" and data.get("node") is not None:
        run["node"], run["at"] = str(data.get("display_node") or data["node"]), now
        return
    if event in _END:
        _runs.pop(pid, None)
        parts = sorted(run["spent"].items(), key=lambda kv: -kv[1])
        shown = " · ".join(f"{name} {secs:.1f}s" for name, secs in parts if secs >= 0.3)
        log.info("FunPack Timing", f"{now - run['start']:.1f}s in nodes: {shown or 'nothing over 0.3s'}")


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
