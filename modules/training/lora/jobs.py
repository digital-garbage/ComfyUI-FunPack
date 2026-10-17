"""The training run in progress, and the record of finished ones.

The training node reports each step here; the editor polls `snapshot(since)` for the loss curve.
One run at a time (ComfyUI runs one graph at a time). A finished run is written to runs/<id>.json
so its graph can be looked at, and compared, later.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid

from ..._core import config, log

MAX_KEPT_LOSSES = 4000          # a saved run keeps a thinned curve: a 100000 step run is not a 100000 number file
_LOCK = threading.Lock()
_ID = re.compile(r"\A[0-9a-f]{12}\Z")
IDLE = {"state": "idle", "run": "", "set": "", "name": "", "step": 0, "total": 0, "started": 0.0, "message": "", "error": "",
        "lora": "", "samples": [], "settings": {}}
_now: dict = dict(IDLE)
_loss: list[float] = []
_stop = False


def _dir():
    path = config.TRAINING_DIR / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def trail(message: str) -> None:
    """A stage of the run in the Log, which is kept on disk as it is written: when ComfyUI dies mid-run, the last line says where."""
    rss = ""
    try:
        with open("/proc/self/statm") as f:
            rss = f" [process {int(f.read().split()[1]) * 4096 / 1024 ** 3:.1f} GiB in memory]"
    except OSError:
        pass
    log.info("FunPack LoRA training", message + rss)


def begin(set_name: str, name: str, total: int, settings: dict, prompt_id: str = "") -> str:
    """Start a record. A run that was still marked running (ComfyUI restarted under it) is simply replaced."""
    global _stop
    with _LOCK:
        _stop = False
        _loss.clear()
        _now.clear()
        _now.update(IDLE, state="training", run=uuid.uuid4().hex[:12], set=set_name, name=name, total=int(total), started=time.time(),
                    settings=dict(settings), prompt_id=prompt_id, message="Loading the model and the training data")
        return _now["run"]


def step(loss: float) -> None:
    with _LOCK:
        if _now["state"] == "training":
            _loss.append(float(loss))
            _now["step"] = len(_loss)
            _now["message"] = ""


def say(message: str, state: str | None = None) -> None:
    trail(message)
    with _LOCK:
        _now["message"] = message
        if state and _now["state"] not in ("done", "error", "stopped"):
            _now["state"] = state


def samples(urls: list) -> None:
    with _LOCK:
        _now["samples"] = list(urls)


def thin(values, keep: int = MAX_KEPT_LOSSES) -> list[float]:
    """At most `keep` numbers, evenly spread, the last one always kept."""
    if len(values) <= keep:
        return list(values)
    stride = len(values) / keep
    out = [values[int(i * stride)] for i in range(keep - 1)]
    return out + [values[-1]]


def finish(state: str, lora: str = "", error: str = "") -> None:
    """state: done, stopped (the person stopped it) or error. Writes the run's record."""
    with _LOCK:
        if _now["run"] == "" or _now["state"] in ("done", "error", "stopped"):
            return
        _now.update(state=state, lora=lora, error=error, message="")
        record = {k: _now.get(k) for k in ("run", "set", "name", "total", "started", "settings", "lora", "samples", "error")}
        record.update(state=state, steps=len(_loss), seconds=round(time.time() - _now["started"], 1),
                      final_loss=_loss[-1] if _loss else None, loss=thin(_loss))
    try:
        (_dir() / f"{record['run']}.json").write_text(json.dumps(record), encoding="utf-8")
    except OSError:
        pass                                          # the run is over either way; the history just lacks it


def snapshot(since: int = 0) -> dict:
    with _LOCK:
        since = max(0, min(int(since), len(_loss)))
        return {**{k: v for k, v in _now.items() if k != "settings"}, "loss": _loss[since:], "loss_from": since,
                "elapsed": round(time.time() - _now["started"], 1) if _now["started"] and _now["state"] == "training" else 0}


def history() -> list[dict]:
    out = []
    for p in _dir().glob("*.json"):
        if not _ID.match(p.stem):
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(r, dict):
            out.append({k: r.get(k) for k in ("run", "set", "name", "state", "steps", "total", "seconds", "final_loss", "started", "lora")})
    return sorted(out, key=lambda r: r.get("started") or 0, reverse=True)


def record(run_id: str) -> dict | None:
    if not _ID.match(str(run_id)):
        return None
    try:
        r = json.loads((_dir() / f"{run_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return r if isinstance(r, dict) else None


def forget(run_id: str) -> bool:
    if not _ID.match(str(run_id)):
        return False
    path = _dir() / f"{run_id}.json"
    if not path.is_file():
        return False
    path.unlink()
    return True


def request_stop() -> bool:
    """Ask the running training to end at its next step. False when nothing is training."""
    global _stop
    with _LOCK:
        if _now["state"] != "training":
            return False
        _stop = True
        _now["message"] = "Stopping"
        return True


def stop_requested() -> bool:
    return _stop
