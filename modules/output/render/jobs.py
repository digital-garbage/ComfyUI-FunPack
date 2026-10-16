"""Background jobs (final render, combined export): ffmpeg can outlast a tunnel's request
timeout, so the request returns an id and the page polls for it."""

import asyncio
import uuid
from collections import OrderedDict

MAX_JOBS = 50                         # finished jobs are kept this long for the poll that reads them

_jobs: "OrderedDict[str, dict]" = OrderedDict()


def new() -> str:
    job = uuid.uuid4().hex
    _jobs[job] = {"state": "queued"}
    while len(_jobs) > MAX_JOBS:
        _jobs.popitem(last=False)
    return job


def get(job: str) -> dict | None:
    return _jobs.get(job)


async def run(job: str, fn, *args, error) -> None:
    """Run the blocking `fn(*args)` off the event loop and record how it ended. `error` is the
    exception class whose message is meant for the person; anything else is reported plainly."""
    _jobs[job] = {"state": "running"}
    try:
        result = await asyncio.to_thread(fn, *args)
        _jobs[job] = {"state": "done", **result}
    except error as exc:
        _jobs[job] = {"state": "error", "detail": str(exc)}
    except Exception as exc:                          # noqa: BLE001
        _jobs[job] = {"state": "error", "detail": f"{type(exc).__name__}: {exc}"}
