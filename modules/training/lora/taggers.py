"""Automatic tagging with ONNX taggers (the WD family: a model.onnx and a selected_tags.csv).

Downloaded on request from huggingface.co into the pack's training folder, run with onnxruntime
(the add-on's package), and never loaded until a set is tagged. A tagger answers with tags only;
what to do with them (append, replace, fill empty captions) is the editor's choice.
"""

from __future__ import annotations

import csv
import re
import threading
import urllib.request

from ..._core import config, log

SOURCE = "FunPack LoRA training"
FILES = ("model.onnx", "selected_tags.csv")
HOST = "https://huggingface.co"
_REPO = re.compile(r"\A[A-Za-z0-9][\w.\-]{0,95}/[A-Za-z0-9][\w.\-]{0,95}\Z")

#: Ready-made choices. A person can also name any repo that has the same two files.
TAGGERS = (
    {"id": "wd_vit_v3", "title": "WD ViT v3", "repo": "SmilingWolf/wd-vit-tagger-v3", "hint": "The usual choice: good and fast."},
    {"id": "wd_swinv2_v3", "title": "WD SwinV2 v3", "repo": "SmilingWolf/wd-swinv2-tagger-v3", "hint": "A little more accurate, a little slower."},
    {"id": "wd_convnext_v3", "title": "WD ConvNeXt v3", "repo": "SmilingWolf/wd-convnext-tagger-v3", "hint": "Another balanced pick."},
    {"id": "wd_vit_large_v3", "title": "WD ViT Large v3", "repo": "SmilingWolf/wd-vit-large-tagger-v3", "hint": "Most accurate and the biggest download."},
)
CATEGORY_GENERAL, CATEGORY_CHARACTER, CATEGORY_RATING = 0, 4, 9
KAOMOJI = frozenset({"0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"})

_state: dict[str, dict] = {}
_sessions: dict[str, tuple] = {}
_lock = threading.Lock()


def spec_of(tagger_id: str):
    """A listed tagger, or a custom one named `owner/name`; None when it is neither."""
    for t in TAGGERS:
        if t["id"] == tagger_id:
            return dict(t)
    if _REPO.match(str(tagger_id)):
        return {"id": tagger_id, "title": tagger_id, "repo": tagger_id, "hint": "A repo you named."}
    return None


def folder(tagger_id: str):
    return config.TRAINING_DIR / "taggers" / tagger_id.replace("/", "__")


def ready(tagger_id: str) -> bool:
    return all((folder(tagger_id) / f).is_file() and (folder(tagger_id) / f).stat().st_size > 0 for f in FILES)


def listing() -> list[dict]:
    rows = [dict(t) for t in TAGGERS]
    for d in sorted((config.TRAINING_DIR / "taggers").glob("*")) if (config.TRAINING_DIR / "taggers").is_dir() else []:
        tid = d.name.replace("__", "/", 1)
        if "/" in tid and not any(t["id"] == tid for t in rows) and spec_of(tid):
            rows.append(spec_of(tid))
    for row in rows:
        s = _state.get(row["id"]) or {}
        row["state"] = "downloading" if s.get("state") == "downloading" else "ready" if ready(row["id"]) else "error" if s.get("state") == "error" else "absent"
        row["done"], row["total"], row["error"] = s.get("done", 0), s.get("total", 0), s.get("error", "")
    return rows


def _fetch(url: str, dest, progress) -> None:
    """Stream a URL to a file through a .part name, so a cut connection leaves nothing that looks finished."""
    req = urllib.request.Request(url, headers={"User-Agent": "FunPack"})
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(req, timeout=60) as res, open(part, "wb") as out:
        total = int(res.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = res.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            progress(done, total)
    if total and done != total:
        part.unlink(missing_ok=True)
        raise OSError(f"the download stopped at {done} of {total} bytes")
    part.replace(dest)


def download(tagger_id: str, fetch=None) -> bool:
    """Start the download in a thread. False when it is unknown or already running."""
    spec = spec_of(tagger_id)
    fetch = fetch or _fetch
    with _lock:
        if spec is None or (_state.get(tagger_id) or {}).get("state") == "downloading":
            return False
        _state[tagger_id] = {"state": "downloading", "done": 0, "total": 0, "error": ""}

    def work():
        try:
            target = folder(tagger_id)
            target.mkdir(parents=True, exist_ok=True)
            base = 0
            for name in FILES:
                def progress(done, total, base=base):
                    _state[tagger_id].update(done=base + done, total=base + total if total else 0)
                fetch(f"{HOST}/{spec['repo']}/resolve/main/{name}", target / name, progress)
                base = _state[tagger_id]["done"]
            _sessions.pop(tagger_id, None)
            _state[tagger_id] = {"state": "ready", "done": base, "total": base, "error": ""}
            log.info(SOURCE, f"tagger {spec['title']} downloaded")
        except Exception as exc:  # noqa: BLE001 -- shown beside the tagger, and in the Log
            _state[tagger_id] = {"state": "error", "done": 0, "total": 0, "error": f"{type(exc).__name__}: {exc}"}
            log.warning(SOURCE, f"tagger {spec['title']} could not be downloaded: {exc}")

    threading.Thread(target=work, name=f"funpack-tagger-{tagger_id}", daemon=True).start()
    return True


def remove(tagger_id: str) -> bool:
    import shutil
    if spec_of(tagger_id) is None or (_state.get(tagger_id) or {}).get("state") == "downloading":
        return False
    _sessions.pop(tagger_id, None)
    _state.pop(tagger_id, None)
    shutil.rmtree(folder(tagger_id), ignore_errors=True)
    return True


def read_tags(path) -> list[tuple[str, int]]:
    """[(name, category)] in the model's output order."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "name" not in rows[0] or "category" not in rows[0]:
        raise ValueError("selected_tags.csv has no name and category columns")
    return [(r["name"], int(r["category"])) for r in rows]


def session(tagger_id: str):
    """(onnx session, tag list, input size), loaded once."""
    if tagger_id in _sessions:
        return _sessions[tagger_id]
    if not ready(tagger_id):
        raise ValueError("this tagger is not downloaded yet")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ValueError("onnxruntime is not installed: switch on LoRA training in Settings > Modules and restart ComfyUI") from exc
    wanted = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    sess = ort.InferenceSession(str(folder(tagger_id) / "model.onnx"), providers=wanted)
    shape = sess.get_inputs()[0].shape
    size = int(shape[1]) if isinstance(shape[1], int) else 448
    tags = read_tags(folder(tagger_id) / "selected_tags.csv")
    _sessions[tagger_id] = (sess, tags, size)
    return _sessions[tagger_id]


def prepare(img, size: int):
    """The model's input: padded square on white, resized, BGR, 0 to 255, shape (1, size, size, 3)."""
    import numpy as np
    from PIL import Image
    side = max(img.size)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    canvas = canvas.resize((size, size), Image.BICUBIC)
    return np.asarray(canvas, dtype=np.float32)[:, :, ::-1][None].copy()


def tags_of(probs, tags, general=0.35, character=0.85, characters=False, blocked=(), limit=0) -> list[str]:
    """Names above their threshold, most certain first. Rating tags are never used."""
    blocked = {b.lower() for b in blocked}
    picked = []
    for p, (name, cat) in zip(probs, tags):
        if cat == CATEGORY_RATING or (cat == CATEGORY_CHARACTER and not characters):
            continue
        if p < (character if cat == CATEGORY_CHARACTER else general):
            continue
        label = name if name in KAOMOJI else name.replace("_", " ")
        if label.lower() not in blocked:
            picked.append((float(p), label))
    picked.sort(key=lambda x: -x[0])
    out = [label for _p, label in picked]
    return out[:limit] if limit else out


def run(tagger_id: str, img, **options) -> list[str]:
    sess, tags, size = session(tagger_id)
    probs = sess.run(None, {sess.get_inputs()[0].name: prepare(img, size)})[0][0]
    return tags_of(probs, tags, **options)


# One tagging job at a time, polled by the editor. Results are handed over, never written into a set:
# the editor owns the captions on screen, so a tagger cannot overwrite a caption being typed.
job: dict = {"state": "idle", "done": 0, "total": 0, "error": "", "results": {}, "problems": []}


def autotag(tagger_id: str, targets, options: dict, load) -> bool:
    """Tag `targets` [(item id, path)] in a thread. `load(path)` gives a PIL image. False when one is running."""
    with _lock:
        if job["state"] == "running":
            return False
        job.update(state="running", done=0, total=len(targets), error="", results={}, problems=[])

    def work():
        try:
            session(tagger_id)
            for iid, path in targets:
                if job["state"] != "running":
                    break
                try:
                    job["results"][iid] = run(tagger_id, load(path), **options)
                except Exception as exc:  # noqa: BLE001 -- one unreadable file does not stop the rest
                    job["problems"].append(f"{path.name}: {exc}")
                job["done"] += 1
            if job["state"] == "running":
                job["state"] = "done"
        except Exception as exc:  # noqa: BLE001
            job.update(state="error", error=str(exc))
            log.warning(SOURCE, f"tagging stopped: {exc}")

    threading.Thread(target=work, name="funpack-autotag", daemon=True).start()
    return True


def cancel() -> None:
    if job["state"] == "running":
        job["state"] = "cancelled"
