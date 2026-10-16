"""Named taste keys on disk, and pairing a run's capture with its rating.

Layout: `<pack>/taste/<key>/<kind>.pt` holds `{"rows": [{"prompt_id", "reward",
"rows": {name: tensor}}]}`; `<kind>.pending.pt` holds the latest run's capture and
`<kind>.<prompt id>.pending.pt` the earlier runs' of the SAME Generate.
A key is a folder, so deleting one is one folder and a kind is one file.

A capture waits for a rating until the next Generate starts (`new_generation`), then is forgotten:
nothing is learned from a clip nobody rated. A Generate can be several runs (one per scene that
starts its own run), so every run of the latest Generate waits, each TAGGED with the ComfyUI prompt
id it came from -- rating a clip whose capture was dropped is refused and said, never paired with
the wrong run. Changing your mind on a clip already recorded updates its row instead of adding a
second one.

Keys are disposable (retrained per rental), so there is no migration from v4's
`refinements/` files.
"""

import json
import os
import re
import shutil
import tempfile
import threading
import zipfile
import zlib
from pathlib import Path

import torch

from ..._core import config

ROOT = config.ROOT / "taste"
_KEY = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}\Z")
_LOCK = threading.RLock()
REWARD = {"liked": 1.0, "disliked": -1.0}
# Each rating's vote shrinks by this per later rating: half the vote sits in the
# last ~7 (v4's RECENCY_DECAY; without it early ratings of one style outvoted a
# new style until out-rated one-for-one).
RECENCY_DECAY = 0.9
# Two liked AND two disliked before a direction exists: one of each is two
# points, not a direction.
MIN_PER_GROUP = 2
# A dislike can say WHICH half went wrong: "image" (the picture was bad) or "composition"
# (the picture was fine, the shot plan was not). Plain disliked blames both.
AXES = ("image", "composition")
MAX_ROWS = 200


def valid(key) -> bool:
    return isinstance(key, str) and bool(_KEY.match(key)) and not key.endswith((".", " ")) \
        and not key.lower().endswith(".json")      # `latest.json` lives beside the keys: a key cannot share its name


def _dir(key):
    if not valid(key):
        raise ValueError(f"{key!r} is not a usable taste key name")
    return ROOT / key


def keys():
    if not ROOT.is_dir():
        return []
    return sorted(p.name for p in ROOT.iterdir() if p.is_dir() and valid(p.name))


def _read(path, default):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        return default


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)


_loaded = {}


def load(key, kind):
    """Rated rows of a kind. Re-read only when the file changed: features ask at
    the start of every run, and a REINS log is tens of MB."""
    path = _dir(key) / f"{kind}.pt"
    try:
        stamp = path.stat().st_mtime_ns
    except FileNotFoundError:
        return {"rows": []}
    hit = _loaded.get(path)
    if hit is None or hit[0] != stamp:
        hit = _loaded[path] = (stamp, _read(path, {"rows": []}))
    return hit[1]


def latest_key():
    """The key the most recent capture went to, or None."""
    return _read_latest().get("key")


def kind_path(key, kind):
    """Where a kind's rated rows live on disk (it may not exist yet)."""
    return _dir(key) / f"{kind}.pt"


def clear_kind(key, kind):
    """Forget every rated row and the waiting captures of one kind."""
    kind_path(key, kind).unlink(missing_ok=True)
    for path in [_pending_path(key, kind), *_dir(key).glob(f"{kind}.*.pending.pt")]:
        path.unlink(missing_ok=True)


def current_prompt_id():
    """The ComfyUI prompt running now -- set before any node executes."""
    try:
        from server import PromptServer
        return getattr(PromptServer.instance, "last_prompt_id", None)
    except Exception:                            # noqa: BLE001 -- tests, headless
        return None


def _latest_path():
    return ROOT / "latest.json"


MAX_PENDING = 64          # earlier runs of one Generate kept waiting, per kind
_PID = re.compile(r"\A[A-Za-z0-9_-]{1,80}\Z")      # a prompt id names a file: only plain ones may


def _pending_path(key, kind, prompt_id=None):
    """The latest run's waiting capture (no id), or an earlier run's of the same Generate."""
    return _dir(key) / (f"{kind}.pending.pt" if prompt_id is None else f"{kind}.{prompt_id}.pending.pt")


def _prune_aside(key, kind):
    aside = sorted(_dir(key).glob(f"{kind}.*.pending.pt"), key=lambda p: p.stat().st_mtime_ns)
    for old in aside[:-MAX_PENDING]:
        old.unlink(missing_ok=True)


def capture(key, kind, rows, prompt_id=None, keep=MAX_ROWS, only=None, mixed=False):
    """This run's capture for `kind`, waiting for a rating until the next Generate starts.

    The previous run's capture of the same Generate is kept aside, not overwritten. Kept in the
    dtype given: a banked latent stays half precision. `keep` caps how many rated rows this kind
    holds, oldest dropped first. `only="liked"`: a kind that learns from liked clips alone does not
    bank a disliked one (it could only push the liked ones out under `keep`). `mixed`: its rows are
    legitimately of different shapes (a clip bank is matched by size), so import does not demand one.
    """
    if not rows:
        return
    prompt_id = prompt_id or current_prompt_id()
    clean = {str(k): v.detach().cpu() for k, v in rows.items()}
    with _LOCK:
        latest = _pending_path(key, kind)
        before = _read(latest, None)
        earlier = before.get("prompt_id") if before else None
        if earlier is not None and earlier != prompt_id:
            if _PID.match(str(earlier)):
                os.replace(latest, _pending_path(key, kind, earlier))
        _write(latest, {"prompt_id": prompt_id, "rows": clean, "keep": int(keep), "only": only, "mixed": bool(mixed)})
        _prune_aside(key, kind)
        state = _read_latest()
        runs = state.get("runs", {})
        kinds = runs.setdefault(prompt_id, {}).setdefault(key, [])
        if kind not in kinds:
            kinds.append(kind)
        ROOT.mkdir(parents=True, exist_ok=True)
        _latest_path().write_text(json.dumps({"key": key, "prompt_id": prompt_id, "runs": runs, "forgot": state.get("forgot", [])}))


def _held_by(path, prompt_id):
    """Whether a waiting capture is `prompt_id`'s. An earlier run's carries its id in the name; only the latest
    (`<kind>.pending.pt`) is opened, and one that cannot be read is nobody's (it must not stop the rest going)."""
    if path.name.count(".") > 2:
        return path.name.split(".")[1] == prompt_id
    try:
        return (_read(path, None) or {}).get("prompt_id") == prompt_id
    except Exception:  # noqa: BLE001
        return False


def new_generation(keep=None):
    """A Generate is starting: every capture still waiting for a rating is forgotten, but `keep`'s (the run this
    Generate already queued, which may have captured before this call). -> how many were dropped."""
    with _LOCK:
        dropped = 0
        for key in keys():
            for path in _dir(key).glob("*.pending.pt"):
                if keep is not None and _held_by(path, keep):
                    continue
                path.unlink(missing_ok=True)
                dropped += 1
        state = _read_latest()
        if state:
            runs = state.get("runs", {})
            # remembered so a late rating can say this was why, and only then
            forgot = [*state.get("forgot", []), *(p for p in runs if p != keep)][-256:]
            _latest_path().write_text(json.dumps({"key": state.get("key"), "runs": {p: v for p, v in runs.items() if p == keep}, "forgot": forgot}))
    return dropped


def _read_latest():
    try:
        return json.loads(_latest_path().read_text())
    except (FileNotFoundError, ValueError):
        return {}


def blind(rows, to):
    """`rows` as a learner that cannot judge the `to` axis sees them: a dislike blamed on
    that axis alone is neutral (reward 0, so it counts as neither liked nor disliked)."""
    if not to:
        return rows
    return [dict(r, reward=0.0) if r.get("axis") == to else r for r in rows]


def _restore_pending(key, kind, prompt_id, rows, keep, only=None, mixed=False):
    """A cleared rating puts the capture back to waiting (until the next Generate), so the clip can be
    rated again. The latest slot if it is free or already this run's, else aside under the prompt id."""
    latest = _pending_path(key, kind)
    held = _read(latest, None)
    if held and held.get("prompt_id") not in (None, prompt_id):
        if not _PID.match(str(prompt_id)):
            return
        latest = _pending_path(key, kind, prompt_id)
    _write(latest, {"prompt_id": prompt_id, "rows": rows, "keep": int(keep), "only": only, "mixed": bool(mixed)})
    _prune_aside(key, kind)


def rate(prompt_id, rating, axis=None):
    """-> {"recorded": [kinds], "updated": [kinds], "why": str|None}.

    `rating` None clears: the row is removed, nothing is learned from that clip -- and while its
    Generate is still the latest, the capture waits again so it can be rated again.
    `axis` only goes with "disliked" (see AXES); any other rating drops it.
    """
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ValueError("a rating names the clip it is about (no prompt id was given)")
    if rating is not None and rating not in REWARD:
        raise ValueError(f"{rating!r} is not a rating")
    if axis is not None and axis not in AXES:
        raise ValueError(f"{axis!r} is not a rating axis")
    if axis is not None and rating != "disliked":
        raise ValueError("only a dislike can name what went wrong")
    out = {"recorded": [], "updated": [], "why": None}
    with _LOCK:
        state = _read_latest()
        entry = state.get("runs", {}).get(prompt_id) or {}
        # A clip already recorded: change or remove its row, in every key/kind.
        for key in keys():
            for path in _dir(key).glob("*.pt"):
                if path.name.endswith(".pending.pt"):
                    continue
                data = _read(path, {"rows": []})
                hit = [r for r in data["rows"] if r.get("prompt_id") == prompt_id]
                if not hit:
                    continue
                if rating is None or (hit[-1].get("only") == "liked" and rating != "liked"):
                    # a liked-only bank holds no dislike: it leaves, and may come back if liked again
                    data["rows"] = [r for r in data["rows"] if r.get("prompt_id") != prompt_id]
                    if path.stem in entry.get(key, []):
                        _restore_pending(key, path.stem, prompt_id, hit[-1]["rows"], hit[-1].get("keep", MAX_ROWS),
                                         hit[-1].get("only"), hit[-1].get("mixed", False))
                else:
                    for r in hit:
                        r["reward"] = REWARD[rating]
                        r.pop("axis", None)
                        if axis:
                            r["axis"] = axis
                _write(path, data)
                out["updated"].append(path.stem)
        if out["updated"]:
            return out

        if not entry:
            out["reason"] = "forgotten" if prompt_id in state.get("forgot", []) else "nothing"   # a label the app can act on; `why` is for people
            out["why"] = ("a new Generate started before this clip was rated, and unrated clips are forgotten then"
                          if out["reason"] == "forgotten" else
                          "this run captured nothing: no learning feature acted on it, or ComfyUI reused an "
                          "earlier result instead of sampling again")
            return out
        if rating is None:
            return out
        for key, kinds in entry.items():
            for kind in kinds:
                where = [_pending_path(key, kind)]
                if _PID.match(str(prompt_id)):
                    where.append(_pending_path(key, kind, prompt_id))
                for pending_path in where:
                    pending = _read(pending_path, None)
                    if pending and pending.get("prompt_id") == prompt_id:
                        break
                else:
                    continue
                data = load(key, kind)
                keep = int(pending.get("keep", MAX_ROWS))
                if pending.get("only") == "liked" and rating != "liked":
                    # nothing here learns from it, but a changed mind may still like it
                    out.setdefault("skipped", []).append(kind)
                    continue
                row = {"prompt_id": prompt_id, "reward": REWARD[rating], "rows": pending["rows"], "keep": keep,
                       "only": pending.get("only"), "mixed": bool(pending.get("mixed"))}
                if axis:
                    row["axis"] = axis
                data["rows"].append(row)
                data["rows"] = data["rows"][-keep:]
                _write(_dir(key) / f"{kind}.pt", data)
                pending_path.unlink(missing_ok=True)
                out["recorded"].append(kind)
        if not out["recorded"] and not out.get("skipped"):
            out["why"] = ("this clip's capture was not kept (more runs waited than are kept, or it was "
                          "cleared); nothing was learned")
    return out


def counts(key, kind, name=None):
    """(liked, disliked) rows for a kind, optionally only rows holding `name`."""
    rows = [r for r in load(key, kind)["rows"] if name is None or str(name) in r["rows"]]
    return (sum(r["reward"] > 0 for r in rows), sum(r["reward"] < 0 for r in rows))


def direction(key, kind, name):
    """-> (unit vector | None, n_liked, n_disliked).

    Recency-weighted mean(liked) - mean(disliked) over the rows holding `name`,
    split by the SIGN of the rating only. Raw rows, not normalised: a hidden
    state's magnitude carries content (v4's trajectory-probe lesson).
    """
    name = str(name)
    rows = [r for r in load(key, kind)["rows"] if name in r["rows"]]
    aged = [(r["rows"][name], r["reward"], RECENCY_DECAY ** (len(rows) - 1 - i))
            for i, r in enumerate(rows)]
    liked = [(d, a) for d, w, a in aged if w > 0]
    disliked = [(d, a) for d, w, a in aged if w < 0]
    if len(liked) < MIN_PER_GROUP or len(disliked) < MIN_PER_GROUP:
        return None, len(liked), len(disliked)

    def wmean(group):
        ds = torch.stack([d for d, _ in group]).float()
        a = torch.tensor([a for _, a in group], dtype=ds.dtype)
        return (ds * a.view(-1, *([1] * (ds.dim() - 1)))).sum(0) / a.sum()

    diff = wmean(liked) - wmean(disliked)
    norm = diff.norm()
    if not torch.isfinite(norm) or norm <= 1e-8:
        return None, len(liked), len(disliked)
    return diff / norm, len(liked), len(disliked)


def _forget_key_runs(key):
    """The key's waiting runs died with its folder; other keys' runs stay ratable."""
    state = _read_latest()
    if not state:
        return
    runs = {}
    for pid, per_key in state.get("runs", {}).items():
        left = {k: v for k, v in per_key.items() if k != key}
        if left:
            runs[pid] = left
    state["runs"] = runs
    if state.get("key") == key:
        state["key"] = None
    _latest_path().write_text(json.dumps(state))


def delete(key):
    with _LOCK:
        shutil.rmtree(_dir(key), ignore_errors=True)
        _forget_key_runs(key)


# --- moving a key between machines -------------------------------------------
#
# A key is a folder of `<kind>.pt` files, so exporting is a zip of them. Importing reads a zip a
# stranger may have made: only plainly named `.pt` files, bounded in count and size, each one
# loadable with weights_only (no pickled code) before anything touches the real folder.

_KIND_FILE = re.compile(r"\A[A-Za-z0-9_]{1,48}\.pt\Z")
MAX_FILES = 64
MAX_BYTES = 4 * 1024 ** 3


def export_key(key, out_path):
    """Zip `key`'s rated rows into `out_path` (waiting captures are not part of what was learned)."""
    folder = _dir(key)
    if not folder.is_dir():
        raise ValueError(f"there is no taste key called {key!r}")
    with _LOCK:
        found = [p for p in sorted(folder.glob("*.pt")) if _KIND_FILE.match(p.name)]
        if not found:
            raise ValueError(f"{key!r} has not learned anything yet: there is nothing to export")
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as zf:
            for path in found:
                zf.write(path, path.name)
    return out_path


def _sound_rows(data):
    """True when `data` is what the learners consume: a dict of rated rows, each with a number
    for a reward and tensors for what was captured. Loading it safely is not enough -- a row of
    the wrong shape would break every later rating."""
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        return False
    shapes = {}
    for row in data["rows"]:
        if not isinstance(row, dict) or isinstance(row.get("reward"), bool) or not isinstance(row.get("reward"), (int, float)):
            return False
        if not isinstance(row.get("prompt_id"), str) or not row["prompt_id"]:
            return False
        if row.get("axis") not in (None, *AXES):
            return False
        captured = row.get("rows")
        if not isinstance(captured, dict) or not all(isinstance(v, torch.Tensor) for v in captured.values()):
            return False
        # What is stacked later must agree: one name, one shape (a key trained across two model
        # families would otherwise abort every sampling that reads it).
        if row.get("mixed") is True:
            continue                                  # a clip bank: rows of different sizes are the point
        for name, v in captured.items():
            if shapes.setdefault(name, tuple(v.shape)) != tuple(v.shape):
                return False
    return True


def import_key(key, zip_path, overwrite=False):
    """-> number of kinds imported. FileExistsError if `key` exists and `overwrite` is off."""
    dest = _dir(key)
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise ValueError("that is not an exported taste key (not a zip file)") from exc
    with zf:
        infos = zf.infolist()
        if not infos or len(infos) > MAX_FILES:
            raise ValueError(f"an exported key holds 1 to {MAX_FILES} files, this has {len(infos)}")
        for info in infos:
            if not _KIND_FILE.match(info.filename):
                raise ValueError(f"{info.filename!r} is not part of a taste key")
        if sum(i.file_size for i in infos) > MAX_BYTES:
            raise ValueError("that key is larger than 4 GB")
        ROOT.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".import-", dir=ROOT))
        try:
            for info in infos:
                target = stage / info.filename
                try:
                    with zf.open(info) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
                except (zipfile.BadZipFile, zlib.error, RuntimeError, NotImplementedError, EOFError) as exc:
                    raise ValueError(f"{info.filename} cannot be read from the zip ({type(exc).__name__}): "
                                     f"it is damaged, encrypted or compressed in an unsupported way") from exc
                try:
                    data = torch.load(target, map_location="cpu", weights_only=True)
                except Exception as exc:                     # noqa: BLE001
                    raise ValueError(f"{info.filename} is not a readable taste file ({type(exc).__name__})") from exc
                if not _sound_rows(data):
                    raise ValueError(f"{info.filename} does not hold rated rows of the shape FunPack reads")
            with _LOCK:
                if dest.exists():
                    if not overwrite:
                        raise FileExistsError(key)
                    shutil.rmtree(dest)
                os.replace(stage, dest)
                # Anything waiting to be rated under this name died with the old folder.
                _forget_key_runs(key)
                return len(infos)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
