import copy
import json
import os
import random
import re
from hashlib import md5
from datetime import datetime, timezone

import folder_paths
import torch
from aiohttp import web
from server import PromptServer

try:
    from .conditioning import (
        normalize_refiner_rating,
        refinement_state_path,
        serializable_to_tensor,
        tensor_to_serializable,
    )
except ImportError:
    from conditioning import (
        normalize_refiner_rating,
        refinement_state_path,
        serializable_to_tensor,
        tensor_to_serializable,
    )


TEMPLATE_NONE = "-None-"
REFINEMENT_KEY_NONE = "-None-"
SCENE_NONE = "-None-"
TEMPLATE_DB_VERSION = 1
SCENE_CATEGORIES = {
    "negative": {"bad", "blurry", "worst", "low", "noise", "deformed", "artifact", "ugly", "broken"},
    "action": {"walk", "walking", "run", "running", "turn", "turning", "dance", "dancing", "jump", "jumping", "move", "moving", "motion", "hold", "holding", "look", "looking", "smile", "smiling"},
    "camera": {"camera", "shot", "closeup", "close-up", "wide", "angle", "zoom", "pan", "dolly", "tracking", "handheld", "focus", "framing", "lens", "viewpoint"},
    "subject": {"woman", "man", "girl", "boy", "person", "character", "robot", "creature", "dragon", "animal", "vehicle", "object", "monster"},
    "appearance": {"hair", "eyes", "face", "skin", "dress", "jacket", "armor", "outfit", "clothing", "pose", "expression", "body", "wearing", "shirt", "coat", "robe", "boots", "hat", "mask"},
    "environment": {"forest", "city", "street", "room", "beach", "mountain", "temple", "sunset", "night", "rain", "snow", "sky", "background", "setting", "landscape", "interior", "exterior"},
    "style": {"anime", "cinematic", "photorealistic", "painterly", "illustration", "stylized", "realistic", "film", "noir", "dramatic", "soft", "lighting", "moody", "neon", "gothic"},
    "quality": {"masterpiece", "best", "quality", "detailed", "sharp", "highres", "high-res", "ultra", "perfect", "clean", "realism", "smooth", "crisp", "polished", "4k", "8k"},
    "details": {"reflection", "reflections", "texture", "textures", "shadow", "shadows", "smoke", "dust", "particles", "prop", "props", "fabric", "glass", "sparkles", "pattern", "grain"},
}
SCENE_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "its", "of", "on", "or", "the", "their", "then",
    "there", "through", "to", "with", "without",
}


def template_store_dir():
    user_dir_getter = getattr(folder_paths, "get_user_directory", None)
    if callable(user_dir_getter):
        base_dir = user_dir_getter()
    else:
        base_dir = getattr(folder_paths, "user_directory", None)
    if not base_dir:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "default", "FunPack")


def template_store_path():
    return os.path.join(template_store_dir(), "templates.json")



def shortcut_store_path():
    return os.path.join(template_store_dir(), "shortcuts.json")


def transition_store_path():
    """On-disk store for custom prompt split markers (formerly transitions.json)."""
    return os.path.join(template_store_dir(), "promptsplit.json")


def _legacy_transition_store_path():
    return os.path.join(template_store_dir(), "transitions.json")


def _migrate_transition_store_if_needed() -> None:
    path = transition_store_path()
    legacy = _legacy_transition_store_path()
    if os.path.exists(path) or not os.path.exists(legacy):
        return
    try:
        os.rename(legacy, path)
    except OSError:
        import shutil
        shutil.copy2(legacy, path)


def refinement_store_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "refinements")


def normalize_refinement_key(value):
    value = str(value or "").strip()
    # Keys are stored readably as <key>.json now, so a user (or a dropped filename)
    # that includes the extension still resolves to the same key.
    if value.lower().endswith(".json"):
        value = value[:-5].strip()
    if not value or value == REFINEMENT_KEY_NONE:
        return ""
    return value


def empty_v2_refinement_state(refinement_key):
    return {
        "version": 2,
        "refinement_key": refinement_key,
        "state_namespace": "clip",
        "global": {
            "total_iterations": 0,
            "avg_reward_ema": 0.0,
            "good_streak": 0,
            "bad_streak": 0,
            "last_rating_label": "Initial discovery",
            "last_missing_axes": [],
            "phrase_memory": {},
            "axis_conditioning_memory": {},
            "lora_weight_memory": {},
            "preferred_context_memory": {},
            "loss_history": [],
        },
        "prompt_histories": {},
        "last_run": None,
    }


def refinement_key_path(refinement_key):
    return refinement_state_path(refinement_key, "clip", prefix="refine_v2")


def normalize_refinement_state(data, fallback_key=""):
    if not isinstance(data, dict):
        return None, ""
    key = normalize_refinement_key(data.get("refinement_key") or fallback_key)
    if not key:
        return None, ""
    state = dict(data)
    state["version"] = 2
    state["refinement_key"] = key
    state["state_namespace"] = "clip"
    state.setdefault("global", {})
    state["global"].setdefault("phrase_memory", {})
    state["global"].setdefault("axis_conditioning_memory", {})
    state["global"].setdefault("lora_weight_memory", {})
    state["global"].setdefault("preferred_context_memory", {})
    state["global"].setdefault("loss_history", [])
    state.setdefault("prompt_histories", {})
    state.setdefault("last_run", None)
    return state, key


def _coerce_refinement_payload(data):
    """Unwrap common export envelopes so any shared / HuggingFace refinement file
    resolves to its inner V2 clip state. The Studio export and the Editor import
    both round-trip through ``{"state": {...}}`` wrappers in some versions."""
    if isinstance(data, dict) and isinstance(data.get("state"), dict):
        data = data["state"]
    return data if isinstance(data, dict) else None


# A refinement key is stored as <key>.json. Sidecars in the same folder are either
# non-.json (value_fn .pt, latents .pt) or carry a mode suffix (<key>.sampler_ctx.json),
# and legacy installs may still hold opaque <prefix>_<md5>.json files. All are
# excluded from the key listing below.
_SIDECAR_JSON_SUFFIXES = (".sampler_ctx.json",)
_LEGACY_HASHED_RE = re.compile(r"_[0-9a-f]{32}$|^[0-9a-f]{32}$")


def load_refinement_key_state(refinement_key, create=False):
    key = normalize_refinement_key(refinement_key)
    if not key:
        return None, "", "No refinement key selected."
    path = refinement_key_path(key)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            state, loaded_key = normalize_refinement_state(data, key)
            if state is None:
                return None, key, f"Refinement key '{key}' is unreadable."
            return state, loaded_key, f"Loaded refinement key '{loaded_key}'."
        except (json.JSONDecodeError, OSError, ValueError):
            return None, key, f"Refinement key '{key}' is unreadable."
    if not create:
        return None, key, f"Refinement key '{key}' does not exist."
    state = empty_v2_refinement_state(key)
    save_refinement_key_state(state, key)
    return state, key, f"Created refinement key '{key}'."


def save_refinement_key_state(state, refinement_key):
    key = normalize_refinement_key(refinement_key)
    if not key:
        return ""
    state, key = normalize_refinement_state(state, key)
    if state is None:
        return ""
    path = refinement_key_path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)
    return path


def refinement_key_names():
    # The key IS the filename: every <key>.json in the folder is a key. Skip mode
    # sidecars, legacy hashed files, and internal dunder keys (e.g. the Absolute
    # taste store).
    keys = set()
    directory = refinement_store_dir()
    if os.path.isdir(directory):
        for filename in os.listdir(directory):
            low = filename.lower()
            if not low.endswith(".json") or low.endswith(_SIDECAR_JSON_SUFFIXES):
                continue
            stem = filename[:-5]
            if stem.startswith("__") or _LEGACY_HASHED_RE.search(stem):
                continue
            if not os.path.isfile(os.path.join(directory, filename)):
                continue
            key = normalize_refinement_key(stem)
            if key:
                keys.add(key)
    return [REFINEMENT_KEY_NONE] + sorted(keys)


def _refinement_key_files(key):
    """Every top-level store file that makes up one refinement key: the canonical
    <key>.json plus its sidecars (<key>.value_fn.pt, <key>.sampler_ctx.json, and any
    other <key>.<mode>.* file). The attn_maps/ banks + velocity store are handled
    separately via clear_refinement_data. Used by delete_refinement_key so a key is
    removed atomically instead of leaving sidecars orphaned."""
    paths = set()
    # Known canonical + sidecar modes (deterministic paths).
    paths.add(refinement_state_path(key, "clip", prefix="refine_v2"))
    paths.add(refinement_state_path(key, "value_fn", prefix="refine_v2", extension="pt"))
    paths.add(refinement_state_path(key, "sampler_ctx", prefix="refine_v2"))
    # Defensive: sweep any other top-level <key>.* sidecar so a future sidecar type
    # can't be orphaned — that orphaning is the whole bug this delete path fixes.
    directory = refinement_store_dir()
    if os.path.isdir(directory):
        try:
            clip_name = os.path.basename(refinement_state_path(key, "clip", prefix="refine_v2"))
            stem = clip_name[:-5] if clip_name.lower().endswith(".json") else clip_name
            for filename in os.listdir(directory):
                if filename == clip_name or filename.startswith(stem + "."):
                    full = os.path.join(directory, filename)
                    if os.path.isfile(full):
                        paths.add(full)
        except OSError:
            pass
    return paths


def delete_refinement_key(refinement_key):
    """Atomically remove EVERY file that makes up a refinement key — the canonical
    <key>.json AND its sidecars (value function, sampler context, blessed attention
    maps / K/V identity banks, creativity latent, velocity-bias store) — plus the
    in-process velocity memory.

    Deleting only <key>.json (e.g. by hand in the folder) used to orphan the value
    function and blessed banks; those keep steering future generations and survive a
    ComfyUI restart, which is the root of 'I cleared the key but generations stayed
    dirty'. This makes deletion match the user's mental model: key gone = gone."""
    key = normalize_refinement_key(refinement_key)
    if not key:
        return {"deleted": "", "removed": 0, "error": "No refinement key selected."}
    removed = 0
    for path in _refinement_key_files(key):
        try:
            if os.path.isfile(path):
                os.remove(path)
                removed += 1
        except OSError as exc:
            print(f"[FunPack] delete key '{key}': could not remove {path}: {exc}")
    # attn_maps banks (blessed/temp maps, attn weights, K/V) + creativity latent +
    # the on-disk AND in-process velocity-bias memory.
    try:
        try:
            from .ltx_enhancements import clear_refinement_data
        except ImportError:
            from ltx_enhancements import clear_refinement_data
        clear_refinement_data(key)
    except Exception as exc:
        print(f"[FunPack] delete key '{key}': enhancement cleanup failed: {exc}")
    return {"deleted": key, "removed": removed}


def _absolute_store_paths():
    try:
        from .conditioning import FUNPACK_ABSOLUTE_KEY
    except ImportError:
        from conditioning import FUNPACK_ABSOLUTE_KEY
    return (
        refinement_state_path(FUNPACK_ABSOLUTE_KEY, "clip", prefix="refine_v2"),
        refinement_state_path(FUNPACK_ABSOLUTE_KEY, "value_fn", prefix="refine_v2", extension="pt"),
    )


def absolute_store_info():
    """Surface the keyless Absolute 'global taste' store — it learns from every rated
    generation across all prompts, is invisible in the key list (dunder name), and is
    only otherwise wiped by Session Reset. Returned so the UI can show + clear it."""
    json_path, vf_path = _absolute_store_paths()
    info = {"exists": os.path.isfile(json_path) or os.path.isfile(vf_path),
            "total_iterations": 0, "liked_count": 0, "bad_count": 0}
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            g = data.get("global", data) if isinstance(data, dict) else {}
            info["total_iterations"] = int(g.get("total_iterations", 0) or 0)
            info["liked_count"] = int((g.get("liked_dir") or {}).get("direction_count", 0) or 0)
            info["bad_count"] = int((g.get("bad_dir") or {}).get("direction_count", 0) or 0)
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return info


def clear_absolute_store():
    """Wipe the keyless Absolute taste store (direction pool + global value function)."""
    removed = 0
    for path in _absolute_store_paths():
        try:
            if os.path.isfile(path):
                os.remove(path)
                removed += 1
        except OSError as exc:
            print(f"[FunPack] clear absolute store: could not remove {path}: {exc}")
    return {"cleared": True, "removed": removed}


def empty_template_db():
    return {
        "version": TEMPLATE_DB_VERSION,
        "source": "ComfyUI-FunPack",
        "templates": {},
    }


def load_template_db():
    path = template_store_path()
    if not os.path.exists(path):
        return empty_template_db()
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError, ValueError):
        return empty_template_db()

    if not isinstance(data, dict):
        return empty_template_db()
    templates = data.get("templates")
    if not isinstance(templates, dict):
        data["templates"] = {}
    data.setdefault("version", TEMPLATE_DB_VERSION)
    data.setdefault("source", "ComfyUI-FunPack")
    return data


def save_template_db(data):
    os.makedirs(template_store_dir(), exist_ok=True)
    with open(template_store_path(), "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def template_names():
    data = load_template_db()
    names = sorted(
        name for name in data.get("templates", {}).keys()
        if isinstance(name, str) and name.strip()
    )
    return [TEMPLATE_NONE] + names


def empty_shortcut_db():
    return {
        "version": 1,
        "source": "ComfyUI-FunPack",
        "shortcuts": {},
        # Managed grouping list for the Composer. Each entry is
        # {"name": <category>, "sub_categories": [<sub>, ...]}. Persisted so a
        # category survives even with no shortcut assigned to it yet.
        "categories": [],
    }


def _clean_label(value):
    return re.sub(r"\s+", " ", str(value or "").strip())


def normalize_category_list(value):
    """List of {"name", "sub_categories": [...]} — deduped (case-insensitive),
    order-preserving, sub-categories nested under their parent category."""
    if not isinstance(value, list):
        return []
    out = []
    index = {}  # lower(name) -> position in out
    for entry in value:
        if isinstance(entry, str):
            entry = {"name": entry, "sub_categories": []}
        if not isinstance(entry, dict):
            continue
        name = _clean_label(entry.get("name"))
        if not name:
            continue
        key = name.lower()
        if key not in index:
            index[key] = len(out)
            out.append({"name": name, "sub_categories": []})
        bucket = out[index[key]]["sub_categories"]
        seen = {s.lower() for s in bucket}
        for sub in entry.get("sub_categories", []) or []:
            sub = _clean_label(sub)
            if sub and sub.lower() not in seen:
                seen.add(sub.lower())
                bucket.append(sub)
    return out


def _union_category(categories, name, sub=""):
    """Ensure `name` (and optional `sub`) exists in the managed list. Mutates + returns."""
    name = _clean_label(name)
    if not name:
        return categories
    key = name.lower()
    entry = next((c for c in categories if c["name"].lower() == key), None)
    if entry is None:
        entry = {"name": name, "sub_categories": []}
        categories.append(entry)
    sub = _clean_label(sub)
    if sub and sub.lower() not in {s.lower() for s in entry["sub_categories"]}:
        entry["sub_categories"].append(sub)
    return categories


def shortcut_key(value):
    value = re.sub(r"[^\w'’.-]+", " ", str(value or "").strip().lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def shortcut_list(value):
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            value = parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            value = re.split(r"[,;\n]+", value)
    if not isinstance(value, list):
        return []
    result = []
    seen = set()
    for item in value:
        text = re.sub(r"\s+", " ", str(item or "").strip())
        key = shortcut_key(text)
        if text and key and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def normalize_shortcut_name(value, fallback=""):
    value = re.sub(r"\s+", " ", str(value or "").strip())
    if value:
        return value
    fallback = re.sub(r"\s+", " ", str(fallback or "").strip())
    return fallback


def normalize_shortcut_item(item, fallback_name=""):
    if not isinstance(item, dict):
        return None
    name = normalize_shortcut_name(item.get("name"), fallback_name)
    triggers = shortcut_list(item.get("triggers", item.get("activation_words", item.get("activation", []))))
    replacements = _shortcut_replacements(item.get("replacements", item.get("replacement", []))) or [""]
    if not name:
        name = triggers[0] if triggers else ""
    if not name or not triggers:
        return None
    created = str(item.get("created_at") or now_iso())
    return {
        "name": name,
        "enabled": bool(item.get("enabled", True)),
        "triggers": triggers,
        "replacements": replacements,
        # Optional per-shortcut refinement key. Empty string = bound to the run's default
        # key (no special handling). A non-default key means "when this shortcut fires,
        # this key is being trained" — Studio steers/rates the scene against it.
        "refinement_key": normalize_refinement_key(item.get("refinement_key", "")),
        # Optional grouping for the Composer (free-text). Sub-category is only
        # meaningful under a category, but we store both verbatim.
        "category": re.sub(r"\s+", " ", str(item.get("category") or "").strip()),
        "sub_category": re.sub(r"\s+", " ", str(item.get("sub_category") or "").strip()),
        "created_at": created,
        "updated_at": str(item.get("updated_at") or now_iso()),
    }


def normalize_shortcut_db(data):
    if not isinstance(data, dict):
        data = empty_shortcut_db()
    shortcuts = data.get("shortcuts", {})
    if isinstance(shortcuts, list):
        shortcuts = {str(index): item for index, item in enumerate(shortcuts)}
    if not isinstance(shortcuts, dict):
        shortcuts = {}
    normalized = {}
    for fallback_name, item in shortcuts.items():
        shortcut = normalize_shortcut_item(item, fallback_name)
        if not shortcut:
            continue
        key = shortcut_key(shortcut["name"]) or shortcut_key(fallback_name)
        if key:
            normalized[key] = shortcut
    data["version"] = 1
    data["source"] = "ComfyUI-FunPack"
    data["shortcuts"] = normalized
    # Managed category list: normalize, then union any category/sub-category a
    # shortcut references so the list never loses a grouping that's in use.
    categories = normalize_category_list(data.get("categories"))
    for shortcut in normalized.values():
        _union_category(categories, shortcut.get("category"), shortcut.get("sub_category"))
    data["categories"] = categories
    return data


def load_shortcut_db():
    path = shortcut_store_path()
    if not os.path.exists(path):
        return empty_shortcut_db()
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError, ValueError):
        return empty_shortcut_db()
    return normalize_shortcut_db(data)


def save_shortcut_db(data):
    data = normalize_shortcut_db(data)
    os.makedirs(template_store_dir(), exist_ok=True)
    with open(shortcut_store_path(), "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


REVOLVER_STORE_FILENAME = "shortcut_revolver.json"


def revolver_store_path():
    return os.path.join(template_store_dir(), REVOLVER_STORE_FILENAME)


def load_revolver():
    """Shortcut-revolver settings + per-shortcut cycle state. Sidecar of the shortcut DB —
    deliberately NOT part of the shortcuts export/import payload. ``state[<shortcut key>]`` =
    {"fp": <fingerprint of the replacement list>, "queue": [remaining replacement indices,
    next one first]}."""
    data = {}
    path = revolver_store_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (json.JSONDecodeError, OSError, ValueError):
            data = {}
    if not isinstance(data, dict):
        data = {}
    state = data.get("state")
    return {
        "enabled": bool(data.get("enabled", False)),
        "random": bool(data.get("random", False)),
        "state": state if isinstance(state, dict) else {},
    }


def save_revolver(data):
    os.makedirs(template_store_dir(), exist_ok=True)
    with open(revolver_store_path(), "w", encoding="utf-8") as file:
        json.dump({
            "enabled": bool(data.get("enabled", False)),
            "random": bool(data.get("random", False)),
            "state": data.get("state") if isinstance(data.get("state"), dict) else {},
        }, file, indent=2)


def revolver_enabled():
    try:
        return bool(load_revolver()["enabled"])
    except Exception:
        return False


def revolver_settings():
    data = load_revolver()
    return {"enabled": data["enabled"], "random": data["random"]}


def set_revolver_settings(enabled=None, random_order=None):
    """Update revolver settings. Any actual change resets the cycle state — the old queues'
    ordering semantics no longer apply once the mode flips."""
    data = load_revolver()
    changed = False
    if enabled is not None and bool(enabled) != data["enabled"]:
        data["enabled"] = bool(enabled)
        changed = True
    if random_order is not None and bool(random_order) != data["random"]:
        data["random"] = bool(random_order)
        changed = True
    if changed:
        data["state"] = {}
        save_revolver(data)
    return {"enabled": data["enabled"], "random": data["random"]}


def _revolver_fingerprint(replacements):
    payload = json.dumps(list(replacements), ensure_ascii=False)
    return md5(payload.encode("utf-8")).hexdigest()[:12]


def revolver_next_replacement(state, key, replacements, random_order):
    """Draw the next replacement for shortcut ``key`` from its revolver queue, mutating
    ``state`` in place. The queue is rebuilt — a fresh full cycle over every replacement —
    when it's empty or the replacement set changed (fingerprint mismatch): sequential
    (first, second, … last) by default, shuffled once per cycle when ``random_order``.
    Either way nothing repeats until the whole set has fired."""
    n = len(replacements)
    fp = _revolver_fingerprint(replacements)
    entry = state.get(key)
    queue = None
    if isinstance(entry, dict) and entry.get("fp") == fp and isinstance(entry.get("queue"), list):
        queue = [int(i) for i in entry["queue"] if isinstance(i, int) and 0 <= int(i) < n]
    if not queue:
        queue = list(range(n))
        if random_order:
            random.shuffle(queue)
    index = queue.pop(0)
    state[key] = {"fp": fp, "queue": queue}
    return replacements[index]


def shortcut_items(data=None):
    data = load_shortcut_db() if data is None else normalize_shortcut_db(data)
    items = []
    for key, item in data.get("shortcuts", {}).items():
        if not isinstance(item, dict):
            continue
        row = dict(item)
        row["key"] = key
        items.append(row)
    return sorted(items, key=lambda item: item.get("name", "").lower())


def save_shortcut_item(payload):
    item = normalize_shortcut_item(payload)
    if item is None:
        raise ValueError("Shortcut activation phrase and replacement phrase are required.")
    data = load_shortcut_db()
    shortcuts = data.setdefault("shortcuts", {})
    key = shortcut_key(item["name"])
    original_key = shortcut_key(payload.get("original_name", ""))
    previous = shortcuts.get(key, {}) if isinstance(shortcuts.get(key), dict) else {}
    if not previous and original_key:
        previous = shortcuts.get(original_key, {}) if isinstance(shortcuts.get(original_key), dict) else {}
    item["created_at"] = str(previous.get("created_at") or item["created_at"])
    item["updated_at"] = now_iso()
    shortcuts[key] = item
    if original_key and original_key != key:
        shortcuts.pop(original_key, None)
    save_shortcut_db(data)
    return key, data


def delete_shortcut_item(name):
    key = shortcut_key(name)
    data = load_shortcut_db()
    if key:
        data.setdefault("shortcuts", {}).pop(key, None)
        save_shortcut_db(data)
    return key, data


def shortcut_categories(data=None):
    data = load_shortcut_db() if data is None else normalize_shortcut_db(data)
    return data.get("categories", [])


def add_shortcut_category(name, sub_category=""):
    """Add a category (and optionally a sub-category under it) to the managed list."""
    name = _clean_label(name)
    if not name:
        raise ValueError("Category name is required.")
    data = load_shortcut_db()
    _union_category(data.setdefault("categories", []), name, sub_category)
    save_shortcut_db(data)
    return load_shortcut_db()


# --- Custom transition DB --------------------------------------------------

def empty_transition_db():
    return {"version": 1, "source": "ComfyUI-FunPack", "transitions": {}}



def normalize_transition_item(item, fallback_name=""):
    if not isinstance(item, dict):
        return None
    trigger = re.sub(r"\s+", " ", str(item.get("trigger") or "").strip())
    if not trigger:
        return None
    name = re.sub(r"\s+", " ", str(item.get("name") or fallback_name or trigger).strip())
    placement = str(item.get("placement") or "global").strip().lower()
    if placement not in ("global", "start", "end", "silent"):
        placement = "global"
    return {
        "name": name,
        "trigger": trigger,
        "placement": placement,
        "enabled": bool(item.get("enabled", True)),
    }


def normalize_transition_db(data):
    if not isinstance(data, dict):
        data = empty_transition_db()
    transitions = data.get("transitions", {})
    if isinstance(transitions, list):
        transitions = {str(i): item for i, item in enumerate(transitions)}
    if not isinstance(transitions, dict):
        transitions = {}
    normalized = {}
    for fallback_name, item in transitions.items():
        entry = normalize_transition_item(item, fallback_name)
        if not entry:
            continue
        key = shortcut_key(entry["name"]) or shortcut_key(fallback_name)
        if key:
            normalized[key] = entry
    data["version"] = 1
    data["source"] = "ComfyUI-FunPack"
    data["transitions"] = normalized
    return data


def load_transition_db():
    _migrate_transition_store_if_needed()
    path = transition_store_path()
    if not os.path.exists(path):
        return empty_transition_db()
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError, ValueError):
        return empty_transition_db()
    return normalize_transition_db(data)


def save_transition_db(data):
    data = normalize_transition_db(data)
    os.makedirs(template_store_dir(), exist_ok=True)
    with open(transition_store_path(), "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def transition_items(data=None):
    data = load_transition_db() if data is None else normalize_transition_db(data)
    items = []
    for key, item in data.get("transitions", {}).items():
        if not isinstance(item, dict):
            continue
        row = dict(item)
        row["key"] = key
        items.append(row)
    return sorted(items, key=lambda item: item.get("name", "").lower())


def save_transition_item(payload):
    entry = normalize_transition_item(payload)
    if not entry:
        raise ValueError("Transition trigger phrase is required.")
    data = load_transition_db()
    key = shortcut_key(entry["name"])
    original_key = shortcut_key(str(payload.get("original_name") or ""))
    transitions = data.setdefault("transitions", {})
    transitions[key] = entry
    if original_key and original_key != key:
        transitions.pop(original_key, None)
    save_transition_db(data)
    return key, data


def delete_transition_item(name):
    key = shortcut_key(name)
    data = load_transition_db()
    if key:
        data.setdefault("transitions", {}).pop(key, None)
        save_transition_db(data)
    return key, data


def load_custom_transition_triggers():
    """Return {trigger: {"placement": override_or_None}} for enabled custom split markers.

    placement is 'start', 'end', 'silent', or None (use global setting).
    """
    data = load_transition_db()
    result = {}
    for item in data.get("transitions", {}).values():
        if not isinstance(item, dict) or not item.get("enabled", True):
            continue
        trigger = re.sub(r"\s+", " ", str(item.get("trigger") or "").strip())
        if not trigger:
            continue
        placement = str(item.get("placement") or "global").strip().lower()
        result[trigger.lower()] = {
            "placement": placement if placement in ("start", "end", "silent") else None,
            "visual_effect": str(item.get("visual_effect") or "none"),
        }
    return result


def _shortcut_trigger_pattern(trigger):
    words = [re.escape(part) for part in re.split(r"\s+", str(trigger or "").strip()) if part]
    if not words:
        return ""
    body = r"\s+".join(words)
    return rf"(?<![\w’’-])({body})(?![\w’’-])"


def _shortcut_replacements(raw):
    """Parse replacement list, allowing empty string as a valid ‘remove’ replacement.

    A replacement is a prose phrase that legitimately contains commas/semicolons, so a raw
    string is split on NEWLINES ONLY (the 'one per line' UI contract). Splitting on commas
    tore one phrase into several variants and re-saving kept re-tearing it."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            raw = raw.split("\n")
    if not isinstance(raw, list):
        return []
    seen = set()
    result = []
    for item in raw:
        text = re.sub(r"\s+", " ", str(item if item is not None else "").strip())
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _cleanup_removed_phrases(text):
    """Fix punctuation and spacing artifacts left by empty-replacement removals."""
    text = re.sub(r"[ \t]+([,;])", r"\1", text)
    text = re.sub(r"([,;])\s*([,;])+", r"\1", text)
    text = re.sub(r"^[\s,;]+", "", text)
    text = re.sub(r"[\s,;]+$", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


# ── prompt variables ($name) ────────────────────────────────────────────────────
# Variables are a project-scoped find/replace layer resolved DEAD LAST — after shortcut
# expansion AND after the transition split — so they can never create or move a scene cut.
# A `$name` token is replaced with the variable's text; the text may itself reference other
# variables (recursive), undefined names are left as literal `$name`, and a variable that
# references itself anywhere in its own chain is a cycle (left literal, never expanded).
_VARIABLE_TOKEN = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_VARIABLE_MAX_DEPTH = 64


def _normalize_variables(variables):
    """Accept either {name: text} or [{"name":, "value":}] and return an ordered-safe dict.
    Names are stripped of a leading `$` and surrounding whitespace; later entries win."""
    out = {}
    if isinstance(variables, dict):
        items = list(variables.items())
    elif isinstance(variables, list):
        items = [
            (v.get("name"), v.get("value", v.get("content", v.get("text", ""))))
            for v in variables if isinstance(v, dict)
        ]
    else:
        items = []
    for name, val in items:
        key = str(name if name is not None else "").lstrip("$").strip()
        if key:
            out[key] = str(val if val is not None else "")
    return out


def resolve_variables(text, variables):
    """Substitute `$name` tokens in `text` from `variables`. Recursive, cycle-safe (a name in
    its own expansion chain is left literal), depth-guarded, undefined -> literal `$name`.
    Returns (resolved_text, sorted list of undefined names referenced)."""
    var_map = _normalize_variables(variables)
    if not var_map:
        return str(text or ""), []
    undefined = set()

    def _expand(s, stack, depth):
        if depth > _VARIABLE_MAX_DEPTH:
            return s

        def _repl(m):
            name = m.group(1)
            if name not in var_map:
                undefined.add(name)
                return m.group(0)            # undefined -> leave literal
            if name in stack:
                return m.group(0)            # cycle -> leave literal, do not recurse
            return _expand(var_map[name], stack | {name}, depth + 1)

        return _VARIABLE_TOKEN.sub(_repl, s)

    return _expand(str(text or ""), frozenset(), 0), sorted(undefined)


def apply_prompt_shortcuts(text, seed=0, shortcut_db=None, revolver_commit=False):
    """Expand shortcut triggers. Multi-replacement picks are seeded-random unless the
    shortcut revolver is enabled (see load_revolver): then each firing draws the next
    replacement from the shortcut's no-repeat cycle. ``revolver_commit`` persists the
    advanced cycle state — True only for real generation; previews peek without saving,
    so they show exactly what the next generation will draw."""
    original = str(text or "")
    if not original:
        return original, []
    data = load_shortcut_db() if shortcut_db is None else normalize_shortcut_db(shortcut_db)
    candidates = []
    for db_key, shortcut in data.get("shortcuts", {}).items():
        if not isinstance(shortcut, dict) or not bool(shortcut.get("enabled", True)):
            continue
        replacements = _shortcut_replacements(shortcut.get("replacements", shortcut.get("replacement", [])))
        if not replacements:
            continue
        sc_key = normalize_refinement_key(shortcut.get("refinement_key", ""))
        for trigger in shortcut_list(shortcut.get("triggers", [])):
            pattern = _shortcut_trigger_pattern(trigger)
            if pattern:
                candidates.append((trigger, pattern, replacements, shortcut.get("name", trigger),
                                   sc_key, str(db_key)))
    if not candidates:
        return original, []

    candidates.sort(key=lambda item: len(shortcut_key(item[0])), reverse=True)
    combined = "|".join(f"(?P<t{index}>{pattern})" for index, (_, pattern, _, _, _, _) in enumerate(candidates))
    if not combined:
        return original, []
    try:
        rng_seed = int(seed or 0)
    except (TypeError, ValueError):
        rng_seed = 0
    if rng_seed == 0:
        rng_seed = int(md5(original.encode("utf-8")).hexdigest()[:12], 16)
    rng = random.Random(rng_seed)
    revolver = load_revolver()
    revolver_on = bool(revolver.get("enabled"))
    revolver_state = revolver.get("state")
    revolver_random = bool(revolver.get("random"))
    revolver_dirty = False
    applied = []
    removals_happened = False

    def replace(match):
        nonlocal removals_happened, revolver_dirty
        for index, (trigger, _, replacements, name, sc_key, db_key) in enumerate(candidates):
            if match.group(f"t{index}") is None:
                continue
            if revolver_on and len(replacements) > 1:
                replacement = revolver_next_replacement(revolver_state, db_key, replacements, revolver_random)
                revolver_dirty = True
            else:
                replacement = rng.choice(replacements)
            applied.append({"name": str(name), "trigger": trigger,
                            "replacement": replacement, "refinement_key": sc_key})
            if not replacement:
                removals_happened = True
            return replacement
        return match.group(0)

    expanded = re.sub(combined, replace, original, flags=re.IGNORECASE | re.UNICODE)
    if removals_happened:
        expanded = _cleanup_removed_phrases(expanded)
    if revolver_dirty and revolver_commit:
        try:
            save_revolver(revolver)
        except OSError:
            pass
    return expanded, applied


def normalize_template_name(value):
    value = str(value or "").strip()
    if not value or value == TEMPLATE_NONE:
        return ""
    return value


def now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_safe(value):
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return copy.deepcopy(value)


def maybe_store_string(template, field, value, update_only):
    if isinstance(value, str) and value.strip():
        template[field] = value
    elif not update_only:
        template.pop(field, None)


def collect_template_payload(
    mode,
    activation_word="",
    refinement_key="",
    positive_prompt="",
    negative_prompt="",
    sigmas=None,
    lora_stack=None,
    update_only=False,
    existing=None,
):
    template = dict(existing or {}) if update_only else {}
    template["mode"] = mode if mode in {"ltx2", "wan"} else "ltx2"
    maybe_store_string(template, "activation_word", activation_word, update_only)
    maybe_store_string(template, "refinement_key", refinement_key, update_only)
    maybe_store_string(template, "positive_prompt", positive_prompt, update_only)
    maybe_store_string(template, "negative_prompt", negative_prompt, update_only)

    if isinstance(sigmas, torch.Tensor):
        template["sigmas"] = tensor_to_serializable(sigmas.detach().cpu())
    elif not update_only:
        template.pop("sigmas", None)

    if isinstance(lora_stack, dict):
        template["lora_stack"] = json_safe(lora_stack)
    elif not update_only:
        template.pop("lora_stack", None)

    return template














def template_field_summary(template):
    fields = []
    for field in ("positive_prompt", "negative_prompt", "activation_word", "refinement_key", "sigmas", "lora_stack"):
        if field in template:
            fields.append(field)
    return ", ".join(fields) if fields else "none"


@PromptServer.instance.routes.get("/funpack/templates")
async def funpack_templates(_):
    return web.json_response(
        {"templates": template_names(), "path": template_store_path()},
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@PromptServer.instance.routes.get("/funpack/templates/export")
async def funpack_templates_export(_):
    data = load_template_db()
    return web.json_response(
        data,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Content-Disposition": "attachment; filename=funpack_templates.json",
        },
    )


@PromptServer.instance.routes.post("/funpack/templates/import")
async def funpack_templates_import(request):
    incoming = await request.json()
    templates = incoming.get("templates") if isinstance(incoming, dict) else None
    if not isinstance(templates, dict):
        return web.json_response({"error": "Imported file does not contain a templates object."}, status=400)

    data = load_template_db()
    current = data.setdefault("templates", {})
    imported = 0
    for name, template in templates.items():
        clean_name = normalize_template_name(name)
        if not clean_name or not isinstance(template, dict):
            continue
        item = dict(template)
        item["name"] = clean_name
        item["updated_at"] = now_iso()
        current[clean_name] = item
        imported += 1
    save_template_db(data)
    return web.json_response({"imported": imported, "templates": template_names()})


@PromptServer.instance.routes.get("/funpack/shortcuts")
async def funpack_shortcuts(_):
    data = load_shortcut_db()
    return web.json_response(
        {
            "path": shortcut_store_path(),
            "data": data,
            "shortcuts": shortcut_items(data),
            "categories": shortcut_categories(data),
        },
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@PromptServer.instance.routes.get("/funpack/shortcuts/export")
async def funpack_shortcuts_export(_):
    data = load_shortcut_db()
    return web.json_response(
        data,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Content-Disposition": "attachment; filename=funpack_shortcuts.json",
        },
    )


@PromptServer.instance.routes.post("/funpack/shortcuts/shortcut")
async def funpack_shortcut_save(request):
    body = await request.json()
    if not isinstance(body, dict):
        return web.json_response({"error": "Shortcut payload must be an object."}, status=400)

    action = str(body.get("action") or "save").lower()
    name = normalize_shortcut_name(body.get("name"))
    if action == "delete":
        if not name:
            return web.json_response({"error": "Shortcut name is required."}, status=400)
        key, data = delete_shortcut_item(name)
        return web.json_response({"deleted": key, "data": data, "shortcuts": shortcut_items(data), "categories": shortcut_categories(data)})

    try:
        key, data = save_shortcut_item(body)
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)
    return web.json_response({"saved": key, "data": data, "shortcuts": shortcut_items(data), "categories": shortcut_categories(data)})


@PromptServer.instance.routes.post("/funpack/shortcuts/import")
async def funpack_shortcuts_import(request):
    incoming = await request.json()
    if not isinstance(incoming, dict):
        return web.json_response({"error": "Imported file is not a shortcut database."}, status=400)
    if "shortcuts" not in incoming:
        return web.json_response({"error": "Imported file does not contain shortcuts."}, status=400)
    imported = normalize_shortcut_db(incoming)
    data = load_shortcut_db()
    shortcuts = data.setdefault("shortcuts", {})
    count = 0
    for key, item in imported.get("shortcuts", {}).items():
        if isinstance(item, dict):
            shortcuts[str(key)] = item
            count += 1
    save_shortcut_db(data)
    data = load_shortcut_db()
    return web.json_response({"imported": count, "data": data, "shortcuts": shortcut_items(data), "categories": shortcut_categories(data)})


@PromptServer.instance.routes.get("/funpack/transitions")
async def funpack_transitions(_):
    data = load_transition_db()
    return web.json_response(
        {"path": transition_store_path(), "data": data, "transitions": transition_items(data)},
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@PromptServer.instance.routes.get("/funpack/transitions/export")
async def funpack_transitions_export(_):
    data = load_transition_db()
    return web.json_response(
        data,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Content-Disposition": "attachment; filename=funpack_promptsplit.json",
        },
    )


@PromptServer.instance.routes.post("/funpack/transitions/transition")
async def funpack_transition_save(request):
    body = await request.json()
    if not isinstance(body, dict):
        return web.json_response({"error": "Transition payload must be an object."}, status=400)
    action = str(body.get("action") or "save").lower()
    name = str(body.get("name") or body.get("trigger") or "").strip()
    if action == "delete":
        if not name:
            return web.json_response({"error": "Transition name is required."}, status=400)
        key, data = delete_transition_item(name)
        return web.json_response({"deleted": key, "data": data, "transitions": transition_items(data)})
    try:
        key, data = save_transition_item(body)
    except ValueError as error:
        return web.json_response({"error": str(error)}, status=400)
    return web.json_response({"saved": key, "data": data, "transitions": transition_items(data)})


@PromptServer.instance.routes.post("/funpack/transitions/import")
async def funpack_transitions_import(request):
    incoming = await request.json()
    if not isinstance(incoming, dict):
        return web.json_response({"error": "Imported file is not a transition database."}, status=400)
    if "transitions" not in incoming:
        return web.json_response({"error": "Imported file does not contain transitions."}, status=400)
    imported_db = normalize_transition_db(incoming)
    data = load_transition_db()
    entries = data.setdefault("transitions", {})
    count = 0
    for key, item in imported_db.get("transitions", {}).items():
        if isinstance(item, dict):
            entries[str(key)] = item
            count += 1
    save_transition_db(data)
    data = load_transition_db()
    return web.json_response({"imported": count, "data": data, "transitions": transition_items(data)})


@PromptServer.instance.routes.post("/funpack/parse_timeline")
async def funpack_parse_timeline(request):
    data = await request.json()
    prompt = str(data.get("prompt") or "")
    seed = int(data.get("seed") or 0)
    try:
        from .conditioning import parse_timeline_segments
    except ImportError:
        from conditioning import parse_timeline_segments
    expanded, _ = apply_prompt_shortcuts(prompt, seed=seed)
    return web.json_response(parse_timeline_segments(expanded))


@PromptServer.instance.routes.get("/funpack/refinement_keys")
async def funpack_refinement_keys(_):
    return web.json_response(
        {"keys": refinement_key_names(), "path": refinement_store_dir()},
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@PromptServer.instance.routes.get("/funpack/refinement_keys/export")
async def funpack_refinement_keys_export(request):
    key = normalize_refinement_key(request.query.get("key", ""))
    state, loaded_key, status = load_refinement_key_state(key, create=False)
    if state is None:
        return web.json_response({"error": status}, status=404)
    return web.json_response(
        state,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Content-Disposition": f"attachment; filename={loaded_key}.json",
        },
    )


def _truthy(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _save_imported_refinement_key(incoming, overwrite=False):
    """Shared tail for the one-shot and chunked import routes. Returns an aiohttp
    response. When a key of the same name already exists and ``overwrite`` is not
    set, replies 409 with ``{"exists": True, "key": ...}`` so the client can ask
    the user to overwrite or cancel."""
    state, key = normalize_refinement_state(_coerce_refinement_payload(incoming) or {})
    if state is None:
        return web.json_response({"error": "Imported file is not a valid V2 refinement key JSON."}, status=400)
    if not overwrite and os.path.exists(refinement_key_path(key)):
        return web.json_response(
            {"exists": True, "key": key,
             "error": f"A refinement key named '{key}' already exists."},
            status=409,
        )
    path = save_refinement_key_state(state, key)
    if not path:
        return web.json_response({"error": "Could not save imported refinement key."}, status=400)
    return web.json_response({"imported": key, "keys": refinement_key_names()})


@PromptServer.instance.routes.post("/funpack/refinement_keys/import")
async def funpack_refinement_keys_import(request):
    incoming = await request.json()
    return _save_imported_refinement_key(incoming, overwrite=_truthy(request.query.get("overwrite")))


# --- Chunked import -------------------------------------------------------------------
# Reverse proxies in front of ComfyUI on rented GPUs (Vast.ai, Runpod) cap the
# request body well below ComfyUI's own 100 MB limit, so a large refinement key
# POSTed in one shot is rejected with HTTP 413 before it ever reaches us. The
# client streams the key in small chunks (each well under any proxy limit) which
# we reassemble on disk, then finalize parses + saves the whole thing.
def _refinement_upload_dir():
    directory = os.path.join(refinement_store_dir(), ".uploads")
    os.makedirs(directory, exist_ok=True)
    return directory


def _refinement_upload_path(upload_id):
    safe = re.sub(r"[^A-Za-z0-9_-]", "", str(upload_id or ""))[:80]
    if not safe:
        return None
    return os.path.join(_refinement_upload_dir(), f"{safe}.part")


@PromptServer.instance.routes.post("/funpack/refinement_keys/import_chunk")
async def funpack_refinement_keys_import_chunk(request):
    part_path = _refinement_upload_path(request.query.get("upload_id"))
    if not part_path:
        return web.json_response({"error": "Missing or invalid upload id."}, status=400)
    try:
        index = int(request.query.get("index", "0"))
    except (TypeError, ValueError):
        return web.json_response({"error": "Invalid chunk index."}, status=400)
    chunk = await request.read()
    # index 0 starts (or restarts) the upload; later chunks append in order.
    with open(part_path, "wb" if index <= 0 else "ab") as file:
        file.write(chunk)
    return web.json_response({"ok": True, "received": os.path.getsize(part_path)})


@PromptServer.instance.routes.post("/funpack/refinement_keys/import_finalize")
async def funpack_refinement_keys_import_finalize(request):
    part_path = _refinement_upload_path(request.query.get("upload_id"))
    if not part_path or not os.path.exists(part_path):
        return web.json_response({"error": "Upload not found or expired."}, status=404)
    try:
        with open(part_path, "r", encoding="utf-8") as file:
            incoming = json.load(file)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        return web.json_response({"error": f"Uploaded data is not valid JSON: {exc}"}, status=400)
    finally:
        try:
            os.remove(part_path)
        except OSError:
            pass
    return _save_imported_refinement_key(incoming, overwrite=_truthy(request.query.get("overwrite")))


@PromptServer.instance.routes.post("/funpack/refinement_keys/delete")
async def funpack_refinement_keys_delete(request):
    key = normalize_refinement_key(request.query.get("key", ""))
    if not key:
        try:
            body = await request.json()
        except Exception:
            body = {}
        key = normalize_refinement_key((body or {}).get("key", ""))
    if not key:
        return web.json_response({"error": "No refinement key specified."}, status=400)
    result = delete_refinement_key(key)
    result["keys"] = refinement_key_names()
    return web.json_response(result, headers={"Cache-Control": "no-store, max-age=0"})


@PromptServer.instance.routes.get("/funpack/refinement_keys/absolute")
async def funpack_refinement_keys_absolute(_):
    return web.json_response(absolute_store_info(),
                             headers={"Cache-Control": "no-store, max-age=0"})


@PromptServer.instance.routes.post("/funpack/refinement_keys/clear_absolute")
async def funpack_refinement_keys_clear_absolute(_):
    result = clear_absolute_store()
    result.update(absolute_store_info())
    return web.json_response(result, headers={"Cache-Control": "no-store, max-age=0"})


@PromptServer.instance.routes.get("/funpack/available_loras")
async def funpack_available_loras(request):
    try:
        import folder_paths as _fp
        loras = _fp.get_filename_list("loras")
    except Exception:
        loras = []
    return web.json_response(
        {"loras": sorted(loras)},
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@PromptServer.instance.routes.get("/funpack/phrase_memory")
async def funpack_phrase_memory(request):
    key = normalize_refinement_key(request.query.get("key", ""))
    if not key:
        return web.json_response({"phrases": [], "key": ""})
    state, loaded_key, _ = load_refinement_key_state(key, create=False)
    if not isinstance(state, dict):
        return web.json_response({"phrases": [], "key": key})
    global_state = state.get("global", {})
    phrase_memory = global_state.get("phrase_memory", {}) if isinstance(global_state, dict) else {}
    phrases = []
    for text, entry in (phrase_memory.items() if isinstance(phrase_memory, dict) else []):
        if not isinstance(entry, dict):
            continue
        clean = str(entry.get("text", text) or text).strip()
        if not clean:
            continue
        phrases.append({
            "text": clean,
            "category": str(entry.get("primary", "") or entry.get("machine_primary", "") or "details"),
            "evidence": int(entry.get("occurrence_count", 0) or 0),
        })
    phrases.sort(key=lambda p: (-p["evidence"], p["text"]))
    return web.json_response(
        {"phrases": phrases, "key": loaded_key or key},
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@PromptServer.instance.routes.get("/funpack/value_function/export")
async def funpack_vf_export(request):
    from conditioning import refinement_state_path
    key = normalize_refinement_key(request.query.get("key", ""))
    if not key:
        return web.json_response({"error": "No refinement key provided."}, status=400)
    path = refinement_state_path(key, "value_fn", prefix="refine_v2", extension="pt")
    if not os.path.exists(path):
        return web.json_response({"error": "No value function found for this key."}, status=404)
    with open(path, "rb") as f:
        data = f.read()
    safe_key = re.sub(r"[^\w\-]", "_", key)[:64]
    return web.Response(
        body=data,
        content_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="funpack_vf_{safe_key}.pt"'},
    )


@PromptServer.instance.routes.post("/funpack/value_function/import")
async def funpack_vf_import(request):
    from conditioning import refinement_state_path
    try:
        from value_function import OnlineValueFunction
    except ImportError:
        return web.json_response({"error": "value_function module not available."}, status=500)
    key = normalize_refinement_key(request.query.get("key", ""))
    if not key:
        return web.json_response({"error": "No refinement key provided."}, status=400)
    raw = await request.read()
    if not raw:
        return web.json_response({"error": "Empty file."}, status=400)
    import io
    try:
        with torch.inference_mode(False):
            vf = OnlineValueFunction.load(io.BytesIO(raw))
    except Exception as e:
        return web.json_response({"error": f"Invalid value function file: {e}"}, status=400)
    dest = refinement_state_path(key, "value_fn", prefix="refine_v2", extension="pt")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(raw)
    return web.json_response({"ok": True, "n_trained": vf.n_trained, "buffer": len(vf.buffer_c)})


class FunPackRefinementKeyLoader:
    CATEGORY = "FunPack/Refinement"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("refinement_key", "status")
    FUNCTION = "load_refinement_key"
    OUTPUT_NODE = True
    DESCRIPTION = "Loads, creates, imports, and exports FunPack Video Refiner V2 refinement keys."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refinement_key": (refinement_key_names(), {"default": REFINEMENT_KEY_NONE}),
                "key_name": ("STRING", {"default": "", "multiline": False}),
                "create_if_missing": ("BOOLEAN", {"default": True}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, refinement_key=None, **kwargs):
        return True

    def load_refinement_key(self, refinement_key, key_name, create_if_missing=True):
        selected = normalize_refinement_key(refinement_key)
        typed = normalize_refinement_key(key_name)
        target = selected or typed
        if not target:
            return ("", "No refinement key selected. Pick an existing key or type a new key name.")
        _, loaded_key, status = load_refinement_key_state(target, create=bool(create_if_missing))
        return (loaded_key if loaded_key else target, status)
