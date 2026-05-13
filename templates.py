import copy
import json
import os
import random
import re
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
SCENE_DB_VERSION = 1
SCENE_STATE_FIELD = "scene_builder"
WILDCARD_RE = re.compile(r"\{([^{}]*\|[^{}]*)\}")
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


def scene_store_path():
    return os.path.join(template_store_dir(), "scenes.json")


def refinement_store_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "refinements")


def normalize_refinement_key(value):
    value = str(value or "").strip()
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
    keys = set()
    directory = refinement_store_dir()
    if os.path.isdir(directory):
        for filename in os.listdir(directory):
            if not filename.startswith("refine_v2_") or not filename.endswith(".json"):
                continue
            path = os.path.join(directory, filename)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                key = normalize_refinement_key(data.get("refinement_key") if isinstance(data, dict) else "")
                if key:
                    keys.add(key)
            except (json.JSONDecodeError, OSError, ValueError):
                continue
    return [REFINEMENT_KEY_NONE] + sorted(keys)


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


def empty_scene_db():
    return {
        "version": SCENE_DB_VERSION,
        "source": "ComfyUI-FunPack",
        "universal_memory": {},
        "scenes": {},
    }


def normalize_scene_db(data):
    if not isinstance(data, dict):
        data = empty_scene_db()
    data.setdefault("version", SCENE_DB_VERSION)
    data.setdefault("source", "ComfyUI-FunPack")
    if not isinstance(data.get("universal_memory"), dict):
        data["universal_memory"] = {}
    if not isinstance(data.get("scenes"), dict):
        data["scenes"] = {}
    return data


def load_scene_db(refinement_key=""):
    key = normalize_refinement_key(refinement_key)
    if key:
        state, _, _ = load_refinement_key_state(key, create=False)
        if isinstance(state, dict):
            return normalize_scene_db(state.get(SCENE_STATE_FIELD, {}))
        return empty_scene_db()

    path = scene_store_path()
    if not os.path.exists(path):
        return empty_scene_db()
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError, ValueError):
        return empty_scene_db()

    return normalize_scene_db(data)


def save_scene_db(data, refinement_key=""):
    key = normalize_refinement_key(refinement_key)
    data = normalize_scene_db(data)
    if key:
        state, _, _ = load_refinement_key_state(key, create=True)
        if isinstance(state, dict):
            state[SCENE_STATE_FIELD] = data
            save_refinement_key_state(state, key)
        return

    os.makedirs(template_store_dir(), exist_ok=True)
    with open(scene_store_path(), "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def scene_names(refinement_key=""):
    data = load_scene_db(refinement_key)
    names = sorted(
        name for name in data.get("scenes", {}).keys()
        if isinstance(name, str) and name.strip()
    )
    return [SCENE_NONE] + names


def normalize_scene_name(value):
    value = str(value or "").strip()
    if not value or value == SCENE_NONE:
        return ""
    return value


def normalize_scene_key(value):
    value = re.sub(r"[^\w'’]+", " ", str(value or "").strip().lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def scene_token_words(text):
    return [
        token.lower()
        for token in re.findall(r"[\w'’.-]+", str(text or ""), flags=re.UNICODE)
        if any(char.isalpha() for char in token) and len(token.strip("._-")) >= 2
    ]


def categorize_scene_phrase(text, source="positive"):
    if source == "negative":
        return "negative"
    words = set(scene_token_words(text))
    best = ("details", 0)
    for category, keywords in SCENE_CATEGORIES.items():
        if category == "negative":
            continue
        score = len(words & keywords)
        if score > best[1]:
            best = (category, score)
    return best[0]


def extract_scene_phrases(text, source="positive"):
    phrases = []
    seen = set()
    for raw in re.split(r"[,;.\n]+", str(text or "")):
        clean = re.sub(r"\s+", " ", raw).strip(" ,;:.").strip()
        if len(clean) < 2:
            continue
        key = clean.lower()
        if key in seen:
            continue
        words = scene_token_words(clean)
        if not words:
            continue
        seen.add(key)
        phrases.append({
            "text": clean,
            "key": key,
            "source": source,
            "category": categorize_scene_phrase(clean, source),
            "tokens": words,
        })
    return phrases


def remember_scene_phrases(data, positive_prompt="", negative_prompt=""):
    memory = data.setdefault("universal_memory", {})
    changed = False
    for source, prompt in (("positive", positive_prompt), ("negative", negative_prompt)):
        for phrase in extract_scene_phrases(prompt, source):
            key = phrase["key"]
            current = memory.setdefault(key, {
                "text": phrase["text"],
                "source": source,
                "category": phrase["category"],
                "tokens": phrase["tokens"],
                "count": 0,
                "created_at": now_iso(),
            })
            current["text"] = current.get("text") or phrase["text"]
            current["source"] = source
            current["category"] = phrase["category"]
            current["tokens"] = phrase["tokens"]
            current["count"] = int(current.get("count", 0) or 0) + 1
            current["updated_at"] = now_iso()
            changed = True
    return changed


def scene_memory_items(data):
    memory = data.get("universal_memory", {}) if isinstance(data, dict) else {}
    items = []
    for key, item in memory.items():
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or key).strip()
        if not text:
            continue
        items.append({
            "text": text,
            "key": str(key),
            "source": item.get("source", "positive"),
            "category": item.get("category", "details"),
            "tokens": item.get("tokens", scene_token_words(text)),
            "count": int(item.get("count", 0) or 0),
            "wildcard_group": str(item.get("wildcard_group") or "").strip(),
        })
    return sorted(items, key=lambda item: (item["category"], item["text"].lower()))


def normalize_scene_memory_items(value):
    if isinstance(value, dict):
        iterable = value.items()
    elif isinstance(value, list):
        iterable = ((None, item) for item in value)
    else:
        return {}

    memory = {}
    for fallback_key, item in iterable:
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict):
            continue
        text = re.sub(r"\s+", " ", str(item.get("text") or fallback_key or "").strip())
        if not text:
            continue
        key = normalize_scene_key(item.get("key") or text) or text.lower()
        source = "negative" if item.get("source") == "negative" or item.get("category") == "negative" else "positive"
        category = str(item.get("category") or categorize_scene_phrase(text, source)).strip().lower()
        if category not in SCENE_CATEGORIES:
            category = "negative" if source == "negative" else "details"
        tokens = scene_token_words(text)
        memory[key] = {
            "text": text,
            "source": source,
            "category": category,
            "tokens": tokens,
            "count": max(0, int(item.get("count", 0) or 0)),
            "created_at": str(item.get("created_at") or now_iso()),
            "updated_at": now_iso(),
            "wildcard_group": str(item.get("wildcard_group") or "").strip(),
        }
    return memory


def normalize_scene_phrase_list(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            value = [value]
    if not isinstance(value, list):
        return []
    result = []
    seen = set()
    for item in value:
        text = item.get("text") if isinstance(item, dict) else item
        clean = re.sub(r"\s+", " ", str(text or "")).strip(" ,;:.").strip()
        key = clean.lower()
        if not clean or key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


def parse_scene_payload(value):
    if isinstance(value, dict):
        payload = value
    elif isinstance(value, str) and value.strip():
        try:
            payload = json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            payload = {}
    else:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return {
        "positive_phrases": normalize_scene_phrase_list(payload.get("positive_phrases", [])),
        "negative_phrases": normalize_scene_phrase_list(payload.get("negative_phrases", [])),
    }


def scene_text_or_payload(text, phrases):
    text = str(text or "").strip()
    if text:
        return text
    return scene_text_from_phrases(phrases)


def scene_payload_from_scene(scene):
    if not isinstance(scene, dict):
        return {"positive_phrases": [], "negative_phrases": []}
    return {
        "positive_phrases": normalize_scene_phrase_list(scene.get("positive_phrases", [])),
        "negative_phrases": normalize_scene_phrase_list(scene.get("negative_phrases", [])),
    }


def scene_text_from_phrases(phrases):
    return ", ".join(normalize_scene_phrase_list(phrases))


def scene_from_texts(name, aliases, output_mode, positive_text, negative_text, refinement_key=""):
    positive_text = str(positive_text or "").strip()
    negative_text = str(negative_text or "").strip()
    return {
        "name": normalize_scene_name(name),
        "aliases": aliases_from_text(aliases),
        "output_mode": output_mode if output_mode in {"Manual", "Auto", "Learning"} else "Manual",
        "refinement_key": normalize_refinement_key(refinement_key),
        "positive_text": positive_text,
        "negative_text": negative_text,
        "positive_phrases": normalize_scene_phrase_list(extract_scene_phrases(positive_text, "positive")),
        "negative_phrases": normalize_scene_phrase_list(extract_scene_phrases(negative_text, "negative")),
    }


def aliases_from_text(value):
    aliases = []
    seen = set()
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = re.split(r"[,;\n]+", str(value or ""))
    for raw in raw_values:
        clean = re.sub(r"\s+", " ", raw).strip()
        key = normalize_scene_key(clean)
        if clean and key not in seen:
            seen.add(key)
            aliases.append(clean)
    return aliases


def scene_exact_match(intent_prompt, scenes):
    intent = normalize_scene_key(intent_prompt)
    if not intent:
        return "", None
    padded = f" {intent} "
    for name, scene in sorted(scenes.items(), key=lambda item: len(item[0]), reverse=True):
        keys = [name]
        if isinstance(scene, dict):
            keys.extend(scene.get("aliases", []) if isinstance(scene.get("aliases"), list) else [])
        for key in keys:
            clean = normalize_scene_key(key)
            if clean and f" {clean} " in padded:
                return name, scene
    return "", None


def scene_fuzzy_match(intent_prompt, scenes):
    intent_words = set(scene_token_words(intent_prompt))
    if not intent_words:
        return "", None
    best = ("", None, 0.0)
    for name, scene in scenes.items():
        keys = [name]
        if isinstance(scene, dict):
            keys.extend(scene.get("aliases", []) if isinstance(scene.get("aliases"), list) else [])
        for key in keys:
            key_words = set(scene_token_words(key))
            if not key_words:
                continue
            overlap = len(intent_words & key_words)
            if overlap < max(1, len(key_words)):
                continue
            score = overlap / max(len(key_words), 1)
            if score > best[2]:
                best = (name, scene, score)
    return best[0], best[1]


def find_scene_for_intent(intent_prompt, scenes):
    name, scene = scene_exact_match(intent_prompt, scenes)
    if name:
        return name, scene, "exact"
    name, scene = scene_fuzzy_match(intent_prompt, scenes)
    if name:
        return name, scene, "fuzzy"
    return "", None, "none"


def scene_field_summary(scene):
    if not isinstance(scene, dict):
        return "none"
    fields = []
    for field in ("positive_phrases", "negative_phrases", "refinement_key"):
        if scene.get(field):
            fields.append(field)
    if scene.get("positive_text"):
        fields.append("positive_text")
    if scene.get("negative_text"):
        fields.append("negative_text")
    return ", ".join(fields) if fields else "none"


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


def resolve_wildcards(text, seed=0):
    text = str(text or "")
    if not text:
        return text

    rng = random.Random(int(seed)) if int(seed or 0) != 0 else random.Random()

    def replace(match):
        choices = [item.strip() for item in match.group(1).split("|")]
        choices = [item for item in choices if item]
        if not choices:
            return match.group(0)
        return rng.choice(choices)

    previous = None
    current = text
    while previous != current:
        previous = current
        current = WILDCARD_RE.sub(replace, current)
    return current


def resolve_scene_database_wildcards(text, scene_db):
    text = str(text or "")
    if not text or not isinstance(scene_db, dict):
        return text

    groups = {}
    for item in scene_db.get("universal_memory", {}).values():
        if not isinstance(item, dict):
            continue
        group = str(item.get("wildcard_group") or "").strip()
        phrase = str(item.get("text") or "").strip()
        if not group or not phrase:
            continue
        groups.setdefault(group, {})[normalize_scene_key(phrase)] = phrase

    if not groups:
        return text

    parts = re.split(r"([,;.\n]+)", text)
    matches_by_group = {}
    for index in range(0, len(parts), 2):
        phrase_key = normalize_scene_key(parts[index])
        if not phrase_key:
            continue
        for group, phrases in groups.items():
            if phrase_key in phrases:
                matches_by_group.setdefault(group, []).append((index, phrase_key))
                break

    changed = False
    for matches in matches_by_group.values():
        unique_keys = sorted({phrase_key for _, phrase_key in matches})
        if len(unique_keys) < 2:
            continue
        keep = random.choice(unique_keys)
        for index, phrase_key in matches:
            if phrase_key != keep:
                parts[index] = ""
                changed = True

    if not changed:
        return text

    output = "".join(parts)
    output = re.sub(r"\s*([,;.])\s*([,;.]\s*)+", r"\1 ", output)
    output = re.sub(r"(?:^|[\s,;.])+\n", "\n", output)
    output = re.sub(r"\s+", " ", output.replace("\n", ", ")).strip(" ,;.")
    return output


def prompt_key_for_refiner(prompt, mode):
    del mode
    return re.sub(r"\s+", " ", str(prompt or "").strip())


def load_refiner_state(refinement_key, mode):
    if not refinement_key:
        return None, "No refinement key stored."
    candidates = (
        (refinement_state_path(refinement_key, "clip", prefix="refine_v2"), "V2 refiner state"),
        (refinement_state_path(refinement_key, mode), "legacy refiner state"),
    )
    for path, label in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            if isinstance(data, dict):
                data["_funpack_state_label"] = label
            return data, f"{label} loaded for key '{refinement_key}'."
        except (json.JSONDecodeError, OSError, ValueError):
            return None, f"{label} for key '{refinement_key}' is unreadable."
    return None, f"No refiner state found for key '{refinement_key}'."


def choose_prompt_history(data, prompt, mode):
    prompt_histories = data.get("prompt_histories", {}) if isinstance(data, dict) else {}
    if not isinstance(prompt_histories, dict) or not prompt_histories:
        return None, "", "No prompt histories in refiner state."

    prompt_key = prompt_key_for_refiner(prompt, mode)
    if prompt_key in prompt_histories and isinstance(prompt_histories[prompt_key], dict):
        return prompt_histories[prompt_key], prompt_key, "Matched exact prompt history."

    last_key = data.get("last_prompt_key") if isinstance(data, dict) else None
    if isinstance(last_key, str) and isinstance(prompt_histories.get(last_key), dict):
        return prompt_histories[last_key], last_key, "Used latest prompt history."

    for key, history in prompt_histories.items():
        if isinstance(history, dict):
            return history, key, "Used first available prompt history."

    return None, "", "No usable prompt history found."


def best_history_conditioning(history):
    if not isinstance(history, dict):
        return None, "No prompt history selected."

    liked = history.get("liked_reference_embeds")
    if liked is not None and int(history.get("liked_reference_count", 0) or 0) > 0:
        return liked, "Loaded liked-average conditioning."

    best_entry = None
    best_score = None
    for entry in history.get("history", []):
        if not isinstance(entry, dict) or entry.get("modified_embeds") is None:
            continue
        profile = normalize_refiner_rating(entry.get("rating_label", entry.get("rating", 0)))
        score = (
            int(profile.get("level", 0)),
            float(profile.get("reward", 0.0)),
            int(entry.get("iteration", 0)),
        )
        if best_score is None or score > best_score:
            best_entry = entry
            best_score = score
    if best_entry is not None:
        return best_entry.get("modified_embeds"), "Loaded best-rated history conditioning."

    reference = history.get("reference_embeds") or history.get("source_conditioning_embeds")
    if reference is not None:
        return reference, "Loaded stored reference conditioning."

    return None, "No conditioning data found in selected history."


def conditioning_from_refiner(refinement_key, mode, prompt):
    data, state_status = load_refiner_state(refinement_key, mode)
    if data is None:
        return None, state_status

    history, _, history_status = choose_prompt_history(data, prompt, mode)
    serialized, conditioning_status = best_history_conditioning(history)
    if serialized is None and isinstance(data, dict):
        global_state = data.get("global") if isinstance(data.get("global"), dict) else {}
        if isinstance(global_state.get("liked_conditioning"), dict):
            serialized = global_state.get("liked_conditioning")
            conditioning_status = "Loaded V2 liked-average conditioning."
        elif isinstance(data.get("last_run"), dict) and isinstance(data["last_run"].get("conditioning"), dict):
            serialized = data["last_run"].get("conditioning")
            conditioning_status = "Loaded V2 latest-run conditioning."
    if serialized is None:
        return None, f"{state_status} {history_status} {conditioning_status}"

    try:
        tensor = serializable_to_tensor(serialized)
    except Exception as error:
        return None, f"{state_status} {history_status} Failed to restore conditioning: {error}"

    return [(tensor, {"pooled_output": None})], f"{state_status} {history_status} {conditioning_status}"


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


@PromptServer.instance.routes.get("/funpack/scenes")
async def funpack_scenes(request):
    key = normalize_refinement_key(request.query.get("key", ""))
    data = load_scene_db(key)
    return web.json_response(
        {
            "scenes": scene_names(key),
            "path": refinement_key_path(key) if key else scene_store_path(),
            "refinement_key": key,
            "data": data,
            "memory": scene_memory_items(data),
        },
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@PromptServer.instance.routes.get("/funpack/scenes/export")
async def funpack_scenes_export(request):
    key = normalize_refinement_key(request.query.get("key", ""))
    data = load_scene_db(key)
    return web.json_response(
        data,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Content-Disposition": f"attachment; filename=funpack_scenes_{key or 'global'}.json",
        },
    )


@PromptServer.instance.routes.post("/funpack/scenes/scene")
async def funpack_scene_save(request):
    key = normalize_refinement_key(request.query.get("key", ""))
    body = await request.json()
    if not isinstance(body, dict):
        return web.json_response({"error": "Scene payload must be an object."}, status=400)

    action = str(body.get("action") or "save").lower()
    name = normalize_scene_name(body.get("name"))
    if not name:
        return web.json_response({"error": "Scene name is required."}, status=400)

    data = load_scene_db(key)
    scenes = data.setdefault("scenes", {})
    if action == "delete":
        scenes.pop(name, None)
        save_scene_db(data, key)
        return web.json_response({"deleted": name, "scenes": scene_names(key), "data": data})

    previous = scenes.get(name, {}) if isinstance(scenes.get(name), dict) else {}
    scene = scene_from_texts(
        name,
        body.get("aliases", previous.get("aliases", [])),
        body.get("mode") or body.get("output_mode") or previous.get("output_mode", "Manual"),
        body.get("positive_text", previous.get("positive_text", "")),
        body.get("negative_text", previous.get("negative_text", "")),
        key,
    )
    scene["created_at"] = previous.get("created_at", now_iso())
    scene["updated_at"] = now_iso()
    scenes[name] = scene
    save_scene_db(data, key)
    return web.json_response({"saved": name, "scenes": scene_names(key), "data": data})


@PromptServer.instance.routes.post("/funpack/scenes/database")
async def funpack_scene_database_save(request):
    key = normalize_refinement_key(request.query.get("key", ""))
    body = await request.json()
    if not isinstance(body, dict):
        return web.json_response({"error": "Database payload must be an object."}, status=400)

    incoming = body.get("universal_memory", body.get("memory", {}))
    memory = normalize_scene_memory_items(incoming)
    data = load_scene_db(key)
    data["universal_memory"] = memory
    save_scene_db(data, key)
    return web.json_response({"saved": len(memory), "data": data, "memory": scene_memory_items(data)})


@PromptServer.instance.routes.post("/funpack/scenes/import")
async def funpack_scenes_import(request):
    key = normalize_refinement_key(request.query.get("key", ""))
    incoming = await request.json()
    if not isinstance(incoming, dict):
        return web.json_response({"error": "Imported file is not a scene database."}, status=400)

    data = load_scene_db(key)
    imported_memory = incoming.get("universal_memory", {})
    if isinstance(imported_memory, dict):
        memory = data.setdefault("universal_memory", {})
        for memory_key, item in imported_memory.items():
            if isinstance(item, dict):
                memory[str(memory_key)] = item

    imported_scenes = incoming.get("scenes", {})
    if not isinstance(imported_scenes, dict):
        return web.json_response({"error": "Imported file does not contain a scenes object."}, status=400)

    scenes = data.setdefault("scenes", {})
    imported = 0
    for name, scene in imported_scenes.items():
        clean_name = normalize_scene_name(name)
        if not clean_name or not isinstance(scene, dict):
            continue
        item = dict(scene)
        item["name"] = clean_name
        item["updated_at"] = now_iso()
        scenes[clean_name] = item
        imported += 1
    save_scene_db(data, key)
    return web.json_response({"imported": imported, "scenes": scene_names(key), "refinement_key": key})


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
            "Content-Disposition": f"attachment; filename=funpack_refinement_{loaded_key}.json",
        },
    )


@PromptServer.instance.routes.post("/funpack/refinement_keys/import")
async def funpack_refinement_keys_import(request):
    incoming = await request.json()
    if isinstance(incoming, dict) and isinstance(incoming.get("state"), dict):
        incoming = incoming["state"]
    state, key = normalize_refinement_state(incoming)
    if state is None:
        return web.json_response({"error": "Imported file is not a valid V2 refinement key JSON."}, status=400)
    path = save_refinement_key_state(state, key)
    if not path:
        return web.json_response({"error": "Could not save imported refinement key."}, status=400)
    return web.json_response({"imported": key, "keys": refinement_key_names()})


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


class FunPackSceneBuilder:
    CATEGORY = "FunPack/Scene"
    RETURN_TYPES = ("STRING", "STRING", "FUNPACK_LORA_STACK", "STRING")
    RETURN_NAMES = (
        "positive_prompt",
        "negative_prompt",
        "lora_stack",
        "status",
    )
    FUNCTION = "build_scene"
    OUTPUT_NODE = True
    DESCRIPTION = "Builds named scene presets from manually selected universal prompt phrases."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_name": ("STRING", {"default": "", "multiline": False}),
                "mode": (["Manual", "Auto", "Learning"], {"default": "Manual"}),
                "scene": (scene_names(), {"default": SCENE_NONE}),
                "aliases": ("STRING", {"default": "", "multiline": False}),
                "action": (["load", "save", "update", "delete"], {"default": "load"}),
                "scene_positive": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Write or arrange positive scene words and phrases here.",
                }),
                "scene_negative": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Write or arrange negative scene words and phrases here.",
                }),
                "refinement_key": ("STRING", {"default": "", "multiline": False}),
                "scene_payload": ("STRING", {"default": "{}", "multiline": False}),
            },
            "optional": {
                "intent_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Connected intent text used for Auto scene detection.",
                }),
                "positive_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Connected positive prompt used only to collect universal scene phrase memory.",
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True,
                    "tooltip": "Connected negative prompt used only to collect universal negative phrase memory.",
                }),
                "refinement_key_input": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "forceInput": True,
                    "tooltip": "Optional linked refinement key, for example from FunPack Refinement Key Loader.",
                }),
                "lora_stack": ("FUNPACK_LORA_STACK", {
                    "tooltip": "Optional current LoRA stack. Scene Builder passes it through unchanged so Refiner can use it for suggestions.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, scene=None, **kwargs):
        return True

    def _manual_scene(self, selected_name, scene_name, aliases, mode, scene_positive, scene_negative, refinement_key, payload):
        target_name = normalize_scene_name(scene_name) or selected_name
        positive_text = scene_text_or_payload(scene_positive, payload["positive_phrases"])
        negative_text = scene_text_or_payload(scene_negative, payload["negative_phrases"])
        return scene_from_texts(target_name, aliases, mode, positive_text, negative_text, refinement_key)

    def _outputs_for_scene(self, name, scene, scene_db=None, lora_stack=None, source="Manual"):
        positive = str(scene.get("positive_text") or scene_text_from_phrases(scene.get("positive_phrases", [])))
        negative = str(scene.get("negative_text") or scene_text_from_phrases(scene.get("negative_phrases", [])))
        positive = resolve_scene_database_wildcards(positive, scene_db)
        negative = resolve_scene_database_wildcards(negative, scene_db)
        lora_count = len(lora_stack.get("loras", [])) if isinstance(lora_stack, dict) else 0
        status = (
            f"Scene Builder {source}: '{name or scene.get('name', '') or 'unsaved'}'. "
            f"Stored fields: {scene_field_summary(scene)}.\n"
            f"Positive phrases: {len(scene.get('positive_phrases', []) or [])}. "
            f"Negative phrases: {len(scene.get('negative_phrases', []) or [])}. "
            f"LoRA stack pass-through: {lora_count} LoRA(s)."
        )
        return (positive, negative, lora_stack, status)

    def _learning_outputs(self, positive_prompt, negative_prompt, lora_stack=None):
        positive = str(positive_prompt or "")
        negative = str(negative_prompt or "")
        lora_count = len(lora_stack.get("loras", [])) if isinstance(lora_stack, dict) else 0
        status = (
            "Scene Builder Learning: pass-through active. "
            f"Collected positive source: {'yes' if positive else 'no'}. "
            f"Collected negative source: {'yes' if negative else 'no'}. "
            f"LoRA stack pass-through: {lora_count} LoRA(s)."
        )
        return (positive, negative, lora_stack, status)

    def _empty_outputs(self, status):
        return ("", "", None, status)

    def build_scene(
        self,
        scene_name,
        mode="Manual",
        scene=SCENE_NONE,
        aliases="",
        action="load",
        scene_positive="",
        scene_negative="",
        refinement_key="",
        scene_payload="{}",
        intent_prompt="",
        positive_prompt="",
        negative_prompt="",
        refinement_key_input="",
        lora_stack=None,
        output_mode=None,
    ):
        linked_key = normalize_refinement_key(refinement_key_input)
        if linked_key:
            refinement_key = linked_key
        selected_name = normalize_scene_name(scene)
        if output_mode in {"Manual", "Auto", "Learning"}:
            mode = output_mode
        mode = mode if mode in {"Manual", "Auto", "Learning"} else "Manual"
        action = action if action in {"load", "save", "update", "delete"} else "load"
        payload = parse_scene_payload(scene_payload)

        data = load_scene_db(refinement_key)
        memory_changed = remember_scene_phrases(data, positive_prompt, negative_prompt)
        scenes = data.setdefault("scenes", {})
        manual = self._manual_scene(
            selected_name,
            scene_name,
            aliases,
            mode,
            scene_positive,
            scene_negative,
            refinement_key,
            payload,
        )

        if action == "delete":
            if not selected_name or selected_name not in scenes:
                if memory_changed:
                    save_scene_db(data, refinement_key)
                return self._empty_outputs("Delete skipped: no selected scene.")
            deleted = scenes.pop(selected_name)
            save_scene_db(data, refinement_key)
            return self._empty_outputs(
                f"Deleted scene '{selected_name}'. Removed fields: {scene_field_summary(deleted)}."
            )

        if action == "save":
            target_name = normalize_scene_name(scene_name) or selected_name
            if not target_name:
                if memory_changed:
                    save_scene_db(data, refinement_key)
                return self._empty_outputs("Save skipped: enter a scene name.")
            manual["name"] = target_name
            manual["created_at"] = scenes.get(target_name, {}).get("created_at", now_iso())
            manual["updated_at"] = now_iso()
            scenes[target_name] = manual
            save_scene_db(data, refinement_key)
            return self._outputs_for_scene(target_name, manual, data, lora_stack, "Manual saved")

        if action == "update":
            target_name = selected_name or normalize_scene_name(scene_name)
            if not target_name:
                if memory_changed:
                    save_scene_db(data, refinement_key)
                return self._empty_outputs("Update skipped: select a scene or enter a scene name.")
            previous = scenes.get(target_name, {})
            manual["name"] = target_name
            manual["created_at"] = previous.get("created_at", now_iso()) if isinstance(previous, dict) else now_iso()
            manual["updated_at"] = now_iso()
            scenes[target_name] = manual
            save_scene_db(data, refinement_key)
            return self._outputs_for_scene(target_name, manual, data, lora_stack, "Manual updated")

        if mode == "Learning":
            if memory_changed:
                save_scene_db(data, refinement_key)
            return self._learning_outputs(positive_prompt, negative_prompt, lora_stack)

        if mode == "Auto":
            matched_name, matched_scene, match_type = find_scene_for_intent(intent_prompt, scenes)
            if matched_name and isinstance(matched_scene, dict):
                if memory_changed:
                    save_scene_db(data, refinement_key)
                return self._outputs_for_scene(matched_name, matched_scene, data, lora_stack, f"Auto {match_type}")

        if memory_changed:
            save_scene_db(data, refinement_key)
        return self._outputs_for_scene(manual.get("name", ""), manual, data, lora_stack, "Manual")
