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
TEMPLATE_DB_VERSION = 1
WILDCARD_RE = re.compile(r"\{([^{}]*\|[^{}]*)\}")


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


class FunPackTemplateManager:
    CATEGORY = "FunPack/Templates"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "SIGMAS", "CONDITIONING", "FUNPACK_LORA_STACK", "STRING", "STRING")
    RETURN_NAMES = (
        "positive_prompt",
        "negative_prompt",
        "activation_word",
        "sigmas",
        "positive_conditioning",
        "lora_stack",
        "refinement_key",
        "status",
    )
    FUNCTION = "manage_template"
    OUTPUT_NODE = True
    DESCRIPTION = "Stores and loads reusable FunPack generation templates."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": (template_names(), {"default": TEMPLATE_NONE}),
                "name": ("STRING", {"default": "", "multiline": False}),
                "action": (["load", "save", "update", "delete"], {"default": "load"}),
                "mode": (["ltx2", "wan"], {"default": "ltx2"}),
                "wildcard_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "activation_word": ("STRING", {"default": "", "multiline": False}),
                "refinement_key": ("STRING", {"default": "", "multiline": False}),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "sigmas": ("SIGMAS",),
                "lora_stack": ("FUNPACK_LORA_STACK",),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, template=None, **kwargs):
        return True

    def _empty_outputs(self, status):
        return ("", "", "", torch.FloatTensor([]), None, None, "", status)

    def _loaded_outputs(self, name, template, wildcard_seed):
        mode = template.get("mode", "ltx2")
        positive_raw = template.get("positive_prompt", "")
        positive_prompt = resolve_wildcards(positive_raw, wildcard_seed)
        negative_prompt = template.get("negative_prompt", "")
        activation_word = template.get("activation_word", "")
        refinement_key = template.get("refinement_key", "")

        sigmas = torch.FloatTensor([])
        if isinstance(template.get("sigmas"), dict):
            try:
                sigmas = serializable_to_tensor(template["sigmas"]).detach().clone().cpu()
            except Exception:
                sigmas = torch.FloatTensor([])

        positive_conditioning = None
        conditioning_status = "No refinement key stored; conditioning not loaded."
        if refinement_key:
            positive_conditioning, conditioning_status = conditioning_from_refiner(
                refinement_key,
                mode,
                positive_raw,
            )

        lora_stack = json_safe(template["lora_stack"]) if isinstance(template.get("lora_stack"), dict) else None
        status = (
            f"Template '{name}' loaded. Stored fields: {template_field_summary(template)}.\n"
            f"Wildcard seed: {wildcard_seed or 'random'}.\n"
            f"{conditioning_status}"
        )
        return (
            positive_prompt,
            negative_prompt,
            activation_word,
            sigmas,
            positive_conditioning,
            lora_stack,
            refinement_key if refinement_key else "",
            status,
        )

    def manage_template(
        self,
        template,
        name,
        action,
        mode,
        wildcard_seed,
        activation_word="",
        refinement_key="",
        positive_prompt="",
        negative_prompt="",
        sigmas=None,
        lora_stack=None,
    ):
        action = action if action in {"load", "save", "update", "delete"} else "load"
        selected_name = normalize_template_name(template)
        new_name = normalize_template_name(name)
        data = load_template_db()
        templates = data.setdefault("templates", {})

        if action == "delete":
            if not selected_name or selected_name not in templates:
                return self._empty_outputs("Delete skipped: no selected template.")
            deleted = templates.pop(selected_name)
            save_template_db(data)
            return self._empty_outputs(
                f"Deleted template '{selected_name}'. Removed fields: {template_field_summary(deleted)}."
            )

        if action == "save":
            target_name = new_name if new_name and new_name not in templates else selected_name
            if not target_name:
                return self._empty_outputs("Save skipped: enter a unique name or select an existing template.")
            saved = collect_template_payload(
                mode,
                activation_word,
                refinement_key,
                positive_prompt,
                negative_prompt,
                sigmas,
                lora_stack,
                update_only=False,
            )
            saved["name"] = target_name
            saved["created_at"] = templates.get(target_name, {}).get("created_at", now_iso())
            saved["updated_at"] = now_iso()
            templates[target_name] = saved
            save_template_db(data)
            return self._loaded_outputs(target_name, saved, wildcard_seed)

        if action == "update":
            if not selected_name or selected_name not in templates:
                return self._empty_outputs("Update skipped: select an existing template.")
            updated = collect_template_payload(
                mode,
                activation_word,
                refinement_key,
                positive_prompt,
                negative_prompt,
                sigmas,
                lora_stack,
                update_only=True,
                existing=templates[selected_name],
            )
            updated["name"] = selected_name
            updated["created_at"] = templates[selected_name].get("created_at", now_iso())
            updated["updated_at"] = now_iso()
            templates[selected_name] = updated
            save_template_db(data)
            return self._loaded_outputs(selected_name, updated, wildcard_seed)

        if not selected_name or selected_name not in templates:
            return self._empty_outputs("No template selected.")
        return self._loaded_outputs(selected_name, templates[selected_name], wildcard_seed)
