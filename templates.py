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
    prompt = (prompt or "").strip()
    if (mode or "ltx2").lower() == "wan":
        return re.sub(r"\s+", " ", prompt)
    return prompt


def load_refiner_state(refinement_key, mode):
    if not refinement_key:
        return None, "No refinement key stored."
    path = refinement_state_path(refinement_key, mode)
    if not os.path.exists(path):
        return None, f"No refiner state found for key '{refinement_key}'."
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file), f"Refiner state loaded for key '{refinement_key}'."
    except (json.JSONDecodeError, OSError, ValueError):
        return None, f"Refiner state for key '{refinement_key}' is unreadable."


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
