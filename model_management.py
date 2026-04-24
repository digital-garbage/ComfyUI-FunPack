import json
import logging
import math
import os
import re
from collections import defaultdict
from hashlib import md5

import comfy.lora
import comfy.lora_convert
import comfy.sd
import comfy.utils
import folder_paths
import torch
from aiohttp import web
from server import PromptServer


LORA_TYPES = ["general", "concept", "style", "quality", "character"]
LORA_STACK_TYPE = "FUNPACK_LORA_STACK"
TRANSFORMER_BLOCK_PATTERN = re.compile(r"(?:^|\.)transformer_blocks\.(\d+)\.")
LTX_IMAGE_MODELS = {"ltxv", "ltxav"}


@PromptServer.instance.routes.get("/funpack/loras")
async def funpack_loras(_):
    return web.json_response(
        ["None"] + folder_paths.get_filename_list("loras"),
        headers={"Cache-Control": "no-store, max-age=0"},
    )


class AnyType(str):
    def __ne__(self, _):
        return False


any_type = AnyType("*")


class FlexibleOptionalInputType(dict):
    def __init__(self, input_type, data=None):
        super().__init__(data or {})
        self.input_type = input_type
        self.data = data or {}

    def __contains__(self, _):
        return True

    def __getitem__(self, key):
        return self.data.get(key, (self.input_type,))

    def get(self, key, default=None):
        return self.data.get(key, default)

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


def normalize_prompt_for_mode(prompt, mode):
    prompt = (prompt or "").strip()
    if (mode or "ltx2").lower() == "wan":
        return re.sub(r"\s+", " ", prompt)
    return prompt


def prompt_key_for_mode(prompt, mode):
    if (mode or "ltx2").lower() == "wan":
        return normalize_prompt_for_mode(prompt, mode)
    return prompt or ""


def lora_state_id(lora_name, lora_type):
    return md5(f"{lora_name}::{lora_type}".encode("utf-8")).hexdigest()[:16]


def refiner_state_path(refinement_key, mode):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    refinements_dir = os.path.join(base_dir, "refinements")
    safe_key = md5(f"{(mode or 'ltx2').lower()}::{refinement_key}".encode("utf-8")).hexdigest()
    return os.path.join(refinements_dir, f"refine_{safe_key}.json")


def coerce_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def safe_float(value, fallback=1.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def patch_target_key(patch_key):
    if isinstance(patch_key, tuple) and patch_key:
        return patch_key[0]
    return patch_key


def transformer_block_index(patch_key):
    target_key = patch_target_key(patch_key)
    if not isinstance(target_key, str):
        return None

    match = TRANSFORMER_BLOCK_PATTERN.search(target_key)
    if not match:
        return None
    return int(match.group(1))


def patch_energy(value):
    if isinstance(value, torch.Tensor):
        return float(value.abs().mean().item())

    weights = getattr(value, "weights", None)
    if weights is not None:
        return patch_energy(weights)

    if isinstance(value, dict):
        return sum(patch_energy(item) for item in value.values())

    if isinstance(value, (list, tuple)):
        return sum(patch_energy(item) for item in value)

    return 0.0


class FunPackApplyLoraWeights:
    """
    Builds a LoRA stack from user base weights, then applies prompt-specific
    suggested weights previously saved by FunPack Gemma Embedding Refiner.
    """

    CATEGORY = "FunPack/Model Management"
    RETURN_TYPES = (LORA_STACK_TYPE, "STRING")
    RETURN_NAMES = ("lora_stack", "status")
    FUNCTION = "apply_lora_weights"
    DESCRIPTION = "Reads Gemma Refiner prompt-specific LoRA suggestions and prepares a LoRA stack for loading."

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        optional = FlexibleOptionalInputType(
            any_type,
            {
                "lora_list": ("STRING", {"default": "[]", "multiline": False}),
                "lora_0": (loras, {"default": "None"}),
                "lora_0_type": (LORA_TYPES, {"default": "general"}),
                "lora_0_base_weight": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "Trainer-recommended model anchor weight.",
                    },
                ),
            },
        )

        return {
            "required": {
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "refinement_key": ("STRING", {"default": "my_style_v1", "multiline": False}),
                "mode": (["ltx2", "wan"], {"default": "ltx2"}),
                "per_block": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "For LTX-mode stacks, derive hidden transformer block strengths from the LoRA itself and apply them automatically.",
                    },
                ),
            },
            "optional": optional,
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, lora_list=None, lora_0=None):
        return True

    def _load_suggestions(self, refinement_key, mode, prompt_key):
        path = refiner_state_path(refinement_key, mode)
        if not os.path.exists(path):
            return {}, "base weights: no refiner state file"

        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (json.JSONDecodeError, OSError, ValueError):
            return {}, "base weights: refiner state unreadable"

        prompt_history = data.get("prompt_histories", {}).get(prompt_key)
        if not prompt_history:
            return {}, "base weights: prompt has no exact-match suggestions"

        suggestions = prompt_history.get("lora_weight_suggestions", {})
        if not suggestions:
            return {}, "base weights: prompt suggestions not available yet"

        return suggestions, "refiner suggestions applied"

    def _entry_from_row(self, index, row):
        if not isinstance(row, dict):
            return None
        if not row.get("on", True):
            return None

        name = row.get("lora", row.get("name", "None"))
        if not name or name == "None":
            return None

        lora_type = row.get("type", row.get("lora_type", "general"))
        if lora_type not in LORA_TYPES:
            lora_type = "general"

        return {
            "slot": index,
            "name": name,
            "type": lora_type,
            "id": lora_state_id(name, lora_type),
            "base_model_weight": safe_float(row.get("strength", row.get("base_weight", 1.0))),
        }

    def _iter_lora_list(self, value):
        if value in (None, "", "[]"):
            return None
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
        if not isinstance(value, list):
            return None

        entries = []
        for index, row in enumerate(value):
            entry = self._entry_from_row(index, row)
            if entry:
                entries.append(entry)
        return entries

    def _iter_slots(self, kwargs):
        listed_entries = self._iter_lora_list(kwargs.get("lora_list"))
        if listed_entries is not None:
            yield from listed_entries
            return

        indexed_slots = set()
        for key in kwargs:
            match = re.fullmatch(r"lora_(\d+)", key)
            if match:
                indexed_slots.add(int(match.group(1)))

        for index in sorted(indexed_slots):
            lora_name = kwargs.get(f"lora_{index}", "None")
            if isinstance(lora_name, dict):
                entry = self._entry_from_row(index, lora_name)
                if entry:
                    yield entry
                continue

            shifted_base_weight = None
            lora_type = kwargs.get(f"lora_{index}_type", "general")
            if not isinstance(lora_name, str) and isinstance(lora_type, str) and lora_type not in LORA_TYPES:
                shifted_base_weight = safe_float(lora_name, 1.0)
                lora_name = lora_type
                lora_type = "general"

            if not lora_name or lora_name == "None":
                continue

            if lora_type not in LORA_TYPES:
                lora_type = "general"
            base_weight = safe_float(kwargs.get(f"lora_{index}_base_weight", 1.0), shifted_base_weight or 1.0)

            yield {
                "slot": index,
                "name": lora_name,
                "type": lora_type,
                "id": lora_state_id(lora_name, lora_type),
                "base_model_weight": base_weight,
            }

    def _get_suggestion(self, suggestions, entry):
        suggestion = suggestions.get(entry["id"])
        if self._suggestion_matches_base(suggestion, entry):
            return suggestion

        for legacy in suggestions.values():
            if (
                isinstance(legacy, dict)
                and legacy.get("name") == entry["name"]
                and legacy.get("type", "general") == entry["type"]
                and self._suggestion_matches_base(legacy, entry)
            ):
                return legacy

        return {}

    def _suggestion_matches_base(self, suggestion, entry):
        if not isinstance(suggestion, dict):
            return False

        saved_base = suggestion.get("base_model_weight")
        if saved_base is None:
            return True

        return abs(float(saved_base) - float(entry["base_model_weight"])) <= 1e-6

    def apply_lora_weights(self, positive_prompt, refinement_key, mode, per_block=False, **kwargs):
        mode = (mode or "ltx2").lower()
        per_block = coerce_bool(per_block)
        prompt_key = prompt_key_for_mode(positive_prompt, mode)
        suggestions, source_message = self._load_suggestions(refinement_key, mode, prompt_key)

        loras = []
        lines = [f"FunPack Apply LoRA Weights | {source_message}"]
        lines.append(f"Per-block application: {'enabled' if per_block else 'disabled'}")
        for entry in self._iter_slots(kwargs):
            suggestion = self._get_suggestion(suggestions, entry)
            model_weight = float(suggestion.get("model_weight", entry["base_model_weight"]))
            source = "suggested" if suggestion else "base"

            stack_entry = dict(entry)
            stack_entry["model_weight"] = model_weight
            stack_entry["source"] = source
            loras.append(stack_entry)

            lines.append(
                f"lora_{entry['slot']}: {entry['name']} [{entry['type']}] "
                f"{source}={model_weight:+.3f} base={entry['base_model_weight']:+.3f}"
            )

        stack = {
            "version": 2,
            "refinement_key": refinement_key,
            "mode": mode,
            "per_block": per_block,
            "positive_prompt": positive_prompt,
            "prompt_key": prompt_key,
            "loras": loras,
        }

        if not loras:
            lines.append("No LoRAs selected.")

        return (stack, "\n".join(lines))


class FunPackLoraLoader:
    """Loads the LoRA stack prepared by FunPack Apply LoRA Weights."""

    CATEGORY = "FunPack/Model Management"
    RETURN_TYPES = ("MODEL", "CLIP", LORA_STACK_TYPE, "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "lora_stack", "status")
    FUNCTION = "load_loras"
    DESCRIPTION = "Loads LoRAs from a FunPack LoRA stack without doing any learning."

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_stack": (LORA_STACK_TYPE,),
            },
            "optional": {
                "clip": ("CLIP",),
            },
        }

    def _load_lora_file(self, lora_name):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        return lora

    def _model_image_model(self, model):
        model_wrapper = getattr(model, "model", None)
        model_config = getattr(model_wrapper, "model_config", None)
        unet_config = getattr(model_config, "unet_config", None)
        if isinstance(unet_config, dict):
            return unet_config.get("image_model")
        return None

    def _per_block_requested(self, entry, lora_stack):
        return coerce_bool(entry.get("per_block", lora_stack.get("per_block", False)))

    def _per_block_supported(self, model, lora_stack, entry):
        if not self._per_block_requested(entry, lora_stack):
            return False

        if (lora_stack.get("mode") or "ltx2").lower() != "ltx2":
            return False

        return self._model_image_model(model) in LTX_IMAGE_MODELS

    def _load_model_lora_patches(self, model, lora):
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        converted_lora = comfy.lora_convert.convert_lora(lora)
        return comfy.lora.load_lora(converted_lora, key_map)

    def _split_model_patches_by_block(self, loaded):
        global_patches = {}
        block_patches = defaultdict(dict)

        for patch_key, patch_value in loaded.items():
            block_index = transformer_block_index(patch_key)
            if block_index is None:
                global_patches[patch_key] = patch_value
                continue
            block_patches[block_index][patch_key] = patch_value

        return global_patches, dict(block_patches)

    def _block_scales_from_patches(self, block_patches):
        block_scores = {}
        for block_index, patches in block_patches.items():
            score = 0.0
            for patch_value in patches.values():
                score += patch_energy(patch_value)
            if score > 0.0:
                block_scores[block_index] = score

        if len(block_scores) < 2:
            return {}

        mean_score = sum(block_scores.values()) / len(block_scores)
        if mean_score <= 0.0:
            return {}

        scales = {}
        for block_index, score in block_scores.items():
            ratio = max(0.0, score / mean_score)
            scales[block_index] = max(0.25, min(1.75, ratio ** 0.5))

        return scales

    def _apply_model_patches(self, model, loaded, model_weight, block_scales=None):
        new_model = model.clone()
        applied = set()
        global_patches, block_patches = self._split_model_patches_by_block(loaded)

        if global_patches:
            applied.update(new_model.add_patches(global_patches, model_weight))

        if block_scales:
            for block_index in sorted(block_patches):
                block_strength = model_weight * block_scales.get(block_index, 1.0)
                applied.update(new_model.add_patches(block_patches[block_index], block_strength))
        else:
            for block_index in sorted(block_patches):
                applied.update(new_model.add_patches(block_patches[block_index], model_weight))

        for patch_key in loaded:
            if patch_key not in applied:
                logging.warning("NOT LOADED %s", patch_key)

        return new_model, len(global_patches), len(block_patches)

    def _load_lora_per_block(self, model, lora, model_weight):
        loaded = self._load_model_lora_patches(model, lora)
        _, block_patches = self._split_model_patches_by_block(loaded)
        block_scales = self._block_scales_from_patches(block_patches)
        if not block_scales:
            return None, "per-block fallback=global"

        new_model, non_block_count, _ = self._apply_model_patches(
            model,
            loaded,
            model_weight,
            block_scales=block_scales,
        )
        min_scale = min(block_scales.values())
        max_scale = max(block_scales.values())
        status = (
            f"per-block blocks={len(block_scales)} non_block={non_block_count} "
            f"range={min_scale:.2f}..{max_scale:.2f}"
        )
        return new_model, status

    def load_loras(self, model, lora_stack, clip=None):
        loras = lora_stack.get("loras", []) if isinstance(lora_stack, dict) else []
        per_block = coerce_bool(lora_stack.get("per_block", False)) if isinstance(lora_stack, dict) else False
        lines = [f"FunPack LoRA Loader | loading {len(loras)} LoRA(s)"]
        lines.append(f"Per-block application: {'enabled' if per_block else 'disabled'}")
        loaded_count = 0

        for entry in loras:
            model_weight = float(entry.get("model_weight", 0.0))
            if model_weight == 0:
                lines.append(f"lora_{entry.get('slot', '?')}: {entry.get('name', '?')} skipped at zero weight")
                continue

            lora = self._load_lora_file(entry["name"])
            apply_status = "global"
            if self._per_block_supported(model, lora_stack, entry):
                per_block_model, apply_status = self._load_lora_per_block(model, lora, model_weight)
                if per_block_model is not None:
                    model = per_block_model
                else:
                    model, clip = comfy.sd.load_lora_for_models(model, clip, lora, model_weight, 0.0)
            elif self._per_block_requested(entry, lora_stack):
                apply_status = "per-block unsupported -> global"
                model, clip = comfy.sd.load_lora_for_models(model, clip, lora, model_weight, 0.0)
            else:
                model, clip = comfy.sd.load_lora_for_models(model, clip, lora, model_weight, 0.0)
            loaded_count += 1
            lines.append(
                f"lora_{entry.get('slot', '?')}: {entry['name']} "
                f"applied={model_weight:+.3f} source={entry.get('source', 'base')} mode={apply_status}"
            )

        if loaded_count == 0:
            lines.append("No LoRAs were applied.")

        return (model, clip, lora_stack, "\n".join(lines))
