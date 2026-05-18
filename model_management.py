import json
import logging
import math
import os
import re
from collections import OrderedDict, defaultdict
from hashlib import md5

import comfy.lora
import comfy.lora_convert
import comfy.sd
import comfy.utils
import folder_paths
import torch
from aiohttp import web
from server import PromptServer


LORA_TYPES = ["general", "action", "style", "quality", "character"]
LORA_STACK_TYPE = "FUNPACK_LORA_STACK"
TRANSFORMER_BLOCK_PATTERN = re.compile(r"(?:^|\.)transformer_blocks\.(\d+)\.")
LTX_IMAGE_MODELS = {"ltxv", "ltxav"}
LORA_RAW_CACHE_SIZE = 12
LORA_PATCH_CACHE_SIZE = 24
LORA_PROFILE_CACHE_SIZE = 24
LORA_BLOCK_TYPE_PROFILES = {
    "character": {"priority": 1.18, "yield": 0.48},
    "action": {"priority": 1.12, "yield": 0.62},
    "concept": {"priority": 1.12, "yield": 0.62},
    "quality": {"priority": 1.04, "yield": 0.72},
    "style": {"priority": 0.96, "yield": 0.96},
    "general": {"priority": 0.90, "yield": 1.12},
}
# Normalized block position zones (0.0=first block, 1.0=last block).
# good: primary contribution zone - blocks here get a mild boost
# bad: blocks here actively work against the type's purpose - suppressed to 0.0
LORA_TYPE_BLOCK_ZONES = {
    "character": {"good": (0.45, 1.00), "bad": (0.00, 0.20)},
    "action":    {"good": (0.20, 0.80), "bad": None},
    "concept":   {"good": (0.20, 0.80), "bad": None},
    "quality":   {"good": None,         "bad": None},
    "style":     {"good": (0.00, 0.50), "bad": (0.65, 1.00)},
    "general":   {"good": None,         "bad": None},
}
LORA_ZONE_BOOST = 1.12
LORA_ZONE_SCALE_CAP = 1.75
# Blocks confirmed as primary semantic focal points (PAG default=14, STG defaults=14,19).
# These are never zeroed by type-zone suppression and never damped by stack pressure.
SEMANTIC_ANCHOR_BLOCKS = frozenset({14, 19})


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


def prompt_key_for_v2(prompt):
    return re.sub(r"\s+", " ", str(prompt or "").strip())


def lora_state_id(lora_name, lora_type):
    return md5(f"{lora_name}::{lora_type}".encode("utf-8")).hexdigest()[:16]


def normalize_lora_type(lora_type):
    lora_type = str(lora_type or "general").strip().lower()
    if lora_type == "concept":
        return "action"
    return lora_type if lora_type in LORA_TYPES else "general"


def refiner_state_path(refinement_key, mode):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    refinements_dir = os.path.join(base_dir, "refinements")
    safe_key = md5(f"{(mode or 'ltx2').lower()}::{refinement_key}".encode("utf-8")).hexdigest()
    return os.path.join(refinements_dir, f"refine_{safe_key}.json")


def refiner_v2_state_path(refinement_key):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    refinements_dir = os.path.join(base_dir, "refinements")
    safe_key = md5(f"clip::{refinement_key}".encode("utf-8")).hexdigest()
    return os.path.join(refinements_dir, f"refine_v2_{safe_key}.json")


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
                "refinement_key_input": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "forceInput": True,
                    "tooltip": "Optional linked refinement key, for example from FunPack Refinement Key Loader. Overrides the refinement_key widget when connected.",
                }),
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
                        "tooltip": "For LTX-mode stacks, analyze LoRA block deltas and balance competing block strengths automatically.",
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

    def _load_suggestions(self, refinement_key, mode, prompt_key, v2_prompt_key):
        for path, key, label in (
            (refiner_v2_state_path(refinement_key), v2_prompt_key, "V2 refiner suggestions applied"),
            (refiner_state_path(refinement_key, mode), prompt_key, "legacy refiner suggestions applied"),
        ):
            if not os.path.exists(path):
                continue

            try:
                with open(path, "r", encoding="utf-8") as file:
                    data = json.load(file)
            except (json.JSONDecodeError, OSError, ValueError):
                continue

            prompt_history = data.get("prompt_histories", {}).get(key)
            if not prompt_history:
                continue

            suggestions = prompt_history.get("lora_weight_suggestions", {})
            if suggestions:
                return suggestions, label

        return {}, "base weights: prompt suggestions not available yet"

    def _entry_from_row(self, index, row):
        if not isinstance(row, dict):
            return None
        if not row.get("on", True):
            return None

        name = row.get("lora", row.get("name", "None"))
        if not name or name == "None":
            return None

        lora_type = normalize_lora_type(row.get("type", row.get("lora_type", "general")))

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
            if not isinstance(lora_name, str) and isinstance(lora_type, str) and lora_type not in LORA_TYPES and lora_type != "concept":
                shifted_base_weight = safe_float(lora_name, 1.0)
                lora_name = lora_type
                lora_type = "general"

            if not lora_name or lora_name == "None":
                continue

            lora_type = normalize_lora_type(lora_type)
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
        linked_refinement_key = str(kwargs.pop("refinement_key_input", "") or "").strip()
        if linked_refinement_key:
            refinement_key = linked_refinement_key
        mode = (mode or "ltx2").lower()
        per_block = coerce_bool(per_block)
        prompt_key = prompt_key_for_mode(positive_prompt, mode)
        v2_prompt_key = prompt_key_for_v2(positive_prompt)
        suggestions, source_message = self._load_suggestions(refinement_key, mode, prompt_key, v2_prompt_key)

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
        self.raw_lora_cache = OrderedDict()
        self.model_patch_cache = OrderedDict()
        self.block_profile_cache = OrderedDict()

    def _cache_get(self, cache, key):
        if key not in cache:
            return None
        value = cache.pop(key)
        cache[key] = value
        return value

    def _cache_put(self, cache, key, value, max_items):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_items:
            cache.popitem(last=False)

    def _lora_file_cache_key(self, lora_name):
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        stat = os.stat(lora_path)
        mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))
        return (lora_path, mtime_ns, stat.st_size)

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
        cache_key = self._lora_file_cache_key(lora_name)
        lora = self._cache_get(self.raw_lora_cache, cache_key)
        if lora is None:
            lora = comfy.utils.load_torch_file(cache_key[0], safe_load=True)
            self._cache_put(self.raw_lora_cache, cache_key, lora, LORA_RAW_CACHE_SIZE)

        return lora, cache_key

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

    def _model_cache_key(self, model):
        model_wrapper = getattr(model, "model", None)
        model_config = getattr(model_wrapper, "model_config", None)
        unet_config = getattr(model_config, "unet_config", None)
        image_model = unet_config.get("image_model") if isinstance(unet_config, dict) else None
        sampling = getattr(model_wrapper, "model_sampling", None)
        return (
            id(model_wrapper),
            type(model_wrapper).__module__,
            type(model_wrapper).__name__,
            type(model_config).__module__ if model_config is not None else None,
            type(model_config).__name__ if model_config is not None else None,
            type(sampling).__module__ if sampling is not None else None,
            type(sampling).__name__ if sampling is not None else None,
            image_model,
        )

    def _load_model_lora_patches(self, model, lora, lora_cache_key, model_cache_key=None):
        if lora_cache_key is None:
            key_map = comfy.lora.model_lora_keys_unet(model.model, {})
            converted_lora = comfy.lora_convert.convert_lora(lora)
            return comfy.lora.load_lora(converted_lora, key_map)

        cache_key = (lora_cache_key, model_cache_key or self._model_cache_key(model))
        loaded = self._cache_get(self.model_patch_cache, cache_key)
        if loaded is not None:
            return loaded

        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        converted_lora = comfy.lora_convert.convert_lora(lora)
        loaded = comfy.lora.load_lora(converted_lora, key_map)
        self._cache_put(self.model_patch_cache, cache_key, loaded, LORA_PATCH_CACHE_SIZE)
        return loaded

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

    def _block_scores_from_patches(self, block_patches):
        block_scores = {}
        for block_index, patches in block_patches.items():
            score = 0.0
            for patch_value in patches.values():
                score += patch_energy(patch_value)
            if score > 0.0:
                block_scores[block_index] = score

        return block_scores

    def _block_scales_from_scores(self, block_scores):
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

    def _block_scales_from_patches(self, block_patches):
        return self._block_scales_from_scores(self._block_scores_from_patches(block_patches))

    def _normalized_block_scores(self, block_scores):
        total = sum(block_scores.values())
        if total <= 0.0:
            return {}

        return {block_index: score / total for block_index, score in block_scores.items()}

    def _top_block_summary(self, normalized_scores, limit=4):
        top_blocks = sorted(normalized_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return ",".join(str(block_index) for block_index, _ in top_blocks) if top_blocks else "none"

    def _block_type_profile(self, entry):
        return LORA_BLOCK_TYPE_PROFILES.get(entry.get("type", "general"), LORA_BLOCK_TYPE_PROFILES["general"])

    def _block_profile_template(self, loaded, patch_cache_key=None):
        if patch_cache_key is not None:
            cached = self._cache_get(self.block_profile_cache, patch_cache_key)
            if cached is not None:
                return cached

        global_patches, block_patches = self._split_model_patches_by_block(loaded)
        block_scores = self._block_scores_from_patches(block_patches)
        base_scales = self._block_scales_from_scores(block_scores)
        normalized_scores = self._normalized_block_scores(block_scores)
        if not base_scales or not normalized_scores:
            return None

        template = {
            "global_count": len(global_patches),
            "block_count": len(block_patches),
            "block_scores": block_scores,
            "base_scales": base_scales,
            "normalized_scores": normalized_scores,
            "top_blocks": self._top_block_summary(normalized_scores),
            "concentration": max(normalized_scores.values()) if normalized_scores else 0.0,
        }
        if patch_cache_key is not None:
            self._cache_put(self.block_profile_cache, patch_cache_key, template, LORA_PROFILE_CACHE_SIZE)
        return template

    def _apply_type_zone_scales(self, scales, entry):
        lora_type = normalize_lora_type(entry.get("type", "general"))
        zone = LORA_TYPE_BLOCK_ZONES.get(lora_type, {})
        good = zone.get("good")
        bad = zone.get("bad")
        if not good and not bad:
            return scales, 0

        block_indices = list(scales.keys())
        if not block_indices:
            return scales, 0
        max_idx = max(block_indices)
        if max_idx == 0:
            return scales, 0

        result = dict(scales)
        suppressed = 0
        for block_index, scale in scales.items():
            pos = block_index / max_idx
            if bad and bad[0] <= pos <= bad[1]:
                if block_index in SEMANTIC_ANCHOR_BLOCKS:
                    continue
                result[block_index] = 0.0
                suppressed += 1
            elif good and good[0] <= pos <= good[1]:
                result[block_index] = min(LORA_ZONE_SCALE_CAP, scale * LORA_ZONE_BOOST)
        return result, suppressed

    def _semantic_anchor_quality(self, block_scores):
        total = sum(block_scores.values())
        if total <= 0.0:
            return 1.0
        anchor_energy = sum(block_scores.get(b, 0.0) for b in SEMANTIC_ANCHOR_BLOCKS)
        anchor_share = anchor_energy / total
        expected = len(SEMANTIC_ANCHOR_BLOCKS) / max(len(block_scores), 1)
        ratio = anchor_share / max(expected, 1e-9)
        return max(0.92, min(1.08, 1.0 + (ratio - 1.0) * 0.08))

    def _lora_block_profile(self, entry, loaded, model_weight, patch_cache_key=None):
        template = self._block_profile_template(loaded, patch_cache_key)
        if template is None:
            return None

        type_profile = self._block_type_profile(entry)
        stack_scales, suppressed = self._apply_type_zone_scales(dict(template["base_scales"]), entry)
        quality = self._semantic_anchor_quality(template["block_scores"])
        stack_scales = {
            k: min(LORA_ZONE_SCALE_CAP, v * quality) if v > 0.0 else v
            for k, v in stack_scales.items()
        }
        return {
            "entry": entry,
            "loaded": loaded,
            "model_weight": model_weight,
            "global_count": template["global_count"],
            "block_count": template["block_count"],
            "block_scores": template["block_scores"],
            "base_scales": template["base_scales"],
            "stack_scales": stack_scales,
            "suppressed_count": suppressed,
            "anchor_quality": quality,
            "normalized_scores": template["normalized_scores"],
            "top_blocks": template["top_blocks"],
            "concentration": template["concentration"],
            "priority": type_profile["priority"],
            "yield": type_profile["yield"],
            "overlap_score": 0.0,
        }

    def _pair_overlap_factor(self, entry, other_entry):
        lora_type = entry.get("type", "general")
        other_type = other_entry.get("type", "general")
        if "quality" in {lora_type, other_type}:
            return 0.72
        lora_type = normalize_lora_type(lora_type)
        other_type = normalize_lora_type(other_type)
        if lora_type in {"style", "general"} and other_type in {"action", "character"}:
            return 1.14
        if lora_type in {"action", "character"} and other_type in {"action", "character"}:
            return 1.08
        return 1.0

    def _block_presence_threshold(self, profile):
        block_count = max(1, len(profile["normalized_scores"]))
        return max(0.003, 0.40 / block_count)

    def _stack_block_scales(self, profiles):
        if len(profiles) < 2:
            return

        block_indices = sorted({block_index for profile in profiles for block_index in profile["normalized_scores"]})
        for block_index in block_indices:
            contributors = [
                profile
                for profile in profiles
                if profile["normalized_scores"].get(block_index, 0.0) >= self._block_presence_threshold(profile)
            ]
            if len(contributors) < 2:
                continue

            for profile in contributors:
                if profile["stack_scales"].get(block_index, 1.0) == 0.0:
                    continue
                own_presence = profile["normalized_scores"].get(block_index, 0.0)
                own_weight = max(0.05, abs(profile["model_weight"]))
                own_signal = own_presence * own_weight * profile["priority"]
                other_signals = []
                other_presence = 0.0
                for other in contributors:
                    if other is profile:
                        continue
                    factor = self._pair_overlap_factor(profile["entry"], other["entry"])
                    presence = other["normalized_scores"].get(block_index, 0.0) * factor
                    other_presence += presence
                    other_signals.append(
                        presence * max(0.05, abs(other["model_weight"])) * other["priority"]
                    )

                strongest_other = max(other_signals) if other_signals else 0.0
                if strongest_other <= 0.0:
                    continue

                overlap_ratio = other_presence / max(own_presence + other_presence, 1e-9)
                advantage = (own_signal - strongest_other) / max(own_signal + strongest_other, 1e-9)
                if advantage >= 0.0:
                    multiplier = 1.0 + min(0.18, advantage * 0.14) * min(1.0, overlap_ratio * 1.25)
                elif block_index in SEMANTIC_ANCHOR_BLOCKS:
                    continue
                else:
                    pressure = min(1.0, overlap_ratio * 1.35)
                    damp = min(0.35, (-advantage) * 0.22 * profile["yield"] * pressure)
                    multiplier = 1.0 - damp

                profile["stack_scales"][block_index] = max(
                    0.18,
                    min(1.90, profile["stack_scales"].get(block_index, 1.0) * multiplier),
                )
                profile["overlap_score"] = max(profile["overlap_score"], overlap_ratio)

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

    def _per_block_status(self, profile):
        scales = profile["stack_scales"]
        w = profile["model_weight"]
        active = {k: v for k, v in scales.items() if v > 0.0}
        min_eff = w * min(active.values()) if active else 0.0
        max_eff = w * max(active.values()) if active else 0.0
        suppressed = profile.get("suppressed_count", 0)
        quality = profile.get("anchor_quality", 1.0)
        mode = "smart-per-block" if profile["overlap_score"] > 0.0 else "per-block"
        parts = [
            f"{mode} blocks={len(scales)} non_block={profile['global_count']}",
            f"range={min_eff:.3f}..{max_eff:.3f}",
            f"top={profile['top_blocks']} overlap={profile['overlap_score']:.2f}",
        ]
        if suppressed:
            parts.insert(2, f"zeroed={suppressed}")
        if quality != 1.0:
            parts.insert(-1, f"anchor_q={quality:.2f}")
        return " ".join(parts)

    def _load_lora_per_block(self, model, lora, model_weight):
        lora_cache_key = None
        loaded = self._load_model_lora_patches(model, lora, lora_cache_key)
        profile = self._lora_block_profile({}, loaded, model_weight)
        if not profile:
            return None, "per-block fallback=global"

        new_model, non_block_count, _ = self._apply_model_patches(
            model,
            loaded,
            model_weight,
            block_scales=profile["stack_scales"],
        )
        scales = profile["stack_scales"]
        min_eff = model_weight * min(scales.values())
        max_eff = model_weight * max(scales.values())
        status = (
            f"per-block blocks={len(scales)} non_block={non_block_count} "
            f"range={min_eff:.3f}..{max_eff:.3f}"
        )
        return new_model, status

    def load_loras(self, model, lora_stack, clip=None):
        loras = lora_stack.get("loras", []) if isinstance(lora_stack, dict) else []
        per_block = coerce_bool(lora_stack.get("per_block", False)) if isinstance(lora_stack, dict) else False
        lines = [f"FunPack LoRA Loader | loading {len(loras)} LoRA(s)"]
        lines.append(f"Per-block application: {'enabled' if per_block else 'disabled'}")
        loaded_count = 0
        model_cache_key = self._model_cache_key(model)
        prepared = []
        per_block_profiles = []

        for entry in loras:
            model_weight = float(entry.get("model_weight", 0.0))
            if model_weight == 0:
                lines.append(f"lora_{entry.get('slot', '?')}: {entry.get('name', '?')} skipped at zero weight")
                continue

            lora, lora_cache_key = self._load_lora_file(entry["name"])
            if self._per_block_supported(model, lora_stack, entry):
                patch_cache_key = (lora_cache_key, model_cache_key)
                loaded = self._load_model_lora_patches(model, lora, lora_cache_key, model_cache_key)
                profile = self._lora_block_profile(entry, loaded, model_weight, patch_cache_key)
                if profile is not None:
                    item = {
                        "entry": entry,
                        "lora": lora,
                        "mode": "per_block",
                        "model_weight": model_weight,
                        "profile": profile,
                    }
                    prepared.append(item)
                    per_block_profiles.append(profile)
                else:
                    prepared.append(
                        {
                            "entry": entry,
                            "loaded": loaded,
                            "mode": "patch_global",
                            "model_weight": model_weight,
                            "status": "per-block fallback=global",
                        }
                    )
            elif self._per_block_requested(entry, lora_stack):
                prepared.append(
                    {
                        "entry": entry,
                        "lora": lora,
                        "mode": "global",
                        "model_weight": model_weight,
                        "status": "per-block unsupported -> global",
                    }
                )
            else:
                prepared.append(
                    {
                        "entry": entry,
                        "lora": lora,
                        "mode": "global",
                        "model_weight": model_weight,
                        "status": "global",
                    }
                )

        self._stack_block_scales(per_block_profiles)

        for item in prepared:
            entry = item["entry"]
            model_weight = item["model_weight"]
            if item["mode"] == "per_block":
                profile = item["profile"]
                model, _, _ = self._apply_model_patches(
                    model,
                    profile["loaded"],
                    model_weight,
                    block_scales=profile["stack_scales"],
                )
                apply_status = self._per_block_status(profile)
            elif item["mode"] == "patch_global":
                apply_status = item["status"]
                model, _, _ = self._apply_model_patches(model, item["loaded"], model_weight)
            else:
                apply_status = item["status"]
                model, clip = comfy.sd.load_lora_for_models(model, clip, item["lora"], model_weight, 0.0)

            loaded_count += 1
            lines.append(
                f"lora_{entry.get('slot', '?')}: {entry['name']} "
                f"applied={model_weight:+.3f} source={entry.get('source', 'base')} mode={apply_status}"
            )

        if loaded_count == 0:
            lines.append("No LoRAs were applied.")

        return (model, clip, lora_stack, "\n".join(lines))
