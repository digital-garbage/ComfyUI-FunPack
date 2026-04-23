import json
import os
import re
from hashlib import md5

import comfy.sd
import comfy.utils
import folder_paths


LORA_TYPES = ["general", "concept", "style", "quality", "character"]
LORA_STACK_TYPE = "FUNPACK_LORA_STACK"


class AnyType(str):
    def __ne__(self, _):
        return False


any_type = AnyType("*")


class FlexibleOptionalInputType(dict):
    def __init__(self, input_type, data=None):
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
            },
            "optional": optional,
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

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

    def _iter_slots(self, kwargs):
        indexed_slots = set()
        for key in kwargs:
            match = re.fullmatch(r"lora_(\d+)", key)
            if match:
                indexed_slots.add(int(match.group(1)))

        for index in sorted(indexed_slots):
            lora_name = kwargs.get(f"lora_{index}", "None")
            if isinstance(lora_name, dict):
                if not lora_name.get("on", True):
                    continue
                name = lora_name.get("lora", "None")
                if not name or name == "None":
                    continue
                lora_type = lora_name.get("type", lora_name.get("lora_type", "general"))
                if lora_type not in LORA_TYPES:
                    lora_type = "general"
                yield {
                    "slot": index,
                    "name": name,
                    "type": lora_type,
                    "id": lora_state_id(name, lora_type),
                    "base_model_weight": float(lora_name.get("strength", lora_name.get("base_weight", 1.0))),
                }
                continue

            if not lora_name or lora_name == "None":
                continue

            lora_type = kwargs.get(f"lora_{index}_type", "general")
            if lora_type not in LORA_TYPES:
                lora_type = "general"

            yield {
                "slot": index,
                "name": lora_name,
                "type": lora_type,
                "id": lora_state_id(lora_name, lora_type),
                "base_model_weight": float(kwargs.get(f"lora_{index}_base_weight", 1.0)),
            }

    def _get_suggestion(self, suggestions, entry):
        suggestion = suggestions.get(entry["id"])
        if suggestion:
            return suggestion

        for legacy in suggestions.values():
            if (
                isinstance(legacy, dict)
                and legacy.get("name") == entry["name"]
                and legacy.get("type", "general") == entry["type"]
            ):
                return legacy

        return {}

    def apply_lora_weights(self, positive_prompt, refinement_key, mode, **kwargs):
        mode = (mode or "ltx2").lower()
        prompt_key = prompt_key_for_mode(positive_prompt, mode)
        suggestions, source_message = self._load_suggestions(refinement_key, mode, prompt_key)

        loras = []
        lines = [f"FunPack Apply LoRA Weights | {source_message}"]
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
            "version": 1,
            "refinement_key": refinement_key,
            "mode": mode,
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

    def load_loras(self, model, lora_stack, clip=None):
        loras = lora_stack.get("loras", []) if isinstance(lora_stack, dict) else []
        lines = [f"FunPack LoRA Loader | loading {len(loras)} LoRA(s)"]
        loaded_count = 0

        for entry in loras:
            model_weight = float(entry.get("model_weight", 0.0))
            if model_weight == 0:
                lines.append(f"lora_{entry.get('slot', '?')}: {entry.get('name', '?')} skipped at zero weight")
                continue

            lora = self._load_lora_file(entry["name"])
            model, clip = comfy.sd.load_lora_for_models(model, clip, lora, model_weight, 0.0)
            loaded_count += 1
            lines.append(
                f"lora_{entry.get('slot', '?')}: {entry['name']} "
                f"applied={model_weight:+.3f} source={entry.get('source', 'base')}"
            )

        if loaded_count == 0:
            lines.append("No LoRAs were applied.")

        return (model, clip, lora_stack, "\n".join(lines))
