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

try:
    from .widgets import field, list_widget, parse_rows
    from . import funpack_log as _log
except ImportError:  # standalone tests import the modules directly
    from widgets import field, list_widget, parse_rows
    import funpack_log as _log

LORA_TYPES = ["general", "action", "style", "quality", "character"]
LORA_STACK_TYPE = "FUNPACK_LORA_STACK"
# Any numbered block container a diffusion transformer might use. Per-block application
# reads the LoRA's own deltas, which is a property of the adapter and not of the
# architecture — so the container name is the only thing that was ever model-specific, and
# it is a list rather than a gate. Longest names first so `transformer_blocks` is not
# matched as `blocks` (it cannot be, the preceding char is `_`, but the order says so).
TRANSFORMER_BLOCK_PATTERN = re.compile(
    r"(?:^|\.)(transformer_blocks|double_blocks|single_blocks|joint_blocks|blocks|layers)"
    r"\.(\d+)\.")
LTX_IMAGE_MODELS = {"ltxv", "ltxav"}
LORA_RAW_CACHE_SIZE = 12
LORA_PATCH_CACHE_SIZE = 24
LORA_PROFILE_CACHE_SIZE = 24
LORA_BLOCK_TYPE_PROFILES = {
    "character": {"priority": 1.18, "yield": 0.48},
    "action": {"priority": 1.12, "yield": 0.62},
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
    "quality":   {"good": None,         "bad": None},
    "style":     {"good": (0.00, 0.50), "bad": (0.65, 1.00)},
    "general":   {"good": None,         "bad": None},
}
LORA_ZONE_BOOST = 1.12
LORA_ZONE_SCALE_CAP = 1.75
# Primary semantic focal points, as a POSITION along the stack rather than an index.
# PAG's default is block 14 and STG's are 14 and 19, both measured on LTX's 28-block DiT;
# 14/27 and 19/27 are those same blocks expressed as a fraction, so a 28-block model still
# resolves to exactly 14 and 19 while a model of any other depth gets the proportionally
# equivalent blocks instead of two indices that mean nothing there.
# These are never zeroed by type-zone suppression and never damped by stack pressure.
SEMANTIC_ANCHOR_POSITIONS = (14.0 / 27.0, 19.0 / 27.0)


def semantic_anchor_blocks(block_indices):
    """The blocks nearest SEMANTIC_ANCHOR_POSITIONS in this model's own stack."""
    indices = sorted(int(i) for i in block_indices)
    if len(indices) < 2:
        return frozenset(indices)
    max_idx = max(indices)
    if max_idx <= 0:
        return frozenset(indices)
    return frozenset(
        min(indices, key=lambda i: abs(i / max_idx - pos))
        for pos in SEMANTIC_ANCHOR_POSITIONS
    )


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
    return int(match.group(2))


def block_container_name(patch_key):
    """Which numbered container this patch lives in, or None.

    Two containers in one model (Flux's double_blocks/single_blocks) both start at 0, so
    their indices would collide into one bucket and one scale would be applied to two
    unrelated blocks. Per-block declines in that case rather than averaging them.
    """
    target_key = patch_target_key(patch_key)
    if not isinstance(target_key, str):
        return None
    match = TRANSFORMER_BLOCK_PATTERN.search(target_key)
    return match.group(1) if match else None


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


def lora_row_fields():
    """The columns of one LoRA row, shared by the list widget and its parser."""
    loras = ["None"] + folder_paths.get_filename_list("loras")
    return [
        field("lora", "combo", label="LoRA", choices=loras, default="None"),
        field("type", "combo", label="type", choices=LORA_TYPES, default="general",
              tooltip="What the LoRA is for. Drives per-block placement when per_block is on."),
        field("strength", "float", label="strength", default=1.0, min=-10.0, max=10.0, step=0.01),
    ]


def lora_list_input(tooltip, allow_empty=False):
    return list_widget("LoRA", lora_row_fields(), add_label="+ Add LoRA…", tooltip=tooltip,
                       allow_empty=allow_empty, picker="lora")


def stack_from_lora_list(value, per_block=False):
    """A LoRA stack straight from a list widget, with no refinement suggestions applied.

    Same row shape FunPack Apply LoRA Weights emits, so the loader cannot tell the two
    apart — which is what lets a LoRA be used without wiring a second node for it.
    """
    entries = []
    for row in parse_rows(value, lora_row_fields(), key="lora"):
        name = row["lora"]
        lora_type = normalize_lora_type(row["type"])
        entries.append({
            "slot": row["index"],
            "name": name,
            "type": lora_type,
            "id": lora_state_id(name, lora_type),
            "base_model_weight": row["strength"],
            "model_weight": row["strength"],
            "source": "list",
        })
    return {
        "version": 2,
        "refinement_key": "",
        "mode": "ltx2",
        "per_block": coerce_bool(per_block),
        "positive_prompt": "",
        "prompt_key": "",
        "loras": entries,
    }


# Wrapper prefixes trainers put in front of otherwise-standard keys. ComfyUI's own key maps
# already cover diffusion_model./lora_unet_/lycoris_ and, for LTX, the bare
# transformer_blocks.* form -- these are the ones that arrive wrapped and match nothing at
# all until the wrapper comes off.
LORA_KEY_PREFIXES = (
    "transformer.",
    "diffusion_model.",
    "model.diffusion_model.",
    "base_model.model.",
    # PEFT adapters trained against a wrapper that names the DiT: RAVEN's streaming LoRA is
    # `base_model.model.dit.<path>`, so stripping only `base_model.model.` leaves `dit.` in
    # front of every key and the whole file matches nothing.
    "base_model.model.dit.",
    "base_model.model.diffusion_model.",
    "lora_model.",
    "net.",
)


def _count_lora_matches(lora, key_map):
    try:
        return len(comfy.lora.load_lora(lora, key_map, log_missing=False))
    except TypeError:  # ComfyUI without the log_missing argument
        return len(comfy.lora.load_lora(lora, key_map))


def _strip_key_prefix(lora, prefix):
    if not any(k.startswith(prefix) for k in lora):
        return None
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in lora.items()}


def _lora_pair_dims(adapter):
    """(rows, cols) of the delta a plain LoRA entry produces, else None.

    comfy's LoRAAdapter holds `weights = (mat1, mat2, alpha, mid, dora_scale, reshape)` and
    merges with `mm(mat1.flatten(1), mat2.flatten(1)).reshape(weight.shape)`, so the delta is
    flattened the same way here rather than assuming a 2-D pair.

    Returns None — abstaining, so nothing is dropped — for the two entries whose merge this
    does not model: a locon `mid` rebuilds mat2 into a different shape, and a `reshape` pads
    the target weight before merging, so the weight it lands in is not the one in the model.
    Anything that is not a plain LoRA at all (LoKr, LoHa, a plain "diff") is left to comfy.
    """
    w = getattr(adapter, "weights", None)
    if not w or len(w) < 2:
        return None
    if len(w) > 3 and w[3] is not None:      # locon mid
        return None
    if len(w) > 5 and w[5] is not None:      # target weight is padded before the merge
        return None
    try:
        return int(w[0].flatten(start_dim=1).shape[0]), int(w[1].flatten(start_dim=1).shape[-1])
    except (AttributeError, IndexError, TypeError):
        return None


def _mismatched_lora_keys(model, patches):
    """Patch keys whose LoRA pair cannot multiply into the weight they name.

    Key matching says the LoRA is FOR this model; it says nothing about the weights fitting.
    A mismatch is only discovered later, inside comfy's merge, as one generic warning per
    key — 51 of them for the case below, mixed into everything else a load prints. Checked
    here so the count reaches the status line the user actually reads.

    Both dimensions have to agree, not just the element count. comfy merges with
    `mm(mat1.flatten(1), mat2.flatten(1)).reshape(weight.shape)` and adds the result — the
    reshape succeeds on ANY pair with the right number of elements, so a pair that is right
    by count and wrong by shape is merged in silence, scrambled. A correctly trained adapter
    always produces (weight.shape[0], weight.shape[1:].numel()) exactly; anything else fitting
    by count is a transposed or differently fused variant, and merging it corrupts the weight.
    That corruption is invisible at load and surfaces as an all-NaN latent mid-render.
    """
    try:
        sd = model.model.state_dict()
    except Exception:  # noqa: BLE001 — diagnostics must never break a load
        return []
    bad = []
    for key, adapter in patches.items():
        dims = _lora_pair_dims(adapter)
        target = sd.get(key)
        if dims is None or target is None or getattr(target, "dim", None) is None:
            continue
        try:
            if target.dim() < 2:
                continue
            want = (int(target.shape[0]), int(target.shape[1:].numel()))
        except Exception:  # noqa: BLE001
            continue
        if dims != want:
            bad.append((key, dims, want))
    return bad


def _adaln_curve_note(model, bad):
    """The specific, common cause — named, because "51 shape mismatches" is not actionable.

    ComfyUI's MiniMax H3 checkpoints come in two forms. The full one derives the adaLN input
    from a `time_embedder` (time_embed_dim 2688); the pruned "curve" one replaces it with a
    precomputed `adaln_t_table` over a much smaller shared basis, so every
    `adaln_proj.linear` is narrower on its input side. A LoRA trained against the full model
    therefore cannot merge into a curve-form checkpoint — and it cannot be projected either,
    because the basis used to build the table is not in the file.
    """
    if not any("adaln_proj" in k for k, _, _ in bad):
        return None
    dm = getattr(getattr(model, "model", None), "diffusion_model", None)
    if not getattr(dm, "use_adaln_curves", False):
        return None
    n = sum(1 for k, _, _ in bad if "adaln_proj" in k)
    return (f"{n} adaLN adapters do NOT apply: this is a curve-form H3 checkpoint (adaLN "
            f"reads a compact time-curve basis), and the LoRA was trained against the "
            f"full-width form. They cannot be projected onto it — the basis that built the "
            f"table is not in the checkpoint. They are DROPPED rather than attempted — each "
            f"attempt builds the full-size delta before discovering it does not fit. The rest "
            f"of the LoRA still applies. For the full effect use a LoRA converted for "
            f"ComfyUI's H3 checkpoint.")


_DROPPED_FRAGMENT = re.compile(r"\| (\d+)/(\d+) DROPPED")


def _dropped_all_of(line):
    """True when a status line says every weight this LoRA matched was dropped for shape."""
    return any(int(m.group(1)) == int(m.group(2)) for m in _DROPPED_FRAGMENT.finditer(line))


def resolve_lora_patches(model, lora, clip=None, name=None):
    """Match a LoRA against a model, trying known wrapper prefixes. -> (patches, note).

    `name` is the file this LoRA came from. It is carried into the log lines because the
    answer to "is this one bad LoRA or the loader" is only visible across a whole stack:
    one file dropping keys is that file, every file dropping every key is the loader.

    ComfyUI's converters handle the tensor-naming dialects (lora_up/lora_A/lora.up/PEFT
    defaults); what they do not handle is a whole state dict nested under a wrapper the
    model's key map never mentions, which is how a diffusers-trained LoRA ends up applying
    zero weights in silence. Every candidate is scored against the real key map and the one
    that matches the most weights wins, so a format nobody anticipated still lands as long
    as its keys are recognisable underneath.
    """
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    converted = comfy.lora_convert.convert_lora(lora)

    best, best_count, best_note = converted, _count_lora_matches(converted, key_map), "as-is"
    for prefix in LORA_KEY_PREFIXES:
        variant = _strip_key_prefix(converted, prefix)
        if variant is None:
            continue
        count = _count_lora_matches(variant, key_map)
        if count > best_count:
            best, best_count, best_note = variant, count, f"stripped {prefix}"

    who = f"{name}: " if name else ""
    patches = comfy.lora.load_lora(best, key_map)
    if not patches:
        logging.warning(
            f"[FunPack] {who}LoRA matched no weights in this model - wrong model family?")
        return patches, "MATCHED NOTHING"
    matched = len(patches)
    note = f"keys={matched} fmt={best_note}"
    # Matched by NAME is not applied: a mismatched pair is dropped during the merge, and
    # a LoRA that reports success while a third of it never lands is the worst of both.
    bad = _mismatched_lora_keys(model, patches)
    if bad:
        # DROPPED, not merely reported. Left in, comfy attempts every one of them and fails
        # per key — and the failure is not free: it materialises the full lora_A @ lora_B
        # delta first, and only then discovers it cannot be reshaped into the weight. For a
        # curve-form H3 checkpoint that is a 96768x2688 tensor per block, 51 times, allocated
        # and thrown away one after another while dynamic VRAM staging streams the model in.
        # Nothing about the render changes — these adapters could never have applied — so the
        # only thing keeping them buys is 51 ERROR lines and the memory churn behind them.
        for key, _, _ in bad:
            patches.pop(key, None)
        curve = _adaln_curve_note(model, bad)
        if curve:
            _log.note_on_change(f"lora:adaln_curve:{name or '?'}", "FunPack", who + curve)
            note += f" | {len(bad)} adaLN adapters DROPPED (curve-form — see log)"
        else:
            # WITH the numbers. "trained against a different variant" is not actionable on
            # its own; the two shapes say which variant, and whether every layer is off by the
            # same factor (a different width — nothing to do) or only some are (a fusion or
            # naming difference — fixable).
            sample = "; ".join(f"{k}: LoRA {d[0]}x{d[1]} into weight {w[0]}x{w[1]}"
                               for k, d, w in bad[:3])
            _log.note_on_change(
                f"lora:shape:{name or '?'}", "FunPack",
                f"{who}{len(bad)} of {matched} matched weights do not have the shape of the "
                f"weight they name, so they are DROPPED before the merge. {sample}. Kept, they "
                f"would not error — comfy reshapes any delta with the right element count and "
                f"adds it scrambled, which shows up later as an all-NaN latent, not as a load "
                f"failure. {len(bad)} of {matched} means "
                + ("this LoRA was trained against a different variant of this architecture."
                   if len(bad) < matched else
                   "NOTHING from this file applies — if every LoRA in the stack says the same, "
                   "suspect the checkpoint the loader built, not the LoRAs."))
            note += f" | {len(bad)}/{matched} DROPPED (shape mismatch)"
    return patches, note


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
                # The canvas draws this with web/funpack_lora_weights.js; the funpack_list
                # spec is what lets the FunPack Editor draw the same rows instead of raw JSON.
                "lora_list": lora_list_input(
                    "LoRAs whose weights this node looks up for the current prompt."),
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
            if not isinstance(lora_name, str) and isinstance(lora_type, str) and lora_type not in LORA_TYPES:
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
                "lora_list": lora_list_input(
                    "LoRAs to apply, top to bottom. Enough on its own — a stack is only "
                    "needed for prompt-specific trained weights. Empty is fine: the model "
                    "passes straight through.",
                    allow_empty=True),
            },
            "optional": {
                "clip": ("CLIP",),
                "lora_stack": (LORA_STACK_TYPE, {
                    # Advanced: LoRAs are chosen in the list above. The stack is for trained,
                    # prompt-specific strengths, so it must not read as the way in.
                    "advanced": True,
                    "tooltip": "Optional stack from FunPack Apply LoRA Weights, carrying "
                               "prompt-specific trained strengths. Its LoRAs are applied "
                               "first, then this node's own list."}),
                "per_block": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Analyze each LoRA's block deltas and balance competing block "
                               "strengths. LTX models only; a wired stack can switch this on "
                               "by itself."}),
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
        """Whether to TRY per-block for this entry. Not a model whitelist any more.

        Per-block scaling is derived from the LoRA's own deltas — which blocks it actually
        put energy into — so what decides whether it can run is whether this adapter names
        numbered blocks, not which architecture it was trained for. That is discovered from
        the patches themselves in _lora_block_profile, which returns None when there are
        fewer than two blocks to compare; the caller then falls back to global and says so.
        Refusing here by image_model meant a model FunPack had never heard of got the
        fallback without anyone ever looking at its deltas.
        """
        return self._per_block_requested(entry, lora_stack)

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

    def _load_model_lora_patches(self, model, lora, lora_cache_key, model_cache_key=None,
                                 name=None):
        """(patches, format note). The note says how the LoRA's keys had to be read."""
        if lora_cache_key is None:
            return resolve_lora_patches(model, lora, name=name)

        cache_key = (lora_cache_key, model_cache_key or self._model_cache_key(model))
        cached = self._cache_get(self.model_patch_cache, cache_key)
        if cached is not None:
            return cached

        result = resolve_lora_patches(model, lora, name=name)
        self._cache_put(self.model_patch_cache, cache_key, result, LORA_PATCH_CACHE_SIZE)
        return result

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

        containers = {block_container_name(k) for k in loaded}
        containers.discard(None)
        if len(containers) > 1:
            # Both containers number from 0, so their blocks would share buckets and one
            # scale would land on two unrelated blocks. Declining is the honest answer; the
            # caller falls back to global at the plain weight.
            _log.note_on_change(
                "lora:block_containers", "FunPack",
                f"per-block declined: this model numbers its blocks in {len(containers)} "
                f"separate stacks ({', '.join(sorted(containers))}), which start at 0 each, "
                f"so a block index does not identify one block. The LoRA applies globally "
                f"at its plain weight instead.")
            return None

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
        anchors = semantic_anchor_blocks(block_indices)
        for block_index, scale in scales.items():
            pos = block_index / max_idx
            if bad and bad[0] <= pos <= bad[1]:
                if block_index in anchors:
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
        anchors = semantic_anchor_blocks(block_scores.keys())
        anchor_energy = sum(block_scores.get(b, 0.0) for b in anchors)
        anchor_share = anchor_energy / total
        expected = len(anchors) / max(len(block_scores), 1)
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
        anchors = semantic_anchor_blocks(block_indices)
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
                elif block_index in anchors:
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

    def _load_lora_per_block(self, model, lora, model_weight, name=None):
        lora_cache_key = None
        loaded, fmt = self._load_model_lora_patches(model, lora, lora_cache_key, name=name)
        profile = self._lora_block_profile({}, loaded, model_weight)
        if not profile:
            return None, f"per-block fallback=global {fmt}"

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
            f"range={min_eff:.3f}..{max_eff:.3f} {fmt}"
        )
        return new_model, status

    def load_loras(self, model, lora_list="[]", lora_stack=None, clip=None, per_block=False):
        # Both sources are honoured: a wired stack carries trained, prompt-specific weights,
        # the list is what the user typed here. Dropping either one silently would make a
        # filled-in field do nothing.
        stack = lora_stack if isinstance(lora_stack, dict) else {}
        own = stack_from_lora_list(lora_list, per_block)
        loras = list(stack.get("loras", [])) + own["loras"]
        per_block = coerce_bool(stack.get("per_block", False)) or coerce_bool(per_block)
        lora_stack = {**own, **stack, "per_block": per_block, "loras": loras}
        if not loras:
            # An empty loader is a wire, not an error: it hands the model on untouched so it
            # can sit in the pipeline permanently, waiting for the run that wants a LoRA.
            return (model, clip, lora_stack, "FunPack LoRA Loader | No active LoRAs")
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
                loaded, fmt = self._load_model_lora_patches(
                    model, lora, lora_cache_key, model_cache_key, name=entry.get("name"))
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
                            "status": f"per-block fallback=global {fmt}",
                        }
                    )
            elif self._per_block_requested(entry, lora_stack):
                prepared.append(
                    {
                        "entry": entry,
                        "lora": lora,
                        "cache_key": lora_cache_key,
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
                        "cache_key": lora_cache_key,
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
                # Same resolver as the per-block path: comfy's own converters, plus the
                # wrapper-prefix search, so an unusual format is not silently a no-op.
                loaded, fmt = self._load_model_lora_patches(
                    model, item["lora"], item.get("cache_key"), model_cache_key,
                    name=entry.get("name"))
                model, _, _ = self._apply_model_patches(model, loaded, model_weight)
                apply_status = f"{item['status']} {fmt}"

            loaded_count += 1
            lines.append(
                f"lora_{entry.get('slot', '?')}: {entry['name']} "
                f"applied={model_weight:+.3f} source={entry.get('source', 'base')} mode={apply_status}"
            )

        if loaded_count == 0:
            lines.append("No LoRAs were applied.")
        else:
            verdict = self._stack_shape_verdict(lines)
            if verdict:
                _log.note_on_change("lora:stack_shape", "FunPack", verdict)
                lines.append(verdict)

        return (model, clip, lora_stack, "\n".join(lines))

    @staticmethod
    def _stack_shape_verdict(lines):
        """One line about the STACK, because the per-file lines cannot answer the question.

        "3 of 400 weights dropped" is a property of one LoRA. "every weight of every LoRA
        dropped" is a property of the model they were all measured against — the same reading
        the user makes by eye across the status lines, made once here so it is not missed.
        """
        applied = [ln for ln in lines if " mode=" in ln]
        if not applied:
            return None
        total = [ln for ln in applied if "DROPPED (shape mismatch)" in ln]
        if not total:
            return None
        whole = [ln for ln in total if _dropped_all_of(ln)]
        if len(whole) == len(applied) and len(applied) > 1:
            return (f"all {len(applied)} LoRAs in this stack matched this model by name and "
                    f"NONE by shape. That is one property shared by every file, so suspect the "
                    f"checkpoint the loader built (a repack whose layout was mis-inferred) "
                    f"before suspecting the LoRAs.")
        return (f"{len(total)} of {len(applied)} LoRAs dropped weights for shape; the rest "
                f"applied. Per-file numbers are on the lines above.")
