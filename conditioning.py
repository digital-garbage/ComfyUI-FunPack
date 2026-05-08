import base64
import gc
import glob
import json
import math
import os
import random
import re
from hashlib import md5
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import folder_paths

LORA_REFINER_TYPE_PROFILES = {
    "general": {"step": 0.025, "max_offset": 0.20, "min_offset": -0.35, "bad_max_offset": 0.45, "bad_min_offset": -1.35, "culprit_bias": 0.28},
    "action": {"step": 0.046, "max_offset": 0.35, "min_offset": -0.45, "bad_max_offset": 0.75, "bad_min_offset": -2.10, "culprit_bias": 0.16},
    "concept": {"step": 0.046, "max_offset": 0.35, "min_offset": -0.45, "bad_max_offset": 0.75, "bad_min_offset": -2.10, "culprit_bias": 0.16},
    "style": {"step": 0.032, "max_offset": 0.28, "min_offset": -0.38, "bad_max_offset": 0.58, "bad_min_offset": -1.55, "culprit_bias": 0.20},
    "quality": {"step": 0.022, "max_offset": 0.18, "min_offset": -0.30, "bad_max_offset": 0.38, "bad_min_offset": -1.20, "culprit_bias": 0.18},
    "character": {"step": 0.024, "max_offset": 0.20, "min_offset": -0.32, "bad_max_offset": 0.42, "bad_min_offset": -1.30, "culprit_bias": 0.10},
}

RATING_LABELS = [
    "-Just forget it-",
    "Perfect",
    "Missing details",
    "Missing concept",
    "Missing quality",
    "Missing details + concept",
    "Missing details + quality",
    "Missing concept + quality",
    "Awful",
]

RATING_PROFILES = {
    "-Just forget it-": {
        "key": "forget",
        "level": 0,
        "legacy_score": 0,
        "legacy_range": "ignored",
        "reward": 0.0,
        "quality_signal": 0.0,
        "concept_signal": 0.0,
        "detail_signal": 0.0,
        "missing_axes": [],
        "prompt_emphasis": 0.0,
        "skip_learning": True,
    },
    "Initial discovery": {
        "key": "discover",
        "level": 5,
        "legacy_score": 6,
        "legacy_range": "discovery",
        "reward": 0.0,
        "quality_signal": 0.0,
        "concept_signal": 0.0,
        "detail_signal": 0.0,
        "missing_axes": [],
        "prompt_emphasis": 0.0,
    },
    "Perfect": {
        "key": "like",
        "level": 8,
        "legacy_score": 10,
        "legacy_range": "keep",
        "reward": 1.0,
        "quality_signal": 1.0,
        "concept_signal": 1.0,
        "detail_signal": 1.0,
        "missing_axes": [],
        "prompt_emphasis": 0.45,
    },
    "Missing details": {
        "key": "missing_details",
        "level": 7,
        "legacy_score": 8,
        "legacy_range": "boost details",
        "reward": 0.45,
        "quality_signal": 0.85,
        "concept_signal": 0.70,
        "detail_signal": -1.0,
        "missing_axes": ["details"],
        "prompt_emphasis": 0.65,
    },
    "Missing concept": {
        "key": "missing_concept",
        "level": 6,
        "legacy_score": 6,
        "legacy_range": "boost concept",
        "reward": 0.10,
        "quality_signal": 0.85,
        "concept_signal": -1.0,
        "detail_signal": 0.20,
        "missing_axes": ["concept"],
        "prompt_emphasis": 0.95,
    },
    "Missing quality": {
        "key": "missing_quality",
        "level": 5,
        "legacy_score": 4,
        "legacy_range": "boost quality",
        "reward": -0.25,
        "quality_signal": -1.0,
        "concept_signal": 0.65,
        "detail_signal": 0.65,
        "missing_axes": ["quality"],
        "prompt_emphasis": 0.55,
    },
    "Missing details + concept": {
        "key": "missing_details_concept",
        "level": 4,
        "legacy_score": 5,
        "legacy_range": "boost details+concept",
        "reward": -0.05,
        "quality_signal": 0.75,
        "concept_signal": -1.0,
        "detail_signal": -1.0,
        "missing_axes": ["details", "concept"],
        "prompt_emphasis": 1.05,
    },
    "Missing details + quality": {
        "key": "missing_details_quality",
        "level": 3,
        "legacy_score": 3,
        "legacy_range": "boost details+quality",
        "reward": -0.35,
        "quality_signal": -1.0,
        "concept_signal": 0.50,
        "detail_signal": -1.0,
        "missing_axes": ["details", "quality"],
        "prompt_emphasis": 0.82,
    },
    "Missing concept + quality": {
        "key": "missing_concept_quality",
        "level": 2,
        "legacy_score": 2,
        "legacy_range": "boost concept+quality",
        "reward": -0.55,
        "quality_signal": -1.0,
        "concept_signal": -1.0,
        "detail_signal": 0.15,
        "missing_axes": ["concept", "quality"],
        "prompt_emphasis": 1.02,
    },
    "Awful": {
        "key": "awful",
        "level": 1,
        "legacy_score": 2,
        "legacy_range": "boost all",
        "reward": -0.85,
        "quality_signal": -1.0,
        "concept_signal": -1.0,
        "detail_signal": -1.0,
        "missing_axes": ["details", "concept", "quality"],
        "prompt_emphasis": 1.20,
        "rollback_on_failure": True,
    },
}

RATING_ALIASES = {
    "I like it": "Perfect",
    "I don't like it": "Awful",
    "Missing details + concepts": "Missing details + concept",
    "Missing concept + details": "Missing details + concept",
    "Missing quality + details": "Missing details + quality",
    "Missing quality + concept": "Missing concept + quality",
    "Missing everything": "Awful",
}

CATEGORY_FEEDBACK_MAP = {
    1: "general",
    2: "concept",
    3: "style",
    4: "quality",
    5: "character",
    6: "details",
}


def _clamp(value, low, high):
    return max(low, min(high, value))


def normalize_refiner_rating(value):
    if isinstance(value, str):
        cleaned = value.strip()
        cleaned = RATING_ALIASES.get(cleaned, cleaned)
        if cleaned in RATING_PROFILES:
            return dict(RATING_PROFILES[cleaned], label=cleaned)
        try:
            value = int(float(cleaned))
        except (TypeError, ValueError):
            return dict(RATING_PROFILES["Missing concept"], label="Missing concept")

    try:
        legacy_score = int(value)
    except (TypeError, ValueError):
        legacy_score = RATING_PROFILES["Missing concept"]["legacy_score"]

    legacy_score = int(_clamp(legacy_score, 1, 10))
    if legacy_score >= 9:
        label = "Perfect"
    elif legacy_score >= 7:
        label = "Missing details"
    elif legacy_score >= 5:
        label = "Missing concept"
    elif legacy_score >= 3:
        label = "Missing quality"
    else:
        label = "Awful"

    profile = dict(RATING_PROFILES[label], label=label)
    profile["legacy_score"] = legacy_score
    return profile


PROTECTED_PHRASE_RE = re.compile(r'"([^"]+)"|“([^”]+)”|\\([^\\]+)\\')
def tensor_to_serializable(t: torch.Tensor) -> dict:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(t)}")
    arr = t.detach().cpu().numpy()
    return {
        "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype)
    }

def serializable_to_tensor(d: dict) -> torch.Tensor:
    arr = np.frombuffer(base64.b64decode(d["data"]), dtype=d["dtype"]).reshape(d["shape"]).copy()
    tensor = torch.from_numpy(arr).to(dtype=torch.float32)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def _safe_float(value, fallback=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback

def _safe_int(value, fallback=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback

def _to_image_tensor(image):
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

def render_refinement_loss_graph(refinement_key, scheduler_mode, mode, total_iterations, latest_learning_loss, points, width=960, height=540):
    image = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    bg = (24, 28, 34)
    panel = (30, 36, 44)
    grid = (64, 76, 92)
    axis = (122, 137, 156)
    text = (235, 239, 244)
    subtext = (168, 180, 194)
    line = (74, 201, 255)
    fill = (74, 201, 255, 72)
    point_color = (255, 181, 71)

    image.paste(bg, (0, 0, width, height))

    draw.rounded_rectangle((16, 16, width - 16, height - 16), radius=18, fill=panel, outline=(49, 58, 70), width=1)
    draw.text((32, 28), "FunPack Refinement Loss", fill=text, font=font)
    draw.text((32, 50), f"Embedding: {refinement_key}", fill=text, font=font)
    draw.text((32, 68), f"Scheduler: {scheduler_mode.upper()}   Mode: {mode.upper()}   Iterations: {total_iterations}   Latest loss: {latest_learning_loss:.4f}", fill=subtext, font=font)

    if not points:
        draw.text((32, 112), "No loss history is available yet.", fill=(255, 160, 122), font=font)
        return _to_image_tensor(image.convert("RGB"))

    left = 70
    top = 92
    right = width - 28
    bottom = height - 58
    graph_width = max(1, right - left)
    graph_height = max(1, bottom - top)

    draw.line((left, top, left, bottom), fill=axis, width=1)
    draw.line((left, bottom, right, bottom), fill=axis, width=1)

    y_values = [_safe_float(point.get("learning_loss"), 0.0) for point in points]
    x_values = [_safe_int(point.get("total_iteration"), index + 1) for index, point in enumerate(points)]

    y_min = min(y_values)
    y_max = max(y_values)
    if abs(y_max - y_min) < 1e-9:
        pad = 0.25 if y_max == 0 else abs(y_max) * 0.15
        y_min -= pad
        y_max += pad
    else:
        pad = max(0.02, (y_max - y_min) * 0.12)
        y_min -= pad
        y_max += pad

    x_min = min(x_values)
    x_max = max(x_values)
    if x_min == x_max:
        x_max = x_min + 1

    for i in range(5):
        y = top + (graph_height * i / 4.0)
        draw.line((left, y, right, y), fill=grid, width=1)
        y_label = y_max - ((y_max - y_min) * i / 4.0)
        draw.text((18, y - 6), f"{y_label:.3f}", fill=subtext, font=font)

    for i in range(5):
        x = left + (graph_width * i / 4.0)
        draw.line((x, top, x, bottom), fill=grid, width=1)
        x_label = round(x_min + ((x_max - x_min) * i / 4.0))
        draw.text((x - 10, bottom + 10), str(x_label), fill=subtext, font=font)

    coords = []
    for x_value, y_value in zip(x_values, y_values):
        norm_x = (x_value - x_min) / max(1e-9, (x_max - x_min))
        norm_y = (y_value - y_min) / max(1e-9, (y_max - y_min))
        px = left + norm_x * graph_width
        py = bottom - norm_y * graph_height
        coords.append((px, py))

    if len(coords) == 1:
        px, py = coords[0]
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=point_color)
    else:
        polygon = [(coords[0][0], bottom)] + coords + [(coords[-1][0], bottom)]
        draw.polygon(polygon, fill=fill)
        draw.line(coords, fill=line, width=3)
        px, py = coords[-1]
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=point_color, outline=(255, 243, 214))

    return _to_image_tensor(image.convert("RGB"))


def refinement_state_path(refinement_key, mode, prefix="refine", extension="json"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    refinements_dir = os.path.join(base_dir, "refinements")
    safe_key = md5(f"{(mode or 'ltx2').lower()}::{refinement_key}".encode("utf-8")).hexdigest()
    return os.path.join(refinements_dir, f"{prefix}_{safe_key}.{extension}")


def clone_latent(latent):
    if not isinstance(latent, dict):
        return None

    cloned = {}
    for key, value in latent.items():
        cloned[key] = value.detach().clone() if isinstance(value, torch.Tensor) else value
    return cloned


def latent_samples(latent):
    if not isinstance(latent, dict):
        return None
    samples = latent.get("samples")
    return samples if isinstance(samples, torch.Tensor) else None


def latent_sample_type_name(latent):
    if not isinstance(latent, dict):
        return "missing"
    samples = latent.get("samples")
    if samples is None:
        return "missing samples"
    shape = tuple(samples.shape) if isinstance(samples, torch.Tensor) else "unknown"
    return f"{type(samples).__module__}.{type(samples).__name__}, shape={shape}"


def latent_is_plain_video_tensor(latent):
    samples = latent_samples(latent)
    if samples is None:
        return False
    if samples.dim() == 5:
        return True
    if samples.dim() == 4 and latent.get("type") != "audio":
        return True
    return False


def cpu_tensor_bundle(latent):
    if not isinstance(latent, dict):
        return {}

    bundle = {}
    for key, value in latent.items():
        if isinstance(value, torch.Tensor):
            bundle[key] = value.detach().cpu().clone()
    return bundle


def latent_from_tensor_bundle(bundle):
    if not isinstance(bundle, dict):
        return None

    latent = {}
    for key, value in bundle.items():
        if key == "_meta":
            continue
        latent[key] = value.detach().clone() if isinstance(value, torch.Tensor) else value
    return latent if latent_samples(latent) is not None else None


class FunPackVideoRefiner:
    LATENT_OUTPUT_INDEX = 6
    NO_LATENT_REFERENCE_ERROR = "No available latent to operate. Please connect reference latent to input of Video Refiner."
    WRONG_LATENT_ERROR = (
        "Video Refiner latent input must be a plain video LATENT with tensor samples. "
        "Audio latents and LTX audio/video combined NestedTensor latents are not supported here. "
        "Connect only the video latent to Video Refiner, then feed the refined video latent into "
        "LTXVConcatAVLatent as video_latent."
    )
    SAVED_LATENT_ONLY_STATUS = (
        "Running refinement on saved latent. Changing reference latent shape and size will cause no effect to generation."
    )

    _tokenizers = {}
    _tokenizer_sources = {
        "ltx2": [
            ("DreamFast/gemma-3-12b-it-heretic-v2", {
                "trust_remote_code": True,
                "use_fast": True,
            }),
        ],
        "wan": [
            ("Wan-AI/Wan2.2-T2V-A14B", {
                "subfolder": "google/umt5-xxl",
                "use_fast": True,
            }),
            ("Wan-AI/Wan2.2-I2V-A14B", {
                "subfolder": "google/umt5-xxl",
                "use_fast": True,
            }),
            ("Wan-AI/Wan2.2-Animate-14B", {
                "subfolder": "google/umt5-xxl",
                "use_fast": True,
            }),
            ("Wan-AI/Wan2.1-T2V-1.3B", {
                "subfolder": "google/umt5-xxl",
                "use_fast": True,
            }),
            ("Runware/Wan2.2-TI2V-5B", {
                "subfolder": "google/umt5-xxl",
                "use_fast": True,
            }),
            ("ddwf/Wan2.2-Animate-14B", {
                "subfolder": "google/umt5-xxl",
                "use_fast": True,
            }),
        ],
    }

    @classmethod
    def _get_tokenizer_sources(cls, mode="ltx2"):
        mode = (mode or "ltx2").lower()
        sources = list(cls._tokenizer_sources.get(mode, cls._tokenizer_sources["ltx2"]))

        if mode == "wan":
            local_sources = []
            models_dir = getattr(folder_paths, "models_dir", None)
            if models_dir:
                local_sources.extend(
                    (path, {})
                    for path in glob.glob(os.path.join(models_dir, "Wan", "*", "google", "umt5-xxl"))
                    if os.path.isdir(path)
                )
                text_encoder_tokenizer = os.path.join(models_dir, "text_encoders", "google", "umt5-xxl")
                if os.path.isdir(text_encoder_tokenizer):
                    local_sources.append((text_encoder_tokenizer, {}))

            if local_sources:
                sources = local_sources + sources

        return sources

    @classmethod
    def _get_tokenizer(cls, mode="ltx2"):
        mode = (mode or "ltx2").lower()
        cached = cls._tokenizers.get(mode)
        if cached is not None:
            return cls._tokenizers[mode]

        sources = cls._get_tokenizer_sources(mode)
        for model_id, kwargs in sources:
            try:
                cls._tokenizers[mode] = AutoTokenizer.from_pretrained(model_id, **kwargs)
                return cls._tokenizers[mode]
            except Exception as e:
                print(f"[FunPackVideoRefiner] Tokenizer load failed for mode '{mode}' from '{model_id}': {e}")

        return None

    def _normalize_prompt_for_mode(self, prompt: str, mode: str) -> str:
        prompt = (prompt or "").strip()
        if mode == "wan":
            return re.sub(r"\s+", " ", prompt)
        return prompt

    def _get_conditioning_seq_len(self, conditioning: torch.Tensor) -> int:
        if not isinstance(conditioning, torch.Tensor) or conditioning.dim() <= 1:
            return 0
        return conditioning.shape[1] if conditioning.dim() == 3 else conditioning.shape[0]

    def _get_conditioning_token_mask(self, conditioning: torch.Tensor):
        if not isinstance(conditioning, torch.Tensor) or conditioning.dim() <= 1:
            return None

        if conditioning.dim() == 3:
            token_energy = conditioning.detach().abs().sum(dim=-1)
            mask = token_energy.gt(1e-12).any(dim=0)
        else:
            token_energy = conditioning.detach().abs().sum(dim=-1)
            mask = token_energy.gt(1e-12)

        if not bool(mask.any()):
            seq_len = self._get_conditioning_seq_len(conditioning)
            return torch.ones(seq_len, dtype=torch.bool, device=conditioning.device)

        return mask

    def _get_effective_seq_len(self, token_mask, fallback_seq_len: int) -> int:
        if token_mask is None:
            return fallback_seq_len
        active_positions = torch.nonzero(token_mask, as_tuple=False).flatten()
        if active_positions.numel() == 0:
            return fallback_seq_len
        return min(fallback_seq_len, int(active_positions[-1].item()) + 1)

    def _mask_to_embedding_dims(self, token_mask, reference: torch.Tensor):
        if token_mask is None or not isinstance(reference, torch.Tensor) or reference.dim() <= 1:
            return None

        token_mask = token_mask.to(device=reference.device, dtype=reference.dtype)
        if reference.dim() == 3:
            return token_mask.view(1, -1, 1)
        return token_mask.view(-1, 1)

    def _masked_sequence_mean(self, conditioning: torch.Tensor, token_mask):
        if not isinstance(conditioning, torch.Tensor):
            return conditioning
        if conditioning.dim() <= 1 or token_mask is None:
            return conditioning

        mask = self._mask_to_embedding_dims(token_mask, conditioning)
        if conditioning.dim() == 3:
            denom = mask.sum(dim=1).clamp_min(1.0)
            return (conditioning * mask).sum(dim=1) / denom

        denom = mask.sum(dim=0).clamp_min(1.0)
        return (conditioning * mask).sum(dim=0) / denom

    def _tokenize_ids(self, tokenizer, text: str, add_special_tokens: bool, max_length: Optional[int] = None):
        if not tokenizer or not text:
            return []

        kwargs = {"add_special_tokens": add_special_tokens}
        if max_length is not None and max_length > 0:
            kwargs["truncation"] = True
            kwargs["max_length"] = max_length

        try:
            tokenized = tokenizer(text, **kwargs)
            input_ids = tokenized.get("input_ids", tokenized)
            if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            return input_ids if isinstance(input_ids, list) else list(input_ids)
        except Exception:
            try:
                return tokenizer.encode(text, **kwargs)
            except Exception:
                return []

    def _iter_prompt_segments(self, prompt: str):
        if not prompt:
            return

        last_end = 0
        for match in PROTECTED_PHRASE_RE.finditer(prompt):
            start, end = match.span()
            if start > last_end:
                yield ("text", prompt[last_end:start])
            protected_text = (match.group(1) or match.group(2) or match.group(3) or "").strip()
            if protected_text:
                yield ("protected", protected_text)
            last_end = end

        if last_end < len(prompt):
            yield ("text", prompt[last_end:])

    def _mask_quoted_text(self, prompt: str):
        if not prompt:
            return ""
        return PROTECTED_PHRASE_RE.sub(lambda match: " " * len(match.group(0)), prompt)

    def _fallback_word_groups_from_prompt(self, prompt: str, seq_len: int, token_mask=None, existing_groups=None):
        if not prompt or seq_len <= 0:
            return []

        effective_seq_len = self._get_effective_seq_len(token_mask, seq_len)
        if effective_seq_len <= 0:
            return []

        active_positions = list(range(effective_seq_len))
        if token_mask is not None:
            active_positions = [
                index
                for index, enabled in enumerate(token_mask[:effective_seq_len].detach().cpu().tolist())
                if enabled
            ]
        if not active_positions:
            return []

        existing = {
            str(group[2]).strip().lower()
            for group in (existing_groups or [])
            if isinstance(group, (list, tuple)) and len(group) >= 3
        }
        candidates = []
        seen = set(existing)
        for segment_type, segment_text in self._iter_prompt_segments(prompt):
            if segment_type == "protected":
                pieces = [segment_text.strip()]
            else:
                pieces = re.findall(r"[\w'’.-]+", segment_text, flags=re.UNICODE)
            for piece in pieces:
                clean = piece.strip(" \t\r\n,;:!?()[]{}\"'")
                lower = clean.lower()
                if lower in seen or not self._is_valuable_token(clean):
                    continue
                seen.add(lower)
                candidates.append(clean)

        if not candidates:
            return []

        fallback = []
        step_count = len(candidates)
        max_slots = len(active_positions)
        for index, word in enumerate(candidates[:max_slots]):
            if step_count == 1:
                pos_index = len(active_positions) // 2
            else:
                pos_index = round(index * (len(active_positions) - 1) / max(1, min(step_count, max_slots) - 1))
            start = int(active_positions[max(0, min(len(active_positions) - 1, pos_index))])
            fallback.append((start, min(effective_seq_len, start + 1), word, []))
        return fallback

    def _build_word_groups(self, prompt: str, tokenizer, seq_len: int, token_mask=None):
        if not prompt or seq_len <= 0:
            return []

        if not tokenizer:
            return self._fallback_word_groups_from_prompt(prompt, seq_len, token_mask)

        effective_seq_len = self._get_effective_seq_len(token_mask, seq_len)
        token_mask_list = None
        if token_mask is not None:
            token_mask_list = token_mask[:effective_seq_len].detach().cpu().tolist()

        full_token_ids = self._tokenize_ids(
            tokenizer,
            prompt,
            add_special_tokens=True,
            max_length=effective_seq_len
        )[:effective_seq_len]

        word_groups = []
        grouped_seen = set()
        for segment_type, segment_text in self._iter_prompt_segments(prompt):
            if segment_type != "protected":
                continue

            clean_phrase = segment_text.strip()
            lower = clean_phrase.lower()
            if lower in grouped_seen or len(clean_phrase) < 3 or not any(c.isalpha() for c in clean_phrase):
                continue

            grouped_seen.add(lower)
            phrase_token_list = self._tokenize_ids(
                tokenizer,
                clean_phrase,
                add_special_tokens=False
            )
            if not phrase_token_list:
                continue

            for start in range(max(0, len(full_token_ids) - len(phrase_token_list) + 1)):
                end = start + len(phrase_token_list)
                if token_mask_list is not None and not all(token_mask_list[start:min(effective_seq_len, end)]):
                    continue
                if full_token_ids[start:end] == phrase_token_list and start < effective_seq_len:
                    word_groups.append((start, min(effective_seq_len, end), clean_phrase, phrase_token_list))
                    break

        raw_words = [w.strip() for w in self._mask_quoted_text(prompt).split() if w.strip()]
        for word in raw_words:
            clean_word = word
            lower = clean_word.lower()
            if lower in grouped_seen or len(clean_word) < 3 or not self._is_valuable_token(clean_word):
                continue

            grouped_seen.add(lower)
            word_token_list = self._tokenize_ids(
                tokenizer,
                clean_word,
                add_special_tokens=False
            )
            if not word_token_list:
                continue

            found = False
            for start in range(max(0, len(full_token_ids) - len(word_token_list) + 1)):
                end = start + len(word_token_list)
                if token_mask_list is not None and not all(token_mask_list[start:min(effective_seq_len, end)]):
                    continue
                if full_token_ids[start:end] == word_token_list and start < effective_seq_len:
                    word_groups.append((start, min(effective_seq_len, end), clean_word, word_token_list))
                    found = True
                    break

            if not found:
                continue

        if len(word_groups) < 2:
            word_groups.extend(
                self._fallback_word_groups_from_prompt(
                    prompt,
                    seq_len,
                    token_mask,
                    existing_groups=word_groups,
                )
            )

        return sorted(
            [group for group in word_groups if group[1] > group[0]],
            key=lambda group: (group[0], group[1], group[2].lower()),
        )

    def _void_empty_status(self, enabled=False):
        return f"Void: {'on' if enabled else 'off'} | idle | tokens: none"

    def _void_token_embedding(self, conditioning: torch.Tensor, start: int, end: int):
        if not isinstance(conditioning, torch.Tensor) or conditioning.dim() <= 1 or end <= start:
            return None
        if conditioning.dim() == 3:
            return conditioning[:, start:end, :].detach().mean(dim=(0, 1))
        return conditioning[start:end, :].detach().mean(dim=0)

    def _void_bank_score(self, item):
        likes = float(item.get("liked_count", 0))
        neutral = float(item.get("neutral_count", 0))
        awful = float(item.get("awful_count", 0))
        wanted = float(item.get("wanted_count", 0))
        missing = float(item.get("missing_count", 0))
        concept_missing = float(item.get("missing_concept_count", 0))
        detail_missing = float(item.get("missing_detail_count", 0))
        quality_missing = float(item.get("missing_quality_count", 0))
        missing_pressure = missing * 0.22 + concept_missing * 0.14 + detail_missing * 0.10 + quality_missing * 0.08
        return float(item.get("score", 0.0)) + likes * 0.40 + neutral * 0.08 + wanted * 0.30 + missing_pressure - awful * 0.95

    def _missing_axes_for_rating_key(self, rating_key):
        for profile in RATING_PROFILES.values():
            if profile.get("key") == rating_key:
                return set(profile.get("missing_axes", []))
        return set()

    def _axis_counter_name(self, axis):
        if axis == "concept":
            return "missing_concept_count"
        if axis == "details":
            return "missing_detail_count"
        if axis == "quality":
            return "missing_quality_count"
        return None

    def _axis_memory_boost(self, missing_axes):
        axes = set(missing_axes or [])
        boost = 0.0
        if "concept" in axes:
            boost += 0.42
        if "details" in axes:
            boost += 0.30
        if "quality" in axes:
            boost += 0.24
        if len(axes) >= 2:
            boost += 0.16
        return boost

    def _apply_missing_memory_pressure(self, item, missing_axes, iter_num):
        axes = set(missing_axes or [])
        if not isinstance(item, dict) or not axes:
            return item
        item["wanted_count"] = max(0, int(item.get("wanted_count", 0)) + 1)
        item["missing_count"] = max(0, int(item.get("missing_count", 0)) + 1)
        for axis in axes:
            counter = self._axis_counter_name(axis)
            if counter:
                item[counter] = max(0, int(item.get(counter, 0)) + 1)
        item["last_missing_iter"] = int(iter_num)
        return item

    def _ensure_void_token_bank(self, global_adaptive):
        bank = global_adaptive.get("void_token_bank")
        if not isinstance(bank, dict):
            bank = {}
        if bank and global_adaptive.get("void_token_bank_validated") is True:
            return bank
        cleaned = {}
        for token, item in bank.items():
            if not isinstance(token, str) or not isinstance(item, dict):
                continue
            vector = item.get("embedding")
            if not isinstance(vector, dict):
                continue
            try:
                shape = vector.get("shape", [])
                if len(shape) != 1 or int(shape[0]) <= 0:
                    continue
                item["score"] = round(max(-2.0, min(8.0, float(item.get("score", 0.0)))), 4)
                item["liked_count"] = max(0, int(item.get("liked_count", 0)))
                item["neutral_count"] = max(0, int(item.get("neutral_count", 0)))
                item["awful_count"] = max(0, int(item.get("awful_count", 0)))
                item["wanted_count"] = max(0, int(item.get("wanted_count", 0)))
                item["missing_count"] = max(0, int(item.get("missing_count", 0)))
                item["missing_concept_count"] = max(0, int(item.get("missing_concept_count", 0)))
                item["missing_detail_count"] = max(0, int(item.get("missing_detail_count", 0)))
                item["missing_quality_count"] = max(0, int(item.get("missing_quality_count", 0)))
                item["last_seen_iter"] = max(0, int(item.get("last_seen_iter", 0)))
                item["last_missing_iter"] = max(0, int(item.get("last_missing_iter", 0)))
                item["sample_count"] = max(0, int(item.get("sample_count", 0)))
                cleaned[token] = item
            except (TypeError, ValueError):
                continue
        global_adaptive["void_token_bank"] = cleaned
        global_adaptive["void_token_bank_validated"] = True
        return cleaned

    def _void_pair_key(self, left_token, right_token):
        return f"{str(left_token).strip().lower()}\t{str(right_token).strip().lower()}"

    def _void_pair_score(self, item):
        likes = float(item.get("liked_count", 0))
        neutral = float(item.get("neutral_count", 0))
        awful = float(item.get("awful_count", 0))
        wanted = float(item.get("wanted_count", 0))
        missing = float(item.get("missing_count", 0))
        return float(item.get("score", 0.0)) + likes * 0.35 + neutral * 0.05 + wanted * 0.22 + missing * 0.16 - awful * 0.85

    def _ensure_void_pair_bank(self, global_adaptive):
        bank = global_adaptive.get("void_token_pairs")
        if not isinstance(bank, dict):
            bank = {}
        if bank and global_adaptive.get("void_token_pairs_validated") is True:
            return bank
        cleaned = {}
        for key, item in bank.items():
            if not isinstance(key, str) or "\t" not in key or not isinstance(item, dict):
                continue
            try:
                left, right = key.split("\t", 1)
                if not self._is_valuable_token(left) or not self._is_valuable_token(right):
                    continue
                item["score"] = round(max(-3.0, min(6.0, float(item.get("score", 0.0)))), 4)
                item["liked_count"] = max(0, int(item.get("liked_count", 0)))
                item["neutral_count"] = max(0, int(item.get("neutral_count", 0)))
                item["awful_count"] = max(0, int(item.get("awful_count", 0)))
                item["wanted_count"] = max(0, int(item.get("wanted_count", 0)))
                item["missing_count"] = max(0, int(item.get("missing_count", 0)))
                item["missing_concept_count"] = max(0, int(item.get("missing_concept_count", 0)))
                item["missing_detail_count"] = max(0, int(item.get("missing_detail_count", 0)))
                item["missing_quality_count"] = max(0, int(item.get("missing_quality_count", 0)))
                item["last_seen_iter"] = max(0, int(item.get("last_seen_iter", 0)))
                item["last_missing_iter"] = max(0, int(item.get("last_missing_iter", 0)))
                cleaned[key] = item
            except (TypeError, ValueError):
                continue
        global_adaptive["void_token_pairs"] = cleaned
        global_adaptive["void_token_pairs_validated"] = True
        return cleaned

    def _lucky_context_score(self, item):
        likes = float(item.get("liked_count", 0))
        neutral = float(item.get("neutral_count", 0))
        awful = float(item.get("awful_count", 0))
        wanted = float(item.get("wanted_count", 0))
        missing = float(item.get("missing_count", 0))
        concept_missing = float(item.get("missing_concept_count", 0))
        detail_missing = float(item.get("missing_detail_count", 0))
        return float(item.get("score", 0.0)) + likes * 0.38 + neutral * 0.06 + wanted * 0.24 + missing * 0.18 + concept_missing * 0.10 + detail_missing * 0.08 - awful * 0.90

    def _ensure_lucky_context_memory(self, global_adaptive):
        memory = global_adaptive.get("lucky_context_memory")
        if not isinstance(memory, dict):
            memory = {}
        if memory and global_adaptive.get("lucky_context_memory_validated") is True:
            return memory

        cleaned = {}
        for anchor, neighbors in memory.items():
            anchor = str(anchor).strip().lower()
            if not self._is_valuable_token(anchor) or not isinstance(neighbors, dict):
                continue

            clean_neighbors = {}
            for neighbor, item in neighbors.items():
                neighbor = str(neighbor).strip().lower()
                if neighbor == anchor or not self._is_valuable_token(neighbor) or not isinstance(item, dict):
                    continue
                try:
                    item["score"] = round(max(-4.0, min(8.0, float(item.get("score", 0.0)))), 4)
                    item["liked_count"] = max(0, int(item.get("liked_count", 0)))
                    item["neutral_count"] = max(0, int(item.get("neutral_count", 0)))
                    item["awful_count"] = max(0, int(item.get("awful_count", 0)))
                    item["wanted_count"] = max(0, int(item.get("wanted_count", 0)))
                    item["missing_count"] = max(0, int(item.get("missing_count", 0)))
                    item["missing_concept_count"] = max(0, int(item.get("missing_concept_count", 0)))
                    item["missing_detail_count"] = max(0, int(item.get("missing_detail_count", 0)))
                    item["missing_quality_count"] = max(0, int(item.get("missing_quality_count", 0)))
                    item["last_seen_iter"] = max(0, int(item.get("last_seen_iter", 0)))
                    item["last_missing_iter"] = max(0, int(item.get("last_missing_iter", 0)))
                    item["co_count"] = max(0, int(item.get("co_count", 0)))
                    clean_neighbors[neighbor] = item
                except (TypeError, ValueError):
                    continue

            if clean_neighbors:
                cleaned[anchor] = clean_neighbors

        global_adaptive["lucky_context_memory"] = cleaned
        global_adaptive["lucky_context_memory_validated"] = True
        return cleaned

    def _phrase_position_score(self, item):
        likes = float(item.get("liked_count", 0))
        neutral = float(item.get("neutral_count", 0))
        awful = float(item.get("awful_count", 0))
        wanted = float(item.get("wanted_count", 0))
        missing = float(item.get("missing_count", 0))
        return float(item.get("score", 0.0)) + likes * 0.45 + neutral * 0.06 + wanted * 0.26 + missing * 0.18 - awful * 0.85

    def _ensure_lucky_phrase_placements(self, global_adaptive):
        memory = global_adaptive.get("lucky_phrase_placements")
        if not isinstance(memory, dict):
            memory = {}
        if memory and global_adaptive.get("lucky_phrase_placements_validated") is True:
            return memory

        cleaned = {}
        for phrase, item in memory.items():
            phrase_key = str(phrase).strip().lower()
            if len(phrase_key) < 3 or not isinstance(item, dict):
                continue
            positions = item.get("positions")
            if not isinstance(positions, dict):
                continue
            clean_positions = {}
            for pos, pos_item in positions.items():
                if not isinstance(pos_item, dict):
                    continue
                try:
                    pos_key = str(max(0, min(31, int(pos))))
                    pos_item["score"] = round(max(-4.0, min(8.0, float(pos_item.get("score", 0.0)))), 4)
                    pos_item["liked_count"] = max(0, int(pos_item.get("liked_count", 0)))
                    pos_item["neutral_count"] = max(0, int(pos_item.get("neutral_count", 0)))
                    pos_item["awful_count"] = max(0, int(pos_item.get("awful_count", 0)))
                    pos_item["wanted_count"] = max(0, int(pos_item.get("wanted_count", 0)))
                    pos_item["missing_count"] = max(0, int(pos_item.get("missing_count", 0)))
                    pos_item["missing_concept_count"] = max(0, int(pos_item.get("missing_concept_count", 0)))
                    pos_item["missing_detail_count"] = max(0, int(pos_item.get("missing_detail_count", 0)))
                    pos_item["missing_quality_count"] = max(0, int(pos_item.get("missing_quality_count", 0)))
                    pos_item["count"] = max(0, int(pos_item.get("count", 0)))
                    pos_item["last_seen_iter"] = max(0, int(pos_item.get("last_seen_iter", 0)))
                    pos_item["last_missing_iter"] = max(0, int(pos_item.get("last_missing_iter", 0)))
                    clean_positions[pos_key] = pos_item
                except (TypeError, ValueError):
                    continue
            if not clean_positions:
                continue
            cleaned[phrase_key] = {
                "text": str(item.get("text", phrase_key)).strip().lower() or phrase_key,
                "positions": clean_positions,
            }

        global_adaptive["lucky_phrase_placements"] = cleaned
        global_adaptive["lucky_phrase_placements_validated"] = True
        return cleaned

    def _rating_memory_deltas(self, rating_key):
        if rating_key == "like":
            return 0.70, 1, 0, 0
        if rating_key == "missing_details":
            return 0.56, 0, 1, 0
        if rating_key == "missing_concept":
            return 0.72, 0, 1, 0
        if rating_key == "missing_quality":
            return 0.48, 0, 1, 0
        if rating_key == "missing_details_concept":
            return 0.84, 0, 1, 0
        if rating_key == "missing_details_quality":
            return 0.68, 0, 1, 0
        if rating_key == "missing_concept_quality":
            return 0.80, 0, 1, 0
        if rating_key == "awful":
            return -1.10, 0, 0, 1
        if rating_key == "discover":
            return 0.0, 0, 1, 0
        return 0.12, 0, 1, 0

    def _update_lucky_context_memory(self, global_adaptive, tokens, rating_key, iter_num):
        memory = self._ensure_lucky_context_memory(global_adaptive)
        ordered_tokens = []
        seen = set()
        for token in tokens or []:
            token = str(token).strip().lower()
            if token in seen or not self._is_valuable_token(token):
                continue
            seen.add(token)
            ordered_tokens.append(token)

        if len(ordered_tokens) < 2:
            return memory

        score_delta, liked_delta, neutral_delta, awful_delta = self._rating_memory_deltas(rating_key)
        missing_axes = self._missing_axes_for_rating_key(rating_key)
        if missing_axes:
            score_delta += self._axis_memory_boost(missing_axes) * 0.55
        # Keep context learning local and ordered. A full all-to-all update makes
        # long Lucky prompts create thousands of JSON entries per click.
        window = 6
        for index, anchor in enumerate(ordered_tokens):
            neighbors = memory.setdefault(anchor, {})
            start = max(0, index - window)
            end = min(len(ordered_tokens), index + window + 1)
            for neighbor_index in range(start, end):
                if neighbor_index == index:
                    continue
                neighbor = ordered_tokens[neighbor_index]
                distance = abs(neighbor_index - index)
                if distance <= 0:
                    continue
                local_weight = 1.0 / float(distance)
                item = neighbors.get(neighbor, {})
                item["score"] = round(max(-4.0, min(8.0, float(item.get("score", 0.0)) + score_delta * local_weight)), 4)
                item["liked_count"] = max(0, int(item.get("liked_count", 0)) + liked_delta)
                item["neutral_count"] = max(0, int(item.get("neutral_count", 0)) + neutral_delta)
                item["awful_count"] = max(0, int(item.get("awful_count", 0)) + awful_delta)
                if missing_axes:
                    item = self._apply_missing_memory_pressure(item, missing_axes, iter_num)
                item["co_count"] = max(0, int(item.get("co_count", 0)) + 1)
                item["last_seen_iter"] = int(iter_num)
                neighbors[neighbor] = item

        return memory

    def _update_lucky_phrase_placements(self, global_adaptive, prompt_text, rating_key, iter_num):
        phrases = self._ordered_prompt_phrases(prompt_text)
        if not phrases:
            return self._ensure_lucky_phrase_placements(global_adaptive)

        memory = self._ensure_lucky_phrase_placements(global_adaptive)
        score_delta, liked_delta, neutral_delta, awful_delta = self._rating_memory_deltas(rating_key)
        missing_axes = self._missing_axes_for_rating_key(rating_key)
        if missing_axes:
            score_delta += self._axis_memory_boost(missing_axes) * 0.70
        for index, phrase in enumerate(phrases[:32]):
            text = str(phrase.get("text", "")).strip().lower()
            if len(text) < 3:
                continue
            entry = memory.setdefault(text, {"text": text, "positions": {}})
            positions = entry.setdefault("positions", {})
            pos_key = str(index)
            item = positions.get(pos_key, {})
            item["score"] = round(max(-4.0, min(8.0, float(item.get("score", 0.0)) + score_delta)), 4)
            item["liked_count"] = max(0, int(item.get("liked_count", 0)) + liked_delta)
            item["neutral_count"] = max(0, int(item.get("neutral_count", 0)) + neutral_delta)
            item["awful_count"] = max(0, int(item.get("awful_count", 0)) + awful_delta)
            if missing_axes:
                item = self._apply_missing_memory_pressure(item, missing_axes, iter_num)
            item["count"] = max(0, int(item.get("count", 0)) + 1)
            item["last_seen_iter"] = int(iter_num)
            positions[pos_key] = item
            entry["text"] = text
            memory[text] = entry

        global_adaptive["lucky_phrase_placements"] = memory
        return memory

    def _update_void_token_pairs(self, global_adaptive, word_groups, rating_key, iter_num):
        pair_bank = self._ensure_void_pair_bank(global_adaptive)
        ordered_tokens = []
        seen_positions = set()
        for start, _, full_word, _ in sorted(word_groups or [], key=lambda group: (group[0], group[1])):
            token = str(full_word).strip().lower()
            if not self._is_valuable_token(token):
                continue
            marker = (int(start), token)
            if marker in seen_positions:
                continue
            seen_positions.add(marker)
            ordered_tokens.append(token)

        if len(ordered_tokens) < 2:
            return pair_bank

        base_delta, liked_delta, neutral_delta, awful_delta = self._rating_memory_deltas(rating_key)
        missing_axes = self._missing_axes_for_rating_key(rating_key)
        score_delta = base_delta * 0.84
        if missing_axes:
            score_delta += self._axis_memory_boost(missing_axes) * 0.48

        for left, right in zip(ordered_tokens, ordered_tokens[1:]):
            if left == right:
                continue
            key = self._void_pair_key(left, right)
            item = pair_bank.get(key, {})
            item["left"] = left
            item["right"] = right
            item["score"] = round(max(-3.0, min(6.0, float(item.get("score", 0.0)) + score_delta)), 4)
            item["liked_count"] = max(0, int(item.get("liked_count", 0)) + liked_delta)
            item["neutral_count"] = max(0, int(item.get("neutral_count", 0)) + neutral_delta)
            item["awful_count"] = max(0, int(item.get("awful_count", 0)) + awful_delta)
            if missing_axes:
                item = self._apply_missing_memory_pressure(item, missing_axes, iter_num)
            item["last_seen_iter"] = int(iter_num)
            pair_bank[key] = item

        self._update_lucky_context_memory(global_adaptive, ordered_tokens, rating_key, iter_num)
        global_adaptive["void_token_pairs"] = pair_bank
        return global_adaptive["void_token_pairs"]

    def _void_pair_is_poor(self, global_adaptive, left_token, right_token):
        if not left_token or not right_token:
            return False
        pair_bank = global_adaptive.get("void_token_pairs")
        if not isinstance(pair_bank, dict):
            return False
        item = pair_bank.get(self._void_pair_key(left_token, right_token))
        if not isinstance(item, dict):
            return False
        awful_count = int(item.get("awful_count", 0))
        positive_count = int(item.get("liked_count", 0)) + int(item.get("neutral_count", 0))
        return awful_count > 0 and self._void_pair_score(item) < -0.05 and awful_count >= max(1, positive_count)

    def _update_void_token_bank(self, global_adaptive, word_groups, cur_positive,
                                rating_profile, iter_num, token_mask=None):
        bank = self._ensure_void_token_bank(global_adaptive)
        rating_key = rating_profile.get("key", "")
        if not word_groups or not isinstance(cur_positive, torch.Tensor) or rating_profile.get("skip_learning"):
            return bank

        score_delta, liked_delta, neutral_delta, awful_delta = self._rating_memory_deltas(rating_key)
        missing_axes = set(rating_profile.get("missing_axes", []))
        if missing_axes:
            score_delta += self._axis_memory_boost(missing_axes)

        token_mask_list = None
        if token_mask is not None:
            token_mask_list = token_mask.detach().cpu().tolist()

        for start, end, full_word, _ in word_groups:
            if not self._is_valuable_token(full_word):
                continue
            if token_mask_list is not None and not all(token_mask_list[start:min(len(token_mask_list), end)]):
                continue
            embedding = self._void_token_embedding(cur_positive, start, end)
            if embedding is None:
                continue

            token = full_word.strip().lower()
            item = bank.get(token, {})
            count = max(0, int(item.get("sample_count", 0)))
            if "embedding" in item and count > 0 and rating_key != "awful":
                try:
                    previous = serializable_to_tensor(item["embedding"]).to(embedding.device)
                    if list(previous.shape) == list(embedding.shape):
                        if rating_key == "like":
                            mix = 0.62
                        elif rating_key == "missing_details":
                            mix = 0.48
                        else:
                            mix = min(0.30, 1.0 / float(count + 1))
                        embedding = previous.lerp(embedding, mix)
                except Exception:
                    pass

            item["embedding"] = tensor_to_serializable(embedding.float().cpu())
            item["score"] = round(max(-2.0, min(8.0, float(item.get("score", 0.0)) + score_delta)), 4)
            item["liked_count"] = max(0, int(item.get("liked_count", 0)) + liked_delta)
            item["neutral_count"] = max(0, int(item.get("neutral_count", 0)) + neutral_delta)
            item["awful_count"] = max(0, int(item.get("awful_count", 0)) + awful_delta)
            if missing_axes:
                item = self._apply_missing_memory_pressure(item, missing_axes, iter_num)
            item["sample_count"] = count + (0 if rating_key == "awful" else 1)
            item["last_seen_iter"] = int(iter_num)
            bank[token] = item

        self._update_void_token_pairs(global_adaptive, word_groups, rating_key, iter_num)

        global_adaptive["void_token_bank"] = bank
        return global_adaptive["void_token_bank"]

    def _seed_void_token_bank(self, global_adaptive, word_groups, cur_positive, iter_num, token_mask=None):
        return self._update_void_token_bank(
            global_adaptive,
            word_groups,
            cur_positive,
            {"key": "discover", "skip_learning": False},
            iter_num,
            token_mask=token_mask,
        )

    def _lucky_groups_from_history(self, history_entry, seq_len):
        if not isinstance(history_entry, dict) or seq_len <= 0:
            return []
        lucky = history_entry.get("lucky") if isinstance(history_entry.get("lucky"), dict) else {}
        injections = lucky.get("injections", [])
        groups = []
        seen = set()
        for item in injections:
            if not isinstance(item, dict):
                continue
            token = str(item.get("token", "")).strip().lower()
            if not self._is_valuable_token(token):
                continue
            try:
                start = int(item.get("start", 0))
                end = int(item.get("end", start + 1))
            except (TypeError, ValueError):
                continue
            start = max(0, min(seq_len - 1, start))
            end = max(start + 1, min(seq_len, end))
            marker = (start, token)
            if marker in seen:
                continue
            seen.add(marker)
            groups.append((start, end, token, []))
        return groups

    def _history_modified_conditioning(self, history_entry, reference, device):
        if not isinstance(history_entry, dict):
            return None
        mod_data = history_entry.get("modified_embeds")
        if mod_data is None:
            return None
        try:
            candidate = serializable_to_tensor(mod_data).to(device)
        except Exception:
            return None
        if not isinstance(reference, torch.Tensor) or list(candidate.shape) != list(reference.shape):
            return None
        return candidate

    def _history_modified_conditioning_any_shape(self, history_entry, device, dtype=None):
        if not isinstance(history_entry, dict):
            return None
        mod_data = history_entry.get("modified_embeds")
        if mod_data is None:
            return None
        try:
            candidate = serializable_to_tensor(mod_data).to(device)
            if dtype is not None:
                candidate = candidate.to(dtype=dtype)
        except Exception:
            return None
        if not isinstance(candidate, torch.Tensor) or candidate.dim() <= 1:
            return None
        return candidate

    def _conditioning_canvas_compatible(self, candidate, template):
        if (
            not isinstance(candidate, torch.Tensor) or
            not isinstance(template, torch.Tensor) or
            candidate.dim() <= 1 or
            template.dim() <= 1 or
            candidate.dim() != template.dim() or
            int(candidate.shape[-1]) != int(template.shape[-1])
        ):
            return False
        if candidate.dim() == 3 and int(candidate.shape[0]) != int(template.shape[0]):
            return False
        return self._get_conditioning_seq_len(candidate) > 0

    def _resize_conditioning_sequence_like(self, source, target):
        if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
            return None
        if source.dim() != target.dim() or source.dim() <= 1:
            return None
        if int(source.shape[-1]) != int(target.shape[-1]):
            return None
        if source.dim() == 3 and int(source.shape[0]) != int(target.shape[0]):
            return None
        if list(source.shape) == list(target.shape):
            return source.to(device=target.device, dtype=target.dtype)

        target_seq = self._get_conditioning_seq_len(target)
        if target_seq <= 0:
            return None

        try:
            working = source.to(device=target.device, dtype=target.dtype)
            if working.dim() == 3:
                resized = F.interpolate(
                    working.transpose(1, 2),
                    size=target_seq,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            else:
                resized = F.interpolate(
                    working.transpose(0, 1).unsqueeze(0),
                    size=target_seq,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).transpose(0, 1)
        except Exception:
            return None

        if list(resized.shape) != list(target.shape):
            return None
        return resized

    def _serialized_tensor_shape(self, payload):
        if not isinstance(payload, dict):
            return None
        shape = payload.get("shape")
        if not isinstance(shape, list) or not shape:
            return None
        try:
            return tuple(int(dim) for dim in shape)
        except (TypeError, ValueError):
            return None

    def _serialized_conditioning_compatible(self, payload, template):
        shape = self._serialized_tensor_shape(payload)
        if (
            shape is None or
            not isinstance(template, torch.Tensor) or
            template.dim() <= 1 or
            len(shape) != int(template.dim()) or
            int(shape[-1]) != int(template.shape[-1])
        ):
            return False
        if len(shape) == 3 and int(shape[0]) != int(template.shape[0]):
            return False
        return (shape[1] if len(shape) == 3 else shape[0]) > 0

    def _serialized_conditioning_seq_len(self, payload):
        shape = self._serialized_tensor_shape(payload)
        if shape is None or len(shape) <= 1:
            return 0
        return int(shape[1] if len(shape) == 3 else shape[0])

    def _select_lucky_memory_canvas(self, prompt_histories, template, device, dtype):
        if not isinstance(prompt_histories, dict) or not isinstance(template, torch.Tensor):
            return None, ""

        candidates = []
        current_seq = self._get_conditioning_seq_len(template)

        def consider(payload, label, quality=0.0, iteration=0):
            if payload is None:
                return
            if not self._serialized_conditioning_compatible(payload, template):
                return
            seq_len = self._serialized_conditioning_seq_len(payload)
            candidates.append({
                "payload": payload,
                "label": label,
                "seq_len": int(seq_len),
                "quality": float(quality),
                "iteration": int(iteration or 0),
            })

        for prompt_key, prompt_state in prompt_histories.items():
            if not isinstance(prompt_state, dict):
                continue
            short_key = str(prompt_key)[:48] or "prompt"
            liked_count = int(prompt_state.get("liked_reference_count", 0) or 0)
            if liked_count > 0:
                consider(
                    prompt_state.get("liked_reference_embeds"),
                    f"liked '{short_key}'",
                    quality=2.0 + min(3.0, liked_count * 0.25),
                    iteration=prompt_state.get("liked_reference_last_iteration", 0),
                )
            consider(prompt_state.get("reference_embeds"), f"reference '{short_key}'", quality=0.5)
            consider(prompt_state.get("source_conditioning_embeds"), f"source '{short_key}'", quality=0.25)

            for entry in prompt_state.get("history", []) or []:
                if not isinstance(entry, dict):
                    continue
                profile = normalize_refiner_rating(entry.get("rating_label", entry.get("rating", 0)))
                key = profile.get("key", "")
                if key in {"forget", "awful"}:
                    continue
                quality = float(profile.get("reward", 0.0))
                if key == "like":
                    quality += 1.5
                elif key.startswith("missing"):
                    quality += 0.45
                consider(
                    entry.get("modified_embeds"),
                    f"history {entry.get('iteration', 0)} '{short_key}'",
                    quality=quality,
                    iteration=entry.get("iteration", 0),
                )

        if not candidates:
            return None, ""

        best = max(
            candidates,
            key=lambda item: (
                item["seq_len"] >= current_seq,
                item["seq_len"],
                item["quality"],
                item["iteration"],
            )
        )
        try:
            canvas = serializable_to_tensor(best["payload"]).to(device=device, dtype=dtype)
        except Exception:
            return None, ""
        if not self._conditioning_canvas_compatible(canvas, template):
            return None, ""
        return canvas.clone(), f"{best['label']} ({best['seq_len']} positions)"

    def _eligible_void_bank_items(self, global_adaptive, embedding_dim):
        bank = self._ensure_void_token_bank(global_adaptive)
        eligible = []
        for token, item in bank.items():
            if int(item.get("awful_count", 0)) >= 2 and int(item.get("liked_count", 0)) <= 0:
                continue
            score = self._void_bank_score(item)
            if score <= 0.35:
                continue
            try:
                vector = serializable_to_tensor(item["embedding"]).float()
            except Exception:
                continue
            if vector.dim() != 1 or int(vector.shape[0]) != int(embedding_dim):
                continue
            eligible.append((token, item, score))
        return eligible

    def _eligible_lucky_bank_items(self, global_adaptive, embedding_dim):
        bank = self._ensure_void_token_bank(global_adaptive)
        candidates = []
        for token, item in bank.items():
            score = self._void_bank_score(item)
            embedding_shape = self._serialized_tensor_shape(item.get("embedding"))
            if embedding_shape is None or len(embedding_shape) != 1 or int(embedding_shape[0]) != int(embedding_dim):
                continue
            candidates.append((token, item, score))

        return candidates

    def _lucky_prompt_anchor_tokens(self, word_groups, token_vectors, global_adaptive):
        context_memory = self._ensure_lucky_context_memory(global_adaptive)
        anchors = []
        seen = set()
        for _, _, full_word, _ in sorted(word_groups or [], key=lambda group: (group[0], group[1])):
            token = str(full_word).strip().lower()
            if token in seen or not self._is_valuable_token(token):
                continue
            if token in token_vectors or token in context_memory:
                anchors.append(token)
                seen.add(token)
        return anchors

    def _lucky_context_neighbors(self, global_adaptive, anchor_tokens, token_vectors, present_tokens=None):
        context_memory = self._ensure_lucky_context_memory(global_adaptive)
        present = set(present_tokens or [])
        scored = {}
        for anchor in anchor_tokens or []:
            for neighbor, item in context_memory.get(anchor, {}).items():
                if neighbor in present or neighbor not in token_vectors:
                    continue
                score = self._lucky_context_score(item)
                if score <= 0.15 and int(item.get("liked_count", 0)) <= 0:
                    continue
                if self._void_pair_is_poor(global_adaptive, anchor, neighbor):
                    continue
                scored[neighbor] = max(scored.get(neighbor, -999.0), score)
        return [
            token for token, _ in sorted(scored.items(), key=lambda pair: pair[1], reverse=True)
        ]

    def _lucky_global_tokens(self, lucky_pool):
        scored = sorted(lucky_pool, key=lambda item: item[2], reverse=True)
        preferred = [
            token for token, item, score in scored
            if score > 0.10 or int(item.get("liked_count", 0)) > 0 or int(item.get("neutral_count", 0)) > 0
        ]
        return preferred or [token for token, _, _ in scored]

    def _lucky_weighted_pick(self, token_scores, candidates, previous_token=None, global_adaptive=None):
        usable = []
        weights = []
        for token in candidates or []:
            token = str(token).strip().lower()
            if not token:
                continue
            if previous_token and global_adaptive is not None and self._void_pair_is_poor(global_adaptive, previous_token, token):
                continue
            usable.append(token)
            weights.append(max(0.05, float(token_scores.get(token, 0.0)) + 1.25))

        if not usable:
            return None

        if len(usable) == 1:
            return usable[0]

        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        index = int(torch.multinomial(weights_tensor, 1).item())
        return usable[index]

    def _lucky_compose_tokens(self, global_adaptive, word_groups, lucky_pool, token_vectors, slot_count):
        token_scores = {token: score for token, _, score in lucky_pool}
        token_items = {token: item for token, item, _ in lucky_pool}
        present_tokens = {
            str(group[2]).strip().lower()
            for group in (word_groups or [])
            if isinstance(group, (list, tuple)) and len(group) >= 3 and self._is_valuable_token(str(group[2]).strip())
        }
        anchor_tokens = self._lucky_prompt_anchor_tokens(word_groups, token_vectors, global_adaptive)
        context_tokens = self._lucky_context_neighbors(
            global_adaptive,
            anchor_tokens,
            token_vectors,
            present_tokens=present_tokens,
        )
        global_tokens = self._lucky_global_tokens(lucky_pool)

        seed_sequence = []
        for token in anchor_tokens + context_tokens + global_tokens:
            if token in token_vectors and token not in seed_sequence:
                seed_sequence.append(token)

        if not seed_sequence:
            seed_sequence = [token for token, _, _ in lucky_pool]

        max_slots = max(0, int(slot_count))
        required_candidates = []
        for index, token in enumerate(anchor_tokens):
            item = token_items.get(token, {})
            missing_pressure = (
                int(item.get("wanted_count", 0)) +
                int(item.get("missing_count", 0)) +
                int(item.get("missing_concept_count", 0)) +
                int(item.get("missing_detail_count", 0)) +
                int(item.get("missing_quality_count", 0))
            )
            # Current prompt anchors are required; missing pressure decides which
            # ones win when the conditioning budget is tight.
            priority = float(token_scores.get(token, 0.0)) + missing_pressure * 0.55 + 0.12 / float(index + 1)
            required_candidates.append((priority, index, token))
        required_candidates = sorted(required_candidates, key=lambda item: (-item[0], item[1]))
        reserve_count = min(len(required_candidates), max_slots)
        if reserve_count > 3:
            reserve_count = min(reserve_count, max(3, min(16, int(math.ceil(max_slots * 0.40)))))
        required_tokens = [
            token for _, _, token in sorted(required_candidates[:reserve_count], key=lambda item: item[1])
        ]

        chosen_tokens = list(required_tokens)
        previous_token = None
        context_hits = 0
        if chosen_tokens:
            previous_token = chosen_tokens[-1]
        for index in range(len(chosen_tokens), max_slots):
            context_index = index - len(required_tokens)
            if context_tokens and 0 <= context_index < len(context_tokens):
                candidates = [context_tokens[context_index]]
            elif seed_sequence and random.random() < 0.72:
                candidates = seed_sequence
            else:
                candidates = [token for token, _, _ in lucky_pool]

            token = self._lucky_weighted_pick(token_scores, candidates, previous_token, global_adaptive)
            if token is None:
                token = self._lucky_weighted_pick(token_scores, [token for token, _, _ in lucky_pool], previous_token, global_adaptive)
            if token is None:
                continue

            if token in context_tokens:
                context_hits += 1
            chosen_tokens.append(token)
            previous_token = token

        if anchor_tokens and context_hits:
            source = "anchors+context"
        elif anchor_tokens:
            source = "prompt anchors"
        elif context_hits:
            source = "context"
        else:
            source = "global preferences"

        return chosen_tokens, {
            "source": source,
            "anchor_tokens": anchor_tokens,
            "required_tokens": required_tokens,
            "context_tokens": context_tokens,
            "context_hits": context_hits,
        }

    def _compose_lucky_prompt_text(self, global_adaptive, word_groups, embedding_dim,
                                   prompt_sequences=None, phrase_sequences=None, target_count=64):
        bank = self._ensure_void_token_bank(global_adaptive)
        lucky_pool = []
        for token, item in bank.items():
            embedding_shape = self._serialized_tensor_shape(item.get("embedding"))
            if embedding_shape is None or len(embedding_shape) != 1 or int(embedding_shape[0]) != int(embedding_dim):
                continue
            lucky_pool.append((token, item, self._void_bank_score(item)))

        if len(lucky_pool) < 2:
            return "", {
                "source": "bank too small",
                "anchor_tokens": [],
                "context_tokens": [],
                "context_hits": 0,
                "token_count": 0,
            }

        token_vectors = {token: True for token, _, _ in lucky_pool}

        if len(token_vectors) < 2:
            return "", {
                "source": "bank unreadable",
                "anchor_tokens": [],
                "context_tokens": [],
                "context_hits": 0,
                "token_count": 0,
            }

        count = max(4, min(int(target_count or 0), 32))
        chosen_tokens, compose_info = self._lucky_compose_tokens(
            global_adaptive,
            word_groups,
            lucky_pool,
            token_vectors,
            count,
        )

        ordered_sequences = []
        if isinstance(prompt_sequences, list):
            for sequence in prompt_sequences:
                if not isinstance(sequence, list):
                    continue
                ordered = [str(token).strip().lower() for token in sequence if str(token).strip().lower() in token_vectors]
                if len(ordered) >= 2:
                    ordered_sequences.append(ordered)

        if ordered_sequences and compose_info.get("source") == "global preferences":
            sequence = random.choice(ordered_sequences)
            chosen_tokens = [
                sequence[index % len(sequence)]
                if random.random() < 0.55 else chosen_tokens[index]
                for index in range(min(len(chosen_tokens), count))
            ]
            compose_info["source"] = "saved prompt order"

        token_scores = {token: score for token, _, score in lucky_pool}
        prompt_tokens = []
        seen = set()
        for token in chosen_tokens:
            token = str(token).strip().lower()
            if token in seen or not self._is_valuable_token(token):
                continue
            prompt_tokens.append(token)
            seen.add(token)

        if not prompt_tokens:
            return "", {
                **compose_info,
                "token_count": 0,
            }

        prompt_phrases = []
        phrase_seen = set()
        chosen_set = set(prompt_tokens)
        if isinstance(phrase_sequences, list):
            phrase_candidates = []
            placement_memory = self._ensure_lucky_phrase_placements(global_adaptive)
            for sequence_index, sequence in enumerate(phrase_sequences[-64:]):
                if not isinstance(sequence, list):
                    continue
                for phrase_index, phrase in enumerate(sequence):
                    if not isinstance(phrase, dict):
                        continue
                    text = str(phrase.get("text", "")).strip()
                    if not text:
                        continue
                    tokens = [
                        str(token).strip().lower()
                        for token in phrase.get("tokens", [])
                        if self._is_valuable_token(str(token).strip())
                    ]
                    if not tokens:
                        continue
                    overlap = chosen_set & set(tokens)
                    if not overlap:
                        continue
                    score = sum(max(0.05, token_scores.get(token, 0.0) + 1.25) for token in overlap)
                    score += 0.18 / float(phrase_index + 1)
                    score += 0.03 / float(max(1, len(phrase_sequences) - sequence_index))
                    preferred_position, placement_score = self._lucky_phrase_preferred_position(
                        placement_memory,
                        text,
                        fallback_position=phrase_index,
                    )
                    score += max(0.0, placement_score) * 0.20
                    phrase_candidates.append((score, preferred_position, phrase_index, text, tokens))

            sorted_candidates = sorted(phrase_candidates, key=lambda item: item[0], reverse=True)
            selected = []
            for score, preferred_position, phrase_index, text, tokens in sorted_candidates:
                key = text.lower()
                if key in phrase_seen:
                    continue
                selected.append({
                    "score": score,
                    "position": preferred_position,
                    "source_position": phrase_index,
                    "text": text,
                    "tokens": tokens,
                })
                phrase_seen.add(key)
                if len(selected) >= 12:
                    break
            prompt_phrases = [
                item["text"]
                for item in sorted(
                    selected,
                    key=lambda item: (item["position"], -item["score"], item["source_position"], item["text"]),
                )
            ]

        if not prompt_phrases:
            prompt_phrases = prompt_tokens

        prompt_text = ", ".join(prompt_phrases)
        return prompt_text, {
            **compose_info,
            "token_count": len(prompt_tokens),
            "tokens": prompt_tokens,
            "phrases": prompt_phrases,
        }

    def _lucky_phrase_preferred_position(self, placement_memory, phrase_text, fallback_position=0):
        phrase_key = str(phrase_text).strip().lower()
        entry = placement_memory.get(phrase_key) if isinstance(placement_memory, dict) else None
        if not isinstance(entry, dict):
            return int(fallback_position), 0.0
        positions = entry.get("positions")
        if not isinstance(positions, dict) or not positions:
            return int(fallback_position), 0.0

        candidates = []
        for pos_key, item in positions.items():
            if not isinstance(item, dict):
                continue
            try:
                pos = max(0, min(31, int(pos_key)))
            except (TypeError, ValueError):
                continue
            score = self._phrase_position_score(item)
            count_bonus = min(0.30, math.log1p(float(item.get("count", 0))) * 0.08)
            candidates.append((score + count_bonus, pos))

        if not candidates:
            return int(fallback_position), 0.0

        candidates = sorted(candidates, key=lambda item: item[0], reverse=True)
        top_score = candidates[0][0]
        close = [item for item in candidates if item[0] >= top_score - 0.18]
        if len(close) > 1 and random.random() < 0.35:
            weights = torch.tensor([max(0.05, item[0] + 1.25) for item in close], dtype=torch.float32)
            selected = close[int(torch.multinomial(weights, 1).item())]
        else:
            selected = candidates[0]
        return int(selected[1]), float(selected[0])

    def _encode_lucky_prompt_conditioning(self, clip, prompt_text, template, device, dtype):
        if clip is None or not prompt_text:
            return None, None, ""
        try:
            tokens = clip.tokenize(prompt_text)
            encoded = clip.encode_from_tokens_scheduled(tokens)
        except Exception as e:
            print(f"[FunPackVideoRefiner] Lucky prompt CLIP/Gemma encode failed: {e}")
            return None, None, f"encode failed: {e}"

        if not isinstance(encoded, list) or not encoded:
            return None, None, "encode returned empty conditioning"

        item = encoded[0]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            cond = item[0]
            meta = item[1] if isinstance(item[1], dict) else {"pooled_output": None}
        else:
            cond = item if isinstance(item, torch.Tensor) else None
            meta = {"pooled_output": None}

        if not isinstance(cond, torch.Tensor):
            return None, None, "encode returned invalid conditioning"

        cond = cond.to(device=device, dtype=dtype)
        if not self._conditioning_canvas_compatible(cond, template):
            return None, None, f"encoded shape {tuple(cond.shape)} incompatible"

        return cond, meta, f"encoded prompt ({self._get_conditioning_seq_len(cond)} positions)"

    def _select_void_tokens(self, global_adaptive, word_groups, cur_positive,
                            word_importance, token_mask=None, max_count=3):
        if not isinstance(cur_positive, torch.Tensor) or cur_positive.dim() <= 1:
            return [], "idle"
        embedding_dim = int(cur_positive.shape[-1])
        eligible = self._eligible_void_bank_items(global_adaptive, embedding_dim)
        if eligible:
            pool = sorted(eligible, key=lambda item: item[2], reverse=True)[:24]
            weights = torch.tensor([max(0.01, item[2]) for item in pool], dtype=torch.float32)
            count = min(max_count, max(1, int(torch.randint(1, min(3, len(pool)) + 1, (1,)).item())))
            picked_indexes = torch.multinomial(weights, count, replacement=False).tolist()
            picked = []
            for index in picked_indexes:
                token, item, _ = pool[index]
                picked.append((token, serializable_to_tensor(item["embedding"]).float()))
            return picked, "bank"

        fallback = []
        token_mask_list = None
        if token_mask is not None:
            token_mask_list = token_mask.detach().cpu().tolist()
        for start, end, full_word, _ in word_groups:
            token = full_word.strip().lower()
            if not self._is_valuable_token(token):
                continue
            if token_mask_list is not None and not all(token_mask_list[start:min(len(token_mask_list), end)]):
                continue
            score = float(word_importance.get(token, 1.0))
            if score < 1.05:
                continue
            embedding = self._void_token_embedding(cur_positive, start, end)
            if embedding is not None:
                fallback.append((token, embedding.float().cpu(), score))
        fallback = sorted(fallback, key=lambda item: item[2], reverse=True)[:max_count]
        return [(token, vector) for token, vector, _ in fallback], "current" if fallback else "empty"

    def _apply_into_the_void(self, new_positive, reference, global_adaptive, word_groups,
                             word_importance, into_the_void=False, im_feeling_lucky=False,
                             token_mask=None, prompt_sequences=None, lucky_canvas=None,
                             lucky_canvas_label="", lucky_prompt_text="", lucky_prompt_tokens=None):
        if not into_the_void and not im_feeling_lucky:
            return new_positive, self._void_empty_status(False), {}
        if not isinstance(new_positive, torch.Tensor) or not isinstance(reference, torch.Tensor):
            return new_positive, "Void: on | unavailable | tokens: none", {}
        if new_positive.dim() <= 1 or list(new_positive.shape) != list(reference.shape):
            return new_positive, "Void: on | incompatible conditioning | tokens: none", {}

        device = new_positive.device
        dtype = new_positive.dtype
        seq_len = new_positive.shape[1] if new_positive.dim() == 3 else new_positive.shape[0]
        active_len = self._get_effective_seq_len(token_mask, seq_len)
        active_len = max(1, min(seq_len, active_len))
        mixed = new_positive.clone()
        status_parts = []
        lucky_metadata = {}
        lucky_canvas_used = False
        lucky_canvas_status = ""

        if (
            im_feeling_lucky and
            self._conditioning_canvas_compatible(lucky_canvas, new_positive)
        ):
            try:
                canvas = lucky_canvas.to(device=device, dtype=dtype)
            except Exception:
                canvas = None
            if isinstance(canvas, torch.Tensor):
                mixed = canvas.clone()
                reference = canvas.clone()
                seq_len = mixed.shape[1] if mixed.dim() == 3 else mixed.shape[0]
                if token_mask is None or int(token_mask.numel()) != int(seq_len):
                    token_mask = self._get_conditioning_token_mask(mixed)
                active_len = self._get_effective_seq_len(token_mask, seq_len)
                active_len = max(1, min(seq_len, active_len))
                lucky_canvas_used = True
                lucky_canvas_status = lucky_canvas_label or f"memory ({seq_len} positions)"

        clamp_base = mixed.clone()

        if into_the_void:
            selected, source = self._select_void_tokens(
                global_adaptive,
                word_groups,
                mixed,
                word_importance,
                token_mask=token_mask,
            )
            if selected:
                positions = torch.linspace(0, active_len - 1, steps=len(selected), device=device).round().long()
                strength = min(0.055, 0.09 / max(1, len(selected)))

                for index, (_, vector) in enumerate(selected):
                    vector = vector.to(device=device, dtype=dtype)
                    if vector.dim() != 1 or int(vector.shape[0]) != int(mixed.shape[-1]):
                        continue
                    pos = int(positions[index].item())
                    if token_mask is not None and pos < token_mask.shape[0] and not bool(token_mask[pos].item()):
                        active_positions = torch.nonzero(token_mask[:active_len], as_tuple=False).flatten()
                        if active_positions.numel() == 0:
                            continue
                        pos = int(active_positions[min(index, active_positions.numel() - 1)].item())
                    if mixed.dim() == 3:
                        mixed[:, pos, :] = mixed[:, pos, :].lerp(vector.view(1, -1), strength)
                    else:
                        mixed[pos, :] = mixed[pos, :].lerp(vector, strength)

                token_list = ", ".join(token for token, _ in selected)
                status_parts.append(f"Void: on | {source} | tokens: {token_list}")
            else:
                status_parts.append(f"Void: on | {source} | tokens: none")
        else:
            status_parts.append("Void: off | idle | tokens: none")

        if im_feeling_lucky:
            active_positions = torch.arange(active_len, device=device)
            if token_mask is not None:
                active_positions = torch.nonzero(token_mask[:active_len], as_tuple=False).flatten()

            prompt_tokens = [
                str(token).strip().lower()
                for token in (lucky_prompt_tokens or [])
                if self._is_valuable_token(str(token).strip())
            ]
            if lucky_canvas_used and prompt_tokens:
                positions = torch.linspace(
                    0,
                    max(0, int(active_positions.numel()) - 1),
                    steps=min(len(prompt_tokens), max(1, int(active_positions.numel()))),
                    device=device,
                ).round().long()
                injections = []
                for index, token in enumerate(prompt_tokens[:int(positions.numel())]):
                    pos = int(active_positions[int(positions[index].item())].item()) if active_positions.numel() > 0 else index
                    injections.append({"token": token, "start": pos, "end": pos + 1})
                lucky_metadata = {
                    "enabled": True,
                    "source": "encoded prompt",
                    "tokens": prompt_tokens,
                    "unique_tokens": sorted(set(prompt_tokens)),
                    "injections": injections,
                    "anchor_tokens": [],
                    "required_tokens": prompt_tokens,
                    "context_tokens": [],
                    "context_hits": 0,
                    "canvas": lucky_canvas_status,
                    "prompt": lucky_prompt_text,
                }
                token_preview = ", ".join(prompt_tokens[:8])
                if len(prompt_tokens) > 8:
                    token_preview += ", ..."
                prompt_preview = re.sub(r"\s+", " ", lucky_prompt_text).strip()
                if len(prompt_preview) > 160:
                    prompt_preview = prompt_preview[:157].rstrip() + "..."
                status_parts.append(
                    f"Lucky: on | encoded prompt | {len(prompt_tokens)} prompt tokens | "
                    f"canvas: {lucky_canvas_status} | prompt: {prompt_preview} | tokens: {token_preview}"
                )
            else:
                lucky_pool = self._eligible_lucky_bank_items(global_adaptive, int(mixed.shape[-1]))
                if len(lucky_pool) < 2:
                    if lucky_canvas_used:
                        lucky_metadata = {
                            "enabled": True,
                            "source": "memory canvas",
                            "tokens": [],
                            "unique_tokens": [],
                            "injections": [],
                            "anchor_tokens": [],
                            "required_tokens": [],
                            "context_tokens": [],
                            "context_hits": 0,
                            "canvas": lucky_canvas_status,
                            "prompt": lucky_prompt_text,
                        }
                        status_parts.append(
                            f"Lucky: on | memory canvas {lucky_canvas_status} | bank too small ({len(lucky_pool)}/2 eligible) | tokens: none"
                        )
                    else:
                        status_parts.append(f"Lucky: on | bank too small ({len(lucky_pool)}/2 eligible) | tokens: none")
                elif active_positions.numel() <= 0:
                    if lucky_canvas_used:
                        lucky_metadata = {
                            "enabled": True,
                            "source": "memory canvas",
                            "tokens": [],
                            "unique_tokens": [],
                            "injections": [],
                            "anchor_tokens": [],
                            "required_tokens": [],
                            "context_tokens": [],
                            "context_hits": 0,
                            "canvas": lucky_canvas_status,
                            "prompt": lucky_prompt_text,
                        }
                        status_parts.append(f"Lucky: on | memory canvas {lucky_canvas_status} | no active positions | tokens: none")
                    else:
                        status_parts.append("Lucky: on | no active positions | tokens: none")
                else:
                    token_membership = {token: True for token, _, _ in lucky_pool}

                    ordered_sequences = []
                    if isinstance(prompt_sequences, list):
                        for sequence in prompt_sequences:
                            if not isinstance(sequence, list):
                                continue
                            ordered = [token for token in sequence if token in token_membership]
                            if len(ordered) >= 2:
                                ordered_sequences.append(ordered)

                    chosen_tokens, compose_info = self._lucky_compose_tokens(
                        global_adaptive,
                        word_groups,
                        lucky_pool,
                        token_membership,
                        int(active_positions.numel()),
                    )
                    if ordered_sequences and compose_info.get("source") == "global preferences":
                        sequence = random.choice(ordered_sequences)
                        chosen_tokens = [
                            sequence[index % len(sequence)]
                            if random.random() < 0.55 else chosen_tokens[index]
                            for index in range(min(len(chosen_tokens), int(active_positions.numel())))
                        ]
                        compose_info["source"] = "saved prompt order"

                    needed_tokens = set(chosen_tokens)
                    token_vectors = {}
                    for token, item, _ in lucky_pool:
                        if token not in needed_tokens:
                            continue
                        try:
                            vector = serializable_to_tensor(item["embedding"]).to(device=device, dtype=dtype)
                        except Exception:
                            continue
                        if vector.dim() == 1 and int(vector.shape[0]) == int(mixed.shape[-1]):
                            token_vectors[token] = vector

                    lucky_field = mixed.clone()
                    picked_tokens = []
                    injections = []
                    for pos_tensor, token in zip(active_positions, chosen_tokens):
                        vector = token_vectors.get(token)
                        if vector is None:
                            continue
                        pos = int(pos_tensor.item())
                        if lucky_field.dim() == 3:
                            lucky_field[:, pos, :] = vector.view(1, -1)
                        else:
                            lucky_field[pos, :] = vector
                        picked_tokens.append(token)
                        injections.append({"token": token, "start": pos, "end": pos + 1})

                    if picked_tokens:
                        strength = 0.030 if into_the_void else 0.040
                        mixed = mixed.lerp(lucky_field, strength)
                        token_preview = []
                        seen_tokens = set()
                        for token in picked_tokens:
                            if token in seen_tokens:
                                continue
                            token_preview.append(token)
                            seen_tokens.add(token)
                            if len(token_preview) >= 8:
                                break
                        token_list = ", ".join(token_preview)
                        if len(seen_tokens) < len(set(picked_tokens)):
                            token_list += ", ..."
                        anchor_preview = ", ".join(compose_info.get("anchor_tokens", [])[:4]) or "none"
                        required_preview = ", ".join(compose_info.get("required_tokens", [])[:6]) or "none"
                        context_preview = ", ".join(compose_info.get("context_tokens", [])[:6]) or "none"
                        lucky_metadata = {
                            "enabled": True,
                            "source": compose_info.get("source", "global preferences"),
                            "tokens": picked_tokens,
                            "unique_tokens": sorted(set(picked_tokens)),
                            "injections": injections,
                            "anchor_tokens": compose_info.get("anchor_tokens", []),
                            "required_tokens": compose_info.get("required_tokens", []),
                            "context_tokens": compose_info.get("context_tokens", []),
                            "context_hits": int(compose_info.get("context_hits", 0)),
                            "canvas": lucky_canvas_status if lucky_canvas_used else "current conditioning",
                            "prompt": lucky_prompt_text,
                        }
                        canvas_phrase = f" | canvas: {lucky_canvas_status}" if lucky_canvas_used else ""
                        prompt_preview = re.sub(r"\s+", " ", lucky_prompt_text).strip()
                        if len(prompt_preview) > 160:
                            prompt_preview = prompt_preview[:157].rstrip() + "..."
                        prompt_phrase = f" | prompt: {prompt_preview}" if prompt_preview else ""
                        status_parts.append(
                            f"Lucky: on | {lucky_metadata['source']} | field {len(picked_tokens)} positions from {len(set(picked_tokens))} tokens | "
                            f"anchors: {anchor_preview} | required: {required_preview} | context add: {context_preview}{canvas_phrase}{prompt_phrase} | tokens: {token_list}"
                        )
                    else:
                        if lucky_canvas_used:
                            lucky_metadata = {
                                "enabled": True,
                                "source": "memory canvas",
                                "tokens": [],
                                "unique_tokens": [],
                                "injections": [],
                                "anchor_tokens": [],
                                "required_tokens": [],
                                "context_tokens": [],
                                "context_hits": 0,
                                "canvas": lucky_canvas_status,
                                "prompt": lucky_prompt_text,
                            }
                            status_parts.append(f"Lucky: on | memory canvas {lucky_canvas_status} | bank unreadable | tokens: none")
                        else:
                            status_parts.append("Lucky: on | bank unreadable | tokens: none")
        else:
            status_parts.append("Lucky: off")

        delta = mixed - clamp_base
        reference_norm = reference.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        max_delta = reference_norm * 0.08
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = torch.minimum(torch.ones_like(delta_norm), max_delta / delta_norm)
        mixed = clamp_base + delta * scale
        mixed = torch.clamp(mixed, min=-60.0, max=60.0)
        norm_factor = clamp_base.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        mixed = mixed / mixed.norm(dim=-1, keepdim=True).clamp_min(1e-8) * norm_factor
        return mixed, " | ".join(status_parts), lucky_metadata

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_conditioning": ("CONDITIONING",),
                "mode": (["ltx2", "wan"], {"default": "ltx2", "label": "Tokenizer Mode"}),
                "rating": (RATING_LABELS, {"default": "Missing concept", "label": "Rating"}),
                "refinement_key": ("STRING", {"default": "my_style_v1", "multiline": False}),
                "scheduler_mode": (["original", "accurate", "aggressive"], {"default": "original"}),
            },
            "optional": {
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Positive prompt"
                }),
                "clip": ("CLIP", {
                    "tooltip": "Optional text encoder. When connected, Lucky composes a learned prompt and re-encodes it before refinement."
                }),
                "sigmas": ("SIGMAS",),
                "sigma_strength": (["off", "subtle", "medium", "strong", "max"], {
                    "default": "subtle",
                    "label": "Sigma Refinement Strength"
                }),
                "reset_session": ("BOOLEAN", {"default": False, "label": "Reset Session (clears ALL history)"}),
                "unlimited_history": ("BOOLEAN", {
                    "default": False,
                    "label": "Unlimited History (never prunes)"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "label": "Exploration Seed"}),
                "feedback_enabled": ("BOOLEAN", {"default": False, "label": "Enable Concept Feedback"}),
                "feedback_rating": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 6,
                    "step": 1,
                    "display": "slider",
                    "label": "Feedback Response (follow the question scale)"
                }),
                "lora_stack": ("FUNPACK_LORA_STACK", {
                    "tooltip": "Optional stack from FunPack LoRA Loader. The refiner uses it to save prompt-specific suggested LoRA weights."
                }),
                "latent": ("LATENT", {
                    "tooltip": "Optional latent to refine. If no saved latent exists for this key, it passes through unchanged."
                }),
                "into_the_void": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Experimental: lightly mixes learned liked token embeddings into the final conditioning for preference discovery."
                }),
                "im_feeling_lucky": ("BOOLEAN", {
                    "default": False,
                    "label": "I'm Feeling Lucky",
                    "tooltip": "Composes conditioning from learned memory first, using stored canvases plus preferred token/context relationships."
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING", "STRING", "IMAGE", "SIGMAS", "LATENT")
    RETURN_NAMES = ("modified_positive", "status", "feedback_question", "training_info", "loss_graph", "refined_sigmas", "refined_latent")
    FUNCTION = "refine"
    CATEGORY = "FunPack/Refinement"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    # =========================================================================
    # SCHEDULER
    # =========================================================================

    def _get_scheduler_factors(self, mode, rating, reward, similarity, iter_num, total_iters,
                               word_importance, word_groups, global_adaptive, device):
        if mode == "original":
            base_lr = 0.18 / (1 + 0.08 * (iter_num ** 0.5))
            confidence = max(0.2, min(1.0, (rating - 3.0) / 5.0))
            if similarity > 0.93:
                confidence *= 0.4
            return base_lr, confidence, 1.0, {}

        confidence = max(0.15, min(1.0, (rating - 2.5) / 6.0))
        if abs(reward) < 0.15 or similarity > 0.94:
            confidence *= 0.35

        prodigy_d = global_adaptive.setdefault("prodigy_d", {})
        d_coef = 1.0 if mode == "accurate" else 1.8
        adaptive_lr = global_adaptive.get("prodigy_lr_base", 1.0)

        word_lr_mult = {}
        for _, _, full_word, _ in word_groups:
            wkey = full_word.lower()
            if wkey not in word_importance:
                continue
            g = abs(reward) + 1e-8
            if wkey not in prodigy_d:
                prodigy_d[wkey] = g
            else:
                prodigy_d[wkey] = 0.9 * prodigy_d[wkey] + 0.1 * g
            word_lr_mult[wkey] = adaptive_lr / (prodigy_d[wkey] ** 0.5 + 1e-8) * d_coef

        current_step = global_adaptive.setdefault("current_step", 0)
        current_step = min(current_step + 1, 500)
        global_adaptive["current_step"] = current_step

        warmup = global_adaptive.get("warmup_steps", 8)
        progress = min(1.0, current_step / max(50, global_adaptive.get("total_steps_estimate", 150)))

        if current_step < warmup:
            lr_scale = current_step / max(1, warmup)
        else:
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * (progress - warmup / (warmup + 50))))

        if mode == "accurate":
            lr_scale *= 0.75
            confidence *= 0.9
            exploration_mult = max(0.3, 1.0 - 0.6 * progress)
        else:
            lr_scale = min(2.2, lr_scale * 1.6)
            confidence = min(1.0, confidence * 1.3)
            exploration_mult = max(0.6, 1.3 - 0.7 * progress)

        return lr_scale, confidence, exploration_mult, word_lr_mult

    # =========================================================================
    # TOKEN / WORD UTILITIES
    # =========================================================================

    def _get_top_tokens(self, token_dict, tokenizer, top_k=10):
        if not token_dict or not tokenizer:
            return "N/A"
        sorted_tokens = sorted(token_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_list = []
        for key, score in sorted_tokens:
            try:
                text = key if isinstance(key, str) else tokenizer.decode([int(key)], skip_special_tokens=True).strip()
                if text:
                    top_list.append(f"{text}({score:.2f})")
            except Exception:
                top_list.append(f"{key}({score:.2f})")
        return ", ".join(top_list) if top_list else "None"

    def _is_valuable_token(self, token_text):
        if not token_text:
            return False
        t = token_text.strip()
        if '<' in t or '>' in t or len(t) < 3:
            return False
        t_lower = t.lower()
        stopwords = {
            "the", "a", "an", "and", "or", "but", "with", "for", "of", "in", "on", "at",
            "to", "from", "by", "is", "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "do", "does", "did", "will", "would", "her", "his", "him", "she",
            "he", "it", "they", "them", "this", "that", "these", "those", "i", "you", "my",
            "your", "our", "their", "me", "us"
        }
        if t_lower in stopwords:
            return False
        if t in {",", ".", "!", "?", ":", ";", "-", "*", "(", ")", "[", "]", "{", "}", "'", "\"", "..."} or t.isdigit():
            return False
        if not any(c.isalpha() for c in t):
            return False
        return True

    def _infer_concept_category(self, phrase_words: list):
        words = set(phrase_words or [])
        if not words:
            return "general"

        category_terms = {
            "quality": {"masterpiece", "best", "quality", "detailed", "highres", "high-res", "ultra", "perfect"},
            "style": {"anime", "cinematic", "photorealistic", "painterly", "illustration", "stylized", "realistic", "film", "noir"},
            "camera": {"closeup", "close-up", "wide", "shot", "angle", "zoom", "pan", "tracking", "dolly", "camera", "focus", "bokeh", "framing"},
            "action": {"running", "walking", "flying", "jumping", "smiling", "turning", "dancing", "moving", "motion", "looking", "holding", "standing", "sitting"},
            "environment": {"forest", "city", "street", "room", "beach", "mountain", "temple", "sunset", "night", "rain", "snow", "sky", "background"},
            "appearance": {"hair", "eyes", "dress", "jacket", "armor", "face", "skin", "beard", "smile", "pose", "outfit"},
            "subject": {"girl", "boy", "woman", "man", "person", "character", "robot", "dragon", "cat", "dog", "bird", "child"},
        }

        best_category, best_score = "general", 0
        for category, terms in category_terms.items():
            score = len(words & terms)
            if score > best_score:
                best_category, best_score = category, score
        return best_category

    def _default_concept_cluster(self, phrase_words: list):
        category = self._infer_concept_category(phrase_words)
        return {
            "label": " ".join((phrase_words or [])[:6]),
            "anchor_words": list(phrase_words or []),
            "category": category,
            "category_source": "auto",
            "category_confidence": 0.35 if category == "general" else 0.65,
            "word_importance": {},
            "presence_target": 1.0,
            "priority_weight": 1.0,
            "overrep_sensitivity": 1.0,
            "stability_weight": 1.0,
            "semantic_fidelity": 1.0,
            "user_affinity": 1.0,
            "question_history": [],
            "last_question_type": None,
            "last_question_iter": 0,
            "usage_count": 0,
            "last_seen_iter": 0,
        }

    def _ensure_concept_cluster_defaults(self, cluster: dict):
        if not isinstance(cluster, dict):
            return self._default_concept_cluster([])

        anchor_words = list(cluster.get("anchor_words", []))
        defaults = self._default_concept_cluster(anchor_words)
        defaults.update(cluster)
        defaults.setdefault("category_source", "auto")
        defaults.setdefault("category_confidence", 0.35 if defaults.get("category") == "general" else 0.65)
        if defaults.get("category_source") != "user" and (
            not defaults.get("category") or defaults.get("category") == "general"
        ):
            defaults["category"] = self._infer_concept_category(defaults.get("anchor_words", []))
            defaults["category_confidence"] = 0.35 if defaults["category"] == "general" else max(0.65, defaults.get("category_confidence", 0.0))
        defaults["question_history"] = list(defaults.get("question_history", []))[-24:]
        return defaults

    def _clip_profile_value(self, value, low=0.5, high=1.8):
        return max(low, min(high, float(value)))

    def _feedback_question_specs(self):
        return {
            "presence": {
                "prompt": "How well is '{label}' represented in the output?",
                "legend": "1=absent  2=weak  3=slightly weak  4=slightly strong  5=perfect  6=overrepresented",
            },
            "priority": {
                "prompt": "How important should '{label}' be relative to the other concepts?",
                "legend": "1=much less important  2=less important  3=slightly less  4=slightly more  5=important  6=top priority",
            },
            "balance": {
                "prompt": "How balanced is '{label}' compared with nearby concepts?",
                "legend": "1=far too weak  2=too weak  3=slightly weak  4=slightly strong  5=balanced  6=overpowering",
            },
            "fidelity": {
                "prompt": "How accurately does '{label}' match what you meant?",
                "legend": "1=wrong  2=mostly wrong  3=slightly off  4=close  5=correct  6=too literal",
            },
            "stability": {
                "prompt": "How stable should '{label}' be across future outputs?",
                "legend": "1=very unstable  2=unstable  3=slightly unstable  4=slightly stable  5=stable  6=very rigid",
            },
            "preference": {
                "prompt": "What is your preference for having '{label}' in future outputs?",
                "legend": "1=strongly less  2=less  3=slightly less  4=slightly more  5=more  6=much more",
            },
            "category": {
                "prompt": "What kind of concept is '{label}'?",
                "legend": "1=general  2=concept  3=style  4=quality  5=character  6=details",
            },
        }

    def _get_concept_mean_importance(self, cluster: dict):
        local_imp = cluster.get("word_importance", {})
        if not local_imp:
            return 1.0
        return sum(local_imp.values()) / max(1, len(local_imp))

    def _get_question_base_weight(self, question_type: str, category: str):
        weights = {
            "presence": {"subject": 1.15, "appearance": 1.12, "action": 1.10, "environment": 1.08, "style": 0.95, "camera": 0.92, "quality": 0.90, "general": 1.0},
            "priority": {"style": 1.15, "camera": 1.12, "quality": 1.08, "environment": 1.02, "subject": 0.98, "appearance": 0.98, "action": 1.0, "general": 1.0},
            "balance": {"style": 1.18, "camera": 1.10, "quality": 1.08, "environment": 1.02, "subject": 0.98, "appearance": 1.0, "action": 1.0, "general": 1.0},
            "fidelity": {"subject": 1.15, "action": 1.12, "environment": 1.08, "appearance": 1.05, "style": 0.95, "camera": 0.92, "quality": 0.90, "general": 1.0},
            "stability": {"action": 1.10, "camera": 1.10, "style": 1.06, "subject": 1.04, "appearance": 1.0, "environment": 1.0, "quality": 0.95, "general": 1.0},
            "preference": {"style": 1.14, "environment": 1.10, "appearance": 1.05, "subject": 1.03, "camera": 1.02, "action": 1.0, "quality": 0.95, "general": 1.0},
        }
        return weights.get(question_type, {}).get(category, 1.0)

    def _score_question_type(self, question_type: str, cluster: dict, category: str,
                             mean_imp: float, neighbor_mean: float, rating_shift: float,
                             similarity: float, iter_num: int):
        uncertainty = max(0.0, 1.0 - min(1.0, abs(mean_imp - cluster.get("presence_target", 1.0))))
        dominance = max(0.0, mean_imp - 1.15)
        neighbor_conflict = abs(mean_imp - neighbor_mean)
        freshness = 1.0
        if cluster.get("last_question_type") == question_type:
            freshness -= 0.35
        if iter_num - int(cluster.get("last_question_iter", 0)) < 3:
            freshness -= 0.20
        freshness = max(0.25, freshness)

        if question_type == "presence":
            score = 0.50 * uncertainty + 0.22 * abs(cluster.get("presence_target", 1.0) - mean_imp) + 0.16 * rating_shift + 0.12 * max(0.0, 1.0 - similarity)
        elif question_type == "priority":
            score = 0.40 * rating_shift + 0.28 * abs(cluster.get("priority_weight", 1.0) - 1.0) + 0.18 * dominance + 0.14 * uncertainty
        elif question_type == "balance":
            score = 0.44 * dominance + 0.28 * neighbor_conflict + 0.16 * abs(cluster.get("overrep_sensitivity", 1.0) - 1.0) + 0.12 * rating_shift
        elif question_type == "fidelity":
            score = 0.38 * (2.0 - cluster.get("semantic_fidelity", 1.0)) + 0.24 * rating_shift + 0.20 * uncertainty + 0.18 * max(0.0, 1.0 - similarity)
        elif question_type == "stability":
            score = 0.35 * rating_shift + 0.30 * abs(cluster.get("stability_weight", 1.0) - 1.0) + 0.20 * max(0.0, 1.0 - similarity) + 0.15 * uncertainty
        elif question_type == "category":
            category_uncertainty = max(0.0, 1.0 - float(cluster.get("category_confidence", 0.35)))
            needs_category_discovery = (
                cluster.get("category_source") != "user" and
                (category == "general" or category_uncertainty >= 0.50)
            )
            if not needs_category_discovery:
                return 0.0
            score = 0.65 + 0.25 * category_uncertainty + 0.10 * rating_shift
        else:  # preference
            score = 0.34 * abs(cluster.get("user_affinity", 1.0) - 1.0) + 0.28 * rating_shift + 0.20 * uncertainty + 0.18 * abs(cluster.get("priority_weight", 1.0) - 1.0)

        return score * self._get_question_base_weight(question_type, category) * freshness

    def _select_feedback_question(self, ordered_concept_ids: list, concept_clusters: dict,
                                  concept_groups: dict, current_concept_labels: dict,
                                  last_rating: int, rating: int,
                                  similarity: float, iter_num: int):
        if not ordered_concept_ids:
            return None

        question_specs = self._feedback_question_specs()
        rating_shift = min(1.0, abs(rating - last_rating) / 4.0)
        candidates = []
        category_candidates = []

        for cid in ordered_concept_ids:
            cluster = concept_clusters.get(cid)
            if not cluster:
                continue
            cluster = self._ensure_concept_cluster_defaults(cluster)
            concept_clusters[cid] = cluster

            category = cluster.get("category", "general")
            mean_imp = self._get_concept_mean_importance(cluster)
            neighbor_ids = self._get_concept_neighbors(cid, ordered_concept_ids, radius=2)
            neighbor_means = [
                self._get_concept_mean_importance(concept_clusters[nid])
                for nid in neighbor_ids if nid in concept_clusters
            ]
            neighbor_mean = sum(neighbor_means) / len(neighbor_means) if neighbor_means else 1.0

            chosen_group_id = None
            for gid, group in concept_groups.items():
                if cid in group.get("concept_ids", []):
                    chosen_group_id = gid
                    break

            for question_type in question_specs.keys():
                score = self._score_question_type(
                    question_type, cluster, category, mean_imp,
                    neighbor_mean, rating_shift, similarity, iter_num
                )
                candidate = {
                    "concept_id": cid,
                    "concept_label": current_concept_labels.get(cid, cluster.get("label", "")),
                    "question_type": question_type,
                    "neighbor_ids": neighbor_ids,
                    "group_id": chosen_group_id,
                    "category": category,
                    "score": score,
                }
                if question_type == "category":
                    category_candidates.append(candidate)
                else:
                    candidates.append(candidate)

        if category_candidates:
            category_candidate = max(category_candidates, key=lambda x: x["score"])
            if category_candidate["score"] >= 0.60:
                return category_candidate

        if not candidates:
            return None

        return max(candidates, key=lambda x: x["score"])

    def _format_feedback_question(self, concept_label: str, question_type: str):
        spec = self._feedback_question_specs().get(question_type, self._feedback_question_specs()["presence"])
        return (
            f"Concept: '{concept_label}'\n"
            f"{spec['prompt'].format(label=concept_label)}\n"
            f"{spec['legend']}"
        )

    # =========================================================================
    # MULTI-LEVEL CONCEPT SYSTEM
    # =========================================================================

    def _parse_concepts(self, prompt: str):
        """
        Level 2 → Level 3 boundary: split the prompt on commas/semicolons into
        concept phrases. Each phrase is returned as a list of significant lowercase
        words, preserving prompt order.

        Example:
          "masterpiece, anime girl, long hair, dark forest"
          -> [["masterpiece"], ["anime","girl"], ["long","hair"], ["dark","forest"]]
        """
        if not prompt:
            return []
        result = []
        for segment_type, segment_text in self._iter_prompt_segments(prompt):
            if segment_type == "protected":
                protected_phrase = segment_text.strip().lower()
                if len(protected_phrase) >= 3 and any(c.isalpha() for c in protected_phrase):
                    result.append([protected_phrase])
                continue

            phrases = [p.strip() for p in re.split(r'[,;]', segment_text) if p.strip()]
            for phrase in phrases:
                words = [w.strip().lower() for w in phrase.split() if self._is_valuable_token(w.strip())]
                if words:
                    result.append(words)
        return result

    def _concept_overlap_words(self, prompt: str):
        words = []
        seen = set()
        for phrase_words in self._parse_concepts(prompt):
            for word in phrase_words:
                clean = str(word).strip().lower()
                if not clean or clean in seen:
                    continue
                seen.add(clean)
                words.append(clean)
        return words

    def _ordered_prompt_words(self, prompt: str):
        words = []
        for phrase_words in self._parse_concepts(prompt):
            for word in phrase_words:
                clean = str(word).strip().lower()
                if clean and self._is_valuable_token(clean):
                    words.append(clean)
        return words

    def _ordered_prompt_phrases(self, prompt: str):
        phrases = []
        if not prompt:
            return phrases
        for phrase_words in self._parse_concepts(prompt):
            clean_words = [
                str(word).strip().lower()
                for word in phrase_words
                if self._is_valuable_token(str(word).strip())
            ]
            if not clean_words:
                continue
            text = " ".join(clean_words)
            text = re.sub(r"\s+", " ", text).strip(" ,;:.")
            if not text or len(text) < 3:
                continue
            phrase_tokens = [
                token.strip().lower()
                for token in re.findall(r"[\w'’.-]+", text, flags=re.UNICODE)
                if self._is_valuable_token(token.strip())
            ]
            if not phrase_tokens:
                phrase_tokens = [text]
            phrases.append({
                "text": text,
                "tokens": phrase_tokens,
            })
        return phrases

    def _lucky_prompt_phrase_sequences(self, prompt_histories: dict):
        if not isinstance(prompt_histories, dict):
            return []

        sequences = []
        seen = set()
        for prompt_key, history_entry in prompt_histories.items():
            if not isinstance(history_entry, dict):
                continue

            history = history_entry.get("history", [])
            good_history = []
            if isinstance(history, list):
                for item in history[-16:]:
                    if not isinstance(item, dict):
                        continue
                    profile = normalize_refiner_rating(item.get("rating_label", item.get("rating", 0)))
                    if int(profile.get("level", 0)) >= 5:
                        good_history.append(item)

            texts = []
            for item in good_history:
                for key in ("prompt_full", "analysis_prompt", "prompt"):
                    text = item.get(key)
                    if isinstance(text, str) and text.strip():
                        texts.append(text)
            if good_history:
                texts.extend(self._history_prompt_texts(prompt_key, history_entry))

            for text in texts:
                phrases = self._ordered_prompt_phrases(text)
                if not phrases:
                    continue
                signature = tuple(phrase["text"] for phrase in phrases)
                if signature in seen:
                    continue
                seen.add(signature)
                sequences.append(phrases[:32])

        return sequences[-48:]

    def _lucky_prompt_sequences(self, prompt_histories: dict):
        if not isinstance(prompt_histories, dict):
            return []

        sequences = []
        seen = set()
        for prompt_key, history_entry in prompt_histories.items():
            if not isinstance(history_entry, dict):
                continue

            history = history_entry.get("history", [])
            good_history = []
            if isinstance(history, list):
                for item in history[-24:]:
                    if not isinstance(item, dict):
                        continue
                    profile = normalize_refiner_rating(item.get("rating_label", item.get("rating", 0)))
                    if int(profile.get("level", 0)) >= 5:
                        good_history.append(item)

            for item in good_history:
                for key in ("prompt_full", "analysis_prompt", "prompt"):
                    text = item.get(key)
                    if not isinstance(text, str) or not text.strip():
                        continue
                    ordered = self._ordered_prompt_words(text)
                    if len(ordered) < 2:
                        continue
                    signature = tuple(ordered)
                    if signature not in seen:
                        seen.add(signature)
                        sequences.append(ordered[:64])

            if good_history:
                for text in self._history_prompt_texts(prompt_key, history_entry):
                    ordered = self._ordered_prompt_words(text)
                    if len(ordered) < 2:
                        continue
                    signature = tuple(ordered)
                    if signature not in seen:
                        seen.add(signature)
                        sequences.append(ordered[:64])

        return sequences[-96:]

    def _history_prompt_texts(self, prompt_key: str, history_entry: dict):
        texts = []
        for key in ("canonical_prompt", "last_positive_prompt"):
            value = history_entry.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value)
        if isinstance(prompt_key, str) and prompt_key.strip():
            texts.append(prompt_key)
        for item in reversed(history_entry.get("history", [])[-4:]):
            value = item.get("prompt") if isinstance(item, dict) else None
            if isinstance(value, str) and value.strip():
                texts.append(value)

        unique = []
        seen = set()
        for text in texts:
            normalized = text.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(normalized)
        return unique

    def _prompt_history_concept_words(self, prompt_key: str, history_entry: dict):
        stored = history_entry.get("prompt_concept_words")
        if isinstance(stored, list) and stored:
            return [str(word).lower() for word in stored if str(word).strip()]

        words = []
        seen = set()
        for text in self._history_prompt_texts(prompt_key, history_entry):
            for word in self._concept_overlap_words(text):
                if word not in seen:
                    seen.add(word)
                    words.append(word)
        return words

    def _conditioning_similarity(self, reference: torch.Tensor, current: torch.Tensor, token_mask=None):
        if not isinstance(reference, torch.Tensor) or not isinstance(current, torch.Tensor):
            return 0.0

        if list(reference.shape) != list(current.shape):
            return 0.0

        current = current.to(reference.device) if current.device != reference.device else current
        try:
            if reference.dim() >= 2 and current.dim() >= 2:
                ref_mean = self._masked_sequence_mean(reference, token_mask)
                cur_mean = self._masked_sequence_mean(current, token_mask)
                return float(F.cosine_similarity(ref_mean, cur_mean, dim=-1).mean().item())
            return float(F.cosine_similarity(
                reference.flatten().unsqueeze(0),
                current.flatten().unsqueeze(0),
                dim=-1
            ).mean().item())
        except Exception:
            return 0.0

    def _conditioning_history_similarity(self, history_entry: dict, raw_positive: torch.Tensor):
        source_data = history_entry.get("source_conditioning_embeds") or history_entry.get("reference_embeds")
        try:
            source_reference = serializable_to_tensor(source_data)
        except Exception:
            return 0.0

        return self._conditioning_similarity(source_reference, raw_positive)

    def _find_prompt_variant_history(self, exact_prompt_key: str, current_words: list,
                                     raw_positive: torch.Tensor, prompt_histories: dict):
        if not exact_prompt_key or not current_words or not prompt_histories:
            return None, None

        current_set = set(current_words)
        if len(current_set) < 2:
            return None, None

        best_key = None
        best_info = None
        best_score = 0.0
        for candidate_key, history_entry in prompt_histories.items():
            if candidate_key == exact_prompt_key or not isinstance(history_entry, dict):
                continue

            candidate_words = self._prompt_history_concept_words(candidate_key, history_entry)
            candidate_set = set(candidate_words)
            if len(candidate_set) < 2:
                continue

            shared = current_set & candidate_set
            shared_count = len(shared)
            union = current_set | candidate_set
            overlap = shared_count / max(1, len(union))
            coverage = shared_count / max(1, min(len(current_set), len(candidate_set)))
            semantic_score = max(overlap, coverage * 0.72)
            conditioning_similarity = self._conditioning_history_similarity(history_entry, raw_positive)

            if conditioning_similarity < 0.68:
                continue
            if shared_count < 3 and not (coverage >= 0.85 and conditioning_similarity >= 0.82):
                continue
            if not (
                (overlap >= 0.46 and conditioning_similarity >= 0.78) or
                (coverage >= 0.68 and conditioning_similarity >= 0.72) or
                (overlap >= 0.60 and conditioning_similarity >= 0.68)
            ):
                continue

            score = semantic_score * 0.58 + conditioning_similarity * 0.42
            if score > best_score:
                best_score = score
                best_key = candidate_key
                best_info = {
                    "score": round(float(score), 4),
                    "overlap": round(float(overlap), 4),
                    "coverage": round(float(coverage), 4),
                    "conditioning_similarity": round(float(conditioning_similarity), 4),
                    "shared_words": sorted(shared)[:12],
                    "matched_prompt_key": candidate_key,
                }

        return best_key, best_info

    def _remember_prompt_variant(self, active: dict, exact_prompt_key: str, positive_prompt: str,
                                 current_words: list, match_info: dict = None):
        active.setdefault("canonical_prompt", positive_prompt or exact_prompt_key)
        existing_words = set(
            str(word).lower()
            for word in active.get("prompt_concept_words", [])
            if str(word).strip()
        )
        for word in current_words:
            existing_words.add(str(word).lower())
        active["prompt_concept_words"] = sorted(existing_words)[:256]

        variants = list(active.get("prompt_variants", []))[-11:]
        variant = {
            "prompt_key": exact_prompt_key,
            "prompt": (positive_prompt or exact_prompt_key)[:240],
        }
        if match_info:
            variant.update({
                "overlap": match_info.get("overlap"),
                "coverage": match_info.get("coverage"),
                "conditioning_similarity": match_info.get("conditioning_similarity"),
            })
        variants.append(variant)
        active["prompt_variants"] = variants

    def _history_entry_prompt_words(self, entry: dict):
        if not isinstance(entry, dict):
            return []
        stored = entry.get("prompt_words")
        if isinstance(stored, list) and stored:
            return [str(word).lower() for word in stored if str(word).strip()]
        prompt_text = entry.get("prompt_full") or entry.get("prompt") or ""
        return self._concept_overlap_words(prompt_text)

    def _retarget_prompt_history_to_source(self, active: dict, old_source: torch.Tensor,
                                           new_source: torch.Tensor):
        if (
            not isinstance(active, dict) or
            not isinstance(old_source, torch.Tensor) or
            not isinstance(new_source, torch.Tensor) or
            list(old_source.shape) != list(new_source.shape)
        ):
            return 0

        shift = new_source.to(old_source.device) - old_source
        migrated = 0

        def _shift_serialized(field_name: str):
            nonlocal migrated
            payload = active.get(field_name)
            if payload is None:
                return
            try:
                tensor = serializable_to_tensor(payload).to(old_source.device)
                if list(tensor.shape) != list(old_source.shape):
                    return
                active[field_name] = tensor_to_serializable(tensor + shift)
                migrated += 1
            except Exception:
                return

        _shift_serialized("reference_embeds")
        _shift_serialized("liked_reference_embeds")

        for item in active.get("history", []):
            if not isinstance(item, dict) or item.get("modified_embeds") is None:
                continue
            try:
                tensor = serializable_to_tensor(item.get("modified_embeds")).to(old_source.device)
                if list(tensor.shape) != list(old_source.shape):
                    continue
                item["modified_embeds"] = tensor_to_serializable(tensor + shift)
                item["retargeted_to_prompt_variant"] = True
                migrated += 1
            except Exception:
                continue

        return migrated

    def _nudge_delta_words(self, words: list, delta: float, concept_clusters: dict,
                           word_importance: dict, iter_num: int, role: str):
        touched = []
        for raw_word in words:
            word = str(raw_word).strip().lower()
            if not self._is_valuable_token(word):
                continue
            cid, is_new = self._match_concept([word], concept_clusters, threshold=0.18)
            if cid is None:
                continue
            if is_new or cid not in concept_clusters:
                concept_clusters[cid] = self._default_concept_cluster([word])
            else:
                concept_clusters[cid] = self._ensure_concept_cluster_defaults(concept_clusters[cid])
                if word not in concept_clusters[cid].get("anchor_words", []):
                    concept_clusters[cid]["anchor_words"].append(word)

            cluster = concept_clusters[cid]
            local_imp = cluster.setdefault("word_importance", {})
            local_imp[word] = max(0.35, min(2.8, float(local_imp.get(word, 1.0)) + delta))
            word_importance[word] = max(0.35, min(2.8, float(word_importance.get(word, 1.0)) + delta * 0.45))
            cluster["presence_target"] = self._clip_profile_value(
                float(cluster.get("presence_target", 1.0)) + delta * 0.28
            )
            cluster["priority_weight"] = self._clip_profile_value(
                float(cluster.get("priority_weight", 1.0)) + delta * 0.20
            )
            if delta > 0:
                cluster["semantic_fidelity"] = self._clip_profile_value(
                    float(cluster.get("semantic_fidelity", 1.0)) + delta * 0.12,
                    low=0.6,
                    high=1.8,
                )
            else:
                cluster["overrep_sensitivity"] = self._clip_profile_value(
                    float(cluster.get("overrep_sensitivity", 1.0)) + abs(delta) * 0.12
                )
            cluster["last_seen_iter"] = iter_num
            cluster["usage_count"] = int(cluster.get("usage_count", 0)) + 1
            cluster["last_prompt_delta_role"] = role
            touched.append(word)
        return touched

    def _missing_axis_matches_category(self, axis: str, category: str):
        category = (category or "general").lower()
        if axis == "concept":
            return category in {"subject", "action", "appearance", "character", "concept", "general"}
        if axis == "details":
            return category in {"camera", "action", "environment", "appearance", "style", "general"}
        if axis == "quality":
            return category in {"quality", "style", "general"}
        return False

    def _apply_missing_axis_prompt_pressure(self, word_groups, word_to_concept, rating_profile,
                                            concept_clusters, word_importance, iter_num):
        missing_axes = set(rating_profile.get("missing_axes", []))
        if not missing_axes or not word_groups:
            return ""

        axis_boosts = {"concept": 0.34, "details": 0.26, "quality": 0.22}
        touched = []
        for _, _, full_word, _ in word_groups:
            word = str(full_word).strip().lower()
            if not self._is_valuable_token(word):
                continue
            cid = word_to_concept.get(word)
            category = "general"
            cluster = None
            if cid and cid in concept_clusters:
                concept_clusters[cid] = self._ensure_concept_cluster_defaults(concept_clusters[cid])
                cluster = concept_clusters[cid]
                category = cluster.get("category", "general")

            matched_axes = [axis for axis in missing_axes if self._missing_axis_matches_category(axis, category)]
            pressure = sum(axis_boosts.get(axis, 0.0) for axis in matched_axes)
            if not pressure:
                pressure = sum(axis_boosts.get(axis, 0.0) for axis in missing_axes) * 0.35
            if not pressure:
                continue

            word_importance[word] = max(0.35, min(2.8, float(word_importance.get(word, 1.0)) + pressure * 0.32))
            if cluster is not None:
                local_imp = cluster.setdefault("word_importance", {})
                local_imp[word] = max(0.35, min(2.8, float(local_imp.get(word, 1.0)) + pressure * 0.55))
                cluster["missing_count"] = int(cluster.get("missing_count", 0)) + 1
                for axis in missing_axes:
                    counter = self._axis_counter_name(axis)
                    if counter:
                        cluster[counter] = int(cluster.get(counter, 0)) + 1
                if "concept" in missing_axes or "details" in missing_axes:
                    cluster["presence_target"] = self._clip_profile_value(
                        float(cluster.get("presence_target", 1.0)) + pressure * 0.18
                    )
                    cluster["priority_weight"] = self._clip_profile_value(
                        float(cluster.get("priority_weight", 1.0)) + pressure * 0.14
                    )
                if "quality" in missing_axes:
                    cluster["semantic_fidelity"] = self._clip_profile_value(
                        float(cluster.get("semantic_fidelity", 1.0)) + pressure * 0.16,
                        low=0.6,
                        high=1.8,
                    )
                    cluster["stability_weight"] = self._clip_profile_value(
                        float(cluster.get("stability_weight", 1.0)) + pressure * 0.12
                    )
                cluster["last_missing_iter"] = iter_num
            touched.append(word)

        if not touched:
            return ""
        axes = "+".join(sorted(missing_axes))
        return f"Missing-axis pressure: {axes} reinforced {', '.join(touched[:10])}{'...' if len(touched) > 10 else ''}."

    def _apply_prompt_delta_attribution(self, history: list, current_words: list,
                                        rating_profile: dict, concept_clusters: dict,
                                        word_importance: dict, iter_num: int):
        if len(history) < 2:
            return ""

        previous_entry = history[-2]
        rated_entry = history[-1]
        previous_words = self._history_entry_prompt_words(previous_entry)
        rated_words = self._history_entry_prompt_words(rated_entry) or current_words
        if not previous_words or not rated_words:
            return ""

        previous_set = set(previous_words)
        rated_set = set(rated_words)
        added = sorted(rated_set - previous_set)
        removed = sorted(previous_set - rated_set)
        if not added and not removed:
            return ""

        previous_profile = normalize_refiner_rating(rated_entry.get("rating_label", rated_entry.get("rating", 0)))
        current_key = rating_profile.get("key", "")
        previous_missing_axes = set(previous_profile.get("missing_axes", []))
        current_missing_axes = set(rating_profile.get("missing_axes", []))
        word_importance = word_importance if isinstance(word_importance, dict) else {}

        boosted = []
        softened = []
        reason = ""
        if current_key == "like" and previous_missing_axes:
            boost = 0.22 + (0.10 if "concept" in previous_missing_axes else 0.0)
            boosted = self._nudge_delta_words(
                added[:24],
                boost,
                concept_clusters,
                word_importance,
                iter_num,
                "added_helped",
            )
            softened = self._nudge_delta_words(
                removed[:24],
                -0.08,
                concept_clusters,
                word_importance,
                iter_num,
                "removed_after_missing",
            )
            reason = f"{previous_profile.get('label', 'missing')} -> {rating_profile.get('label', 'Perfect')}"
        elif current_missing_axes and previous_profile.get("key") == "like":
            boost = 0.14 + (0.08 if "concept" in current_missing_axes else 0.0)
            boosted = self._nudge_delta_words(
                removed[:24],
                boost,
                concept_clusters,
                word_importance,
                iter_num,
                "removed_needed",
            )
            softened = self._nudge_delta_words(
                added[:24],
                -0.10,
                concept_clusters,
                word_importance,
                iter_num,
                "added_hurt",
            )
            reason = f"{previous_profile.get('label', 'Perfect')} -> {rating_profile.get('label', 'missing')}"

        if not boosted and not softened:
            return ""

        parts = [f"Prompt delta: {reason}."]
        if boosted:
            parts.append("boosted " + ", ".join(boosted[:8]))
        if softened:
            parts.append("softened " + ", ".join(softened[:8]))
        return " ".join(parts)

    def _build_prompt_fallback_concept(self, prompt: str, concept_clusters: dict):
        if not prompt:
            return None

        fallback_words = []
        for segment_type, segment_text in self._iter_prompt_segments(prompt):
            if segment_type == "protected":
                protected_phrase = segment_text.strip().lower()
                if len(protected_phrase) >= 3 and any(c.isalpha() for c in protected_phrase):
                    fallback_words.append(protected_phrase)
                continue

            fallback_words.extend(
                w.strip().lower()
                for w in re.split(r'[\s,;]+', segment_text)
                if self._is_valuable_token(w.strip())
            )
            if len(fallback_words) >= 8:
                break

        fallback_words = fallback_words[:8]
        if not fallback_words:
            return None

        fallback_id = "prompt_" + md5("|".join(fallback_words).encode()).hexdigest()[:10]
        if fallback_id not in concept_clusters:
            concept_clusters[fallback_id] = self._default_concept_cluster(fallback_words)
            concept_clusters[fallback_id]["label"] = " ".join(fallback_words[:5])
        else:
            concept_clusters[fallback_id] = self._ensure_concept_cluster_defaults(concept_clusters[fallback_id])

        return fallback_id

    def _force_feedback_fallback(self, ordered_concept_ids: list, concept_clusters: dict,
                                 concept_groups: dict, current_concept_labels: dict,
                                 rating_shift: float, similarity: float):
        if not ordered_concept_ids:
            return None

        ranked = []
        for cid in ordered_concept_ids:
            cluster = concept_clusters.get(cid)
            if not cluster:
                continue
            cluster = self._ensure_concept_cluster_defaults(cluster)
            concept_clusters[cid] = cluster
            mean_imp = self._get_concept_mean_importance(cluster)
            dominance = abs(mean_imp - cluster.get("presence_target", 1.0))
            ranked.append((dominance, cid, cluster))

        if not ranked:
            return None

        _, chosen_cid, cluster = max(ranked, key=lambda x: x[0])
        if rating_shift >= 2.0:
            question_type = "fidelity"
        elif similarity < 0.84:
            question_type = "stability"
        elif cluster.get("category") in {"style", "camera", "quality"}:
            question_type = "balance"
        else:
            question_type = "presence"

        chosen_group_id = None
        for gid, group in concept_groups.items():
            if chosen_cid in group.get("concept_ids", []):
                chosen_group_id = gid
                break

        return {
            "concept_id": chosen_cid,
            "concept_label": current_concept_labels.get(chosen_cid, cluster.get("label", "")),
            "question_type": question_type,
            "neighbor_ids": self._get_concept_neighbors(chosen_cid, ordered_concept_ids, radius=2),
            "group_id": chosen_group_id,
            "category": cluster.get("category", "general"),
            "score": 0.0,
        }

    def _match_concept(self, phrase_words: list, concept_clusters: dict, threshold: float = 0.38):
        """
        Match a phrase to an existing concept cluster using Jaccard similarity on
        anchor words. If no cluster clears `threshold`, a new cluster id is minted.

        Returns (cluster_id, is_new_cluster).
        Tune threshold: lower -> more isolated clusters; higher -> more merging.
        """
        if not phrase_words:
            return None, False
        phrase_set = set(phrase_words)
        best_id, best_score = None, 0.0
        for cid, cluster in concept_clusters.items():
            anchor_set = set(cluster.get("anchor_words", []))
            if not anchor_set:
                continue
            union = len(phrase_set | anchor_set)
            if union == 0:
                continue
            jaccard = len(phrase_set & anchor_set) / union
            if jaccard > best_score:
                best_score, best_id = jaccard, cid
        if best_score >= threshold:
            return best_id, False
        new_id = md5("|".join(sorted(phrase_words)).encode()).hexdigest()[:10]
        return new_id, True

    def _apply_category_feedback(self, cluster: dict, feedback_rating: int, iter_num: int = 0):
        category = CATEGORY_FEEDBACK_MAP.get(int(feedback_rating), "general")
        cluster["category"] = category
        cluster["category_source"] = "user"
        cluster["category_confidence"] = 1.0

        if category in {"concept", "character", "details"}:
            cluster["presence_target"] = self._clip_profile_value(cluster.get("presence_target", 1.0) + 0.10)
            cluster["priority_weight"] = self._clip_profile_value(cluster.get("priority_weight", 1.0) + 0.10)
            cluster["semantic_fidelity"] = self._clip_profile_value(
                cluster.get("semantic_fidelity", 1.0) + 0.08,
                low=0.6,
                high=1.8,
            )
        elif category == "quality":
            cluster["priority_weight"] = self._clip_profile_value(cluster.get("priority_weight", 1.0) + 0.06)
            cluster["overrep_sensitivity"] = self._clip_profile_value(cluster.get("overrep_sensitivity", 1.0) + 0.08)
        elif category == "style":
            cluster["stability_weight"] = self._clip_profile_value(cluster.get("stability_weight", 1.0) + 0.08)
            cluster["user_affinity"] = self._clip_profile_value(cluster.get("user_affinity", 1.0) + 0.05)

        cluster["question_history"] = list(cluster.get("question_history", []))[-23:]
        cluster["question_history"].append({
            "iteration": iter_num,
            "type": "category",
            "rating": feedback_rating,
            "category": category,
        })
        cluster["last_question_type"] = "category"
        cluster["last_question_iter"] = iter_num

    def _build_word_concept_map(self, prompt: str, concept_clusters: dict):
        """
        Parse the prompt into concept phrases (Level 3), match or create clusters,
        and return:
          - word_to_concept: dict mapping each significant word -> cluster_id
          - ordered_concept_ids: list of cluster IDs in prompt order (no duplicates)

        Updates concept_clusters in-place. The ordered list is the backbone for
        group building and neighbour lookup at Level 4.
        """
        word_to_concept = {}
        ordered_concept_ids = []
        current_concept_labels = {}
        for phrase_words in self._parse_concepts(prompt):
            cid, is_new = self._match_concept(phrase_words, concept_clusters)
            if cid is None:
                continue
            phrase_label = " ".join(phrase_words[:6])
            if is_new:
                concept_clusters[cid] = self._default_concept_cluster(phrase_words)
            else:
                concept_clusters[cid] = self._ensure_concept_cluster_defaults(concept_clusters[cid])
                # Expand anchor vocabulary with words newly seen in this phrase
                existing = set(concept_clusters[cid]["anchor_words"])
                for w in phrase_words:
                    if w not in existing:
                        concept_clusters[cid]["anchor_words"].append(w)
                if (
                    concept_clusters[cid].get("category_source") != "user" and
                    concept_clusters[cid].get("category") == "general"
                ):
                    concept_clusters[cid]["category"] = self._infer_concept_category(
                        concept_clusters[cid]["anchor_words"]
                    )
                    concept_clusters[cid]["category_confidence"] = (
                        0.35 if concept_clusters[cid]["category"] == "general" else 0.65
                    )
            concept_clusters[cid]["last_prompt_label"] = phrase_label
            current_concept_labels[cid] = phrase_label
            for w in phrase_words:
                word_to_concept[w] = cid
            if cid not in ordered_concept_ids:
                ordered_concept_ids.append(cid)
        return word_to_concept, ordered_concept_ids, current_concept_labels

    def _build_concept_groups(self, ordered_concept_ids: list, concept_clusters: dict,
                               existing_groups: dict, current_concept_labels: dict,
                               window: int = 3):
        """
        Level 4: group consecutive concept phrases into semantic sentence-level
        units using a non-overlapping sliding window of size `window`.

        Groups model the "sentences" of the prompt — natural semantic blocks
        like (quality tags), (subject + appearance), (setting), (style + tech).

        Existing groups are matched by their concept_ids set and updated in-place
        to preserve their accumulated reward_ema and usage_count. New groups are
        created when a previously unseen concept combination appears.

        Returns the updated groups dict.
        """
        if not ordered_concept_ids:
            return existing_groups

        groups = dict(existing_groups)
        for i in range(0, len(ordered_concept_ids), window):
            chunk = ordered_concept_ids[i:i + window]
            if not chunk:
                continue
            chunk_set = frozenset(chunk)

            # Try to match an existing group by concept set identity
            existing_gid = None
            for gid, g in groups.items():
                if frozenset(g.get("concept_ids", [])) == chunk_set:
                    existing_gid = gid
                    break

            if existing_gid:
                # Preserve the canonical prompt order even if it shifted slightly
                groups[existing_gid]["concept_ids"] = chunk
                groups[existing_gid]["label"] = " | ".join(
                    current_concept_labels.get(cid, concept_clusters.get(cid, {}).get("label", cid))
                    for cid in chunk
                )
            else:
                labels = [current_concept_labels.get(cid, concept_clusters[cid]["label"])
                          for cid in chunk if cid in concept_clusters]
                gid = md5("|".join(chunk).encode()).hexdigest()[:10]
                groups[gid] = {
                    "label": " | ".join(labels),
                    "concept_ids": chunk,
                    "reward_ema": 0.0,
                    "usage_count": 0,
                    "last_seen_iter": 0,
                }

        return groups

    def _get_concept_neighbors(self, concept_id: str, ordered_concept_ids: list,
                                radius: int = 2):
        """
        Return the concept IDs positionally adjacent to `concept_id` within
        `radius` steps in the prompt's ordered concept list.

        Adjacent concepts compete for embedding space: if "anime style" is
        overrepresented, it likely crowds its neighbours "cinematic lighting"
        and "detailed background". This adjacency is what makes the neighbour
        signal in _apply_concept_feedback meaningful.
        """
        if concept_id not in ordered_concept_ids:
            return []
        idx = ordered_concept_ids.index(concept_id)
        neighbors = []
        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue
            ni = idx + offset
            if 0 <= ni < len(ordered_concept_ids):
                neighbors.append(ordered_concept_ids[ni])
        return neighbors

    def _get_dominant_concept(self, ordered_concept_ids: list, concept_clusters: dict):
        """
        Return (cluster_id, avg_importance, label) for the concept with the
        highest mean word importance — i.e. what the embedding currently weighs
        most heavily. This is the "main thing forming the video" signal.
        """
        best_cid, best_score, best_label = None, -1.0, ""
        for cid in ordered_concept_ids:
            if cid not in concept_clusters:
                continue
            imp_vals = list(concept_clusters[cid]["word_importance"].values())
            if not imp_vals:
                continue
            avg = sum(imp_vals) / len(imp_vals)
            if avg > best_score:
                best_score, best_cid, best_label = avg, cid, concept_clusters[cid]["label"]
        return best_cid, best_score, best_label

    def _apply_concept_feedback(self, concept_id: str, feedback_rating: int,
                                 question_type: str,
                                 concept_clusters: dict, neighbor_ids: list,
                                 word_importance: dict, concept_groups: dict,
                                 iter_num: int = 0):
        """
        Multi-level feedback propagation.

        Level 3 — rated concept phrase:
          All word importances inside the rated cluster shift by `direct_delta`.
          This is the primary, high-confidence signal.

        Level 3 — neighbour concepts (adjacent phrases in the prompt):
          Receive a dampened signal modelled on embedding-space competition.
          - Concept being boosted (absent/weak): neighbours get a small inhibitory
            nudge, giving the boosted concept more semantic room.
          - Concept being reduced (overrepresented): neighbours also receive a mild
            reduction — the whole semantic area is too heavy.
          Magnitude: 22% of direct_delta; same sign for overrep, opposite for absent.

        Level 4 — concept groups:
          The reward_ema of every group containing the rated concept is updated,
          enabling group-level health tracking over time.

        Level 2 — global word_importance fallback:
          Each change is also written at 40% strength into the flat global dict
          so the scheduler and prodigy system retain a valid signal.

        Feedback scale:
          1 = absent           -> direct_delta = +0.90 (strong boost)
          2 = weak             -> direct_delta = +0.50
          3 = slightly weak    -> direct_delta = +0.20
          4 = slightly strong  -> direct_delta = -0.15
          5 = perfect          -> direct_delta = +0.04 (small stability reward)
          6 = overrepresented  -> direct_delta = -0.55 (strong reduction)
        """
        if concept_id not in concept_clusters:
            return

        cluster = self._ensure_concept_cluster_defaults(concept_clusters[concept_id])
        concept_clusters[concept_id] = cluster

        question_type = question_type or "presence"
        if question_type == "category":
            self._apply_category_feedback(cluster, feedback_rating, iter_num)
            return

        direct_deltas = {
            "presence": {1: 0.90, 2: 0.50, 3: 0.20, 4: -0.15, 5: 0.04, 6: -0.55},
            "priority": {1: -0.25, 2: -0.12, 3: -0.05, 4: 0.08, 5: 0.18, 6: 0.30},
            "balance": {1: 0.72, 2: 0.40, 3: 0.16, 4: -0.10, 5: 0.00, 6: -0.62},
            "fidelity": {1: 0.25, 2: 0.16, 3: 0.08, 4: 0.02, 5: 0.00, 6: -0.06},
            "stability": {1: 0.10, 2: 0.06, 3: 0.03, 4: 0.00, 5: -0.02, 6: -0.06},
            "preference": {1: -0.30, 2: -0.18, 3: -0.08, 4: 0.10, 5: 0.22, 6: 0.36},
        }
        direct_delta = direct_deltas.get(question_type, direct_deltas["presence"]).get(feedback_rating, 0.0)
        centered = (feedback_rating - 3.5) / 2.5

        # --- Level 3: rated concept ---
        local_imp = cluster["word_importance"]
        for wkey in list(local_imp.keys()):
            local_imp[wkey] = max(0.3, min(2.8, local_imp[wkey] + direct_delta))
            if wkey in word_importance:
                word_importance[wkey] = max(0.3, min(2.8, word_importance[wkey] + direct_delta * 0.4))

        if question_type == "presence":
            cluster["presence_target"] = self._clip_profile_value(
                cluster.get("presence_target", 1.0) + direct_delta * 0.22
            )
        elif question_type == "priority":
            cluster["priority_weight"] = self._clip_profile_value(
                cluster.get("priority_weight", 1.0) + centered * 0.18
            )
        elif question_type == "balance":
            cluster["overrep_sensitivity"] = self._clip_profile_value(
                cluster.get("overrep_sensitivity", 1.0) + max(0.0, centered) * 0.24 - max(0.0, -centered) * 0.12
            )
            cluster["presence_target"] = self._clip_profile_value(
                cluster.get("presence_target", 1.0) + direct_delta * 0.12
            )
        elif question_type == "fidelity":
            cluster["semantic_fidelity"] = self._clip_profile_value(
                cluster.get("semantic_fidelity", 1.0) + (centered * 0.16),
                low=0.6, high=1.8
            )
        elif question_type == "stability":
            cluster["stability_weight"] = self._clip_profile_value(
                cluster.get("stability_weight", 1.0) + centered * 0.18
            )
        elif question_type == "preference":
            cluster["user_affinity"] = self._clip_profile_value(
                cluster.get("user_affinity", 1.0) + centered * 0.20
            )
            cluster["priority_weight"] = self._clip_profile_value(
                cluster.get("priority_weight", 1.0) + centered * 0.08
            )
            cluster["presence_target"] = self._clip_profile_value(
                cluster.get("presence_target", 1.0) + centered * 0.10
            )

        # --- Level 3: neighbour concepts ---
        # Boosted concept -> inhibit neighbours (give it semantic space).
        # Reduced concept -> also soften neighbours (the whole area is too heavy).
        if direct_delta > 0:
            neighbor_delta = -direct_delta * 0.22
        else:
            neighbor_delta = direct_delta * 0.22

        for nid in neighbor_ids:
            if nid not in concept_clusters or nid == concept_id:
                continue
            concept_clusters[nid] = self._ensure_concept_cluster_defaults(concept_clusters[nid])
            n_local = concept_clusters[nid]["word_importance"]
            for wkey in list(n_local.keys()):
                n_local[wkey] = max(0.3, min(2.8, n_local[wkey] + neighbor_delta))
                if wkey in word_importance:
                    word_importance[wkey] = max(0.3, min(2.8,
                                                word_importance[wkey] + neighbor_delta * 0.4))

        # --- Level 4: concept group reward EMA ---
        # Maps feedback 1-6 onto a [-1, +1] reward signal.
        group_reward_map = {1: -1.0, 2: -0.5, 3: -0.1, 4: 0.4, 5: 1.0, 6: -0.6}
        normalized = group_reward_map.get(feedback_rating, 0.0)
        for gid, g in concept_groups.items():
            if concept_id in g.get("concept_ids", []):
                g["reward_ema"] = 0.75 * g.get("reward_ema", 0.0) + 0.25 * normalized
                g["usage_count"] = g.get("usage_count", 0) + 1

        cluster["question_history"] = list(cluster.get("question_history", []))[-23:]
        cluster["question_history"].append({
            "iteration": iter_num,
            "type": question_type,
            "rating": feedback_rating,
        })
        cluster["last_question_type"] = question_type
        cluster["last_question_iter"] = iter_num

    # =========================================================================
    # SIGMA REFINEMENT
    # =========================================================================

    def _ensure_sigma_state_defaults(self, global_adaptive: dict):
        global_adaptive.setdefault("sigma_profile", [0.0] * 32)
        global_adaptive.setdefault("last_applied_sigma_profile", [0.0] * 32)
        global_adaptive.setdefault("sigma_iterations", 0)
        global_adaptive.setdefault("sigma_avg_reward_ema", 0.0)
        global_adaptive.setdefault("sigma_exploration_base", 0.035)
        global_adaptive.setdefault("sigma_history", [])

    def _resolve_sigma_strength(self, strength_mode: str):
        strength_map = {
            "off": 0.0,
            "subtle": 0.18,
            "medium": 0.35,
            "strong": 0.60,
            "max": 1.0,
        }
        return strength_map.get((strength_mode or "subtle").lower(), 0.18)

    def _sigma_resample_profile(self, profile, target_len):
        if target_len <= 0:
            return np.zeros((0,), dtype=np.float32)

        arr = np.asarray(profile, dtype=np.float32)
        if arr.size == 0:
            arr = np.zeros((32,), dtype=np.float32)
        if arr.size == target_len:
            return arr.copy()
        if arr.size == 1:
            return np.full((target_len,), float(arr[0]), dtype=np.float32)

        src = np.linspace(0.0, 1.0, arr.size)
        dst = np.linspace(0.0, 1.0, target_len)
        return np.interp(dst, src, arr).astype(np.float32)

    def _sigma_smooth_noise(self, noise):
        if noise.size <= 2:
            return noise
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        smoothed = np.convolve(noise, kernel, mode="same")
        smoothed = np.convolve(smoothed, kernel, mode="same")
        return smoothed.astype(np.float32)

    def _sigma_enforce_monotonic(self, tuned: torch.Tensor, original: torch.Tensor):
        out = tuned.clone()
        count = int(out.shape[0])
        if count <= 2:
            return out

        eps = max(1e-6, float((original[0] - original[-1]).abs().item()) * 1e-6)

        for i in range(1, count - 1):
            max_allowed = out[i - 1] - eps
            if out[i] >= max_allowed:
                out[i] = max_allowed

        out[-1] = original[-1]
        for i in range(count - 2, 0, -1):
            min_allowed = out[i + 1] + eps
            if out[i] <= min_allowed:
                out[i] = min_allowed

        out[0] = original[0]
        out[-1] = original[-1]
        return out

    def _refine_sigma_schedule(self, sigmas, rating: int, global_adaptive: dict, strength_mode: str, seed: int,
                               rating_profile: Optional[dict] = None):
        if not isinstance(sigmas, torch.Tensor):
            return torch.FloatTensor([]), "Sigma refinement inactive."

        original_sigmas = sigmas.detach().clone()
        if original_sigmas.numel() <= 2:
            return original_sigmas, "Sigma refinement skipped: schedule too short."

        self._ensure_sigma_state_defaults(global_adaptive)

        rating_profile = rating_profile or normalize_refiner_rating(rating)
        rating_key = rating_profile.get("key", "")
        reward = float(rating_profile.get("reward", (rating - 5.5) / 4.5))
        sigma_iterations = int(global_adaptive.get("sigma_iterations", 0))

        profile = np.asarray(global_adaptive.get("sigma_profile", [0.0] * 32), dtype=np.float32)
        last_applied = np.asarray(global_adaptive.get("last_applied_sigma_profile", [0.0] * 32), dtype=np.float32)
        sigma_lr = 0.22 / (1.0 + 0.08 * math.sqrt(max(1, sigma_iterations)))
        profile = np.clip(profile + (last_applied * reward * sigma_lr), -1.0, 1.0)
        profile = np.clip(profile - profile.mean(), -1.0, 1.0)

        sigma_avg_reward_ema = float(global_adaptive.get("sigma_avg_reward_ema", 0.0))
        sigma_avg_reward_ema = 0.85 * sigma_avg_reward_ema + 0.15 * reward
        sigma_exploration_base = float(global_adaptive.get("sigma_exploration_base", 0.035))
        sigma_exploration_scale = max(0.18, 1.0 - max(0.0, sigma_avg_reward_ema) * 0.7)

        if seed != 0:
            np.random.seed(seed % (2 ** 32))
        noise = np.random.normal(
            0.0,
            sigma_exploration_base * sigma_exploration_scale,
            size=profile.shape[0]
        ).astype(np.float32)
        applied_profile = np.clip(profile + self._sigma_smooth_noise(noise), -1.0, 1.0)
        applied_profile = np.clip(applied_profile - applied_profile.mean(), -1.0, 1.0)

        middle_profile = self._sigma_resample_profile(applied_profile, max(0, original_sigmas.shape[0] - 2))
        middle_profile_tensor = torch.tensor(middle_profile, dtype=original_sigmas.dtype, device=original_sigmas.device)

        tuned_sigmas = original_sigmas.clone()
        sigma_strength = self._resolve_sigma_strength(strength_mode)
        for idx in range(1, int(original_sigmas.shape[0]) - 1):
            delta = float(middle_profile_tensor[idx - 1].item())
            current = original_sigmas[idx]
            prev_sigma = original_sigmas[idx - 1]
            next_sigma = original_sigmas[idx + 1]
            if delta >= 0.0:
                tuned_sigmas[idx] = current + (prev_sigma - current) * sigma_strength * delta
            else:
                tuned_sigmas[idx] = current + (current - next_sigma) * sigma_strength * delta

        tuned_sigmas = self._sigma_enforce_monotonic(tuned_sigmas, original_sigmas)

        global_adaptive["sigma_profile"] = profile.tolist()
        global_adaptive["last_applied_sigma_profile"] = applied_profile.tolist()
        global_adaptive["sigma_iterations"] = sigma_iterations + 1
        global_adaptive["sigma_avg_reward_ema"] = sigma_avg_reward_ema
        global_adaptive["sigma_exploration_base"] = max(
            0.012,
            min(0.05, sigma_exploration_base * (0.97 if rating_key == "like" else 1.02))
        )
        sigma_history = list(global_adaptive.get("sigma_history", []))[-119:]
        sigma_history.append({
            "iteration": sigma_iterations + 1,
            "rating": int(rating),
            "rating_label": rating_profile.get("label", str(rating)),
            "reward": round(float(reward), 6),
            "lr": round(float(sigma_lr), 6),
        })
        global_adaptive["sigma_history"] = sigma_history

        mean_shift = float((tuned_sigmas[1:-1] - original_sigmas[1:-1]).abs().mean().item()) if tuned_sigmas.numel() > 2 else 0.0
        max_shift = float((tuned_sigmas[1:-1] - original_sigmas[1:-1]).abs().max().item()) if tuned_sigmas.numel() > 2 else 0.0
        sigma_status = (
            f"Sigma: iter {global_adaptive['sigma_iterations']} | strength {strength_mode} ({sigma_strength:.2f}) | "
            f"mean shift {mean_shift:.6f} | max shift {max_shift:.6f} | "
            f"endpoints preserved ({float(original_sigmas[0].item()):.6f} -> {float(original_sigmas[-1].item()):.6f})"
        )
        return tuned_sigmas, sigma_status

    # =========================================================================
    # LATENT REFINEMENT
    # =========================================================================

    def _resize_tensor_like(self, tensor: torch.Tensor, reference: torch.Tensor):
        if not isinstance(tensor, torch.Tensor) or not isinstance(reference, torch.Tensor):
            return None

        out = tensor.to(device=reference.device, dtype=reference.dtype)
        if list(out.shape) == list(reference.shape):
            return out
        if out.dim() != reference.dim() or out.dim() not in {4, 5}:
            return None

        batch = reference.shape[0]
        if out.shape[0] < batch:
            reps = [1] * out.dim()
            reps[0] = math.ceil(batch / max(1, out.shape[0]))
            out = out.repeat(*reps)
        out = out[:batch]

        channels = reference.shape[1]
        if out.shape[1] < channels:
            pad_shape = list(out.shape)
            pad_shape[1] = channels - out.shape[1]
            out = torch.cat([out, torch.zeros(pad_shape, device=out.device, dtype=out.dtype)], dim=1)
        out = out[:, :channels]

        if out.dim() == 4:
            return F.interpolate(out, size=reference.shape[-2:], mode="bilinear", align_corners=False)
        return F.interpolate(out, size=reference.shape[-3:], mode="trilinear", align_corners=False)

    def _load_saved_latent(self, refinement_key, mode):
        path = refinement_state_path(refinement_key, mode, prefix="latent", extension="pt")
        if not os.path.exists(path):
            return None, "Latent: no saved latent for this key."

        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            return None, f"Latent: failed to load saved latent ({e})."

        if not isinstance(data, dict) or not isinstance(data.get("samples"), torch.Tensor):
            return None, "Latent: saved latent file is invalid."

        return data, "Latent: saved latent loaded."

    def _value_links_to_output(self, value, source_id, output_index):
        if isinstance(value, (list, tuple)):
            if len(value) >= 2 and str(value[0]) == source_id:
                try:
                    if int(value[1]) == output_index:
                        return True
                except (TypeError, ValueError):
                    pass
            return any(self._value_links_to_output(item, source_id, output_index) for item in value)

        if isinstance(value, dict):
            return any(self._value_links_to_output(item, source_id, output_index) for item in value.values())

        return False

    def _is_output_connected(self, prompt, unique_id, output_index):
        if prompt is None or unique_id is None:
            return False

        source_id = str(unique_id)
        prompt_nodes = prompt.get("output", prompt) if isinstance(prompt, dict) else {}
        if not isinstance(prompt_nodes, dict):
            return False

        for node in prompt_nodes.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            if isinstance(inputs, dict) and self._value_links_to_output(inputs, source_id, output_index):
                return True

        workflow = prompt.get("workflow") if isinstance(prompt, dict) else None
        links = workflow.get("links", []) if isinstance(workflow, dict) else []
        for link in links:
            if isinstance(link, (list, tuple)) and len(link) >= 3:
                try:
                    if str(link[1]) == source_id and int(link[2]) == output_index:
                        return True
                except (TypeError, ValueError):
                    continue
            elif isinstance(link, dict):
                try:
                    if str(link.get("origin_id")) == source_id and int(link.get("origin_slot")) == output_index:
                        return True
                except (TypeError, ValueError):
                    continue

        return False

    def _latent_refinement_disabled(self, latent):
        return clone_latent(latent), "Latent: disabled because latent refinement input/output path is not fully connected."

    def _raise_wrong_latent(self, latent):
        sample_type = latent_sample_type_name(latent)
        latent_type = latent.get("type", "unspecified") if isinstance(latent, dict) else "missing"
        raise ValueError(f"{self.WRONG_LATENT_ERROR} Received samples={sample_type}, type={latent_type}.")

    def _save_latent_reference(self, latent, refinement_key, mode):
        samples = latent_samples(latent)
        if samples is None:
            return False

        path = refinement_state_path(refinement_key, mode, prefix="latent", extension="pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        bundle = cpu_tensor_bundle(latent)
        bundle["_meta"] = {
            "refinement_key": refinement_key,
            "mode": (mode or "ltx2").lower(),
            "samples_shape": list(samples.shape),
        }
        torch.save(bundle, path)
        return True

    def _delete_latent_reference(self, refinement_key, mode):
        path = refinement_state_path(refinement_key, mode, prefix="latent", extension="pt")
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return False
        except OSError:
            return False

    def _refine_latent(self, latent, refinement_key, mode, rating, reward, global_adaptive,
                       rating_profile: Optional[dict] = None):
        if isinstance(latent, dict) and not latent_is_plain_video_tensor(latent):
            self._raise_wrong_latent(latent)

        current = latent_samples(latent)
        saved_bundle, load_status = self._load_saved_latent(refinement_key, mode)

        if current is None:
            if saved_bundle is None:
                raise ValueError(self.NO_LATENT_REFERENCE_ERROR)

            saved_latent = latent_from_tensor_bundle(saved_bundle)
            saved_samples = latent_samples(saved_latent)
            if saved_samples is None:
                raise ValueError(self.NO_LATENT_REFERENCE_ERROR)

            latent_state = global_adaptive.setdefault("latent_refinement", {})
            latent_state["last_shape"] = list(saved_samples.shape)
            latent_state["saved_shape"] = list(saved_samples.shape)
            latent_state["status"] = self.SAVED_LATENT_ONLY_STATUS
            return saved_latent, latent_state["status"]

        refined_latent = clone_latent(latent)
        if saved_bundle is None:
            self._save_latent_reference(latent, refinement_key, mode)
            global_adaptive.setdefault("latent_refinement", {})["last_shape"] = list(current.shape)
            return refined_latent, f"{load_status} Current latent saved as reference."

        raw_saved_samples = saved_bundle.get("samples")
        if list(raw_saved_samples.shape) != list(current.shape):
            self._save_latent_reference(latent, refinement_key, mode)
            latent_state = global_adaptive.setdefault("latent_refinement", {})
            latent_state.pop("momentum", None)
            latent_state["last_shape"] = list(current.shape)
            latent_state["saved_shape"] = list(raw_saved_samples.shape)
            return refined_latent, "Latent: input shape changed. Reference rewritten, passthrough."

        saved_samples = raw_saved_samples.to(device=current.device, dtype=current.dtype)
        latent_state = global_adaptive.setdefault("latent_refinement", {})
        latent_state["last_shape"] = list(current.shape)
        latent_state["saved_shape"] = list(raw_saved_samples.shape)

        momentum_data = latent_state.get("momentum")
        if isinstance(momentum_data, dict):
            try:
                momentum = serializable_to_tensor(momentum_data).to(device=current.device, dtype=current.dtype)
                momentum = self._resize_tensor_like(momentum, current)
            except Exception:
                momentum = None
        else:
            momentum = None
        if momentum is None:
            momentum = torch.zeros_like(current)

        nonzero_mask = current.ne(0) & saved_samples.ne(0)
        if not bool(nonzero_mask.any()):
            refined_latent["samples"] = current
            latent_state["status"] = "Latent: only zero-valued positions available, passthrough."
            return refined_latent, latent_state["status"]

        rating_profile = rating_profile or normalize_refiner_rating(rating)
        rating_key = rating_profile.get("key", "")
        missing_axes = set(rating_profile.get("missing_axes", []))
        if rating_key == "like":
            lr = 0.035
        elif missing_axes:
            lr = 0.018
            if "details" in missing_axes:
                lr += 0.006
            if "concept" in missing_axes:
                lr += 0.012
            if "quality" in missing_axes:
                lr += 0.035
            lr = min(0.065, lr)
        elif rating_key == "dislike":
            lr = 0.055
        else:
            lr = 0.018
        target_delta = (saved_samples - current) * nonzero_mask.to(dtype=current.dtype)
        update = target_delta * (reward * lr)
        momentum = 0.82 * momentum + 0.18 * update
        max_step = max(0.001, min(0.08, float(current.detach().abs().mean().item()) * 0.18))
        latent_delta = torch.clamp(momentum, min=-max_step, max=max_step)

        refined_samples = current + latent_delta
        refined_samples = torch.where(nonzero_mask, refined_samples, current)
        refined_samples = torch.nan_to_num(refined_samples, nan=0.0)
        range_min = torch.minimum(current, saved_samples)
        range_max = torch.maximum(current, saved_samples)
        refined_samples = torch.maximum(torch.minimum(refined_samples, range_max), range_min)
        refined_latent["samples"] = refined_samples

        latent_state["momentum"] = tensor_to_serializable(momentum.detach().cpu())
        latent_state["last_rating"] = int(rating)
        latent_state["last_rating_label"] = rating_profile.get("label", str(rating))
        latent_state["last_reward"] = round(float(reward), 4)
        latent_state["nonzero_ratio"] = round(float(nonzero_mask.float().mean().item()), 6)
        latent_state["status"] = (
            f"Latent: adjusted shape {tuple(current.shape)} | nonzero {latent_state['nonzero_ratio']:.1%} | "
            f"max step {max_step:.5f} | bounded to input/reference range"
        )
        return refined_latent, latent_state["status"]

    # =========================================================================
    # LORA WEIGHT SUGGESTIONS
    # =========================================================================

    def _lora_state_id(self, lora_name: str, lora_type: str):
        return md5(f"{lora_name}::{lora_type}".encode("utf-8")).hexdigest()[:16]

    def _lora_words(self, text: str):
        return {
            word.strip().lower()
            for word in re.split(r"[\s,;:_\\/\-().\[\]{}]+", text or "")
            if self._is_valuable_token(word.strip())
        }

    def _ensure_lora_memory(self, memory: dict, lora_entry: dict):
        lora_type = lora_entry.get("type", "general")
        lora_id = lora_entry.get("id") or self._lora_state_id(lora_entry.get("name", ""), lora_type)
        state = memory.setdefault(
            lora_id,
            {
                "name": lora_entry.get("name", ""),
                "type": lora_type,
                "offset_ratio": 0.0,
                "stable_offset_ratio": None,
                "reward_ema": 0.0,
                "culprit_score": 0.0,
                "good_streak": 0,
                "bad_streak": 0,
                "culprit_hits": 0,
                "iterations": 0,
            },
        )
        state["name"] = lora_entry.get("name", "")
        state["type"] = lora_type
        state.setdefault("offset_ratio", 0.0)
        state.setdefault("stable_offset_ratio", None)
        state.setdefault("reward_ema", 0.0)
        state.setdefault("culprit_score", 0.0)
        state.setdefault("good_streak", 0)
        state.setdefault("bad_streak", 0)
        state.setdefault("culprit_hits", 0)
        state.setdefault("iterations", 0)
        return lora_id, state

    def _score_lora_prompt_relation(self, lora_entry, ordered_concept_ids, concept_clusters, current_concept_labels):
        lora_type = lora_entry.get("type", "general")
        lora_words = self._lora_words(lora_entry.get("name", ""))

        best_score = 0.35 if lora_type == "general" else 0.0
        best_labels = []
        best_importance = 1.0

        for cid in ordered_concept_ids:
            cluster = concept_clusters.get(cid)
            if not cluster:
                continue
            cluster = self._ensure_concept_cluster_defaults(cluster)
            label = current_concept_labels.get(cid, cluster.get("label", cid))
            label_words = self._lora_words(label)
            anchor_words = set(cluster.get("anchor_words", []))
            prompt_words = label_words | anchor_words
            category = cluster.get("category", "general")

            overlap = len(lora_words & prompt_words) / max(1, len(lora_words)) if lora_words else 0.0
            category_score = 0.62 if lora_type == category else 0.0
            if lora_type in {"action", "concept"} and category in {"concept", "details", "subject", "appearance", "action", "environment"}:
                category_score = max(category_score, 0.58)
            if lora_type == "character" and category in {"character", "subject", "appearance"}:
                category_score = max(category_score, 0.72)
            if lora_type == "quality" and category == "quality":
                category_score = 0.78
            if lora_type == "style" and category in {"style", "camera"}:
                category_score = max(category_score, 0.58)

            score = max(category_score, overlap)
            if score > best_score:
                best_score = score
                best_labels = [label]
                best_importance = self._get_concept_mean_importance(cluster)
            elif score > 0 and abs(score - best_score) < 1e-6:
                best_labels.append(label)

        return min(1.0, best_score), best_labels[:4], best_importance

    def _update_lora_weight_suggestions(self, lora_stack, active, global_adaptive, ordered_concept_ids,
                                        concept_clusters, current_concept_labels, rating, reward,
                                        rating_profile: Optional[dict] = None):
        if not isinstance(lora_stack, dict) or not lora_stack.get("loras"):
            return "LoRA suggestions: no FunPack LoRA stack connected."

        rating_profile = rating_profile or normalize_refiner_rating(rating)
        rating_key = rating_profile.get("key", "")
        missing_axes = set(rating_profile.get("missing_axes", []))
        missing_count = len(missing_axes)
        memory = global_adaptive.setdefault("lora_weight_memory", {})
        suggestions = {}
        status_parts = []
        relation_cache = {}
        top_concept_relation = 0.0
        concept_lora_count = 0

        for entry in lora_stack.get("loras", []):
            lora_type = entry.get("type", "general")
            lora_id = entry.get("id") or self._lora_state_id(entry.get("name", ""), lora_type)
            relation, matched_labels, concept_importance = self._score_lora_prompt_relation(
                entry,
                ordered_concept_ids,
                concept_clusters,
                current_concept_labels,
            )
            relation_cache[lora_id] = (relation, matched_labels, concept_importance)
            if lora_type in {"action", "concept", "character"}:
                concept_lora_count += 1
                top_concept_relation = max(top_concept_relation, relation)

        for entry in lora_stack.get("loras", []):
            lora_type = entry.get("type", "general")
            profile = LORA_REFINER_TYPE_PROFILES.get(lora_type, LORA_REFINER_TYPE_PROFILES["general"])
            lora_id, state = self._ensure_lora_memory(memory, entry)
            relation, matched_labels, concept_importance = relation_cache.get(lora_id, (0.0, [], 1.0))

            state["reward_ema"] = 0.84 * float(state.get("reward_ema", 0.0)) + 0.16 * reward
            if rating_key == "like":
                state["good_streak"] = int(state.get("good_streak", 0)) + 1
                state["bad_streak"] = 0
            elif rating_key == "dislike" or {"concept", "quality"}.issubset(missing_axes):
                state["bad_streak"] = int(state.get("bad_streak", 0)) + 1
                state["good_streak"] = 0
            else:
                state["good_streak"] = 0
                state["bad_streak"] = 0

            offset = float(state.get("offset_ratio", 0.0))
            stable_offset = state.get("stable_offset_ratio")
            culprit_score = float(state.get("culprit_score", 0.0))
            culprit_hits = int(state.get("culprit_hits", 0))
            if stable_offset is not None and (rating_key == "like" or missing_axes == {"details"}):
                offset = 0.72 * offset + 0.28 * float(stable_offset)

            step = profile["step"] * max(0.15, relation)
            max_offset = profile["max_offset"]
            min_offset = profile["min_offset"]
            effective_relation = max(relation, profile.get("culprit_bias", 0.0))
            base_model = float(entry.get("base_model_weight", entry.get("model_weight", 1.0)))
            base_abs = abs(base_model)
            is_concept_lora = lora_type in {"action", "concept"}
            concept_match_strength = 1.0
            if is_concept_lora:
                concept_match_strength += 0.30 if matched_labels else 0.0
                concept_match_strength += max(0.0, relation - 0.30) * 1.35

            is_primary_concept_lora = (
                lora_type in {"action", "concept", "character"} and
                (
                    concept_lora_count == 1 or
                    (top_concept_relation > 0.0 and relation >= max(0.30, top_concept_relation - 1e-6))
                )
            )

            if rating_key == "like":
                value_mult = 0.75 + min(1.4, max(0.5, concept_importance)) * 0.25
                offset += step * (0.45 + max(0.0, reward)) * value_mult
                culprit_score *= 0.72
                if state["good_streak"] >= 3:
                    if stable_offset is None:
                        stable_offset = offset
                    else:
                        stable_offset = 0.78 * float(stable_offset) + 0.22 * offset
                    state["stable_offset_ratio"] = _clamp(stable_offset, min_offset, max_offset)
                    offset = state["stable_offset_ratio"]
            elif missing_axes:
                if "concept" in missing_axes or "quality" in missing_axes:
                    state["stable_offset_ratio"] = None

                culprit_score *= 0.74 if rating_key == "awful" else 0.82
                culprit_hits = max(0, culprit_hits - 1)
                axis_boost = 0.0

                if "details" in missing_axes:
                    if relation >= 0.12 and lora_type in {"action", "concept", "character", "general", "style"}:
                        value_mult = 0.85 + min(1.2, max(0.5, concept_importance)) * 0.20
                        axis_boost += (0.26 + relation * 0.34) * value_mult
                    elif lora_type == "quality":
                        axis_boost += 0.06

                if "concept" in missing_axes:
                    if lora_type in {"action", "concept", "character"}:
                        if is_primary_concept_lora:
                            axis_boost += (0.95 + abs(reward) * 0.45) * concept_match_strength
                        elif relation > 0.08:
                            axis_boost += 0.34 + relation * 0.40
                    elif lora_type == "general" and relation >= 0.25:
                        axis_boost += 0.25 + relation * 0.25

                if "quality" in missing_axes:
                    if lora_type == "quality":
                        axis_boost += 0.90 + abs(reward) * 0.35
                    elif lora_type == "style" and relation >= 0.25:
                        axis_boost += 0.20 + relation * 0.20
                    elif lora_type == "general" and relation >= 0.20:
                        axis_boost += 0.16 + relation * 0.15

                if axis_boost > 0.0:
                    if rating_key == "awful":
                        axis_boost *= 1.25
                    max_offset = max(max_offset, profile["bad_max_offset"] * (0.62 + 0.10 * missing_count))
                    relation_floor = 0.35 if missing_count >= 2 else 0.22
                    boost_step = profile["step"] * max(relation_floor, effective_relation)
                    offset += boost_step * axis_boost
                elif relation <= 0.05:
                    offset *= 0.97
                else:
                    offset *= 0.94
            elif rating_key == "dislike":
                severity = _clamp((5.0 - float(rating)) / 4.0, 0.0, 1.0)
                culprit_signal = max(0.20, effective_relation) * (0.70 + base_abs * 0.30)
                if is_concept_lora:
                    culprit_signal *= concept_match_strength
                culprit_score = _clamp(culprit_score * 0.72 + severity * culprit_signal, 0.0, 2.5)
                culprit_hits = culprit_hits + 1 if culprit_score >= 0.45 else max(0, culprit_hits - 1)
                max_offset = profile["bad_max_offset"] if rating <= 2 else max(max_offset, profile["bad_max_offset"] * 0.72)
                bad_floor_strength = min(1.0, 0.45 + 0.35 * culprit_score + 0.14 * state["bad_streak"])
                if is_concept_lora:
                    bad_floor_strength = min(1.0, bad_floor_strength + 0.18 * concept_match_strength)
                min_offset = min(min_offset, profile["bad_min_offset"] * bad_floor_strength)
                offset -= step * (1.0 + severity * 3.0) * max(0.4, effective_relation) * max(0.8, 0.85 + culprit_score)
                if state["bad_streak"] >= 2:
                    offset -= step * (0.65 + severity * 1.8) * max(0.25, effective_relation)
                if is_concept_lora and state["bad_streak"] >= 2:
                    offset -= step * (0.85 + severity * 2.4) * max(0.50, effective_relation) * concept_match_strength
                if state["bad_streak"] >= 2:
                    state["stable_offset_ratio"] = None
                if is_concept_lora and matched_labels and state["bad_streak"] >= 2 and culprit_score >= 0.70:
                    offset = min(offset, -1.0)
                if state["bad_streak"] >= 3 and culprit_score >= 0.85 and abs(1.0 + offset) < 0.08:
                    offset = -1.0
            else:
                culprit_score *= 0.92
                culprit_hits = max(0, culprit_hits - 1)
                if relation <= 0.05:
                    offset *= 0.94
                elif stable_offset is not None:
                    offset = 0.65 * offset + 0.35 * float(stable_offset)
                else:
                    offset *= 0.90

            offset = _clamp(offset, min_offset, max_offset)
            state["offset_ratio"] = offset
            state["culprit_score"] = culprit_score
            state["culprit_hits"] = culprit_hits
            state["iterations"] = int(state.get("iterations", 0)) + 1

            model_weight = base_model * (1.0 + offset)
            suspect = culprit_score >= 0.65 or model_weight <= 0.0 or culprit_hits >= 2
            action = (
                "invert" if model_weight < 0.0 else
                "mute" if model_weight == 0.0 else
                "hold" if abs(model_weight - base_model) < 1e-6 else
                "reduce" if abs(model_weight) < abs(base_model) else
                "boost"
            )
            suggestions[lora_id] = {
                "name": entry.get("name", ""),
                "type": lora_type,
                "model_weight": model_weight,
                "base_model_weight": base_model,
                "offset_ratio": offset,
                "culprit_score": culprit_score,
                "culprit_hits": culprit_hits,
                "suspect": suspect,
                "action": action,
                "relation": relation,
                "matched_concepts": matched_labels,
                "rating": int(rating),
                "rating_label": rating_profile.get("label", str(rating)),
                "rating_range": rating_profile.get("legacy_range", ""),
                "good_streak": state.get("good_streak", 0),
                "bad_streak": state.get("bad_streak", 0),
                "stable": state.get("stable_offset_ratio") is not None,
            }

            match_text = ",".join(matched_labels) if matched_labels else "none"
            status_parts.append(
                f"{entry.get('name', '?')}[{lora_type}] rel={relation:.2f} "
                f"offset={offset:+.3f} next={model_weight:+.3f} "
                f"sus={culprit_score:.2f}{' !' if suspect else ''} "
                f"match={match_text}"
            )

        active["lora_weight_suggestions"] = suggestions
        active["last_lora_stack"] = lora_stack
        if not status_parts:
            return "LoRA suggestions: stack empty."
        return f"LoRA suggestions ({rating_profile.get('label', str(rating))}): " + " | ".join(status_parts)

    # =========================================================================
    # MAIN REFINE
    # =========================================================================

    def refine(self, positive_conditioning, mode: str, rating: int, refinement_key: str,
               scheduler_mode: str = "original", positive_prompt: str = "",
               clip=None, reset_session: bool = False, unlimited_history: bool = False,
               seed: int = 0, feedback_enabled: bool = False, feedback_rating: int = 3,
               sigmas=None, sigma_strength: str = "subtle", lora_stack=None, latent=None,
               into_the_void: bool = False, im_feeling_lucky: bool = False,
               prompt=None, unique_id=None):

        mode = (mode or "ltx2").lower()
        if mode not in self._tokenizer_sources:
            mode = "ltx2"
        rating_profile = normalize_refiner_rating(rating)
        rating_label = rating_profile.get("label", str(rating))
        rating = int(rating_profile.get("legacy_score", 6))
        reward = float(rating_profile.get("reward", (rating - 5.5) / 4.5))

        if seed != 0:
            torch.manual_seed(seed)
            random.seed(seed)

        refinements_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "refinements")
        os.makedirs(refinements_dir, exist_ok=True)

        json_file = refinement_state_path(refinement_key, mode)
        fallback_loss_graph = render_refinement_loss_graph(
            refinement_key=refinement_key,
            scheduler_mode=scheduler_mode,
            mode=mode,
            total_iterations=0,
            latest_learning_loss=0.0,
            points=[],
        )
        fallback_sigmas = sigmas.detach().clone() if isinstance(sigmas, torch.Tensor) else torch.FloatTensor([])
        latent_output_connected = self._is_output_connected(prompt, unique_id, self.LATENT_OUTPUT_INDEX)
        fallback_latent = clone_latent(latent)
        fallback_latent_status = "Latent: not evaluated before conditioning validation."

        if not positive_conditioning or not isinstance(positive_conditioning, list) or len(positive_conditioning) == 0:
            return (positive_conditioning, "ERROR: Empty positive CONDITIONING input", "", "ERROR: No positive conditioning", fallback_loss_graph, fallback_sigmas, fallback_latent)

        item = positive_conditioning[0]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            raw_positive = item[0]
            positive_meta = item[1] if isinstance(item[1], dict) else {"pooled_output": None}
        else:
            raw_positive = item if isinstance(item, torch.Tensor) else None
            positive_meta = {"pooled_output": None}

        if not isinstance(raw_positive, torch.Tensor):
            return (positive_conditioning, "ERROR: No positive embedding tensor found", "", "ERROR: Invalid embedding", fallback_loss_graph, fallback_sigmas, fallback_latent)

        if rating_profile.get("skip_learning"):
            pending_cleared = False
            if os.path.exists(json_file):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("pending_feedback") is not None:
                        data["pending_feedback"] = None
                        with open(json_file, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                        pending_cleared = True
                except (json.JSONDecodeError, OSError, ValueError):
                    pending_cleared = False
            pending_status = " Pending feedback queue was cleared." if pending_cleared else ""
            status = (
                f"Feedback ignored | Mode {mode.upper()} | Rating {rating_label}\n"
                "No learning history, LoRA weights, sigma schedule, or latent reference was updated."
                f"{pending_status}"
            )
            training_info = (
                f"Rating: {rating_label}\n"
                "This run was intentionally forgotten. The connected conditioning, sigmas, and latent pass through unchanged."
                f"{pending_status}"
            )
            return (positive_conditioning, status, "", training_info, fallback_loss_graph, fallback_sigmas, fallback_latent)

        analysis_prompt = self._normalize_prompt_for_mode(positive_prompt, mode)
        prompt_key = analysis_prompt if mode == "wan" else positive_prompt
        exact_prompt_key = prompt_key
        if im_feeling_lucky:
            prompt_key = "__lucky_memory__"
        current_prompt_words = self._concept_overlap_words(analysis_prompt)
        prompt_variant_match = None

        # ====================== STATE TEMPLATES ======================
        # Single source of truth for both reset and corrupt-recovery paths.
        def _fresh_global():
            state = {
                "word_importance": {},
                "concept_clusters": {},
                "concept_groups": {},
                "exploration_base": 0.08,
                "momentum": None,
                "avg_reward_ema": 0.0,
                "good_ratio": 0.0,
                "dynamic_sim_threshold": 0.82,
                "last_feedback_concept": None,
                "feedbacked_concepts": [],
                "feedback_memory": {
                    "recent_questions": [],
                    "rating_change_events": [],
                },
                "loss_history": [],
                "lora_weight_memory": {},
                "void_token_bank": {},
                "void_token_pairs": {},
                "lucky_context_memory": {},
                "lucky_phrase_placements": {},
                "mode": mode,
                "scheduler_mode": scheduler_mode,
                "prodigy_d": {},
                "prodigy_lr_base": 1.0,
                "warmup_steps": 8,
                "total_steps_estimate": 150,
                "current_step": 0,
            }
            self._ensure_sigma_state_defaults(state)
            return state

        def _fresh_data():
            return {
                "refinement_key": refinement_key,
                "global_adaptive": _fresh_global(),
                "prompt_histories": {
                    prompt_key: {
                        "canonical_prompt": positive_prompt,
                        "prompt_concept_words": current_prompt_words,
                        "source_prompt_key": exact_prompt_key,
                        "source_conditioning_embeds": tensor_to_serializable(raw_positive),
                        "reference_embeds": tensor_to_serializable(raw_positive),
                        "liked_reference_embeds": None,
                        "liked_reference_count": 0,
                        "history": [],
                        "last_rating": rating,
                        "last_rating_label": rating_label
                    }
                },
                "last_prompt_key": prompt_key,
                "pending_feedback": None
            }

        def _seed_fresh_prompt_discovery(data):
            if not analysis_prompt or not isinstance(data, dict):
                return False
            global_state = data.get("global_adaptive")
            if not isinstance(global_state, dict):
                return False
            seq_len = self._get_conditioning_seq_len(raw_positive)
            token_mask = self._get_conditioning_token_mask(raw_positive) if mode == "wan" else None
            discovery_groups = self._build_word_groups(
                analysis_prompt,
                None,
                seq_len,
                token_mask=token_mask,
            )
            if discovery_groups:
                self._seed_void_token_bank(
                    global_state,
                    discovery_groups,
                    raw_positive,
                    0,
                    token_mask=token_mask,
                )
            self._update_lucky_phrase_placements(global_state, analysis_prompt, "discover", 0)
            return bool(discovery_groups)

        # ====================== RESET / NEW SESSION ======================
        lucky_bootstrap = False
        fresh_discovery_seeded = False
        if reset_session or not os.path.exists(json_file):
            if reset_session:
                self._delete_latent_reference(refinement_key, mode)
            data = _fresh_data()
            fresh_discovery_seeded = _seed_fresh_prompt_discovery(data)
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            if im_feeling_lucky:
                lucky_bootstrap = True
            else:
                if latent_output_connected:
                    fallback_latent, fallback_latent_status = self._refine_latent(
                        latent,
                        refinement_key,
                        mode,
                        rating,
                        reward,
                        {},
                        rating_profile,
                    )
                else:
                    fallback_latent, fallback_latent_status = self._latent_refinement_disabled(latent)
                return (positive_conditioning, "New session started - Reference saved", "", f"New session started. Reference embedding saved.\n{fallback_latent_status}", fallback_loss_graph, fallback_sigmas, fallback_latent)
        else:
            # ====================== SAFE JSON LOAD ======================
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError, ValueError) as e:
                print(f"[FunPackVideoRefiner] Corrupt session file, resetting: {e}")
                try:
                    os.remove(json_file)
                except OSError:
                    pass
                data = _fresh_data()
                fresh_discovery_seeded = _seed_fresh_prompt_discovery(data)
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                if im_feeling_lucky:
                    lucky_bootstrap = True
                else:
                    if latent_output_connected:
                        fallback_latent, fallback_latent_status = self._refine_latent(
                            latent,
                            refinement_key,
                            mode,
                            rating,
                            reward,
                            {},
                            rating_profile,
                        )
                    else:
                        fallback_latent, fallback_latent_status = self._latent_refinement_disabled(latent)
                    return (positive_conditioning, "Session file was corrupt - Reset and started fresh", "", f"Session reset due to corrupt file\n{fallback_latent_status}", fallback_loss_graph, fallback_sigmas, fallback_latent)

        if lucky_bootstrap:
            rating_profile = dict(RATING_PROFILES["Initial discovery"], label="Initial discovery")
            rating_label = rating_profile["label"]
            rating = int(rating_profile.get("legacy_score", 6))
            reward = float(rating_profile.get("reward", 0.0))

        global_adaptive = data["global_adaptive"]
        previous_prompt_key_for_rating = data.get("last_prompt_key")
        # Migrate sessions created before the multi-level concept system was added
        global_adaptive.setdefault("concept_clusters", {})
        global_adaptive.setdefault("concept_groups", {})
        global_adaptive.setdefault("word_importance", {})
        global_adaptive.setdefault("feedbacked_concepts", [])
        global_adaptive.setdefault("feedback_memory", {"recent_questions": [], "rating_change_events": []})
        global_adaptive.setdefault("loss_history", [])
        global_adaptive.setdefault("lora_weight_memory", {})
        self._ensure_void_token_bank(global_adaptive)
        self._ensure_void_pair_bank(global_adaptive)
        self._ensure_lucky_context_memory(global_adaptive)
        self._ensure_lucky_phrase_placements(global_adaptive)
        global_adaptive.setdefault("mode", mode)
        self._ensure_sigma_state_defaults(global_adaptive)
        for cid in list(global_adaptive["concept_clusters"].keys()):
            global_adaptive["concept_clusters"][cid] = self._ensure_concept_cluster_defaults(
                global_adaptive["concept_clusters"][cid]
            )

        prompt_histories = data.get("prompt_histories", {})
        tokenizer = self._get_tokenizer(mode)

        if prompt_key not in prompt_histories and not im_feeling_lucky:
            matched_prompt_key, prompt_variant_match = self._find_prompt_variant_history(
                exact_prompt_key,
                current_prompt_words,
                raw_positive,
                prompt_histories,
            )
            if matched_prompt_key:
                prompt_key = matched_prompt_key

        # ====================== FEEDBACK STATE MACHINE (CONCEPT-LEVEL) ======================
        # Operates on full concept phrases rather than individual words.
        # Rating a concept updates all its tracked words proportionally and sends a
        # dampened neighbour signal to adjacent phrases. See _apply_concept_feedback
        # for the full multi-level propagation model.
        feedback_question_output = ""
        pending = data.get("pending_feedback")
        if not feedback_enabled:
            if pending is not None:
                data["pending_feedback"] = None
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            feedback_question_output = "Feedback disabled. Queue cleared."
        else:
            if pending is not None:
                # Discard stale or incompatible pending feedback before we even
                # consider applying it to the current prompt.
                if pending.get("type") != "concept" or pending.get("prompt_key") != prompt_key:
                    data["pending_feedback"] = None
                    pending = None
                    with open(json_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
            else:
                feedback_question_output = "Feedback enabled. A concept question will appear after this generation."

        # ====================== PROMPT HISTORY SETUP ======================
        if prompt_key in prompt_histories:
            active = prompt_histories[prompt_key]
            is_new_prompt = False
        else:
            is_new_prompt = True
            active = {
                "canonical_prompt": positive_prompt,
                "prompt_concept_words": current_prompt_words,
                "source_prompt_key": exact_prompt_key,
                "source_conditioning_embeds": tensor_to_serializable(raw_positive),
                "reference_embeds": tensor_to_serializable(raw_positive),
                "liked_reference_embeds": None,
                "liked_reference_count": 0,
                "history": [],
                "last_rating": rating,
                "last_rating_label": rating_label
            }
            prompt_histories[prompt_key] = active
        self._remember_prompt_variant(
            active,
            exact_prompt_key,
            positive_prompt,
            current_prompt_words,
            prompt_variant_match,
        )

        # Safe source-conditioning loading. This remains the comparison anchor
        # even after liked results become the active refinement reference.
        source_data = active.get("source_conditioning_embeds") or active.get("reference_embeds")
        try:
            source_reference = serializable_to_tensor(source_data)
        except Exception as e:
            print(f"[FunPackVideoRefiner] Failed to load source conditioning: {e}. Resetting for this prompt.")
            source_reference = raw_positive.clone()
            active["source_conditioning_embeds"] = tensor_to_serializable(source_reference)
            active["reference_embeds"] = tensor_to_serializable(source_reference)
            active["liked_reference_embeds"] = None
            active["liked_reference_count"] = 0
            active["history"] = []
            is_new_prompt = True

        device = source_reference.device
        cur_positive = raw_positive.to(device) if raw_positive.device != device else raw_positive
        source_changed = False
        source_change_reason = ""
        source_retarget_status = ""

        # Shape mismatch guard
        if source_reference.shape != cur_positive.shape:
            print(f"[FunPackVideoRefiner] Source conditioning shape {source_reference.shape} != current {cur_positive.shape}. Resetting source reference.")
            source_reference = cur_positive.clone()
            active["source_conditioning_embeds"] = tensor_to_serializable(source_reference)
            active["reference_embeds"] = tensor_to_serializable(source_reference)
            active["liked_reference_embeds"] = None
            active["liked_reference_count"] = 0
            active["history"] = []
            is_new_prompt = True
            source_changed = True
            source_change_reason = "shape"

        seq_len = self._get_conditioning_seq_len(cur_positive)
        active_token_mask = self._get_conditioning_token_mask(cur_positive) if mode == "wan" else None

        source_similarity = self._conditioning_similarity(source_reference, cur_positive, active_token_mask)
        source_prompt_key = active.get("source_prompt_key")
        prompt_source_changed = bool(source_prompt_key and source_prompt_key != exact_prompt_key)
        if exact_prompt_key != prompt_key:
            prompt_source_changed = True
        conditioning_source_changed = (
            not source_changed and not is_new_prompt and source_similarity < 0.985
        )
        if im_feeling_lucky:
            prompt_source_changed = False
            conditioning_source_changed = False
        if prompt_source_changed or conditioning_source_changed:
            can_retarget_variant = (
                prompt_variant_match is not None and
                prompt_source_changed and
                not source_changed and
                list(source_reference.shape) == list(cur_positive.shape)
            )
            if can_retarget_variant:
                migrated = self._retarget_prompt_history_to_source(active, source_reference, cur_positive)
                source_reference = cur_positive.clone()
                active["source_conditioning_embeds"] = tensor_to_serializable(source_reference)
                active["source_prompt_key"] = exact_prompt_key
                is_new_prompt = False
                source_similarity = 1.0
                source_retarget_status = f" | prompt variant retargeted ({migrated} anchors)"
            else:
                source_reference = cur_positive.clone()
                active["source_conditioning_embeds"] = tensor_to_serializable(source_reference)
                active["reference_embeds"] = tensor_to_serializable(source_reference)
                active["liked_reference_embeds"] = None
                active["liked_reference_count"] = 0
                active["history"] = []
                is_new_prompt = True
                source_changed = True
                source_change_reason = "prompt" if prompt_source_changed else "conditioning"
                source_similarity = 1.0

        active["source_prompt_key"] = exact_prompt_key

        # ====================== WORD GROUPING (Level 2) ======================
        word_groups = self._build_word_groups(
            analysis_prompt,
            tokenizer,
            seq_len,
            token_mask=active_token_mask
        )

        # ====================== CONCEPT CLUSTER + GROUP SETUP (Levels 3 & 4) ======================
        concept_clusters = global_adaptive["concept_clusters"]
        if analysis_prompt:
            word_to_concept, ordered_concept_ids, current_concept_labels = self._build_word_concept_map(
                analysis_prompt, concept_clusters
            )
        else:
            word_to_concept, ordered_concept_ids, current_concept_labels = {}, [], {}

        if feedback_enabled and not ordered_concept_ids and analysis_prompt:
            fallback_cid = self._build_prompt_fallback_concept(analysis_prompt, concept_clusters)
            if fallback_cid:
                ordered_concept_ids = [fallback_cid]
                current_concept_labels[fallback_cid] = concept_clusters[fallback_cid].get("label", analysis_prompt[:64])
                for w in concept_clusters[fallback_cid].get("anchor_words", []):
                    word_to_concept.setdefault(w, fallback_cid)

        if feedback_enabled and pending is not None and pending.get("concept_id") not in set(ordered_concept_ids):
            data["pending_feedback"] = None
            pending = None
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        # Build or update concept groups from the current ordered concept list
        concept_groups = self._build_concept_groups(
            ordered_concept_ids, concept_clusters,
            global_adaptive["concept_groups"], current_concept_labels, window=3
        )
        global_adaptive["concept_groups"] = concept_groups

        # Track which groups are active in this iteration
        active_group_ids = {
            gid for gid, g in concept_groups.items()
            if any(cid in ordered_concept_ids for cid in g.get("concept_ids", []))
        }

        # ====================== SCHEDULER SETUP ======================
        global_adaptive["scheduler_mode"] = scheduler_mode
        history = active.get("history", [])
        iter_num = len(history) + 1
        total_iters = sum(len(p.get("history", [])) for p in prompt_histories.values())
        feedback_memory = global_adaptive.setdefault("feedback_memory", {"recent_questions": [], "rating_change_events": []})
        rating_key = rating_profile.get("key", "")
        rated_positive = None
        if history and not source_changed:
            rated_positive = self._history_modified_conditioning(history[-1], source_reference, device)

        bank_rated_prompt = analysis_prompt
        bank_rated_lucky_prompt = ""
        bank_rated_word_groups = word_groups
        bank_rated_positive = rated_positive
        bank_rated_token_mask = active_token_mask
        if history and isinstance(history[-1].get("lucky"), dict):
            bank_rated_lucky_prompt = str(history[-1].get("lucky", {}).get("prompt", "")).strip()

        if previous_prompt_key_for_rating in prompt_histories:
            previous_active = prompt_histories.get(previous_prompt_key_for_rating)
            previous_history = previous_active.get("history", []) if isinstance(previous_active, dict) else []
            if previous_history:
                try:
                    previous_source = serializable_to_tensor(
                        previous_active.get("source_conditioning_embeds") or previous_active.get("reference_embeds")
                    ).to(device)
                except Exception:
                    previous_source = None
                previous_positive = self._history_modified_conditioning_any_shape(previous_history[-1], device, dtype=cur_positive.dtype)
                if previous_positive is not None:
                    previous_prompt = (
                        previous_history[-1].get("analysis_prompt") or
                        previous_history[-1].get("prompt_full") or
                        previous_history[-1].get("prompt") or
                        previous_active.get("canonical_prompt", "")
                    )
                    previous_seq_len = self._get_conditioning_seq_len(previous_positive)
                    previous_token_mask = self._get_conditioning_token_mask(previous_positive) if mode == "wan" else None
                    previous_word_groups = self._build_word_groups(
                        previous_prompt,
                        tokenizer,
                        previous_seq_len,
                        token_mask=previous_token_mask,
                    )
                    previous_word_groups = previous_word_groups + self._lucky_groups_from_history(
                        previous_history[-1],
                        previous_seq_len,
                    )
                    if previous_word_groups:
                        bank_rated_prompt = previous_prompt
                        if isinstance(previous_history[-1].get("lucky"), dict):
                            bank_rated_lucky_prompt = str(previous_history[-1].get("lucky", {}).get("prompt", "")).strip()
                        bank_rated_word_groups = previous_word_groups
                        bank_rated_positive = previous_positive
                        bank_rated_token_mask = previous_token_mask

        if bank_rated_positive is not None:
            self._update_void_token_bank(
                global_adaptive,
                bank_rated_word_groups,
                bank_rated_positive,
                rating_profile,
                iter_num,
                token_mask=bank_rated_token_mask,
            )
            self._update_lucky_phrase_placements(global_adaptive, bank_rated_prompt, rating_key, iter_num)
            if bank_rated_lucky_prompt:
                self._update_lucky_phrase_placements(global_adaptive, bank_rated_lucky_prompt, rating_key, iter_num)

        should_seed_current_prompt_after_lucky = (
            bool(analysis_prompt) and
            not (lucky_bootstrap and fresh_discovery_seeded) and
            (
                analysis_prompt != bank_rated_prompt or
                bank_rated_positive is None
            )
        )

        if feedback_enabled and pending is not None:
            self._apply_concept_feedback(
                pending["concept_id"],
                feedback_rating,
                pending.get("question_type", "presence"),
                global_adaptive["concept_clusters"],
                pending.get("neighbor_ids", []),
                global_adaptive["word_importance"],
                global_adaptive["concept_groups"],
                pending.get("iteration", 0)
            )
            global_adaptive["last_feedback_concept"] = pending.get("concept_label", "")
            data["pending_feedback"] = None
            pending = None
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        liked_reference = None
        liked_reference_count = int(active.get("liked_reference_count", 0) or 0)
        liked_data = active.get("liked_reference_embeds")
        if liked_reference_count > 0 and liked_data is not None:
            try:
                candidate = serializable_to_tensor(liked_data).to(device)
                if list(candidate.shape) == list(source_reference.shape):
                    liked_reference = candidate
                else:
                    liked_reference_count = 0
                    active["liked_reference_embeds"] = None
                    active["liked_reference_count"] = 0
            except Exception:
                liked_reference_count = 0
                active["liked_reference_embeds"] = None
                active["liked_reference_count"] = 0

        liked_reference_updated = False
        if rating_key == "like" and not source_changed:
            rated_reference = rated_positive
            if rated_reference is None:
                rated_reference = source_reference.clone()
            if liked_reference is None or liked_reference_count <= 0:
                liked_reference = rated_reference.clone()
                liked_reference_count = 1
            else:
                liked_reference = (
                    liked_reference * liked_reference_count + rated_reference
                ) / float(liked_reference_count + 1)
                liked_reference_count += 1
            active["liked_reference_embeds"] = tensor_to_serializable(liked_reference)
            active["liked_reference_count"] = liked_reference_count
            active["liked_reference_last_iteration"] = iter_num
            liked_reference_updated = True

        if liked_reference is not None and liked_reference_count > 0:
            reference = liked_reference.clone()
            reference_mode = "liked average"
        else:
            reference = source_reference.clone()
            reference_mode = "source"

        rollback_reference_found = False
        rollback_rating_label = ""
        rollback_requested = bool(rating_profile.get("rollback_on_failure")) or rating_key == "dislike"
        if rollback_requested and not source_changed and len(history) > 1:
            current_level = int(rating_profile.get("level", 1))
            for entry in reversed(history[:-1]):
                entry_profile = normalize_refiner_rating(entry.get("rating_label", entry.get("rating", 0)))
                if int(entry_profile.get("level", 0)) <= current_level:
                    continue
                mod_data = entry.get("modified_embeds")
                if mod_data is None:
                    continue
                try:
                    candidate = serializable_to_tensor(mod_data).to(device)
                    if list(candidate.shape) == list(source_reference.shape):
                        reference = candidate.clone()
                        reference_mode = "rollback"
                        rollback_reference_found = True
                        rollback_rating_label = entry_profile.get("label", str(entry.get("rating", "")))
                        break
                except Exception:
                    continue
        if (
            rollback_requested and
            not source_changed and
            not rollback_reference_found and
            liked_reference is not None and
            liked_reference_count > 0
        ):
            reference = liked_reference.clone()
            reference_mode = "rollback"
            rollback_reference_found = True
            rollback_rating_label = "liked average"

        active["reference_embeds"] = tensor_to_serializable(reference)
        similarity = self._conditioning_similarity(reference, cur_positive, active_token_mask)

        last_rating = active.get("last_rating", 5)
        last_rating_profile = normalize_refiner_rating(active.get("last_rating_label", last_rating))
        if "last_rating_label" not in active:
            last_rating_profile = normalize_refiner_rating(last_rating)
        last_reward = float(last_rating_profile.get("reward", (last_rating - 5.5) / 4.5))
        rating_shift = abs(rating - last_rating)
        feedback_memory["rating_change_events"] = list(feedback_memory.get("rating_change_events", []))[-15:]
        feedback_memory["rating_change_events"].append({
            "iteration": iter_num,
            "rating": rating,
            "rating_label": rating_label,
            "shift": rating_shift,
            "prompt_key": prompt_key,
        })

        word_importance = global_adaptive["word_importance"]
        missing_axis_status = self._apply_missing_axis_prompt_pressure(
            word_groups,
            word_to_concept,
            rating_profile,
            concept_clusters,
            word_importance,
            iter_num,
        )
        prompt_delta_status = self._apply_prompt_delta_attribution(
            history,
            current_prompt_words,
            rating_profile,
            concept_clusters,
            word_importance,
            iter_num,
        )
        prompt_emphasis = float(rating_profile.get("prompt_emphasis", reward))

        lr_scale, confidence, exploration_mult, word_lr_mult = self._get_scheduler_factors(
            scheduler_mode, rating, reward, similarity, iter_num, total_iters,
            word_importance, word_groups, global_adaptive, device
        )

        expl = global_adaptive["exploration_base"] * exploration_mult

        # ====================== WORD IMPORTANCE UPDATE (concept-aware, Levels 2-3) ======================
        for start, end, full_word, word_token_list in word_groups:
            if not self._is_valuable_token(full_word):
                continue
            wkey = full_word.lower()

            base_lr = 0.22 / (1 + 0.07 * (len(history) ** 0.5))
            group_delta = prompt_emphasis * base_lr * lr_scale * confidence
            if wkey in word_lr_mult:
                group_delta *= word_lr_mult[wkey]

            # Level 3: update per-concept importance (primary, context-isolated signal)
            cid = word_to_concept.get(wkey)
            if cid and cid in concept_clusters:
                concept_clusters[cid] = self._ensure_concept_cluster_defaults(concept_clusters[cid])
                local_imp = concept_clusters[cid]["word_importance"]
                profile_gain = (
                    concept_clusters[cid]["priority_weight"] *
                    (0.82 + 0.18 * concept_clusters[cid]["user_affinity"]) *
                    (0.80 + 0.20 * concept_clusters[cid]["presence_target"])
                )
                group_delta *= max(0.55, min(1.85, profile_gain))
                if wkey not in local_imp:
                    local_imp[wkey] = 1.0
                local_imp[wkey] = max(0.35, min(2.8, local_imp[wkey] + group_delta))
                concept_clusters[cid]["usage_count"] = concept_clusters[cid].get("usage_count", 0) + 1
                concept_clusters[cid]["last_seen_iter"] = iter_num

            # Level 2: dampened global fallback so the scheduler/prodigy system
            # retains a signal without cross-context word-meaning contamination
            if wkey not in word_importance:
                word_importance[wkey] = 1.0
            word_importance[wkey] = max(0.35, min(2.8, word_importance[wkey] + group_delta * 0.4))

        # Update active group usage counters (Level 4)
        for gid in active_group_ids:
            if gid in concept_groups:
                concept_groups[gid]["last_seen_iter"] = iter_num
                concept_groups[gid]["usage_count"] = concept_groups[gid].get("usage_count", 0) + 1

        # ====================== CORE REFINEMENT ======================
        momentum = global_adaptive.get("momentum")
        if momentum is None or not isinstance(momentum, dict):
            momentum = torch.zeros_like(reference)
        else:
            momentum = serializable_to_tensor(momentum).to(device)
            if list(momentum.shape) != list(reference.shape):
                momentum = torch.zeros_like(reference)

        avg_reward_ema = global_adaptive["avg_reward_ema"]
        avg_reward_ema = 0.85 * avg_reward_ema + 0.15 * reward
        global_adaptive["avg_reward_ema"] = avg_reward_ema

        good_ratio = global_adaptive["good_ratio"]
        if rating_key == "like":
            good_ratio = 0.9 * good_ratio + 0.1 * 1.0
            expl = max(0.015, expl * 0.96)
        elif set(rating_profile.get("missing_axes", [])) == {"details"}:
            good_ratio = 0.9 * good_ratio + 0.1 * 0.55
            expl = min(0.12, expl * 1.02)
        else:
            good_ratio = 0.9 * good_ratio + 0.1 * 0.0
            expl = min(0.12, expl * 1.08)
        global_adaptive["good_ratio"] = good_ratio

        sim_threshold = global_adaptive["dynamic_sim_threshold"]
        is_close = (not is_new_prompt) and (similarity >= sim_threshold)

        if (is_close and history) or (rollback_requested and rollback_reference_found and history):
            last_entry = history[-1]
            mod_data = last_entry.get("modified_embeds")
            if mod_data is not None:
                try:
                    prev_modified = serializable_to_tensor(mod_data).to(device)
                    if list(prev_modified.shape) != list(reference.shape):
                        prev_modified = torch.zeros_like(reference)
                except Exception:
                    prev_modified = torch.zeros_like(reference)
            else:
                prev_modified = torch.zeros_like(reference)
            prev_delta = prev_modified - reference
            noise_scale = expl * (1.0 - avg_reward_ema * 0.7)
            if reference_mode == "liked average" and rating_key == "like":
                noise_scale = 0.0
            noise = torch.randn_like(reference) * noise_scale
            if rollback_requested and rollback_reference_found:
                new_delta = (-prev_delta * 0.9) + (momentum * 0.25)
            else:
                multiplier = max(0.05, 1.0 + reward * 1.45)
                if rating == last_rating and rating_key == "like":
                    multiplier += 0.35
                new_delta = (prev_delta * multiplier) + noise + (momentum * 0.6)
        else:
            good_deltas = []
            for entry in history:
                entry_profile = normalize_refiner_rating(entry.get("rating_label", entry.get("rating", 0)))
                if entry_profile.get("key") not in {"like", "missing_details"}:
                    continue
                mod_data = entry.get("modified_embeds")
                if mod_data is None:
                    continue
                try:
                    mod = serializable_to_tensor(mod_data).to(device)
                    if list(mod.shape) == list(reference.shape):
                        good_deltas.append(mod - reference)
                except Exception:
                    continue
            if good_deltas:
                new_delta = torch.stack(good_deltas).mean(dim=0) * (0.7 + reward * 0.4)
            elif reference_mode == "liked average" and rating_key == "like":
                new_delta = torch.zeros_like(reference)
            else:
                new_delta = torch.randn_like(reference) * expl * 0.45

        # Apply concept-aware importance to delta (Levels 2-3)
        # Each token position is scaled by the concept-local importance of the word
        # occupying that position, with global importance as fallback.
        if cur_positive.dim() > 1:
            seq_len = cur_positive.shape[1] if cur_positive.dim() == 3 else cur_positive.shape[0]
            importance_tensor = torch.ones(seq_len, device=device)
            profile_tensor = torch.ones(seq_len, device=device)
            if active_token_mask is not None:
                importance_tensor = importance_tensor * active_token_mask.to(device=device, dtype=importance_tensor.dtype)
                profile_tensor = profile_tensor * active_token_mask.to(device=device, dtype=profile_tensor.dtype)
            for start, end, full_word, _ in word_groups:
                wkey = full_word.lower()
                cid = word_to_concept.get(wkey)
                if cid and cid in concept_clusters:
                    concept_clusters[cid] = self._ensure_concept_cluster_defaults(concept_clusters[cid])
                    imp = concept_clusters[cid]["word_importance"].get(
                        wkey, word_importance.get(wkey, 1.0)
                    )
                    profile_mult = (
                        concept_clusters[cid]["priority_weight"] *
                        (0.84 + 0.16 * concept_clusters[cid]["user_affinity"]) *
                        (0.86 + 0.14 * concept_clusters[cid]["presence_target"])
                    )
                    profile_mult *= (
                        1.0 -
                        0.10 * (concept_clusters[cid]["stability_weight"] - 1.0) -
                        0.06 * (concept_clusters[cid]["semantic_fidelity"] - 1.0)
                    )
                    profile_mult *= (
                        1.0 -
                        0.10 * max(0.0, concept_clusters[cid]["overrep_sensitivity"] - 1.0)
                    )
                    profile_mult = max(0.55, min(1.80, profile_mult))
                else:
                    imp = word_importance.get(wkey, 1.0)
                    profile_mult = 1.0
                importance_tensor[start:end] = imp
                profile_tensor[start:end] = profile_mult
            new_delta = new_delta * importance_tensor.unsqueeze(-1)
            new_delta = new_delta * profile_tensor.unsqueeze(-1)

        token_mask_nd = self._mask_to_embedding_dims(active_token_mask, reference)
        if token_mask_nd is not None:
            new_delta = new_delta * token_mask_nd

        # Final safety guard
        if new_delta.shape != reference.shape:
            new_delta = torch.zeros_like(reference)

        new_positive = reference + new_delta
        new_positive = torch.clamp(new_positive, min=-60.0, max=60.0)
        norm_factor = reference.norm(dim=-1, keepdim=True) + 1e-8
        new_positive = new_positive / (new_positive.norm(dim=-1, keepdim=True) + 1e-8) * norm_factor
        lucky_canvas, lucky_canvas_label = (None, "")
        lucky_prompt_text = ""
        lucky_prompt_tokens = []
        lucky_encoded_meta = None
        lucky_prompt_sequences = []
        lucky_phrase_sequences = []
        if im_feeling_lucky:
            lucky_prompt_sequences = self._lucky_prompt_sequences(prompt_histories)
            lucky_phrase_sequences = self._lucky_prompt_phrase_sequences(prompt_histories)
            if clip is not None:
                lucky_prompt_text, lucky_prompt_info = self._compose_lucky_prompt_text(
                    global_adaptive,
                    word_groups,
                    int(cur_positive.shape[-1]),
                    prompt_sequences=lucky_prompt_sequences,
                    phrase_sequences=lucky_phrase_sequences,
                    target_count=self._get_conditioning_seq_len(cur_positive),
                )
                lucky_prompt_tokens = list(lucky_prompt_info.get("tokens", [])) if isinstance(lucky_prompt_info, dict) else []
                encoded_canvas, encoded_meta, encoded_status = self._encode_lucky_prompt_conditioning(
                    clip,
                    lucky_prompt_text,
                    cur_positive,
                    device,
                    cur_positive.dtype,
                )
                if encoded_canvas is not None:
                    encoded_delta = self._resize_conditioning_sequence_like(new_delta, encoded_canvas)
                    if encoded_delta is not None:
                        refined_encoded_canvas = encoded_canvas + encoded_delta
                        refined_encoded_canvas = torch.clamp(refined_encoded_canvas, min=-60.0, max=60.0)
                        encoded_norm = encoded_canvas.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                        refined_encoded_canvas = (
                            refined_encoded_canvas /
                            refined_encoded_canvas.norm(dim=-1, keepdim=True).clamp_min(1e-8) *
                            encoded_norm
                        )
                    else:
                        refined_encoded_canvas = encoded_canvas
                    lucky_canvas = refined_encoded_canvas
                    lucky_canvas_label = encoded_status
                    lucky_encoded_meta = encoded_meta
                elif encoded_status:
                    print(f"[FunPackVideoRefiner] Lucky CLIP/Gemma canvas unavailable: {encoded_status}")

            if lucky_canvas is None:
                lucky_canvas, lucky_canvas_label = self._select_lucky_memory_canvas(
                    prompt_histories,
                    cur_positive,
                    device,
                    cur_positive.dtype,
                )
        new_positive, void_status, lucky_metadata = self._apply_into_the_void(
            new_positive,
            reference,
            global_adaptive,
            word_groups,
            word_importance,
            into_the_void=into_the_void,
            im_feeling_lucky=im_feeling_lucky,
            token_mask=active_token_mask,
            prompt_sequences=lucky_prompt_sequences,
            lucky_canvas=lucky_canvas,
            lucky_canvas_label=lucky_canvas_label,
            lucky_prompt_text=lucky_prompt_text,
            lucky_prompt_tokens=lucky_prompt_tokens if lucky_encoded_meta is not None else [],
        )
        if should_seed_current_prompt_after_lucky:
            self._seed_void_token_bank(
                global_adaptive,
                word_groups,
                cur_positive,
                iter_num,
                token_mask=active_token_mask,
            )
            self._update_lucky_phrase_placements(global_adaptive, analysis_prompt, "discover", iter_num)

        # Update momentum
        momentum = 0.75 * momentum + 0.25 * (new_delta * reward)
        global_adaptive["momentum"] = tensor_to_serializable(momentum)

        if avg_reward_ema > 0.3:
            sim_threshold = max(0.75, sim_threshold - 0.002)
        global_adaptive["dynamic_sim_threshold"] = sim_threshold
        global_adaptive["exploration_base"] = expl

        # Prune stale concept clusters and groups to prevent unbounded JSON growth
        if len(concept_clusters) > 64:
            concept_clusters = {
                cid: c for cid, c in concept_clusters.items()
                if iter_num - c.get("last_seen_iter", 0) < 500
            }
            global_adaptive["concept_clusters"] = concept_clusters

        if len(concept_groups) > 32:
            concept_groups = {
                gid: g for gid, g in concept_groups.items()
                if iter_num - g.get("last_seen_iter", 0) < 500
            }
            global_adaptive["concept_groups"] = concept_groups

        # ====================== HISTORY ENTRY ======================
        history_entry = {
            "iteration": iter_num,
            "rating": rating,
            "rating_label": rating_label,
            "rating_range": rating_profile.get("legacy_range", ""),
            "reward": round(reward, 3),
            "modified_embeds": tensor_to_serializable(new_positive),
            "similarity": round(similarity, 4),
            "source_similarity": round(source_similarity, 4),
            "reference_mode": reference_mode,
            "liked_reference_count": int(liked_reference_count),
            "liked_reference_updated": bool(liked_reference_updated),
            "rollback_from_rating": rollback_rating_label,
            "prompt": positive_prompt[:180],
            "prompt_full": positive_prompt,
            "analysis_prompt": analysis_prompt,
            "prompt_words": current_prompt_words,
            "exact_prompt_key": exact_prompt_key,
            "into_the_void": bool(into_the_void),
            "im_feeling_lucky": bool(im_feeling_lucky),
            "void_status": void_status,
            "lucky": lucky_metadata,
        }
        history.append(history_entry)

        if not unlimited_history and len(history) > 200:
            sorted_hist = sorted(history, key=lambda x: x.get("rating", 0), reverse=True)
            top = sorted_hist[:40]
            recent = history[-120:]
            seen_iters = {e["iteration"] for e in top}
            history = top + [e for e in recent if e["iteration"] not in seen_iters]

        active["history"] = history
        active["last_rating"] = rating
        active["last_rating_label"] = rating_label
        active["last_positive_prompt"] = positive_prompt
        active["last_exact_prompt_words"] = current_prompt_words
        data["last_prompt_key"] = prompt_key
        data["prompt_histories"] = prompt_histories
        data["global_adaptive"] = global_adaptive

        refined_sigmas, sigma_status = self._refine_sigma_schedule(
            sigmas,
            rating,
            global_adaptive,
            sigma_strength,
            seed,
            rating_profile,
        )
        if latent_output_connected:
            refined_latent, latent_status = self._refine_latent(
                latent,
                refinement_key,
                mode,
                rating,
                reward,
                global_adaptive,
                rating_profile,
            )
        else:
            refined_latent, latent_status = self._latent_refinement_disabled(latent)

        # ====================== INTELLIGENT CONCEPT FEEDBACK SELECTION ======================
        # Chooses one concept/question pair per run using category-aware scoring.
        # Different question types learn different aspects of user preference:
        # presence, priority, balance, fidelity, stability, and preference.
        if feedback_enabled and data.get("pending_feedback") is None and ordered_concept_ids:
            selected_question = self._select_feedback_question(
                ordered_concept_ids,
                concept_clusters,
                concept_groups,
                current_concept_labels,
                last_rating,
                rating,
                similarity,
                iter_num
            )

            if selected_question is None:
                selected_question = self._force_feedback_fallback(
                    ordered_concept_ids,
                    concept_clusters,
                    concept_groups,
                    current_concept_labels,
                    rating_shift,
                    similarity
                )

            if selected_question:
                data["pending_feedback"] = {
                    "type": "concept",
                    "concept_id": selected_question["concept_id"],
                    "concept_label": selected_question["concept_label"],
                    "question_type": selected_question["question_type"],
                    "neighbor_ids": selected_question["neighbor_ids"],
                    "group_id": selected_question["group_id"],
                    "prompt_key": prompt_key,
                    "iteration": iter_num
                }
                feedback_memory["recent_questions"] = list(feedback_memory.get("recent_questions", []))[-15:]
                feedback_memory["recent_questions"].append({
                    "iteration": iter_num,
                    "concept_id": selected_question["concept_id"],
                    "question_type": selected_question["question_type"],
                    "prompt_key": prompt_key,
                })
                global_adaptive["last_feedback_concept"] = selected_question["concept_label"]
                feedback_question_output = self._format_feedback_question(
                    selected_question["concept_label"],
                    selected_question["question_type"]
                )
            else:
                feedback_question_output = (
                    "There is enough information collected on current concepts. "
                    "Node will ask you again in case if rating changes significantly."
                )
        elif feedback_enabled and data.get("pending_feedback") is None:
            feedback_question_output = (
                "There is enough information collected on current concepts. "
                "Node will ask you again in case if rating changes significantly."
            )

        # ====================== TRAINING INFO ======================
        current_top = self._get_top_tokens(word_importance, tokenizer, 10)

        # Per-concept phrase summaries (Level 3)
        active_concept_parts = []
        for cid in ordered_concept_ids:
            if cid not in concept_clusters:
                continue
            c = concept_clusters[cid]
            top_local = self._get_top_tokens(c["word_importance"], tokenizer, 4)
            active_concept_parts.append(
                f"[{current_concept_labels.get(cid, c['label'])}/{c.get('category', 'general')}: "
                f"top={top_local} | p={c.get('presence_target', 1.0):.2f} "
                f"prio={c.get('priority_weight', 1.0):.2f} "
                f"stab={c.get('stability_weight', 1.0):.2f} "
                f"cat={c.get('category_source', 'auto')}]"
            )
        concept_line = "; ".join(active_concept_parts) if active_concept_parts else "none"

        # Concept group health summaries (Level 4)
        group_parts = []
        for gid in active_group_ids:
            g = concept_groups.get(gid)
            if g:
                ema = g.get("reward_ema", 0.0)
                health_icon = "✅" if ema > 0.3 else "⚠️" if ema > -0.3 else "🔄"
                group_parts.append(f"{health_icon} {g['label']} (ema={ema:+.2f})")
        group_line = "; ".join(group_parts) if group_parts else "none"

        # Dominant concept: the phrase the model currently weights most heavily
        dom_cid, dom_score, dom_label = self._get_dominant_concept(
            ordered_concept_ids, concept_clusters
        )
        dom_label = current_concept_labels.get(dom_cid, dom_label) if dom_cid else dom_label
        dominant_line = f"'{dom_label}' (avg_imp={dom_score:.2f})" if dom_cid else "undetermined"

        lora_suggestion_status = self._update_lora_weight_suggestions(
            lora_stack,
            active,
            global_adaptive,
            ordered_concept_ids,
            concept_clusters,
            current_concept_labels,
            rating,
            reward,
            rating_profile,
        )
        if im_feeling_lucky:
            learned_prompt_memories = sum(
                1 for key, item in prompt_histories.items()
                if key != "__lucky_memory__" and isinstance(item, dict) and item.get("history")
            )
            prompt_history_status = (
                "Prompt history: Lucky memory stream "
                f"({len(history)} Lucky updates, {learned_prompt_memories} learned prompt memories)."
            )
        elif prompt_variant_match:
            prompt_history_status = (
                "Prompt history: reused similar enhanced prompt "
                f"(overlap {prompt_variant_match.get('overlap', 0.0):.0%}, "
                f"coverage {prompt_variant_match.get('coverage', 0.0):.0%}, "
                f"conditioning {prompt_variant_match.get('conditioning_similarity', 0.0):.2f})."
            )
        elif exact_prompt_key != prompt_key:
            prompt_history_status = "Prompt history: reused compatible prompt variant."
        else:
            prompt_history_status = "Prompt history: exact prompt."

        normalized_reward = max(0.0, min(1.0, (avg_reward_ema + 1.0) / 2.0))
        stability_factor = max(0.0, 1.0 - similarity)
        raw_loss = (1.0 - normalized_reward) * 0.7 + stability_factor * 0.3
        learning_loss = max(0.02, raw_loss * (1.0 - min(0.95, good_ratio * 0.8)))
        reference_status = (
            f"Reference: {reference_mode} | liked anchors {liked_reference_count} | "
            f"source similarity {source_similarity:.4f}"
        )
        if liked_reference_updated:
            reference_status += " | liked average updated"
        if rollback_reference_found:
            reference_status += f" | rollback from {rollback_rating_label or 'higher rating'}"
        if source_changed:
            reference_status += f" | source refreshed ({source_change_reason or 'changed'})"
        if source_retarget_status:
            reference_status += source_retarget_status
        missing_axis_line = f"{missing_axis_status}\n" if missing_axis_status else ""
        prompt_delta_line = f"{prompt_delta_status}\n" if prompt_delta_status else ""

        training_info = (
            f"Mode: {mode.upper()} | Scheduler: {scheduler_mode.upper()} | Step: {global_adaptive['current_step']} | "
            f"Rating: {rating_label} ({rating_profile.get('legacy_range', rating)}) | "
            f"EMA Reward: {avg_reward_ema:+.3f} | Confidence: {confidence:.2f} | LR Scale: {lr_scale:.3f}\n"
            f"Exploration: {expl:.3f} | Similarity: {similarity:.4f} | Good Ratio: {good_ratio:.1%} | Prompt Emphasis: {prompt_emphasis:+.2f}\n"
            f"**Learning Loss: {learning_loss:.4f}** (lower is better)\n"
            f"{reference_status}\n"
            f"Dominant concept: {dominant_line}\n"
            f"{void_status}\n"
            f"Concept phrases ({len(concept_clusters)} total): {concept_line}\n"
            f"Concept groups ({len(concept_groups)} total): {group_line}\n"
            f"Global top words: {current_top}\n"
            f"{missing_axis_line}"
            f"{prompt_delta_line}"
            f"{prompt_history_status}\n"
            f"{lora_suggestion_status}\n"
            f"{sigma_status}\n"
            f"{latent_status}"
        )

        if scheduler_mode == "accurate":
            training_info += "\n[Accurate] Conservative • Prodigy + Cosine"
        elif scheduler_mode == "aggressive":
            training_info += "\n[Aggressive] Fast style locking"

        global_total_iterations = sum(len(p.get("history", [])) for p in prompt_histories.values())
        loss_history = list(global_adaptive.get("loss_history", []))[-511:]
        loss_history.append({
            "total_iteration": global_total_iterations,
            "learning_loss": round(float(learning_loss), 6),
            "rating": int(rating),
            "rating_label": rating_label,
            "similarity": round(float(similarity), 6),
            "scheduler_mode": scheduler_mode,
            "mode": mode,
        })
        global_adaptive["loss_history"] = loss_history

        loss_graph = render_refinement_loss_graph(
            refinement_key=refinement_key,
            scheduler_mode=scheduler_mode,
            mode=mode,
            total_iterations=global_total_iterations,
            latest_learning_loss=float(learning_loss),
            points=loss_history[-256:],
        )

        # ====================== STATUS ======================
        trend = "↑" if reward > last_reward else "↓" if reward < last_reward else "→"
        health = (
            "🚀 Strong convergence" if avg_reward_ema > 0.6 else
            "✅ Learning well" if avg_reward_ema > 0.3 else
            "⚠️ Still exploring" if avg_reward_ema > -0.2 else
            "🔄 Heavy correction"
        )
        pending_feedback = data.get("pending_feedback")
        if not feedback_enabled:
            feedback_state = "Feedback off"
        elif pending_feedback is not None:
            feedback_state = f"Feedback queued for '{pending_feedback.get('concept_label', 'concept')}'"
        else:
            feedback_state = "Feedback ready"

        suggestions_snapshot = active.get("lora_weight_suggestions", {})
        suspect_count = sum(1 for item in suggestions_snapshot.values() if item.get("suspect"))
        if isinstance(lora_stack, dict) and lora_stack.get("loras"):
            lora_state = f"LoRA active ({len(lora_stack.get('loras', []))} loaded"
            if suspect_count > 0:
                lora_state += f", {suspect_count} suspect"
            lora_state += ")"
        else:
            lora_state = "LoRA idle"

        sigma_state = "Sigma active" if isinstance(sigmas, torch.Tensor) and sigma_strength != "off" else (
            "Sigma connected (off)" if isinstance(sigmas, torch.Tensor) else "Sigma idle"
        )

        latent_state = (
            "Latent active" if latent_output_connected and refined_latent is not None else
            "Latent armed" if latent_output_connected else
            "Latent idle"
        )

        if im_feeling_lucky:
            learned_prompt_memories = sum(
                1 for key, item in prompt_histories.items()
                if key != "__lucky_memory__" and isinstance(item, dict) and item.get("history")
            )
            lucky_bank_size = len(global_adaptive.get("void_token_bank", {}))
            lucky_phrase_size = len(global_adaptive.get("lucky_phrase_placements", {}))
            session_line = (
                f"Session: Lucky memory {len(history)} update(s), {global_total_iterations} total update(s), "
                f"{learned_prompt_memories} learned prompt memory(s), "
                f"{lucky_bank_size} token(s), {lucky_phrase_size} phrase placement(s)"
            )
        else:
            session_line = (
                f"Session: {len(prompt_histories)} prompt(s), {global_total_iterations} total update(s), "
                f"{len(history)} history item(s) on this prompt"
            )

        status = (
            f"{health} | Mode {mode.upper()} | Rating {rating_label} ({rating_profile.get('legacy_range', rating)}) {trend} | Iter {iter_num}\n"
            f"{session_line}\n"
            f"Reference: {reference_mode} ({liked_reference_count} liked) | Focus: {dominant_line} | {feedback_state}\n"
            f"Systems: {lora_state} | {sigma_state} | {latent_state} | {void_status}"
        )

        # ====================== FINAL SAVE ======================
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # ====================== RETURN ======================
        output_positive_meta = lucky_encoded_meta if isinstance(lucky_encoded_meta, dict) else positive_meta
        modified_positive = [(new_positive, output_positive_meta)]

        return (modified_positive, status, feedback_question_output, training_info, loss_graph, refined_sigmas, refined_latent)


FunPackGemmaEmbeddingRefiner = FunPackVideoRefiner


V2_RATING_LABELS = [
    "-Just forget it-",
    "Perfect",
    "Missing details",
    "Missing action",
    "Missing quality",
    "Missing details + action",
    "Missing details + quality",
    "Missing action + quality",
    "Awful",
]

V2_RATING_PROFILES = {
    "-Just forget it-": {"key": "forget", "reward": 0.0, "level": 0, "missing_axes": [], "skip_learning": True},
    "Initial discovery": {"key": "discover", "reward": 0.0, "level": 4, "missing_axes": []},
    "Perfect": {"key": "like", "reward": 1.0, "level": 8, "missing_axes": []},
    "Missing details": {"key": "missing_details", "reward": 0.35, "level": 6, "missing_axes": ["details"]},
    "Missing action": {"key": "missing_action", "reward": 0.05, "level": 5, "missing_axes": ["action"]},
    "Missing quality": {"key": "missing_quality", "reward": -0.30, "level": 4, "missing_axes": ["quality"]},
    "Missing details + action": {"key": "missing_details_action", "reward": -0.10, "level": 3, "missing_axes": ["details", "action"]},
    "Missing details + quality": {"key": "missing_details_quality", "reward": -0.40, "level": 2, "missing_axes": ["details", "quality"]},
    "Missing action + quality": {"key": "missing_action_quality", "reward": -0.55, "level": 1, "missing_axes": ["action", "quality"]},
    "Awful": {"key": "awful", "reward": -0.90, "level": 0, "missing_axes": ["details", "action", "quality"]},
}

V2_FEEDBACK_AXES = ("details", "action", "quality")

V2_RATING_ALIASES = {
    "I like it": "Perfect",
    "I don't like it": "Awful",
    "Missing concept": "Missing action",
    "Missing concepts": "Missing action",
    "Missing details + concept": "Missing details + action",
    "Missing details + concepts": "Missing details + action",
    "Missing concept + details": "Missing details + action",
    "Missing concepts + details": "Missing details + action",
    "Missing concept + quality": "Missing action + quality",
    "Missing concepts + quality": "Missing action + quality",
    "Missing quality + concept": "Missing action + quality",
    "Missing quality + concepts": "Missing action + quality",
    "Missing quality + action": "Missing action + quality",
    "Missing action + details": "Missing details + action",
    "Missing everything": "Awful",
}


def normalize_refiner_v2_rating(value):
    if isinstance(value, str):
        cleaned = V2_RATING_ALIASES.get(value.strip(), value.strip())
        if cleaned in V2_RATING_PROFILES:
            return dict(V2_RATING_PROFILES[cleaned], label=cleaned)
        try:
            value = int(float(cleaned))
        except (TypeError, ValueError):
            return dict(V2_RATING_PROFILES["Missing action"], label="Missing action")

    try:
        legacy_score = int(value)
    except (TypeError, ValueError):
        legacy_score = 6

    legacy_score = int(_clamp(legacy_score, 1, 10))
    if legacy_score >= 9:
        label = "Perfect"
    elif legacy_score >= 7:
        label = "Missing details"
    elif legacy_score >= 5:
        label = "Missing action"
    elif legacy_score >= 3:
        label = "Missing quality"
    else:
        label = "Awful"

    profile = dict(V2_RATING_PROFILES[label], label=label)
    profile["legacy_score"] = legacy_score
    return profile


class FunPackVideoRefinerV2(FunPackVideoRefiner):
    CATEGORY = "FunPack/Refinement"
    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("modified_positive", "status", "training_info", "loss_graph")
    FUNCTION = "refine_v2"
    DESCRIPTION = "Prompt-owned Video Refiner V2. Encodes through the connected CLIP, learns from ratings, and writes LoRA suggestions without sigma/latent/feedback systems."

    V2_STATE_PREFIX = "refine_v2"
    ACTION_LORA_TYPES = {"action", "concept"}
    CATEGORY_DESCRIPTIONS = {
        "action": "physical actions, body movement, animation, motion, gestures, moving subjects",
        "camera": "camera motion, framing, zoom, pan, dolly, close-up, wide shot, focus behavior",
        "subject": "main person, object, creature, vehicle, character, or scene subject",
        "appearance": "character appearance, clothing, face, hair, pose, anatomy, visible traits",
        "environment": "place, location, background, weather, room, landscape, setting",
        "style": "visual style, art direction, lighting style, color grading, cinematic look",
        "quality": "sharpness, detail, realism, clean image quality, low noise, high resolution",
        "details": "small objects, props, reflections, texture, secondary prompt details",
    }
    CATEGORY_KEYWORDS = {
        "action": {
            "walk", "walking", "run", "running", "turn", "turning", "dance", "dancing", "jump",
            "jumping", "fly", "flying", "move", "moving", "motion", "gesture", "gesturing",
            "hold", "holding", "reach", "reaching", "look", "looking", "blink", "blinking",
            "smile", "smiling", "sit", "sitting", "stand", "standing", "kneel", "kneeling",
            "sway", "swaying", "spin", "spinning", "fall", "falling", "grab", "grabbing",
            "throw", "throwing", "catch", "catching", "climb", "climbing", "fight", "fighting",
            "talk", "talking", "speak", "speaking", "wave", "waving", "crawl", "crawling",
            "drive", "driving", "ride", "riding", "open", "opening", "close", "closing",
        },
        "camera": {
            "camera", "shot", "closeup", "close-up", "wide", "angle", "zoom", "pan", "panning",
            "dolly", "tracking", "handheld", "focus", "framing", "push", "push-in", "pull",
            "pull-back", "tilt", "orbit", "orbiting", "crane", "rack", "lens", "viewpoint",
            "perspective", "overhead", "low-angle", "high-angle", "close-up", "macro",
        },
        "subject": {
            "woman", "man", "girl", "boy", "person", "character", "robot", "creature", "dragon",
            "animal", "dog", "cat", "car", "vehicle", "object", "sculpture", "monster",
            "child", "baby", "adult", "crowd", "dancer", "runner", "warrior", "knight",
        },
        "appearance": {
            "hair", "eyes", "face", "skin", "dress", "jacket", "armor", "outfit", "clothing",
            "hands", "pose", "expression", "beard", "makeup", "body", "anatomy", "wearing",
            "wears", "dressed", "clothed", "costume", "shirt", "coat", "robe", "boots",
            "hat", "helmet", "gloves", "mask", "tattoo", "freckles", "scar", "silhouette",
            "flowing", "curly", "straight", "long", "short", "blonde", "brunette",
        },
        "environment": {
            "forest", "city", "street", "room", "kitchen", "beach", "mountain", "temple",
            "sunset", "night", "rain", "snow", "sky", "background", "studio", "tabletop",
            "setting", "environment", "landscape", "interior", "exterior", "alley", "road",
            "field", "desert", "ocean", "sea", "lake", "river", "waterfall", "clouds",
            "weather", "fog", "mist", "storm", "snowfall", "backdrop", "horizon",
        },
        "style": {
            "anime", "cinematic", "photorealistic", "painterly", "illustration", "stylized",
            "realistic", "film", "noir", "vintage", "dramatic", "soft", "lighting",
            "moody", "surreal", "documentary", "editorial", "watercolor", "sketch",
            "render", "rendered", "monochrome", "pastel", "neon", "gothic",
        },
        "quality": {
            "masterpiece", "best", "quality", "detailed", "sharp", "highres", "high-res",
            "ultra", "perfect", "clean", "realism", "realistic", "smooth", "crisp",
            "high-detail", "highly-detailed", "ultra-detailed", "noise-free", "polished",
            "refined", "clear", "high-resolution", "8k", "4k",
        },
        "details": {
            "reflection", "reflections", "texture", "textures", "shadow", "shadows", "smoke",
            "dust", "particles", "small", "tiny", "prop", "props", "fabric", "glass",
            "sparkles", "embers", "debris", "scratches", "cracks", "drops", "droplets",
            "pattern", "patterns", "grain", "details", "ornament", "ornaments",
        },
    }
    CATEGORY_PRIORITY = {
        "action": 0.018,
        "camera": 0.014,
        "subject": 0.008,
        "appearance": 0.012,
        "environment": 0.012,
        "style": 0.006,
        "quality": 0.004,
        "details": 0.002,
    }
    ACTION_SUFFIX_STEMS = {
        "walk", "run", "turn", "danc", "jump", "fly", "mov", "gestur", "hold", "reach",
        "look", "blink", "smil", "sit", "stand", "kneel", "sway", "spin", "fall",
        "grab", "throw", "catch", "climb", "fight", "talk", "speak", "wav", "crawl",
        "driv", "rid", "open", "clos",
    }
    APPEARANCE_CONTEXT_WORDS = {
        "hair", "eyes", "face", "skin", "dress", "jacket", "armor", "outfit", "clothing",
        "shirt", "coat", "robe", "boots", "hat", "helmet", "gloves", "mask", "body",
        "anatomy", "beard", "makeup", "costume",
    }
    ENVIRONMENT_CONTEXT_WORDS = {
        "background", "setting", "environment", "landscape", "room", "street", "forest",
        "city", "beach", "mountain", "temple", "sky", "studio", "interior", "exterior",
        "backdrop", "horizon", "weather",
    }
    QUALITY_CONTEXT_WORDS = {
        "quality", "detail", "details", "detailed", "sharp", "clean", "crisp", "realism",
        "resolution", "highres", "high-resolution", "smooth", "polished",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_prompt": ("STRING", {"multiline": True, "default": "", "placeholder": "Positive prompt"}),
                "clip": ("CLIP", {"tooltip": "Connected text encoder used for all V2 prompt encoding and similarity checks."}),
                "rating": (V2_RATING_LABELS, {"default": "Missing action", "label": "Rating"}),
                "refinement_key": ("STRING", {"default": "my_style_v2", "multiline": False}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "label": "Exploration Seed"}),
                "reset_session": ("BOOLEAN", {"default": False, "label": "Reset V2 Session"}),
                "lora_stack": ("FUNPACK_LORA_STACK", {"tooltip": "Optional stack from FunPack LoRA Loader. V2 writes prompt-specific suggested weights."}),
                "im_feeling_lucky": ("BOOLEAN", {
                    "default": False,
                    "label": "I'm Feeling Lucky",
                    "tooltip": "Compose a learned prompt from V2 phrase memory, then encode it through the connected CLIP.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def _v2_state_path(self, refinement_key):
        return refinement_state_path(refinement_key, "clip", prefix=self.V2_STATE_PREFIX)

    def _v2_empty_state(self, refinement_key):
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
                "loss_history": [],
            },
            "prompt_histories": {},
            "last_run": None,
        }

    def _v2_load_state(self, refinement_key, reset_session=False):
        path = self._v2_state_path(refinement_key)
        if reset_session or not os.path.exists(path):
            return self._v2_empty_state(refinement_key), "fresh"
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            if not isinstance(data, dict) or int(data.get("version", 0)) != 2:
                return self._v2_empty_state(refinement_key), "reset invalid"
            data.setdefault("global", {})
            data.setdefault("prompt_histories", {})
            data.setdefault("last_run", None)
            data["global"].setdefault("axis_conditioning_memory", {})
            data["global"].setdefault("phrase_memory", {})
            data["global"].setdefault("lora_weight_memory", {})
            data["global"].setdefault("loss_history", [])
            return data, "loaded"
        except (json.JSONDecodeError, OSError, ValueError):
            return self._v2_empty_state(refinement_key), "reset unreadable"

    def _v2_save_state(self, data, refinement_key):
        path = self._v2_state_path(refinement_key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    def _v2_prompt_key(self, prompt):
        return re.sub(r"\s+", " ", str(prompt or "").strip())

    def _v2_extract_conditioning(self, encoded):
        if not isinstance(encoded, list) or not encoded:
            return None, {"pooled_output": None}
        item = encoded[0]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            cond = item[0]
            meta = item[1] if isinstance(item[1], dict) else {"pooled_output": None}
        else:
            cond = item if isinstance(item, torch.Tensor) else None
            meta = {"pooled_output": None}
        return cond, meta

    def _v2_encode_prompt(self, clip, prompt_text):
        if clip is None:
            return None, {"pooled_output": None}, "CLIP missing"
        prompt_text = str(prompt_text or "").strip()
        if not prompt_text:
            return None, {"pooled_output": None}, "prompt empty"
        try:
            encoded = clip.encode_from_tokens_scheduled(clip.tokenize(prompt_text))
        except Exception as error:
            return None, {"pooled_output": None}, f"encode failed: {error}"
        cond, meta = self._v2_extract_conditioning(encoded)
        if not isinstance(cond, torch.Tensor):
            return None, {"pooled_output": None}, "encode returned invalid conditioning"
        return cond, meta, f"encoded {self._get_conditioning_seq_len(cond)} positions"

    def _v2_conditioning_vector(self, conditioning):
        if not isinstance(conditioning, torch.Tensor) or conditioning.dim() <= 1:
            return None
        if conditioning.dim() == 3:
            return conditioning.detach().float().mean(dim=(0, 1))
        return conditioning.detach().float().mean(dim=0)

    def _v2_phrase_words(self, text):
        return [
            word.strip().lower()
            for word in re.findall(r"[\w'’.-]+", str(text or ""), flags=re.UNICODE)
            if self._is_valuable_token(word.strip())
        ]

    def _v2_has_action_suffix(self, words):
        for word in words:
            if word.endswith("ing"):
                stem = word[:-3]
            elif word.endswith("ed"):
                stem = word[:-2]
            else:
                continue
            if stem in self.ACTION_SUFFIX_STEMS:
                return True
            if stem.endswith("n") and stem[:-1] in self.ACTION_SUFFIX_STEMS:
                return True
            if stem.endswith("p") and stem[:-1] in self.ACTION_SUFFIX_STEMS:
                return True
        return False

    def _v2_rebalance_anchored_scores(self, scores, words):
        words = set(words or [])
        has_action = bool(words & self.CATEGORY_KEYWORDS["action"]) or self._v2_has_action_suffix(words)
        if words & self.APPEARANCE_CONTEXT_WORDS:
            scores["appearance"] = max(scores["appearance"], 0.66)
            if not has_action:
                scores["action"] = min(scores["action"], 0.40)
        if words & self.ENVIRONMENT_CONTEXT_WORDS:
            scores["environment"] = max(scores["environment"], 0.66)
            if not has_action:
                scores["action"] = min(scores["action"], 0.36)
        if words & self.QUALITY_CONTEXT_WORDS:
            scores["quality"] = max(scores["quality"], 0.64)
            if not has_action:
                scores["action"] = min(scores["action"], 0.34)
        if has_action:
            scores["action"] = max(scores["action"], 0.66)
        return scores

    def _v2_heuristic_scores(self, phrase):
        words = set(self._v2_phrase_words(phrase))
        scores = {category: 0.0 for category in self.CATEGORY_DESCRIPTIONS}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            hits = len(words & keywords)
            if hits:
                scores[category] += min(0.92, 0.48 + hits * 0.18)

        text = str(phrase or "").lower()
        if self._v2_has_action_suffix(words):
            scores["action"] = max(scores["action"], 0.52)
        if "camera" in text and re.search(r"\b(move|moves|moving|push|pull|pan|zoom|track|dolly|tilt)", text):
            scores["camera"] = max(scores["camera"], 0.78)
            scores["action"] = max(scores["action"], 0.48)
        if re.search(r"\b(slowly|quickly|gentle|smooth|fast)\b", text) and scores["action"] > 0.0:
            scores["action"] = min(0.92, scores["action"] + 0.12)
        scores = self._v2_rebalance_anchored_scores(scores, words)
        return scores

    def _v2_scores_primary(self, scores):
        if not scores:
            return "details", 0.0
        primary, confidence = max(
            scores.items(),
            key=lambda item: (item[1], self.CATEGORY_PRIORITY.get(item[0], 0.0)),
        )
        return primary, float(confidence)

    def _v2_clip_similarity_scores(self, clip, phrase, category_vectors):
        phrase_cond, _, _ = self._v2_encode_prompt(clip, phrase)
        phrase_vector = self._v2_conditioning_vector(phrase_cond)
        if phrase_vector is None:
            return {}
        scores = {}
        for category, vector in category_vectors.items():
            if not isinstance(vector, torch.Tensor) or vector.shape != phrase_vector.shape:
                continue
            sim = F.cosine_similarity(phrase_vector.unsqueeze(0), vector.to(phrase_vector.device).unsqueeze(0), dim=-1)
            scores[category] = float(((sim.item() + 1.0) * 0.5))
        return scores

    def _v2_category_vectors(self, clip):
        vectors = {}
        for category, description in self.CATEGORY_DESCRIPTIONS.items():
            cond, _, _ = self._v2_encode_prompt(clip, description)
            vector = self._v2_conditioning_vector(cond)
            if vector is not None:
                vectors[category] = vector.cpu()
        return vectors

    def _v2_merge_clip_category_scores(self, heuristic_scores, clip_scores):
        if not clip_scores:
            return heuristic_scores
        merged = dict(heuristic_scores)
        heuristic_primary, heuristic_confidence = self._v2_scores_primary(heuristic_scores)
        ranked_clip = sorted(clip_scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked_clip:
            return merged
        clip_primary, clip_confidence = ranked_clip[0]
        clip_runner_up = ranked_clip[1][1] if len(ranked_clip) > 1 else 0.0
        clip_margin = float(clip_confidence) - float(clip_runner_up)

        if heuristic_confidence >= 0.60:
            return merged
        if heuristic_confidence >= 0.44 and clip_primary != heuristic_primary and clip_margin < 0.075:
            return merged

        mean_clip = sum(float(value) for value in clip_scores.values()) / max(1, len(clip_scores))
        for category, score in clip_scores.items():
            relative = max(0.0, float(score) - mean_clip)
            merged[category] = max(float(merged.get(category, 0.0)), min(0.56, relative * 2.40))
        if clip_margin >= 0.045:
            merged[clip_primary] = max(float(merged.get(clip_primary, 0.0)), min(0.68, 0.48 + clip_margin * 2.20))
        return merged

    def _v2_category_template(self, value=0.0):
        return {category: float(value) for category in self.CATEGORY_DESCRIPTIONS}

    def _v2_clean_category_scores(self, scores, default=0.0):
        cleaned = self._v2_category_template(default)
        if isinstance(scores, dict):
            for category in cleaned:
                try:
                    cleaned[category] = float(scores.get(category, cleaned[category]))
                except (TypeError, ValueError):
                    cleaned[category] = float(default)
        return cleaned

    def _v2_axis_categories(self, axis):
        if axis == "action":
            return {"action", "camera"}
        if axis == "details":
            return {"details", "appearance", "environment", "camera", "action"}
        if axis == "quality":
            return {"quality", "style"}
        return set()

    def _v2_axis_category_weight(self, axis, category, scores=None):
        if category not in self._v2_axis_categories(axis):
            return 0.0
        if axis == "details" and category == "details":
            return 1.18
        if axis == "details" and category == "action":
            return 0.58
        if axis == "details" and category in {"appearance", "environment"}:
            return 0.72
        if axis == "action" and category == "camera":
            return 0.58
        if axis == "details" and category == "camera":
            return 0.64
        scores = scores or {}
        primary, _ = self._v2_scores_primary(scores)
        return 1.0 if category == primary else 0.78

    def _v2_effective_category_scores(self, machine_scores, category_weights, evidence_count=0):
        machine = self._v2_clean_category_scores(machine_scores)
        learned = self._v2_clean_category_scores(category_weights)
        evidence_count = max(0, int(evidence_count or 0))
        learned_power = min(1.85, 0.30 + evidence_count * 0.18)
        effective = {}
        for category in machine:
            value = float(machine.get(category, 0.0)) + float(learned.get(category, 0.0)) * learned_power
            effective[category] = round(max(-1.0, min(8.0, value)), 6)
        return effective

    def _v2_position_bucket(self, index, total):
        total = max(1, int(total or 1))
        index = max(0, int(index or 0))
        if total <= 2:
            return "solo" if total == 1 else ("early" if index == 0 else "late")
        ratio = index / float(max(1, total - 1))
        if ratio < 0.34:
            return "early"
        if ratio > 0.66:
            return "late"
        return "mid"

    def _v2_context_for_phrase(self, phrases, index, text, window=3):
        phrases = phrases or []
        index = max(0, min(len(phrases) - 1, int(index or 0))) if phrases else 0
        anchor_words = set(self._v2_phrase_words(text))
        context_words = []
        for offset in range(max(0, index - window), min(len(phrases), index + window + 1)):
            if offset == index:
                continue
            item = phrases[offset]
            item_text = item.get("text", "") if isinstance(item, dict) else str(item)
            for word in self._v2_phrase_words(item_text):
                if word in anchor_words or word in context_words:
                    continue
                context_words.append(word)
        bucket = self._v2_position_bucket(index, len(phrases))
        return {
            "anchor_words": sorted(anchor_words),
            "context_words": context_words[:10],
            "position_bucket": bucket,
            "window": int(window),
        }

    def _v2_context_signature(self, context):
        if not isinstance(context, dict):
            return ""
        words = sorted({
            word
            for word in context.get("context_words", [])
            if self._is_valuable_token(str(word).strip())
        })[:8]
        bucket = str(context.get("position_bucket", "mid"))
        return f"{bucket}|{','.join(words)}" if words else bucket

    def _v2_context_similarity(self, left, right):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return 0.0
        left_words = {
            word for word in left.get("context_words", [])
            if self._is_valuable_token(str(word).strip())
        }
        right_words = {
            word for word in right.get("context_words", [])
            if self._is_valuable_token(str(word).strip())
        }
        if not left_words and not right_words:
            word_score = 0.0
        else:
            word_score = len(left_words & right_words) / float(max(1, len(left_words | right_words)))
        bucket_bonus = 0.12 if left.get("position_bucket") == right.get("position_bucket") else 0.0
        return max(0.0, min(1.0, word_score + bucket_bonus))

    def _v2_ensure_context_sense(self, entry, context):
        signature = self._v2_context_signature(context)
        if not signature:
            return None
        senses = entry.setdefault("context_senses", {})
        initial_weights = self._v2_clean_category_scores(entry.get("category_weights", {}))
        sense = senses.setdefault(signature, {
            "signature": signature,
            "context": context,
            "category_weights": {
                category: round(float(value) * 0.35, 6)
                for category, value in initial_weights.items()
            },
            "effective_category_scores": self._v2_clean_category_scores(entry.get("effective_category_scores", {})),
            "category_evidence_count": 0,
            "occurrence_count": 0,
            "last_seen_iter": 0,
            "rating_evidence": {},
        })
        sense["context"] = context
        sense["category_weights"] = self._v2_clean_category_scores(sense.get("category_weights", {}))
        sense["effective_category_scores"] = self._v2_clean_category_scores(sense.get("effective_category_scores", {}))
        sense.setdefault("category_evidence_count", 0)
        sense.setdefault("occurrence_count", 0)
        sense.setdefault("last_seen_iter", 0)
        sense.setdefault("rating_evidence", {})
        return sense

    def _v2_prune_context_senses(self, entry, limit=24):
        senses = entry.get("context_senses", {})
        if not isinstance(senses, dict) or len(senses) <= limit:
            return

        def rank(item):
            _, sense = item
            if not isinstance(sense, dict):
                return (-1, -1, -1)
            return (
                int(sense.get("category_evidence_count", 0)),
                int(sense.get("occurrence_count", 0)),
                int(sense.get("last_seen_iter", 0)),
            )

        kept = dict(sorted(senses.items(), key=rank, reverse=True)[:limit])
        entry["context_senses"] = kept

    def _v2_best_context_sense(self, entry, context):
        if not isinstance(entry, dict) or not isinstance(context, dict):
            return None, 0.0
        senses = entry.get("context_senses", {})
        if not isinstance(senses, dict):
            return None, 0.0
        signature = self._v2_context_signature(context)
        if signature and isinstance(senses.get(signature), dict):
            return senses[signature], 1.0
        best_sense = None
        best_score = 0.0
        for sense in senses.values():
            if not isinstance(sense, dict):
                continue
            score = self._v2_context_similarity(context, sense.get("context", {}))
            if score > best_score:
                best_sense = sense
                best_score = score
        if best_score < 0.22:
            return None, 0.0
        return best_sense, best_score

    def _v2_blend_category_weights(self, target, source, scale=1.0):
        source = self._v2_clean_category_scores(source)
        for category, value in source.items():
            target[category] = float(target.get(category, 0.0)) + float(value) * float(scale)
        return target

    def _v2_ensure_phrase_memory_entry(self, memory, text, phrase=None):
        text = str(text or "").strip().lower()
        phrase = phrase or {}
        primary = phrase.get("primary") or phrase.get("machine_primary") or "details"
        entry = memory.setdefault(text, {
            "text": text,
            "tokens": phrase.get("tokens", []),
            "primary": primary,
            "machine_primary": primary,
            "category_scores": phrase.get("category_scores", {}),
            "clip_heuristic_scores": phrase.get("clip_heuristic_scores", phrase.get("category_scores", {})),
            "category_weights": self._v2_category_template(0.0),
            "effective_category_scores": phrase.get("effective_category_scores", phrase.get("category_scores", {})),
            "category_evidence_count": 0,
            "rating_evidence": {},
            "context_senses": {},
            "occurrence_count": 0,
            "score": 0.0,
            "liked_count": 0,
            "missing_count": 0,
            "satisfied_count": 0,
            "resolved_count": 0,
            "regressed_count": 0,
            "bad_count": 0,
            "wanted_axes": {},
            "satisfied_axes": {},
            "resolved_axes": {},
            "positions": {},
        })
        entry.setdefault("text", text)
        entry.setdefault("tokens", phrase.get("tokens", []))
        entry.setdefault("primary", primary)
        entry.setdefault("machine_primary", phrase.get("machine_primary", entry.get("primary", primary)))
        entry["category_scores"] = self._v2_clean_category_scores(entry.get("category_scores", phrase.get("category_scores", {})))
        entry["clip_heuristic_scores"] = self._v2_clean_category_scores(
            entry.get("clip_heuristic_scores", entry.get("category_scores", {}))
        )
        entry["category_weights"] = self._v2_clean_category_scores(entry.get("category_weights", {}))
        entry["effective_category_scores"] = self._v2_clean_category_scores(
            entry.get("effective_category_scores", entry.get("category_scores", {}))
        )
        entry.setdefault("category_evidence_count", 0)
        entry.setdefault("rating_evidence", {})
        entry.setdefault("context_senses", {})
        entry.setdefault("occurrence_count", 0)
        entry.setdefault("score", 0.0)
        entry.setdefault("liked_count", 0)
        entry.setdefault("missing_count", 0)
        entry.setdefault("satisfied_count", 0)
        entry.setdefault("resolved_count", 0)
        entry.setdefault("regressed_count", 0)
        entry.setdefault("bad_count", 0)
        entry.setdefault("wanted_axes", {})
        entry.setdefault("satisfied_axes", {})
        entry.setdefault("resolved_axes", {})
        entry.setdefault("positions", {})
        return entry

    def _v2_apply_learned_category_scores(self, item, phrase_memory):
        if not isinstance(phrase_memory, dict):
            return item
        text = str(item.get("text", "")).strip().lower()
        context = item.get("context", {})
        entry = phrase_memory.get(text)
        machine_scores = self._v2_clean_category_scores(item.get("category_scores", {}))
        weights = self._v2_category_template(0.0)
        evidence_count = 0
        context_status = "none"
        if isinstance(entry, dict):
            sense, sense_similarity = self._v2_best_context_sense(entry, context)
            entry_scale = 0.35 if isinstance(sense, dict) else 1.0
            self._v2_blend_category_weights(weights, entry.get("category_weights", {}), entry_scale)
            evidence_count += int(int(entry.get("category_evidence_count", 0)) * entry_scale)
            if isinstance(sense, dict):
                self._v2_blend_category_weights(
                    weights,
                    sense.get("category_weights", {}),
                    1.20 + sense_similarity * 0.65,
                )
                evidence_count += int(int(sense.get("category_evidence_count", 0)) * max(0.35, sense_similarity))
                context_status = "exact" if sense_similarity >= 0.99 else f"near:{sense_similarity:.2f}"

        for word in self._v2_phrase_words(text):
            token_entry = phrase_memory.get(word)
            if not isinstance(token_entry, dict) or token_entry is entry:
                continue
            self._v2_blend_category_weights(weights, token_entry.get("category_weights", {}), 0.32)
            evidence_count += int(int(token_entry.get("category_evidence_count", 0)) * 0.22)
            sense, sense_similarity = self._v2_best_context_sense(token_entry, context)
            if isinstance(sense, dict):
                self._v2_blend_category_weights(
                    weights,
                    sense.get("category_weights", {}),
                    0.65 + sense_similarity * 0.55,
                )
                evidence_count += int(int(sense.get("category_evidence_count", 0)) * max(0.42, sense_similarity))
                context_status = "exact" if sense_similarity >= 0.99 else f"near:{sense_similarity:.2f}"

        effective = self._v2_effective_category_scores(machine_scores, weights, evidence_count)
        primary, confidence = self._v2_scores_primary(effective)
        machine_primary, machine_confidence = self._v2_scores_primary(machine_scores)
        item["clip_heuristic_scores"] = machine_scores
        item["machine_primary"] = machine_primary
        item["machine_confidence"] = round(float(machine_confidence), 4)
        item["category_weights"] = weights
        item["category_evidence_count"] = evidence_count
        item["effective_category_scores"] = effective
        item["category_scores"] = effective
        item["primary"] = primary
        item["confidence"] = round(float(confidence), 4)
        item["source"] = "rating_weighted" if evidence_count > 0 else item.get("source", "heuristic")
        item["context_signature"] = self._v2_context_signature(context)
        item["context_source"] = context_status
        return item

    def _v2_concept_units_for_run(self, last_run):
        units = []
        seen = set()

        def add(text, kind, phrase=None, position=0, context=None):
            clean = str(text or "").strip().lower()
            context = context or {}
            seen_key = f"{clean}:{self._v2_context_signature(context)}"
            if len(clean) < 3 or seen_key in seen or not any(c.isalpha() for c in clean):
                return
            words = self._v2_phrase_words(clean)
            if not words:
                return
            seen.add(seen_key)
            source = phrase or {}
            scores = source.get("clip_heuristic_scores", source.get("category_scores", self._v2_heuristic_scores(clean)))
            primary, confidence = self._v2_scores_primary(scores)
            units.append({
                "text": clean,
                "kind": kind,
                "tokens": source.get("tokens", words),
                "position": int(position),
                "context": context,
                "context_signature": self._v2_context_signature(context),
                "category_scores": self._v2_clean_category_scores(scores),
                "clip_heuristic_scores": self._v2_clean_category_scores(scores),
                "effective_category_scores": self._v2_clean_category_scores(source.get("effective_category_scores", scores)),
                "primary": source.get("primary", primary),
                "machine_primary": source.get("machine_primary", primary),
                "confidence": source.get("confidence", round(float(confidence), 4)),
            })

        for index, phrase in enumerate(last_run.get("phrases", []) or []):
            if not isinstance(phrase, dict):
                continue
            text = str(phrase.get("text", "")).strip().lower()
            phrase_context = phrase.get("context") or self._v2_context_for_phrase(last_run.get("phrases", []), index, text)
            add(text, "phrase", phrase, phrase.get("position", index), phrase_context)
            words = self._v2_phrase_words(text)
            for word_index, word in enumerate(words):
                token_context = self._v2_context_for_phrase(last_run.get("phrases", []), index, word)
                add(word, "token", phrase, phrase.get("position", index) + word_index, token_context)
            for size in (2, 3):
                for start in range(0, max(0, len(words) - size + 1)):
                    add(" ".join(words[start:start + size]), "ngram", phrase, phrase.get("position", index) + start, phrase_context)

        prompt = str(last_run.get("prompt", "") or "")
        prompt_segments = [
            {"text": segment.strip()}
            for segment in re.split(r"[,.;\n]+", prompt)
            if segment.strip()
        ]
        for position, segment in enumerate(prompt_segments):
            add(segment.get("text", ""), "prompt_phrase", None, position, self._v2_context_for_phrase(prompt_segments, position, segment.get("text", "")))
        return units[:96]

    def _v2_classify_phrases(self, clip, phrases, global_state=None):
        phrase_items = []
        uncertain = []
        phrase_memory = (global_state or {}).get("phrase_memory", {}) if isinstance(global_state, dict) else {}
        phrases = phrases or []
        for index, phrase in enumerate(phrases):
            text = str(phrase.get("text", "")).strip().lower()
            if not text:
                continue
            scores = self._v2_heuristic_scores(text)
            primary, confidence = self._v2_scores_primary(scores)
            context = self._v2_context_for_phrase(phrases, index, text)
            item = {
                "text": text,
                "tokens": list(phrase.get("tokens", [])),
                "position": index,
                "context": context,
                "context_signature": self._v2_context_signature(context),
                "category_scores": scores,
                "clip_heuristic_scores": dict(scores),
                "machine_primary": primary,
                "machine_confidence": round(float(confidence), 4),
                "category_weights": self._v2_category_template(0.0),
                "category_evidence_count": 0,
                "effective_category_scores": dict(scores),
                "primary": primary,
                "confidence": round(float(confidence), 4),
                "source": "heuristic",
            }
            if confidence < 0.62:
                uncertain.append(item)
            phrase_items.append(item)

        if uncertain and clip is not None:
            category_vectors = self._v2_category_vectors(clip)
            for item in uncertain:
                clip_scores = self._v2_clip_similarity_scores(clip, item["text"], category_vectors)
                if not clip_scores:
                    continue
                merged = self._v2_merge_clip_category_scores(item["category_scores"], clip_scores)
                primary, confidence = self._v2_scores_primary(merged)
                item["category_scores"] = merged
                item["clip_heuristic_scores"] = dict(merged)
                item["machine_primary"] = primary
                item["machine_confidence"] = round(float(confidence), 4)
                item["effective_category_scores"] = dict(merged)
                item["primary"] = primary
                item["confidence"] = round(float(confidence), 4)
                item["source"] = "clip_similarity"

        for item in phrase_items:
            self._v2_apply_learned_category_scores(item, phrase_memory)

        return phrase_items

    def _v2_axis_matches_category(self, axis, category):
        category = (category or "details").lower()
        return category in self._v2_axis_categories(axis)

    def _v2_order_axes(self, axes):
        axis_set = {axis for axis in axes or [] if axis in V2_FEEDBACK_AXES}
        return [axis for axis in V2_FEEDBACK_AXES if axis in axis_set]

    def _v2_axes_for_category(self, category):
        return {
            axis
            for axis in V2_FEEDBACK_AXES
            if self._v2_axis_matches_category(axis, category)
        }

    def _v2_axes_for_scores(self, scores):
        primary, confidence = self._v2_scores_primary(scores)
        axes = set(self._v2_axes_for_category(primary))
        for category, score in (scores or {}).items():
            if float(score or 0.0) >= max(0.28, float(confidence) * 0.72):
                axes |= self._v2_axes_for_category(category)
        return axes

    def _v2_axis_feedback(self, rating_profile, previous_missing_axes=None):
        key = rating_profile.get("key", "")
        if rating_profile.get("skip_learning") or key == "discover":
            return {
                "missing_axes": [],
                "satisfied_axes": [],
                "resolved_axes": [],
                "regressed_axes": [],
            }

        all_axes = set(V2_FEEDBACK_AXES)
        missing_axes = set(rating_profile.get("missing_axes", [])) & all_axes
        if key == "awful":
            missing_axes = set(all_axes)
        elif key == "like":
            missing_axes = set()

        has_previous_axis_signal = previous_missing_axes is not None
        previous_missing = set(previous_missing_axes or []) & all_axes
        satisfied_axes = all_axes - missing_axes
        resolved_axes = previous_missing - missing_axes
        regressed_axes = missing_axes - previous_missing if has_previous_axis_signal else set()

        return {
            "missing_axes": self._v2_order_axes(missing_axes),
            "satisfied_axes": self._v2_order_axes(satisfied_axes),
            "resolved_axes": self._v2_order_axes(resolved_axes),
            "regressed_axes": self._v2_order_axes(regressed_axes),
        }

    def _v2_axis_feedback_status(self, axis_feedback):
        if not isinstance(axis_feedback, dict):
            return "Axis feedback: none."

        def fmt(name):
            values = self._v2_order_axes(axis_feedback.get(name, []))
            return ", ".join(values) if values else "none"

        return (
            f"Axis feedback: missing {fmt('missing_axes')} | "
            f"satisfied {fmt('satisfied_axes')} | "
            f"resolved {fmt('resolved_axes')} | "
            f"regressed {fmt('regressed_axes')}."
        )

    def _v2_top_category_summary(self, scores, limit=3):
        cleaned = self._v2_clean_category_scores(scores)
        ranked = [
            (category, value)
            for category, value in sorted(cleaned.items(), key=lambda item: item[1], reverse=True)
            if abs(float(value)) > 0.0001
        ]
        if not ranked:
            return "neutral"
        return ", ".join(f"{category}:{value:.2f}" for category, value in ranked[:limit])

    def _v2_training_guidance(self, has_previous_run, learning_profile, axis_feedback, phrases, memory_status, lora_status):
        if not has_previous_run:
            return (
                "Guidance: first V2 run only seeds the current prompt. "
                "Set the rating after reviewing this output, then run again with the same refinement key to train."
            )
        if learning_profile.get("skip_learning"):
            return (
                "Guidance: '-Just forget it-' skipped learning. Use it for workflow/model/seed failures; "
                "use a missing-axis rating when the prompt should teach the node."
            )

        missing_axes = set(axis_feedback.get("missing_axes", [])) if isinstance(axis_feedback, dict) else set()
        resolved_axes = set(axis_feedback.get("resolved_axes", [])) if isinstance(axis_feedback, dict) else set()
        regressed_axes = set(axis_feedback.get("regressed_axes", [])) if isinstance(axis_feedback, dict) else set()
        phrase_axes = set()
        context_sources = set()
        weak_contexts = 0
        for phrase in phrases or []:
            phrase_axes |= self._v2_axes_for_scores(
                phrase.get("effective_category_scores", phrase.get("category_scores", {})) or
                {phrase.get("primary", "details"): 1.0}
            )
            context_source = str(phrase.get("context_source", "none"))
            context_sources.add(context_source)
            if context_source == "none":
                weak_contexts += 1

        notes = []
        if missing_axes:
            if not (missing_axes & phrase_axes):
                notes.append(
                    "missing axes did not strongly match current prompt concepts; make the missing action/detail/quality words more explicit"
                )
            else:
                notes.append(
                    "missing axes matched prompt concepts; repeated consistent ratings will strengthen those category weights"
                )
        if resolved_axes:
            notes.append("resolved axes were preserved as positive evidence")
        if regressed_axes:
            notes.append("regressed axes were marked as needing repair")
        if weak_contexts and phrases:
            notes.append("some concepts have no learned context yet; repeat similar phrasing if you want stable sense learning")
        if "exact" in context_sources:
            notes.append("at least one concept matched an exact learned context")
        if lora_status.startswith("LoRA suggestions: no"):
            notes.append("no LoRA stack connected, so only prompt/conditioning memory trained")
        if "no prompt phrases stored" in memory_status:
            notes.append("no valuable prompt concepts were found; add concrete descriptive words")
        return "Guidance: " + "; ".join(notes[:4]) + "." if notes else "Guidance: training signal looks usable."

    def _v2_category_diagnostics(self, phrases, limit=8):
        parts = []
        for item in (phrases or [])[:limit]:
            text = item.get("text", "")
            machine = item.get("machine_primary", item.get("primary", "details"))
            effective = item.get("primary", "details")
            context = item.get("context_source", "none")
            evidence = int(item.get("category_evidence_count", 0))
            top = self._v2_top_category_summary(item.get("effective_category_scores", item.get("category_scores", {})), limit=2)
            parts.append(f"{text}: {machine}->{effective}, ctx={context}, e={evidence}, top={top}")
        return "Category diagnostics: " + (" | ".join(parts) if parts else "none")

    def _v2_update_phrase_memory(self, global_state, last_run, rating_profile, iter_num, axis_feedback=None):
        if not isinstance(last_run, dict) or rating_profile.get("skip_learning"):
            return "Lucky memory: no learning update."
        memory = global_state.setdefault("phrase_memory", {})
        concepts = self._v2_concept_units_for_run(last_run)
        axis_feedback = axis_feedback or self._v2_axis_feedback(
            rating_profile,
            global_state.get("last_missing_axes", []),
        )
        missing_axes = set(axis_feedback.get("missing_axes", []))
        satisfied_axes = set(axis_feedback.get("satisfied_axes", []))
        resolved_axes = set(axis_feedback.get("resolved_axes", []))
        regressed_axes = set(axis_feedback.get("regressed_axes", []))
        reward = float(rating_profile.get("reward", 0.0))
        touched = []
        trained = []
        kind_counts = {}
        for phrase in concepts:
            text = str(phrase.get("text", "")).strip().lower()
            if not text:
                continue
            entry = self._v2_ensure_phrase_memory_entry(memory, text, phrase)
            entry["kind"] = phrase.get("kind", entry.get("kind", "phrase"))
            kind_counts[entry["kind"]] = int(kind_counts.get(entry["kind"], 0)) + 1
            entry["machine_primary"] = phrase.get("machine_primary", entry.get("machine_primary", entry.get("primary", "details")))
            entry["clip_heuristic_scores"] = self._v2_clean_category_scores(
                phrase.get("clip_heuristic_scores", phrase.get("category_scores", entry.get("clip_heuristic_scores", {})))
            )
            context = phrase.get("context", {})
            context_sense = self._v2_ensure_context_sense(entry, context)
            if isinstance(context_sense, dict):
                context_sense["occurrence_count"] = int(context_sense.get("occurrence_count", 0)) + 1
                context_sense["last_seen_iter"] = int(iter_num)
            entry["tokens"] = phrase.get("tokens", entry.get("tokens", []))
            entry["occurrence_count"] = int(entry.get("occurrence_count", 0)) + 1
            entry.setdefault("first_seen_iter", int(iter_num))
            entry.setdefault("positions", {})[str(int(phrase.get("position", 0)))] = (
                int(entry.setdefault("positions", {}).get(str(int(phrase.get("position", 0))), 0)) + 1
            )

            machine_scores = self._v2_clean_category_scores(entry.get("clip_heuristic_scores", {}))
            weights = self._v2_clean_category_scores(entry.get("category_weights", {}))
            effective_before = self._v2_effective_category_scores(
                machine_scores,
                weights,
                int(entry.get("category_evidence_count", 0)),
            )
            phrase_axes = self._v2_axes_for_scores(effective_before)
            matched_missing = phrase_axes & missing_axes
            matched_satisfied = phrase_axes & satisfied_axes
            matched_resolved = phrase_axes & resolved_axes
            matched_regressed = phrase_axes & regressed_axes
            rating_evidence = entry.setdefault("rating_evidence", {})
            weights_before = dict(weights)

            def bump_evidence(name, axes):
                axis_counts = rating_evidence.setdefault(name, {})
                for axis in self._v2_order_axes(axes):
                    axis_counts[axis] = int(axis_counts.get(axis, 0)) + 1

            def train_axis(axis, amount, prefer_current=True):
                for category in self._v2_axis_categories(axis):
                    relation = self._v2_axis_category_weight(axis, category, effective_before if prefer_current else machine_scores)
                    weights[category] = float(weights.get(category, 0.0)) + amount * relation

            if rating_profile.get("key") == "like":
                delta = 0.55
                entry["liked_count"] = int(entry.get("liked_count", 0)) + 1
                rating_evidence["liked"] = int(rating_evidence.get("liked", 0)) + 1
                entry["satisfied_count"] = int(entry.get("satisfied_count", 0)) + 1
                satisfied_axis_counts = entry.setdefault("satisfied_axes", {})
                for axis in V2_FEEDBACK_AXES:
                    satisfied_axis_counts[axis] = int(satisfied_axis_counts.get(axis, 0)) + 1
                bump_evidence("satisfied_axes", V2_FEEDBACK_AXES)
                for category, score in effective_before.items():
                    if float(score) >= 0.28:
                        weights[category] = float(weights.get(category, 0.0)) + 0.13
                for axis in V2_FEEDBACK_AXES:
                    train_axis(axis, 0.035)
            elif rating_profile.get("key") == "awful":
                delta = -0.90
                entry["bad_count"] = int(entry.get("bad_count", 0)) + 1
                rating_evidence["bad"] = int(rating_evidence.get("bad", 0)) + 1
                primary_before, _ = self._v2_scores_primary(effective_before)
                weights[primary_before] = float(weights.get(primary_before, 0.0)) - 0.045
                bump_evidence("missing_axes", missing_axes)
                for axis in self._v2_order_axes(missing_axes):
                    train_axis(axis, 0.045, prefer_current=False)
            elif missing_axes:
                for axis in self._v2_order_axes(missing_axes - matched_missing):
                    train_axis(axis, 0.14, prefer_current=False)
                if matched_missing:
                    delta = 0.34 + (0.08 if matched_regressed else 0.0)
                    entry["missing_count"] = int(entry.get("missing_count", 0)) + 1
                    for axis in self._v2_order_axes(matched_missing):
                        train_axis(axis, 0.18 + (0.05 if axis in matched_regressed else 0.0))
                elif matched_resolved:
                    delta = 0.46
                    entry["resolved_count"] = int(entry.get("resolved_count", 0)) + 1
                    for axis in self._v2_order_axes(matched_resolved):
                        train_axis(axis, 0.15)
                elif matched_satisfied:
                    delta = 0.20
                    entry["satisfied_count"] = int(entry.get("satisfied_count", 0)) + 1
                    for axis in self._v2_order_axes(matched_satisfied):
                        train_axis(axis, 0.075)
                else:
                    delta = 0.04

                wanted_axes = entry.setdefault("wanted_axes", {})
                for axis in self._v2_order_axes(matched_missing):
                    wanted_axes[axis] = int(wanted_axes.get(axis, 0)) + 1
                satisfied_axis_counts = entry.setdefault("satisfied_axes", {})
                for axis in self._v2_order_axes(matched_satisfied):
                    satisfied_axis_counts[axis] = int(satisfied_axis_counts.get(axis, 0)) + 1
                resolved_axis_counts = entry.setdefault("resolved_axes", {})
                for axis in self._v2_order_axes(matched_resolved):
                    resolved_axis_counts[axis] = int(resolved_axis_counts.get(axis, 0)) + 1
                bump_evidence("missing_axes", missing_axes)
                bump_evidence("satisfied_axes", satisfied_axes)
                bump_evidence("resolved_axes", resolved_axes)
                bump_evidence("regressed_axes", regressed_axes)
            else:
                delta = reward * 0.20
                for category, score in effective_before.items():
                    if float(score) >= 0.32:
                        weights[category] = float(weights.get(category, 0.0)) + reward * 0.035

            entry["category_weights"] = {
                category: round(max(-1.5, min(2.5, float(value))), 6)
                for category, value in weights.items()
            }
            entry["category_evidence_count"] = int(entry.get("category_evidence_count", 0)) + (
                0 if rating_profile.get("key") == "discover" else 1
            )
            if isinstance(context_sense, dict) and rating_profile.get("key") != "discover":
                sense_weights = self._v2_clean_category_scores(context_sense.get("category_weights", {}))
                for category, value in entry["category_weights"].items():
                    sense_weights[category] = float(sense_weights.get(category, 0.0)) + (
                        float(value) - float(weights_before.get(category, 0.0))
                    ) * 1.15
                context_sense["category_weights"] = {
                    category: round(max(-1.5, min(2.5, float(value))), 6)
                    for category, value in sense_weights.items()
                }
                context_sense["category_evidence_count"] = int(context_sense.get("category_evidence_count", 0)) + 1
                context_sense["effective_category_scores"] = self._v2_effective_category_scores(
                    machine_scores,
                    context_sense["category_weights"],
                    int(context_sense.get("category_evidence_count", 0)),
                )
                sense_evidence = context_sense.setdefault("rating_evidence", {})
                sense_evidence[rating_profile.get("key", "unknown")] = (
                    int(sense_evidence.get(rating_profile.get("key", "unknown"), 0)) + 1
                )
                self._v2_prune_context_senses(entry)
            entry["effective_category_scores"] = self._v2_effective_category_scores(
                machine_scores,
                entry["category_weights"],
                int(entry.get("category_evidence_count", 0)),
            )
            primary, confidence = self._v2_scores_primary(entry["effective_category_scores"])
            entry["primary"] = primary
            entry["confidence"] = round(float(confidence), 4)
            entry["category_scores"] = dict(entry["effective_category_scores"])
            entry["score"] = round(max(-3.0, min(6.0, float(entry.get("score", 0.0)) + delta)), 4)
            entry["last_seen_iter"] = int(iter_num)
            memory[text] = entry
            touched.append(text)
            if len(trained) < 8 and entry.get("kind") in {"phrase", "token", "ngram"}:
                trained.append(
                    f"{text}[{entry.get('kind')}:{entry['machine_primary']}->{primary}, "
                    f"ctx={phrase.get('context_signature', 'none') or 'none'}, "
                    f"e={int(entry.get('category_evidence_count', 0))}, "
                    f"top={self._v2_top_category_summary(entry['effective_category_scores'], limit=2)}]"
                )

        if not touched:
            return "Lucky memory: no prompt phrases stored."
        kind_summary = ", ".join(f"{name}:{count}" for name, count in sorted(kind_counts.items())) or "none"
        return (
            f"Category memory trained: {len(touched)} concept unit(s) ({kind_summary}). "
            f"Sample: {', '.join(trained) if trained else ', '.join(touched[:6])}. "
            f"{self._v2_axis_feedback_status(axis_feedback)}"
        )

    def _v2_shape_compatible(self, payload, conditioning):
        return self._serialized_conditioning_compatible(payload, conditioning)

    def _v2_store_conditioning_average(self, slot, payload):
        count = int(slot.get("count", 0))
        existing = slot.get("conditioning")
        if count <= 0 or not isinstance(existing, dict):
            slot["conditioning"] = payload
            slot["count"] = 1
            return
        try:
            previous = serializable_to_tensor(existing)
            current = serializable_to_tensor(payload).to(previous.device)
            if list(previous.shape) != list(current.shape):
                slot["conditioning"] = payload
                slot["count"] = 1
                return
            averaged = (previous * count + current) / float(count + 1)
            slot["conditioning"] = tensor_to_serializable(averaged.cpu())
            slot["count"] = count + 1
        except Exception:
            slot["conditioning"] = payload
            slot["count"] = 1

    def _v2_update_axis_conditioning_memory(self, global_state, payload, axis_feedback):
        if not isinstance(payload, dict):
            return
        memory = global_state.setdefault("axis_conditioning_memory", {})
        for axis in self._v2_order_axes(axis_feedback.get("satisfied_axes", [])):
            axis_slot = memory.setdefault(axis, {})
            positive = axis_slot.setdefault("positive", {})
            self._v2_store_conditioning_average(positive, payload)
        for axis in self._v2_order_axes(axis_feedback.get("resolved_axes", [])):
            axis_slot = memory.setdefault(axis, {})
            positive = axis_slot.setdefault("positive", {})
            self._v2_store_conditioning_average(positive, payload)
            axis_slot["resolved_count"] = int(axis_slot.get("resolved_count", 0)) + 1
        for axis in self._v2_order_axes(axis_feedback.get("missing_axes", [])):
            axis_slot = memory.setdefault(axis, {})
            negative = axis_slot.setdefault("negative", {})
            self._v2_store_conditioning_average(negative, payload)
        for axis in self._v2_order_axes(axis_feedback.get("regressed_axes", [])):
            axis_slot = memory.setdefault(axis, {})
            axis_slot["regressed_count"] = int(axis_slot.get("regressed_count", 0)) + 1

    def _v2_update_conditioning_memory(self, global_state, last_run, rating_profile, axis_feedback=None):
        if not isinstance(last_run, dict) or rating_profile.get("skip_learning"):
            return
        payload = last_run.get("conditioning")
        if not isinstance(payload, dict):
            return
        key = rating_profile.get("key", "")
        axis_feedback = axis_feedback or self._v2_axis_feedback(
            rating_profile,
            global_state.get("last_missing_axes", []),
        )
        self._v2_update_axis_conditioning_memory(global_state, payload, axis_feedback)
        if key == "like":
            count = int(global_state.get("liked_conditioning_count", 0))
            if count <= 0 or not isinstance(global_state.get("liked_conditioning"), dict):
                global_state["liked_conditioning"] = payload
                global_state["liked_conditioning_count"] = 1
                return
            try:
                previous = serializable_to_tensor(global_state["liked_conditioning"])
                current = serializable_to_tensor(payload).to(previous.device)
                if list(previous.shape) != list(current.shape):
                    global_state["liked_conditioning"] = payload
                    global_state["liked_conditioning_count"] = 1
                    return
                averaged = (previous * count + current) / float(count + 1)
                global_state["liked_conditioning"] = tensor_to_serializable(averaged.cpu())
                global_state["liked_conditioning_count"] = count + 1
            except Exception:
                global_state["liked_conditioning"] = payload
                global_state["liked_conditioning_count"] = 1
        elif key == "awful" or float(rating_profile.get("reward", 0.0)) < -0.35:
            global_state["bad_conditioning"] = payload
            global_state["bad_conditioning_count"] = int(global_state.get("bad_conditioning_count", 0)) + 1

    def _v2_update_streaks(self, global_state, rating_profile):
        reward = float(rating_profile.get("reward", 0.0))
        avg = float(global_state.get("avg_reward_ema", 0.0))
        avg = 0.86 * avg + 0.14 * reward
        global_state["avg_reward_ema"] = round(avg, 6)
        if rating_profile.get("key") == "like":
            global_state["good_streak"] = int(global_state.get("good_streak", 0)) + 1
            global_state["bad_streak"] = 0
        elif reward < -0.25:
            global_state["bad_streak"] = int(global_state.get("bad_streak", 0)) + 1
            global_state["good_streak"] = 0
        else:
            global_state["good_streak"] = 0
            global_state["bad_streak"] = 0
        global_state["last_rating_label"] = rating_profile.get("label", "")
        global_state["last_missing_axes"] = list(rating_profile.get("missing_axes", []))

    def _v2_auto_strength(self, global_state):
        avg = float(global_state.get("avg_reward_ema", 0.0))
        good = int(global_state.get("good_streak", 0))
        bad = int(global_state.get("bad_streak", 0))
        strength = 0.030
        if good >= 3 or avg > 0.55:
            strength *= 0.42
        elif good >= 1 or avg > 0.25:
            strength *= 0.68
        if bad >= 3 or avg < -0.45:
            strength *= 2.20
        elif bad >= 1 or avg < -0.15:
            strength *= 1.45
        return max(0.008, min(0.085, strength))

    def _v2_apply_conditioning_payload(self, mixed, payload, strength):
        if not self._v2_shape_compatible(payload, mixed):
            return mixed
        try:
            target = serializable_to_tensor(payload).to(device=mixed.device, dtype=mixed.dtype)
            return mixed.lerp(target, strength)
        except Exception:
            return mixed

    def _v2_repel_conditioning_payload(self, mixed, payload, strength):
        if not self._v2_shape_compatible(payload, mixed):
            return mixed
        try:
            target = serializable_to_tensor(payload).to(device=mixed.device, dtype=mixed.dtype)
            return mixed + (mixed - target) * strength
        except Exception:
            return mixed

    def _v2_apply_conditioning_memory(self, conditioning, global_state, rating_profile, axis_feedback=None):
        if not isinstance(conditioning, torch.Tensor):
            return conditioning, "Adaptation: unavailable."
        original = conditioning.clone()
        mixed = conditioning.clone()
        strength = self._v2_auto_strength(global_state)
        axis_feedback = axis_feedback or self._v2_axis_feedback(
            rating_profile,
            global_state.get("last_missing_axes", []),
        )
        liked_payload = global_state.get("liked_conditioning")
        if self._v2_shape_compatible(liked_payload, mixed):
            try:
                liked = serializable_to_tensor(liked_payload).to(device=mixed.device, dtype=mixed.dtype)
                mixed = mixed.lerp(liked, strength)
            except Exception:
                pass
        axis_memory = global_state.get("axis_conditioning_memory", {})
        axis_actions = []
        missing_axes = set(axis_feedback.get("missing_axes", []))
        satisfied_axes = set(axis_feedback.get("satisfied_axes", []))
        for axis in V2_FEEDBACK_AXES:
            axis_slot = axis_memory.get(axis, {}) if isinstance(axis_memory, dict) else {}
            positive = axis_slot.get("positive", {}).get("conditioning") if isinstance(axis_slot, dict) else None
            negative = axis_slot.get("negative", {}).get("conditioning") if isinstance(axis_slot, dict) else None
            if axis in missing_axes:
                before = mixed
                mixed = self._v2_apply_conditioning_payload(mixed, positive, min(0.070, strength * 1.10))
                mixed = self._v2_repel_conditioning_payload(mixed, negative, min(0.055, strength * 0.82))
                if mixed is not before:
                    axis_actions.append(f"{axis}:repair")
            elif axis in satisfied_axes:
                before = mixed
                mixed = self._v2_apply_conditioning_payload(mixed, positive, min(0.026, strength * 0.34))
                if mixed is not before:
                    axis_actions.append(f"{axis}:preserve")
        bad_payload = global_state.get("bad_conditioning")
        if self._v2_shape_compatible(bad_payload, mixed) and float(rating_profile.get("reward", 0.0)) < 0.0:
            try:
                bad = serializable_to_tensor(bad_payload).to(device=mixed.device, dtype=mixed.dtype)
                repel = (mixed - bad) * min(0.055, strength * 0.72)
                mixed = mixed + repel
            except Exception:
                pass
        delta = mixed - original
        original_norm = original.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        max_delta = original_norm * 0.075
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = torch.minimum(torch.ones_like(delta_norm), max_delta / delta_norm)
        mixed = original + delta * scale
        mixed = torch.clamp(mixed, min=-60.0, max=60.0)
        mixed = mixed / mixed.norm(dim=-1, keepdim=True).clamp_min(1e-8) * original_norm
        return mixed, (
            f"Adaptation: automatic strength {strength:.3f} | "
            f"good streak {int(global_state.get('good_streak', 0))} | "
            f"bad streak {int(global_state.get('bad_streak', 0))} | "
            f"reward ema {float(global_state.get('avg_reward_ema', 0.0)):+.3f} | "
            f"axis memory {', '.join(axis_actions) if axis_actions else 'idle'}"
        )

    def _v2_emphasized_prompt(self, prompt, phrases, global_state, rating_profile):
        missing_axes = set(global_state.get("last_missing_axes", [])) | set(rating_profile.get("missing_axes", []))
        if not missing_axes:
            return prompt, "Prompt emphasis: none."
        candidates = []
        for phrase in phrases:
            phrase_axes = self._v2_axes_for_scores(
                phrase.get("effective_category_scores", phrase.get("category_scores", {})) or
                {phrase.get("primary", "details"): 1.0}
            )
            if phrase_axes & missing_axes:
                candidates.append(phrase.get("text", ""))
        candidates = [item for item in candidates if item]
        if not candidates:
            return prompt, "Prompt emphasis: no matching current phrases."
        emphasized = []
        seen = set()
        for text in candidates:
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            emphasized.append(text)
            if len(emphasized) >= 3:
                break
        return f"{prompt}, {', '.join(emphasized)}", f"Prompt emphasis: repeated {', '.join(emphasized)}."

    def _v2_compose_lucky_prompt(self, prompt, phrases, global_state):
        memory = global_state.get("phrase_memory", {})
        scored = []
        for text, entry in memory.items():
            if not isinstance(entry, dict):
                continue
            score = float(entry.get("score", 0.0))
            score += min(1.0, int(entry.get("liked_count", 0)) * 0.14)
            score += min(1.2, sum(int(v) for v in entry.get("wanted_axes", {}).values()) * 0.10)
            score -= min(1.6, int(entry.get("bad_count", 0)) * 0.22)
            if score <= 0.10:
                continue
            scored.append((score, entry.get("primary", "details"), str(entry.get("text", text)).strip()))
        scored = sorted(scored, key=lambda item: (item[0], item[1] == "action"), reverse=True)
        current_texts = [str(item.get("text", "")).strip().lower() for item in phrases]
        selected = []
        seen = set()
        for phrase in current_texts:
            if phrase and phrase not in seen:
                selected.append(phrase)
                seen.add(phrase)
        for _, _, text in scored:
            key = text.lower()
            if not key or key in seen:
                continue
            selected.append(text)
            seen.add(key)
            if len(selected) >= 14:
                break
        if not selected:
            return prompt, "Lucky: on | memory empty, used current prompt."
        lucky_prompt = ", ".join(selected)
        return lucky_prompt, f"Lucky: on | composed {len(selected)} phrase(s): {', '.join(selected[:8])}{'...' if len(selected) > 8 else ''}."

    def _v2_lora_type(self, lora_type):
        lora_type = str(lora_type or "general").strip().lower()
        return "action" if lora_type == "concept" else lora_type

    def _v2_lora_words(self, text):
        return {
            word.strip().lower()
            for word in re.split(r"[\s,;:_\\/\-().\[\]{}]+", text or "")
            if self._is_valuable_token(word.strip())
        }

    def _v2_lora_feedback_axes(self, lora_type):
        lora_type = self._v2_lora_type(lora_type)
        if lora_type == "action":
            return {"action"}
        if lora_type == "quality":
            return {"quality"}
        if lora_type == "style":
            return {"quality", "details"}
        if lora_type == "character":
            return {"action", "details"}
        return set(V2_FEEDBACK_AXES)

    def _v2_update_lora_suggestions(self, lora_stack, prompt_history, global_state, phrases, rating_profile, axis_feedback=None):
        if not isinstance(lora_stack, dict) or not lora_stack.get("loras"):
            return "LoRA suggestions: no FunPack LoRA stack connected."
        memory = global_state.setdefault("lora_weight_memory", {})
        phrase_by_category = {}
        all_phrase_words = set()
        for phrase in phrases:
            scores = (
                phrase.get("effective_category_scores", phrase.get("category_scores", {})) or
                {phrase.get("primary", "details"): 1.0}
            )
            primary, confidence = self._v2_scores_primary(scores)
            words = self._v2_lora_words(phrase.get("text", ""))
            all_phrase_words |= words
            phrase_by_category.setdefault(primary, set()).update(words)
            for category, score in scores.items():
                if category != primary and float(score or 0.0) >= max(0.30, confidence * 0.72):
                    phrase_by_category.setdefault(category, set()).update(words)
        suggestions = {}
        parts = []
        axis_feedback = axis_feedback or self._v2_axis_feedback(
            rating_profile,
            global_state.get("last_missing_axes", []),
        )
        missing_axes = set(axis_feedback.get("missing_axes", []))
        satisfied_axes = set(axis_feedback.get("satisfied_axes", []))
        resolved_axes = set(axis_feedback.get("resolved_axes", []))
        reward = float(rating_profile.get("reward", 0.0))
        for entry in lora_stack.get("loras", []):
            name = entry.get("name", "")
            raw_type = entry.get("type", "general")
            lora_type = self._v2_lora_type(raw_type)
            lora_id = entry.get("id") or self._lora_state_id(name, raw_type)
            lora_words = self._v2_lora_words(name)
            relation = 0.20 if lora_type == "general" else 0.0
            if lora_words and all_phrase_words:
                relation = max(relation, len(lora_words & all_phrase_words) / max(1, len(lora_words)))
            if lora_type == "action":
                relation = max(relation, 0.62 if phrase_by_category.get("action") else 0.0)
            elif lora_type == "quality":
                relation = max(relation, 0.62 if phrase_by_category.get("quality") else 0.0)
            elif lora_type == "style":
                relation = max(relation, 0.54 if phrase_by_category.get("style") or phrase_by_category.get("camera") else 0.0)
            elif lora_type == "character":
                relation = max(relation, 0.58 if phrase_by_category.get("subject") or phrase_by_category.get("appearance") else 0.0)

            state = memory.setdefault(lora_id, {"name": name, "type": lora_type, "offset_ratio": 0.0, "iterations": 0})
            offset = float(state.get("offset_ratio", 0.0))
            step = 0.036 if lora_type == "action" else 0.026
            lora_axes = self._v2_lora_feedback_axes(lora_type)
            if rating_profile.get("key") == "like":
                offset += step * max(0.18, relation) * 0.55
            elif rating_profile.get("key") == "awful":
                offset -= step * max(0.22, relation) * 1.20
            elif lora_axes & missing_axes:
                offset += step * max(0.28, relation) * 1.35
            elif lora_axes & resolved_axes:
                offset += step * max(0.18, relation) * 0.62
            elif lora_axes & satisfied_axes:
                offset += step * max(0.12, relation) * 0.18
            else:
                offset += step * reward * max(0.12, relation)
            offset = _clamp(offset, -0.75, 0.55)
            state["offset_ratio"] = round(float(offset), 6)
            state["iterations"] = int(state.get("iterations", 0)) + 1
            state["type"] = lora_type
            state["name"] = name
            base_model = float(entry.get("base_model_weight", entry.get("model_weight", 1.0)))
            model_weight = base_model * (1.0 + offset)
            action = (
                "mute" if model_weight == 0.0 else
                "reduce" if abs(model_weight) < abs(base_model) else
                "hold" if abs(model_weight - base_model) < 1e-6 else
                "boost"
            )
            suggestions[lora_id] = {
                "name": name,
                "type": lora_type,
                "model_weight": model_weight,
                "base_model_weight": base_model,
                "offset_ratio": offset,
                "relation": relation,
                "action": action,
                "rating_label": rating_profile.get("label", ""),
                "rating_range": (
                    "reduce awful"
                    if rating_profile.get("key") == "awful" else
                    "boost " + "+".join(self._v2_order_axes(lora_axes & missing_axes))
                    if lora_axes & missing_axes else
                    "preserve " + "+".join(self._v2_order_axes(lora_axes & resolved_axes))
                    if lora_axes & resolved_axes else
                    rating_profile.get("key", "")
                ),
            }
            parts.append(f"{name}[{lora_type}] rel={relation:.2f} next={model_weight:+.3f}")
        prompt_history["lora_weight_suggestions"] = suggestions
        return "LoRA suggestions: " + " | ".join(parts)

    def refine_v2(self, positive_prompt, clip, rating, refinement_key,
                  seed=0, reset_session=False, lora_stack=None, im_feeling_lucky=False):
        if seed != 0:
            torch.manual_seed(seed)
            random.seed(seed)

        rating_profile = normalize_refiner_v2_rating(rating)
        rating_label = rating_profile.get("label", str(rating))
        state, state_status = self._v2_load_state(refinement_key, reset_session=reset_session)
        global_state = state.setdefault("global", {})
        global_state.setdefault("phrase_memory", {})
        global_state.setdefault("axis_conditioning_memory", {})
        global_state.setdefault("lora_weight_memory", {})
        global_state.setdefault("loss_history", [])
        has_previous_run = isinstance(state.get("last_run"), dict)
        learning_profile = rating_profile if has_previous_run else dict(V2_RATING_PROFILES["Initial discovery"], label="Initial discovery")
        previous_rating_label = str(global_state.get("last_rating_label", "Initial discovery"))
        previous_missing_axes = (
            list(global_state.get("last_missing_axes", []))
            if previous_rating_label != "Initial discovery" else
            None
        )
        axis_feedback = self._v2_axis_feedback(learning_profile, previous_missing_axes)

        memory_status = self._v2_update_phrase_memory(
            global_state,
            state.get("last_run"),
            learning_profile,
            int(global_state.get("total_iterations", 0)) + 1,
            axis_feedback,
        )
        self._v2_update_conditioning_memory(global_state, state.get("last_run"), learning_profile, axis_feedback)
        if has_previous_run and not learning_profile.get("skip_learning"):
            self._v2_update_streaks(global_state, learning_profile)

        analysis_prompt = self._v2_prompt_key(positive_prompt)
        phrases = self._v2_classify_phrases(clip, self._ordered_prompt_phrases(analysis_prompt), global_state)
        if im_feeling_lucky:
            prompt_to_encode, lucky_status = self._v2_compose_lucky_prompt(analysis_prompt, phrases, global_state)
            encoded_role = "lucky prompt"
        else:
            prompt_to_encode, emphasis_status = self._v2_emphasized_prompt(analysis_prompt, phrases, global_state, learning_profile)
            lucky_status = f"Lucky: off | trained memory only. {emphasis_status}"
            encoded_role = "current prompt"

        cond, meta, encode_status = self._v2_encode_prompt(clip, prompt_to_encode)
        fallback_graph = render_refinement_loss_graph(refinement_key, "v2", "clip", 0, 0.0, [])
        if not isinstance(cond, torch.Tensor):
            status = f"ERROR: V2 could not encode prompt | {encode_status}"
            training_info = f"Rating: {rating_label}\n{memory_status}\n{lucky_status}\n{encode_status}"
            return ([], status, training_info, fallback_graph)

        refined, adaptation_status = self._v2_apply_conditioning_memory(cond, global_state, learning_profile, axis_feedback)
        prompt_key = self._v2_prompt_key(analysis_prompt)
        prompt_history = state.setdefault("prompt_histories", {}).setdefault(prompt_key, {
            "canonical_prompt": analysis_prompt,
            "history": [],
        })
        lora_status = self._v2_update_lora_suggestions(
            lora_stack,
            prompt_history,
            global_state,
            phrases,
            learning_profile,
            axis_feedback,
        )

        global_state["total_iterations"] = int(global_state.get("total_iterations", 0)) + 1
        learning_loss = max(0.02, (1.0 - max(-1.0, min(1.0, float(global_state.get("avg_reward_ema", 0.0))))) * 0.5)
        loss_history = list(global_state.get("loss_history", []))[-511:]
        loss_history.append({
            "total_iteration": int(global_state["total_iterations"]),
            "learning_loss": round(float(learning_loss), 6),
            "rating": int(rating_profile.get("legacy_score", 6)),
            "rating_label": rating_label,
            "similarity": 1.0,
            "scheduler_mode": "automatic",
            "mode": "clip",
        })
        global_state["loss_history"] = loss_history

        phrase_preview = "; ".join(
            (
                f"{item['text']}["
                f"{item.get('machine_primary', item['primary'])}->{item['primary']}:"
                f"{item['confidence']:.2f}/{item['source']}/"
                f"ctx={item.get('context_source', 'none')}/"
                f"e{int(item.get('category_evidence_count', 0))}]"
            )
            for item in phrases[:10]
        ) or "none"
        category_diagnostics = self._v2_category_diagnostics(phrases)
        training_guidance = self._v2_training_guidance(
            has_previous_run,
            learning_profile,
            axis_feedback,
            phrases,
            memory_status,
            lora_status,
        )
        prompt_preview = re.sub(r"\s+", " ", prompt_to_encode).strip()
        if len(prompt_preview) > 240:
            prompt_preview = prompt_preview[:237].rstrip() + "..."

        prompt_history["canonical_prompt"] = analysis_prompt
        prompt_history.setdefault("history", [])
        prompt_history["history"] = list(prompt_history.get("history", []))[-119:]
        prompt_history["history"].append({
            "iteration": int(global_state["total_iterations"]),
            "rating_label": rating_label if has_previous_run else "Unrated",
            "encoded_role": encoded_role,
            "prompt": prompt_to_encode,
            "phrases": phrases,
        })

        state["last_run"] = {
            "prompt": analysis_prompt,
            "encoded_prompt": prompt_to_encode,
            "conditioning": tensor_to_serializable(refined.detach().cpu()),
            "phrases": phrases,
            "rating_label": "Unrated",
            "iteration": int(global_state["total_iterations"]),
        }
        state["global"] = global_state
        self._v2_save_state(state, refinement_key)

        loss_graph = render_refinement_loss_graph(
            refinement_key=refinement_key,
            scheduler_mode="v2-auto",
            mode="clip",
            total_iterations=int(global_state["total_iterations"]),
            latest_learning_loss=float(learning_loss),
            points=loss_history[-256:],
        )
        status = (
            f"V2 {state_status} | CLIP-owned | Rating {rating_label} | Iter {global_state['total_iterations']} | "
            f"Learning {'trained previous run' if has_previous_run and not learning_profile.get('skip_learning') else 'not applied'}\n"
            f"{adaptation_status}\n"
            f"{training_guidance}\n"
            f"{lucky_status}"
        )
        training_info = (
            f"Encoded: {encoded_role} | {encode_status}\n"
            f"Learning target: {'previous run' if has_previous_run else 'none yet'} | "
            f"Applied: {'yes' if has_previous_run and not learning_profile.get('skip_learning') else 'no'} | "
            f"Reason: {'rating skipped learning' if learning_profile.get('skip_learning') else 'waiting for a rated previous run' if not has_previous_run else 'rating trained memory'}\n"
            f"{self._v2_axis_feedback_status(axis_feedback)}\n"
            f"Prompt: {prompt_preview}\n"
            f"{memory_status}\n"
            f"{category_diagnostics}\n"
            f"{training_guidance}\n"
            f"{adaptation_status}\n"
            f"{lucky_status}\n"
            f"Category compact view: {phrase_preview}\n"
            f"{lora_status}"
        )
        return ([(refined, meta)], status, training_info, loss_graph)


class FunPackSaveRefinementLatent:
    CATEGORY = "FunPack/Refinement"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "status")
    FUNCTION = "save_latent"
    DESCRIPTION = "Saves a latent tensor bundle under a refinement key for FunPack Video Refiner latent refinement."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "refinement_key": ("STRING", {"default": "my_style_v1", "multiline": False}),
                "mode": (["ltx2", "wan"], {"default": "ltx2"}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def save_latent(self, latent, refinement_key, mode):
        if isinstance(latent, dict) and not latent_is_plain_video_tensor(latent):
            FunPackVideoRefiner()._raise_wrong_latent(latent)

        samples = latent_samples(latent)
        if samples is None:
            return (clone_latent(latent), "No latent samples tensor found.")

        FunPackVideoRefiner()._save_latent_reference(latent, refinement_key, mode)
        return (clone_latent(latent), f"Saved latent for key '{refinement_key}' ({mode}) shape={tuple(samples.shape)}")


class FunPackPromptCombiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "main_prompt": ("STRING", {
                    "multiline": True,
                    "default": "masterpiece, best quality, detailed background"
                }),
                "delimiter": ("STRING", {
                    "multiline": False,
                    "default": ",",
                }),
            },
            "optional": {
                "prompt1": ("STRING", {"default": "", "multiline": True}),
                "prompt2": ("STRING", {"default": "", "multiline": True}),
                "prompt3": ("STRING", {"default": "", "multiline": True}),
                "prompt4": ("STRING", {"default": "", "multiline": True}),
                "prompt5": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("out1", "out2", "out3", "out4", "out5", "random")
    FUNCTION = "combine"
    CATEGORY = "FunPack"
    OUTPUT_NODE = False

    def combine(self, main_prompt, delimiter=",",
                prompt1="", prompt2="", prompt3="", prompt4="", prompt5=""):

        main = main_prompt.strip()

        def merge(base, delim, addon):
            addon = addon.strip()
            if not addon:
                return base
            if not base:
                return addon
            return f"{base}{delim}{addon}"

        results = []
        for p in (prompt1, prompt2, prompt3, prompt4, prompt5):
            combined = merge(main, delimiter, p)
            results.append(combined)
        
        random_choice = random.choice(results)

        return (*results, random_choice)


class FunPackLorebookEnhancer:
    """
    Injects context from SillyTavern-style lorebook JSON files.
    Always appends activated entries to the END of the prompt.
    Supports multiple lorebooks, constants, selective filtering, probability, etc.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "The brave explorer enters an ancient forest temple at twilight, discovering glowing runes on the walls."
                }),
            },
            "optional": {
                "lorebook_1": ("STRING", {"default": "", "multiline": False}),
                "lorebook_2": ("STRING", {"default": "", "multiline": False}),
                "lorebook_3": ("STRING", {"default": "", "multiline": False}),
                "lorebook_4": ("STRING", {"default": "", "multiline": False}),
                "entry_delimiter": ("STRING", {"default": "", "multiline": True}),
                "context_history": ("STRING", {"multiline": True, "default": ""}),
                "scan_depth": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 12,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "injected_content")
    FUNCTION = "enhance"
    CATEGORY = "FunPack"
    OUTPUT_NODE = True

    def _match_keys(self, keys, text):
        if not keys:
            return False
        if isinstance(keys, str):
            keys = [k.strip() for k in keys.split(",") if k.strip()]
        text = text.lower()
        for key in keys:
            key = key.strip()
            if not key:
                continue
            if key.startswith("/") and key.endswith("/"):
                try:
                    pattern = re.compile(key[1:-1], re.IGNORECASE)
                    if pattern.search(text):
                        return True
                except:
                    continue
            elif key.lower() in text:
                return True
        return False

    def _match_secondary(self, secs, text, logic):
        if not secs:
            return True
        if isinstance(secs, str):
            secs = [s.strip() for s in secs.split(",") if s.strip()]
        matches = [self._match_keys([s], text) for s in secs]

        if logic == 0:   return any(matches)
        elif logic == 1: return all(matches)
        elif logic == 2: return not any(matches)
        elif logic == 3: return not all(matches)
        return True

    def _process_lorebook(self, path, scan_text, activated):
        if not path or not os.path.exists(path):
            print(f"[Lorebook Enhancer] File not found: {path}")
            return activated

        try:
            with open(path, 'r', encoding='utf-8') as f:
                lorebook = json.load(f)

            entries = lorebook.get("entries", [])
            if isinstance(entries, dict):
                entries = list(entries.values())

            for entry in entries:
                if not entry.get("enabled", True):
                    continue

                is_constant = entry.get("constant", False)

                is_constant = entry.get("constant", False)

                if not is_constant:
                    keys = entry.get("keys", entry.get("key", []))              # Fixed: use "keys" (plural) first
                    if not self._match_keys(keys, scan_text):
                        continue

                if entry.get("selective", False):
                    sec_keys = entry.get("keysecondary", []) or entry.get("secondary_keys", [])
                    logic = entry.get("selectiveLogic", 0)
                    if not self._match_secondary(sec_keys, scan_text, logic):
                        continue

                if entry.get("selective", False):
                    sec_keys = entry.get("keysecondary", []) or entry.get("secondary_keys", [])
                    logic = entry.get("selectiveLogic", 0)
                    if not self._match_secondary(sec_keys, scan_text, logic):
                        continue

                prob = entry.get("extensions", {}).get("probability", 100)
                if random.randint(1, 100) > prob:
                    continue

                activated.append(entry)

        except json.JSONDecodeError as e:
            print(f"[Lorebook Enhancer] JSON decode error in {path}: {str(e)}")
        except Exception as e:
            print(f"[Lorebook Enhancer] Failed to process {path}: {type(e).__name__}: {str(e)}")

        return activated

    def enhance(self, prompt,
                lorebook_1="", lorebook_2="", lorebook_3="", lorebook_4="",
                context_history="", scan_depth=4, entry_delimiter=""):

        full_text = (context_history + "\n" + prompt).lower()
        lines = full_text.splitlines()
        scan_text = "\n".join(lines[-scan_depth:]) if scan_depth > 0 else full_text

        activated = []
        for path in [lorebook_1, lorebook_2, lorebook_3, lorebook_4]:
            if path.strip():
                activated = self._process_lorebook(path.strip(), scan_text, activated)

        if not activated:
            return (prompt, "No lorebook entries were triggered.")

        activated.sort(key=lambda e: e.get("insertion_order", e.get("order", 0)))

        injected = []
        enhanced = prompt.strip()  # clean up any trailing space

        for entry in activated:
            content = entry.get("content", "").strip()
            if not content:
                continue
            prefixed_content = f"{entry_delimiter}{content}" if entry_delimiter else content

            # Always append to the end - this is the requested behavior
            if enhanced:
                enhanced += "\n" + prefixed_content
            else:
                enhanced = prefixed_content

            source = entry.get("comment") or entry.get("name") or f"uid:{entry.get('uid','?')}" or "unnamed"
            injected.append(f"[{source}] {prefixed_content}")

        injected_text = "\n\n".join(injected) if injected else "No content injected"
        return (enhanced, injected_text)

class FunPackPromptEnhancer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "user_prompt": ("STRING", {"multiline": True, "default": "A photo of a [subject] in a [setting]. [action]."}),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "<You are a creative AI assistant tasked with describing videos.\n\nDescribe the video by detailing the following aspects:\n1. The main content and theme of the video.\n2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.\n3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.\n4. background environment, light, style and atmosphere.\n5. camera angles, movements, and transitions used in the video:"
                }),
                "model_path_type": (["Local Safetensors", "HuggingFace Pretrained"],),
                "model_path": ("STRING", {"multiline": False, "default": "mlabonne/NeuralLlama-3-8B-Instruct-abliterated"}),
                "llm_safetensors_file": (folder_paths.get_filename_list('clip'),),
                "top_p": ("FLOAT", {"min": 0.0, "max": 2.0, "step": 0.05, "default": 0.75}),
                "top_k": ("INT", {"min": 0, "max": 1000, "step": 1, "default": 40}),
                "temperature": ("FLOAT", {"min": 0.0, "max": 2.0, "step": 0.01, "default": 0.6}),
                "max_new_tokens": ("INT", {"min": 64, "max": 4096, "step": 64, "default": 512}),
                "repetition_penalty": ("FLOAT", {"min": 0.0, "max": 3.0, "step": 0.01, "default": 1.0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "enhance_prompt"
    CATEGORY = "FunPack"

    def enhance_prompt(self, user_prompt, system_prompt, model_path_type, model_path, llm_safetensors_file, top_p, top_k, temperature, max_new_tokens, repetition_penalty):
        llm_model = None
        llm_tokenizer = None
        llm_model_device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[FunPackPromptEnhancer] Starting prompt enhancement...")

        try:
            if model_path_type == "HuggingFace Pretrained":
                print(f"[FunPackPromptEnhancer] Loading LLM from HuggingFace pretrained: {model_path}")
                llm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                llm_model = AutoModelForCausalLM.from_pretrained(model_path, ignore_mismatched_sizes=True, trust_remote_code=True)
            elif model_path_type == "Local Safetensors":
                print(f"[FunPackPromptEnhancer] Loading LLM from local safetensors file: {llm_safetensors_file}")
                full_safetensors_path = folder_paths.get_full_path('clip', llm_safetensors_file)

                llm_tokenizer = AutoTokenizer.from_pretrained("xtuner/llava-llama-3-8b-v1_1-transformers", trust_remote_code=True)

                config = AutoConfig.from_pretrained("xtuner/llava-llama-3-8b-v1_1-transformers", trust_remote_code=True)
                model_base = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

                state_dict = load_file(full_safetensors_path, device="cpu")
                model_base.load_state_dict(state_dict, strict=False)
                llm_model = model_base

            llm_model = llm_model.eval().to(torch.bfloat16 if llm_model_device == "cuda" else torch.float32).to(llm_model_device).requires_grad_(False)
            print(f"[FunPackPromptEnhancer] LLM model loaded successfully to {llm_model_device}!")

            # Model detection to apply correct chat template
            llm_tokenizer.chat_template = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.first %}{{ '<|begin_of_text|>' + content }}{% else %}{{ content }}{% endif %}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' if add_generation_prompt else '' }}"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            if llm_tokenizer.pad_token_id is None:
                llm_tokenizer.pad_token = llm_tokenizer.eos_token
                llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id

            llm_tokens = llm_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True
            ).to(llm_model_device)

            print("[FunPackPromptEnhancer] Generating enhanced prompt...")
            with torch.no_grad():
                generated_ids = llm_model.generate(
                    **llm_tokens,
                    do_sample=True,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=llm_tokenizer.pad_token_id
                )

            output_text = llm_tokenizer.decode(generated_ids[0][llm_tokens['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"[FunPackPromptEnhancer] Enhanced prompt generated: {output_text}")

            return (output_text,)

        except Exception as e:
            print(f"[FunPackPromptEnhancer] Error during prompt enhancement: {e}")
            raise

        finally:
            if llm_model is not None:
                del llm_model
                llm_model = None
            if llm_tokenizer is not None:
                del llm_tokenizer
                llm_tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("[FunPackPromptEnhancer] LLM model and tokenizer unloaded and memory cleared.")


class FunPackStoryWriter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "user_prompt": ("STRING", {"multiline": True, "default": "A photo of a [subject] in a [setting]. [action]."}),
                "prompt1": ("STRING", {"multiline": False, "default": ""}),
                "prompt2": ("STRING", {"multiline": False, "default": ""}),
                "prompt3": ("STRING", {"multiline": False, "default": ""}),
                "prompt4": ("STRING", {"multiline": False, "default": ""}),
                "prompt5": ("STRING", {"multiline": False, "default": ""}),
                "story_system_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "sequence_system_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "model_path_type": (["Local Safetensors", "HuggingFace Pretrained"],),
                "model_path": ("STRING", {"multiline": False, "default": "mlabonne/NeuralLlama-3-8B-Instruct-abliterated"}),
                "llm_safetensors_file": (folder_paths.get_filename_list('clip'),),
                "prompt_count": ("INT", {"min": 1, "max": 5, "step": 1, "default": 3}),
                "top_p": ("FLOAT", {"min": 0.0, "max": 2.0, "step": 0.05, "default": 0.75}),
                "top_k": ("INT", {"min": 0, "max": 1000, "step": 1, "default": 40}),
                "min_p": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.1}),
                "temperature": ("FLOAT", {"min": 0.0, "max": 2.0, "step": 0.01, "default": 0.6}),
                "max_new_tokens": ("INT", {"min": 64, "max": 4096, "step": 64, "default": 512}),
                "repetition_penalty": ("FLOAT", {"min": 0.0, "max": 3.0, "step": 0.01, "default": 1.0}),
                "mode": (["Sequences from story", "Sequences from user prompt"],),
                "vision_input": ("STRING", {"multiline": True, "default": "Put outputs of your VL model here to make the Story Writer aware of the starting image."}),
                "sanity_check": ("BOOLEAN", {"default": True, "label": "Enable Sanity Check"}),
                "sanity_check_system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Analyze the given sequence and perform a correction, if the sequence does not match the given requirements:\n1. The sequence is related to given user's prompt.\n2. The sequence contains only physically possible actions.\n3. The sequence contains information about characters, their appearances, positioning, actions, camera angle, focus and zoom.\n4. The sequence is fully describing the requested action.\n\nOutput ONLY corrected sequence, or return it unchanged if it matches the requirements. No additional text except for sequence is allowed."
                }),
                "disable_continuity": ("BOOLEAN", {"default": False, "label": "Enable/disable continuity - if enabled, does not provide the history of previously generated sequences when generating new one."}),
                "provide_current_id": ("BOOLEAN", {"default": True, "label": "If true, provides current sequence ID to the model even if continuity is disabled."}),
            }
        }

    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING",)
    RETURN_NAMES = ("prompt1","prompt2","prompt3","prompt4","prompt5",)
    FUNCTION = "write_story"
    CATEGORY = "FunPack"

    def write_story(self, user_prompt, prompt1, prompt2, prompt3, prompt4, prompt5, story_system_prompt, sequence_system_prompt, model_path_type, model_path, llm_safetensors_file, prompt_count, top_p, top_k, min_p, temperature, max_new_tokens, repetition_penalty, mode, vision_input, sanity_check, sanity_check_system_prompt, disable_continuity, provide_current_id):
        llm_model = None
        llm_tokenizer = None
        llm_model_device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[FunPackPromptEnhancer] Making initial story...")

        try:
            if model_path_type == "HuggingFace Pretrained":
                print(f"[FunPackStoryWriter] Loading LLM from HuggingFace pretrained: {model_path}")
                llm_tokenizer = AutoTokenizer.from_pretrained(model_path, ignore_mismatched_sizes=True, trust_remote_code=True)
                llm_model = AutoModelForCausalLM.from_pretrained(model_path, ignore_mismatched_sizes=True, trust_remote_code=True)
            elif model_path_type == "Local Safetensors":
                print(f"[FunPackStoryWriter] Loading LLM from local safetensors file: {llm_safetensors_file}")
                full_safetensors_path = folder_paths.get_full_path('clip', llm_safetensors_file)

                llm_tokenizer = AutoTokenizer.from_pretrained("xtuner/llava-llama-3-8b-v1_1-transformers", ignore_mismatched_sizes=True, trust_remote_code=True)

                config = AutoConfig.from_pretrained("xtuner/llava-llama-3-8b-v1_1-transformers", trust_remote_code=True)
                model_base = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

                state_dict = load_file(full_safetensors_path, device="cpu")
                model_base.load_state_dict(state_dict, strict=False)
                llm_model = model_base

            llm_model = llm_model.eval().to(torch.bfloat16 if llm_model_device == "cuda" else torch.float32).to(llm_model_device).requires_grad_(False)
            print(f"[FunPackStoryWriter] LLM model loaded successfully to {llm_model_device}!")

            # Applying correct chat template
            llm_tokenizer.chat_template = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.first %}{{ '<|begin_of_text|>' + content }}{% else %}{{ content }}{% endif %}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' if add_generation_prompt else '' }}"""

            # Inside write_story method, after model loading and template/pad fix

            outputs = [""] * 5

            prompts = [""] * 5

            prompts[0] = prompt1
            prompts[1] = prompt2
            prompts[2] = prompt3
            prompts[3] = prompt4
            prompts[4] = prompt5

            recommended_loras = None

            # ── Initialize messages ONCE, depending on mode ────────────────────────────
            if mode == "Sequences from story":
                messages = [
                    {"role": "system", "content": story_system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

            # Generate hidden story
                llm_tokens = llm_tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt", tokenize=True
                ).to(llm_model_device)

                print("[FunPackStoryWriter] Generating hidden story...")
                with torch.no_grad():
                    generated_ids = llm_model.generate(
                        **llm_tokens,
                        do_sample=True,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=llm_tokenizer.pad_token_id,
                        eos_token_id=llm_tokenizer.eos_token_id,
                    )

                story = llm_tokenizer.decode(generated_ids[0][llm_tokens['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                print(f"[FunPackStoryWriter] Hidden story: {story[:150]}...")

                messages.append({"role": "assistant", "content": story})

            else:  # "Sequences from user prompt"
                messages = [
                    {"role": "system", "content": sequence_system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

            # ── Now generate sequences — only add new instruction + append output ─────
            for seq_idx in range(prompt_count):
                # Add **only** the fresh sequence instruction each time
                if disable_continuity == True and provide_current_id == False:
                    messages = [
                        {"role": "system", "content": sequence_system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]

                    if prompts[seq_idx]:
                        messages.append({"role": "user", "content": f"""User's request for current scene only: {prompts[seq_idx]}"""})

                elif disable_continuity == True and provide_current_id == True:
                    messages = [
                        {"role": "system", "content": sequence_system_prompt},
                        {"role": "user", "content": f"""Current request ID: {seq_idx+1}\nUser's instruction for all requests: {user_prompt}"""}
                    ]

                    if prompts[seq_idx]:
                        messages.append({"role": "user", "content": f"""User's request for current scene only: {prompts[seq_idx]}"""})

                else:
                    messages = [
                        {"role": "system", "content": sequence_system_prompt},
                        {"role": "user", "content": f"""Total amount of requests in this batch: {prompt_count}\nCurrently generating request ID {seq_idx+1} out of {prompt_count}\nRequests left in queue: {prompt_count - seq_idx - 1}\nUser's instruction for all requests: {user_prompt}"""},
                        {"role": "assistant", "content": f"""History:{chr(10).join([f"ID {i}: {text}" for i, text in enumerate(outputs[:seq_idx])]) if seq_idx > 0 else "No history available."}"""}
                    ]
                    if prompts[seq_idx]:
                        messages.append({"role": "user", "content": f"""User's request for current scene only: {prompts[seq_idx]}"""})

                if vision_input is not None:
                    messages.append({"role": "user", "content": f"""Reference image description (this is the starting image in the video batch): {vision_input}"""})

                llm_tokens = llm_tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt", tokenize=True
                ).to(llm_model_device)

                print(f"[FunPackStoryWriter] Generating ID {seq_idx + 1}/{prompt_count}...")
                with torch.no_grad():
                    generated_ids = llm_model.generate(
                        **llm_tokens,
                        do_sample=True,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=llm_tokenizer.pad_token_id,
                        eos_token_id=llm_tokenizer.eos_token_id,
                    )

                seq_text = llm_tokenizer.decode(generated_ids[0][llm_tokens['input_ids'].shape[1]:], skip_special_tokens=True).strip()

                # Append generated sequence — this is what chains everything
                messages.append({"role": "assistant", "content": seq_text})

                if sanity_check == False:
                    print(f"[FunPackStoryWriter] ID {seq_idx + 1} (sanity check skipped): {seq_text}...")

                # Performing sanity check - comparing sequence text to user's prompt according to rules in sanity check system prompt.
                if sanity_check == True:
                    if mode == "Sequences from story" and disable_continuity == False:
                        sanity_messages = [
                            {"role": "system", "content": sanity_check_system_prompt},
                            {"role": "user", "content": f"""Original story: {story}\n
                            Rules: {story_system_prompt}\n
                            User's instruction: {user_prompt}\n
                             Previous response: {outputs[seq_idx] if seq_idx > 0 else "No history available."}\n
                             Response to validate and correct if rules were broken: {seq_text}"""}
                        ]
                    elif mode == "Sequences from user prompt" and disable_continuity == False:
                        sanity_messages = [
                            {"role": "system", "content": sanity_check_system_prompt},
                            {"role": "user", "content": f"""User's instruction: {user_prompt}\n
                            Rules: {sequence_system_prompt}\n
                             Previous response: {outputs[seq_idx] if seq_idx > 0 else "No history available."}\n
                             Response to validate and correct if rules were broken: {seq_text}"""}
                        ]
                    else:
                        sanity_messages = [
                            {"role": "system", "content": sanity_check_system_prompt},
                            {"role": "user", "content": f"""User's instruction: {user_prompt}
                            Rules: {sequence_system_prompt}
                            Response to validate and correct if rules were broken: {seq_text}"""}
                        ]
                    llm_tokens = llm_tokenizer.apply_chat_template(
                        sanity_messages, add_generation_prompt=True, return_tensors="pt", tokenize=True
                        ).to(llm_model_device)

                    print(f"[FunPackStoryWriter] Performing sanity check on ID {seq_idx + 1}/{prompt_count}...")
                    with torch.no_grad():
                        generated_ids = llm_model.generate(
                            **llm_tokens,
                            do_sample=False,
                            top_p=top_p,
                            top_k=top_k,
                            min_p=min_p,
                            temperature=temperature,
                            max_new_tokens=1024,
                            repetition_penalty=1.05,
                            pad_token_id=llm_tokenizer.pad_token_id,
                            eos_token_id=llm_tokenizer.eos_token_id,
                        )

                    seq_text = llm_tokenizer.decode(generated_ids[0][llm_tokens['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                    print(f"[FunPackStoryWriter] ID {seq_idx + 1} (sanity check performed): {seq_text}...")

                outputs[seq_idx] = seq_text

            return tuple(outputs)

        except Exception as e:
            print(f"[FunPackStoryWriter] Error during prompt enhancement: {e}")
            raise

        finally:
            if llm_model is not None:
                del llm_model
                llm_model = None
            if llm_tokenizer is not None:
                del llm_tokenizer
                llm_tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("[FunPackStoryWriter] LLM model and tokenizer unloaded and memory cleared.")
