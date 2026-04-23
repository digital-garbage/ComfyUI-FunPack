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
    "concept": {"step": 0.046, "max_offset": 0.35, "min_offset": -0.45, "bad_max_offset": 0.75, "bad_min_offset": -2.10, "culprit_bias": 0.16},
    "style": {"step": 0.032, "max_offset": 0.28, "min_offset": -0.38, "bad_max_offset": 0.58, "bad_min_offset": -1.55, "culprit_bias": 0.20},
    "quality": {"step": 0.022, "max_offset": 0.18, "min_offset": -0.30, "bad_max_offset": 0.38, "bad_min_offset": -1.20, "culprit_bias": 0.18},
    "character": {"step": 0.024, "max_offset": 0.20, "min_offset": -0.32, "bad_max_offset": 0.42, "bad_min_offset": -1.30, "culprit_bias": 0.10},
}


def _clamp(value, low, high):
    return max(low, min(high, value))


QUOTED_TEXT_RE = re.compile(r'"([^"]+)"|“([^”]+)”')


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
        for match in QUOTED_TEXT_RE.finditer(prompt):
            start, end = match.span()
            if start > last_end:
                yield ("text", prompt[last_end:start])
            quoted_text = (match.group(1) or match.group(2) or "").strip()
            if quoted_text:
                yield ("quote", quoted_text)
            last_end = end

        if last_end < len(prompt):
            yield ("text", prompt[last_end:])

    def _mask_quoted_text(self, prompt: str):
        if not prompt:
            return ""
        return QUOTED_TEXT_RE.sub(lambda match: " " * len(match.group(0)), prompt)

    def _build_word_groups(self, prompt: str, tokenizer, seq_len: int, token_mask=None):
        if not tokenizer or not prompt or seq_len <= 0:
            return []

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
            if segment_type != "quote":
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

        return sorted(
            [group for group in word_groups if group[1] > group[0]],
            key=lambda group: (group[0], group[1], group[2].lower()),
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_conditioning": ("CONDITIONING",),
                "mode": (["ltx2", "wan"], {"default": "ltx2", "label": "Tokenizer Mode"}),
                "rating": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "slider",
                    "label": "Rating (1=horrific, 10=masterpiece)"
                }),
                "refinement_key": ("STRING", {"default": "my_style_v1", "multiline": False}),
                "scheduler_mode": (["original", "accurate", "aggressive"], {"default": "original"}),
            },
            "optional": {
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Positive prompt"
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
        if not defaults.get("category") or defaults.get("category") == "general":
            defaults["category"] = self._infer_concept_category(defaults.get("anchor_words", []))
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
                candidates.append({
                    "concept_id": cid,
                    "concept_label": current_concept_labels.get(cid, cluster.get("label", "")),
                    "question_type": question_type,
                    "neighbor_ids": neighbor_ids,
                    "group_id": chosen_group_id,
                    "category": category,
                    "score": score,
                })

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
            if segment_type == "quote":
                quoted_phrase = segment_text.strip().lower()
                if len(quoted_phrase) >= 3 and any(c.isalpha() for c in quoted_phrase):
                    result.append([quoted_phrase])
                continue

            phrases = [p.strip() for p in re.split(r'[,;]', segment_text) if p.strip()]
            for phrase in phrases:
                words = [w.strip().lower() for w in phrase.split() if self._is_valuable_token(w.strip())]
                if words:
                    result.append(words)
        return result

    def _build_prompt_fallback_concept(self, prompt: str, concept_clusters: dict):
        if not prompt:
            return None

        fallback_words = []
        for segment_type, segment_text in self._iter_prompt_segments(prompt):
            if segment_type == "quote":
                quoted_phrase = segment_text.strip().lower()
                if len(quoted_phrase) >= 3 and any(c.isalpha() for c in quoted_phrase):
                    fallback_words.append(quoted_phrase)
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
                if concept_clusters[cid].get("category") == "general":
                    concept_clusters[cid]["category"] = self._infer_concept_category(
                        concept_clusters[cid]["anchor_words"]
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

    def _refine_sigma_schedule(self, sigmas, rating: int, global_adaptive: dict, strength_mode: str, seed: int):
        if not isinstance(sigmas, torch.Tensor):
            return torch.FloatTensor([]), "Sigma refinement inactive."

        original_sigmas = sigmas.detach().clone()
        if original_sigmas.numel() <= 2:
            return original_sigmas, "Sigma refinement skipped: schedule too short."

        self._ensure_sigma_state_defaults(global_adaptive)

        reward = (rating - 5.5) / 4.5
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
            min(0.05, sigma_exploration_base * (0.97 if rating >= 8 else 1.02))
        )
        sigma_history = list(global_adaptive.get("sigma_history", []))[-119:]
        sigma_history.append({
            "iteration": sigma_iterations + 1,
            "rating": int(rating),
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

    def _refine_latent(self, latent, refinement_key, mode, rating, reward, global_adaptive):
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

        lr = 0.035 if rating >= 7 else 0.055 if rating <= 4 else 0.018
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
                                        concept_clusters, current_concept_labels, rating, reward):
        if not isinstance(lora_stack, dict) or not lora_stack.get("loras"):
            return "LoRA suggestions: no FunPack LoRA stack connected."

        memory = global_adaptive.setdefault("lora_weight_memory", {})
        suggestions = {}
        status_parts = []

        for entry in lora_stack.get("loras", []):
            lora_type = entry.get("type", "general")
            profile = LORA_REFINER_TYPE_PROFILES.get(lora_type, LORA_REFINER_TYPE_PROFILES["general"])
            lora_id, state = self._ensure_lora_memory(memory, entry)
            relation, matched_labels, concept_importance = self._score_lora_prompt_relation(
                entry,
                ordered_concept_ids,
                concept_clusters,
                current_concept_labels,
            )

            state["reward_ema"] = 0.84 * float(state.get("reward_ema", 0.0)) + 0.16 * reward
            if rating >= 8:
                state["good_streak"] = int(state.get("good_streak", 0)) + 1
                state["bad_streak"] = 0
            elif rating <= 4:
                state["bad_streak"] = int(state.get("bad_streak", 0)) + 1
                state["good_streak"] = 0
            else:
                state["good_streak"] = 0
                state["bad_streak"] = 0

            offset = float(state.get("offset_ratio", 0.0))
            stable_offset = state.get("stable_offset_ratio")
            culprit_score = float(state.get("culprit_score", 0.0))
            culprit_hits = int(state.get("culprit_hits", 0))
            if stable_offset is not None and rating >= 6:
                offset = 0.72 * offset + 0.28 * float(stable_offset)

            step = profile["step"] * max(0.15, relation)
            max_offset = profile["max_offset"]
            min_offset = profile["min_offset"]
            effective_relation = max(relation, profile.get("culprit_bias", 0.0))
            base_model = float(entry.get("base_model_weight", entry.get("model_weight", 1.0)))
            base_abs = abs(base_model)
            is_concept_lora = lora_type == "concept"
            concept_match_strength = 1.0
            if is_concept_lora:
                concept_match_strength += 0.30 if matched_labels else 0.0
                concept_match_strength += max(0.0, relation - 0.30) * 1.35

            if rating >= 8:
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
            elif rating <= 4:
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
        return "LoRA suggestions: " + " | ".join(status_parts)

    # =========================================================================
    # MAIN REFINE
    # =========================================================================

    def refine(self, positive_conditioning, mode: str, rating: int, refinement_key: str,
               scheduler_mode: str = "original", positive_prompt: str = "",
               reset_session: bool = False, unlimited_history: bool = False,
               seed: int = 0, feedback_enabled: bool = False, feedback_rating: int = 3,
               sigmas=None, sigma_strength: str = "subtle", lora_stack=None, latent=None,
               prompt=None, unique_id=None):

        mode = (mode or "ltx2").lower()
        if mode not in self._tokenizer_sources:
            mode = "ltx2"

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

        analysis_prompt = self._normalize_prompt_for_mode(positive_prompt, mode)
        prompt_key = analysis_prompt if mode == "wan" else positive_prompt

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
                        "reference_embeds": tensor_to_serializable(raw_positive),
                        "history": [],
                        "last_rating": rating
                    }
                },
                "last_prompt_key": prompt_key,
                "pending_feedback": None
            }

        # ====================== RESET / NEW SESSION ======================
        if reset_session or not os.path.exists(json_file):
            if reset_session:
                self._delete_latent_reference(refinement_key, mode)
            if latent_output_connected:
                fallback_latent, fallback_latent_status = self._refine_latent(
                    latent,
                    refinement_key,
                    mode,
                    rating,
                    (rating - 5.5) / 4.5,
                    {},
                )
            else:
                fallback_latent, fallback_latent_status = self._latent_refinement_disabled(latent)
            data = _fresh_data()
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return (positive_conditioning, "New session started - Reference saved", "", f"New session started. Reference embedding saved.\n{fallback_latent_status}", fallback_loss_graph, fallback_sigmas, fallback_latent)

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
            if latent_output_connected:
                fallback_latent, fallback_latent_status = self._refine_latent(
                    latent,
                    refinement_key,
                    mode,
                    rating,
                    (rating - 5.5) / 4.5,
                    {},
                )
            else:
                fallback_latent, fallback_latent_status = self._latent_refinement_disabled(latent)
            data = _fresh_data()
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return (positive_conditioning, "Session file was corrupt - Reset and started fresh", "", f"Session reset due to corrupt file\n{fallback_latent_status}", fallback_loss_graph, fallback_sigmas, fallback_latent)

        global_adaptive = data["global_adaptive"]
        # Migrate sessions created before the multi-level concept system was added
        global_adaptive.setdefault("concept_clusters", {})
        global_adaptive.setdefault("concept_groups", {})
        global_adaptive.setdefault("word_importance", {})
        global_adaptive.setdefault("feedbacked_concepts", [])
        global_adaptive.setdefault("feedback_memory", {"recent_questions": [], "rating_change_events": []})
        global_adaptive.setdefault("loss_history", [])
        global_adaptive.setdefault("lora_weight_memory", {})
        global_adaptive.setdefault("mode", mode)
        self._ensure_sigma_state_defaults(global_adaptive)
        for cid in list(global_adaptive["concept_clusters"].keys()):
            global_adaptive["concept_clusters"][cid] = self._ensure_concept_cluster_defaults(
                global_adaptive["concept_clusters"][cid]
            )

        prompt_histories = data.get("prompt_histories", {})
        tokenizer = self._get_tokenizer(mode)

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
                "reference_embeds": tensor_to_serializable(raw_positive),
                "history": [],
                "last_rating": rating
            }
            prompt_histories[prompt_key] = active

        # Safe reference loading
        try:
            old_reference = serializable_to_tensor(active["reference_embeds"])
        except Exception as e:
            print(f"[FunPackVideoRefiner] Failed to load reference embedding: {e}. Resetting for this prompt.")
            old_reference = raw_positive.clone()
            active["reference_embeds"] = tensor_to_serializable(old_reference)
            active["history"] = []
            is_new_prompt = True

        device = old_reference.device
        cur_positive = raw_positive.to(device) if raw_positive.device != device else raw_positive

        # Shape mismatch guard
        if old_reference.shape != cur_positive.shape:
            print(f"[FunPackVideoRefiner] Reference shape {old_reference.shape} != current {cur_positive.shape}. Resetting reference.")
            old_reference = cur_positive.clone()
            active["reference_embeds"] = tensor_to_serializable(old_reference)
            active["history"] = []
            is_new_prompt = True

        seq_len = self._get_conditioning_seq_len(cur_positive)
        active_token_mask = self._get_conditioning_token_mask(cur_positive) if mode == "wan" else None

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

        # Safe similarity calculation
        try:
            if old_reference.dim() >= 2 and cur_positive.dim() >= 2:
                ref_mean = self._masked_sequence_mean(old_reference, active_token_mask)
                cur_mean = self._masked_sequence_mean(cur_positive, active_token_mask)
                similarity = F.cosine_similarity(ref_mean, cur_mean, dim=-1).mean().item()
            else:
                similarity = F.cosine_similarity(
                    old_reference.flatten().unsqueeze(0),
                    cur_positive.flatten().unsqueeze(0),
                    dim=-1
                ).mean().item()
        except Exception:
            similarity = 0.0

        reward = (rating - 5.5) / 4.5
        last_rating = active.get("last_rating", 5)
        last_reward = (last_rating - 5.5) / 4.5
        rating_shift = abs(rating - last_rating)
        feedback_memory["rating_change_events"] = list(feedback_memory.get("rating_change_events", []))[-15:]
        feedback_memory["rating_change_events"].append({
            "iteration": iter_num,
            "rating": rating,
            "shift": rating_shift,
            "prompt_key": prompt_key,
        })

        word_importance = global_adaptive["word_importance"]

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
            group_delta = reward * base_lr * lr_scale * confidence
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
        reference = cur_positive.clone()

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
        if rating >= 8:
            good_ratio = 0.9 * good_ratio + 0.1 * 1.0
            expl = max(0.015, expl * 0.96)
        else:
            good_ratio = 0.9 * good_ratio + 0.1 * 0.0
            expl = min(0.12, expl * 1.08)
        global_adaptive["good_ratio"] = good_ratio

        sim_threshold = global_adaptive["dynamic_sim_threshold"]
        is_close = (not is_new_prompt) and (similarity >= sim_threshold)

        if is_close and history:
            last_entry = history[-1]
            mod_data = last_entry.get("modified_embeds")
            if mod_data is not None:
                try:
                    prev_modified = serializable_to_tensor(mod_data).to(device)
                    if list(prev_modified.shape) != list(old_reference.shape):
                        prev_modified = torch.zeros_like(old_reference)
                except Exception:
                    prev_modified = torch.zeros_like(old_reference)
            else:
                prev_modified = torch.zeros_like(old_reference)
            prev_delta = prev_modified - old_reference
            multiplier = max(0.05, 1.0 + reward * 1.45)
            if rating == last_rating and rating >= 8:
                multiplier += 0.35
            noise = torch.randn_like(old_reference) * (expl * (1.0 - avg_reward_ema * 0.7))
            new_delta = (prev_delta * multiplier) + noise + (momentum * 0.6)
        else:
            good_deltas = []
            for entry in history:
                if entry.get("rating", 0) >= 7:
                    mod_data = entry.get("modified_embeds")
                    if mod_data is None:
                        continue
                    try:
                        mod = serializable_to_tensor(mod_data).to(device)
                        if list(mod.shape) == list(old_reference.shape):
                            good_deltas.append(mod - old_reference)
                    except Exception:
                        continue
            if good_deltas:
                new_delta = torch.stack(good_deltas).mean(dim=0) * (0.7 + reward * 0.4)
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
            "reward": round(reward, 3),
            "modified_embeds": tensor_to_serializable(new_positive),
            "similarity": round(similarity, 4),
            "prompt": positive_prompt[:180]
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
        active["last_positive_prompt"] = positive_prompt
        data["last_prompt_key"] = prompt_key
        data["prompt_histories"] = prompt_histories
        data["global_adaptive"] = global_adaptive

        refined_sigmas, sigma_status = self._refine_sigma_schedule(
            sigmas,
            rating,
            global_adaptive,
            sigma_strength,
            seed,
        )
        if latent_output_connected:
            refined_latent, latent_status = self._refine_latent(
                latent,
                refinement_key,
                mode,
                rating,
                reward,
                global_adaptive,
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
                f"stab={c.get('stability_weight', 1.0):.2f}]"
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
        )

        normalized_reward = max(0.0, min(1.0, (avg_reward_ema + 1.0) / 2.0))
        stability_factor = max(0.0, 1.0 - similarity)
        raw_loss = (1.0 - normalized_reward) * 0.7 + stability_factor * 0.3
        learning_loss = max(0.02, raw_loss * (1.0 - min(0.95, good_ratio * 0.8)))

        training_info = (
            f"Mode: {mode.upper()} | Scheduler: {scheduler_mode.upper()} | Step: {global_adaptive['current_step']} | "
            f"EMA Reward: {avg_reward_ema:+.3f} | Confidence: {confidence:.2f} | LR Scale: {lr_scale:.3f}\n"
            f"Exploration: {expl:.3f} | Similarity: {similarity:.4f} | Good Ratio: {good_ratio:.1%}\n"
            f"**Learning Loss: {learning_loss:.4f}** (lower is better)\n"
            f"Dominant concept: {dominant_line}\n"
            f"Concept phrases ({len(concept_clusters)} total): {concept_line}\n"
            f"Concept groups ({len(concept_groups)} total): {group_line}\n"
            f"Global top words: {current_top}\n"
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

        status = (
            f"Mode {mode.upper()} | Iter {iter_num} | Total iters {global_total_iterations} | "
            f"History {len(history)} | Unique prompts {len(prompt_histories)}\n"
            f"Rating {rating}/10 {trend} | Sim {similarity:.3f} | Reward {reward:+.2f} | "
            f"EMA {avg_reward_ema:+.2f} | Good ratio {good_ratio:.0%} | Expl {expl:.3f} | {health}\n"
            f"Dominant: {dominant_line} | Groups: {group_line}\n"
            f"{lora_suggestion_status}\n"
            f"{sigma_status}\n"
            f"{latent_status}"
        )

        # ====================== FINAL SAVE ======================
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # ====================== RETURN ======================
        modified_positive = [(new_positive, positive_meta)]

        return (modified_positive, status, feedback_question_output, training_info, loss_graph, refined_sigmas, refined_latent)


FunPackGemmaEmbeddingRefiner = FunPackVideoRefiner


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
            },
            "optional": {
                "prompt1": ("STRING", {"default": "", "multiline": True}),
                "prompt2": ("STRING", {"default": "", "multiline": True}),
                "prompt3": ("STRING", {"default": "", "multiline": True}),
                "prompt4": ("STRING", {"default": "", "multiline": True}),
                "prompt5": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("out1", "out2", "out3", "out4", "out5")
    FUNCTION = "combine"
    CATEGORY = "FunPack"
    OUTPUT_NODE = False

    def combine(self, main_prompt,
                prompt1="", prompt2="", prompt3="", prompt4="", prompt5=""):

        main = main_prompt.strip()

        def merge(base, addon):
            addon = addon.strip()
            if not addon:
                return base
            if not base:
                return addon
            # You can change separator style here
            #return f"{base}, {addon}"          # ← comma style (most common in SD)
            return f"{base}\n{addon}"        # ← new line style
            # return f"{base} | {addon}"       # ← pipe style, etc.

        results = []
        for p in (prompt1, prompt2, prompt3, prompt4, prompt5):
            combined = merge(main, p)
            results.append(combined)

        # Always return exactly 5 strings (even if some are just the main prompt)
        return tuple(results)

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
