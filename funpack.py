import os
import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlavaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file, safe_open
import glob
import comfy.model_management as mm
import comfy.sd as sd
import folder_paths
import nodes
import tempfile
import gc # Import garbage collector
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import folder_paths
from comfy.utils import ProgressBar
import comfy.clip_vision
import math
import json
from comfy.utils import ProgressBar
import random
import re
import base64
from hashlib import md5
import time

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
    arr = np.frombuffer(base64.b64decode(d["data"]), dtype=d["dtype"]).reshape(d["shape"])
    tensor = torch.from_numpy(arr).to(dtype=torch.float32)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor
    
class FunPackGemmaEmbeddingRefiner:
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
                print(f"[FunPackGemmaEmbeddingRefiner] Tokenizer load failed for mode '{mode}' from '{model_id}': {e}")

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
        raw_words = [w.strip() for w in prompt.split() if w.strip()]
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

        return [group for group in word_groups if group[1] > group[0]]

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
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("modified_positive", "status", "feedback_question", "training_info")
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
        phrases = [p.strip() for p in re.split(r'[,;]', prompt) if p.strip()]
        result = []
        for phrase in phrases:
            words = [w.strip().lower() for w in phrase.split() if self._is_valuable_token(w.strip())]
            if words:
                result.append(words)
        return result

    def _build_prompt_fallback_concept(self, prompt: str, concept_clusters: dict):
        if not prompt:
            return None

        fallback_words = [
            w.strip().lower()
            for w in re.split(r'[\s,;]+', prompt)
            if self._is_valuable_token(w.strip())
        ][:8]
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
    # MAIN REFINE
    # =========================================================================

    def refine(self, positive_conditioning, mode: str, rating: int, refinement_key: str,
               scheduler_mode: str = "original", positive_prompt: str = "",
               reset_session: bool = False, unlimited_history: bool = False,
               seed: int = 0, feedback_enabled: bool = False, feedback_rating: int = 3):

        mode = (mode or "ltx2").lower()
        if mode not in self._tokenizer_sources:
            mode = "ltx2"

        if seed != 0:
            torch.manual_seed(seed)
            random.seed(seed)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        refinements_dir = os.path.join(base_dir, "refinements")
        os.makedirs(refinements_dir, exist_ok=True)

        safe_key = md5(f"{mode}::{refinement_key}".encode("utf-8")).hexdigest()
        json_file = os.path.join(refinements_dir, f"refine_{safe_key}.json")

        if not positive_conditioning or not isinstance(positive_conditioning, list) or len(positive_conditioning) == 0:
            return (positive_conditioning, "ERROR: Empty positive CONDITIONING input", "", "ERROR: No positive conditioning")

        item = positive_conditioning[0]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            raw_positive = item[0]
            positive_meta = item[1] if isinstance(item[1], dict) else {"pooled_output": None}
        else:
            raw_positive = item if isinstance(item, torch.Tensor) else None
            positive_meta = {"pooled_output": None}

        if not isinstance(raw_positive, torch.Tensor):
            return (positive_conditioning, "ERROR: No positive embedding tensor found", "", "ERROR: Invalid embedding")

        analysis_prompt = self._normalize_prompt_for_mode(positive_prompt, mode)
        prompt_key = analysis_prompt if mode == "wan" else positive_prompt

        # ====================== STATE TEMPLATES ======================
        # Single source of truth for both reset and corrupt-recovery paths.
        def _fresh_global():
            return {
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
                "mode": mode,
                "scheduler_mode": scheduler_mode,
                "prodigy_d": {},
                "prodigy_lr_base": 1.0,
                "warmup_steps": 8,
                "total_steps_estimate": 150,
                "current_step": 0,
            }

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
            data = _fresh_data()
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return (positive_conditioning, "✓ New session started – Reference saved", "", "✓ New session started. Reference embedding saved.")

        # ====================== SAFE JSON LOAD ======================
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, ValueError) as e:
            print(f"[FunPackGemmaEmbeddingRefiner] Corrupt session file, resetting: {e}")
            try:
                os.remove(json_file)
            except OSError:
                pass
            data = _fresh_data()
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return (positive_conditioning, "⚠️ Session file was corrupt – Reset and started fresh", "", "⚠️ Session reset due to corrupt file")

        global_adaptive = data["global_adaptive"]
        # Migrate sessions created before the multi-level concept system was added
        global_adaptive.setdefault("concept_clusters", {})
        global_adaptive.setdefault("concept_groups", {})
        global_adaptive.setdefault("word_importance", {})
        global_adaptive.setdefault("feedbacked_concepts", [])
        global_adaptive.setdefault("feedback_memory", {"recent_questions": [], "rating_change_events": []})
        global_adaptive.setdefault("mode", mode)
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
            print(f"[FunPackGemmaEmbeddingRefiner] Failed to load reference embedding: {e}. Resetting for this prompt.")
            old_reference = raw_positive.clone()
            active["reference_embeds"] = tensor_to_serializable(old_reference)
            active["history"] = []
            is_new_prompt = True

        device = old_reference.device
        cur_positive = raw_positive.to(device) if raw_positive.device != device else raw_positive

        # Shape mismatch guard
        if old_reference.shape != cur_positive.shape:
            print(f"[FunPackGemmaEmbeddingRefiner] Reference shape {old_reference.shape} != current {cur_positive.shape}. Resetting reference.")
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
            f"Global top words: {current_top}"
        )

        if scheduler_mode == "accurate":
            training_info += "\n[Accurate] Conservative • Prodigy + Cosine"
        elif scheduler_mode == "aggressive":
            training_info += "\n[Aggressive] Fast style locking"

        # ====================== STATUS ======================
        trend = "↑" if reward > last_reward else "↓" if reward < last_reward else "→"
        health = (
            "🚀 Strong convergence" if avg_reward_ema > 0.6 else
            "✅ Learning well" if avg_reward_ema > 0.3 else
            "⚠️ Still exploring" if avg_reward_ema > -0.2 else
            "🔄 Heavy correction"
        )

        status = (
            f"Mode {mode.upper()} | Iter {iter_num} | Total iters {total_iters} | "
            f"History {len(history)} | Unique prompts {len(prompt_histories)}\n"
            f"Rating {rating}/10 {trend} | Sim {similarity:.3f} | Reward {reward:+.2f} | "
            f"EMA {avg_reward_ema:+.2f} | Good ratio {good_ratio:.0%} | Expl {expl:.3f} | {health}\n"
            f"Dominant: {dominant_line} | Groups: {group_line}"
        )

        # ====================== FINAL SAVE ======================
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # ====================== RETURN ======================
        modified_positive = [(new_positive, positive_meta)]

        return (modified_positive, status, feedback_question_output, training_info)
        
# Constants from StoryMem
IMAGE_FACTOR = 28
VIDEO_MIN_PIXELS = 48 * IMAGE_FACTOR * IMAGE_FACTOR  # 37,632
MIN_FRAME_SIMILARITY = 0.9
MAX_KEYFRAME_NUM = 3
ADAPTIVE_ALPHA = 0.01
HPSV3_QUALITY_THRESHOLD = 3.0

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
            
class FunPackVideoStitch:
    CATEGORY = "FunPack"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("STITCHED",)
    FUNCTION = "stitch"
    INPUT_TYPES = lambda: {
        "required": {
            "blend_frames": ("INT", {"default": 8, "min": 1, "max": 64}),
        },
        "optional": {
            "video1": ("IMAGE",),
            "video2": ("IMAGE",),
            "video3": ("IMAGE",),
            "video4": ("IMAGE",),
            "video5": ("IMAGE",),
            "video6": ("IMAGE",),
            "video7": ("IMAGE",),
            "video8": ("IMAGE",),
        }
    }

    def linear_blend(self, batch_a, batch_b, blend_frames):
        if blend_frames == 1:
            blended_frame = 0.5 * batch_a[-1] + 0.5 * batch_b[0]
            return blended_frame.unsqueeze(0)

        blended = []
        for i in range(blend_frames):
            alpha = i / (blend_frames - 1)
            blended_frame = (1 - alpha) * batch_a[-blend_frames + i] + alpha * batch_b[i]
            blended.append(blended_frame.unsqueeze(0))
        return torch.cat(blended, dim=0)

    def stitch(self, blend_frames, video1=None, video2=None, video3=None, video4=None, video5=None, video6=None, video7=None, video8=None):
        input_videos = [video1, video2, video3, video4, video5, video6, video7, video8]
        video_batches = [v for v in input_videos if v is not None]

        if len(video_batches) < 2:
            raise ValueError("VideoStitch requires at least 2 connected video inputs.")

        output_frames = []

        for i in range(len(video_batches) - 1):
            batch_a = video_batches[i]
            batch_b = video_batches[i + 1]

            if batch_a.shape[0] < blend_frames or batch_b.shape[0] < blend_frames:
                raise ValueError(f"Each video batch must have at least {blend_frames} frames.")

            stable_a = batch_a[:-blend_frames]
            stable_b = batch_b[blend_frames:]
            transition = self.linear_blend(batch_a, batch_b, blend_frames)

            if i == 0:
                output_frames.append(stable_a)
            output_frames.append(transition)
            output_frames.append(stable_b if i == len(video_batches) - 2 else batch_b[blend_frames:-blend_frames])

        final_video = torch.cat(output_frames, dim=0)
        return (final_video,)
        
class FunPackContinueVideo:
    CATEGORY = "FunPack"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("CONTINUED",)
    FUNCTION = "continue_video"
    INPUT_TYPES = lambda: {
        "required": {
            "images": ("IMAGE",),
            "frame_count": ("INT", {"default": 1, "min": 1, "max": 9999}),
        }
    }

    def continue_video(self, images, frame_count):
        total_frames = images.shape[0]

        if frame_count > total_frames:
            raise ValueError(f"Cannot extract {frame_count} frames from video with only {total_frames} frames.")

        continued = images[-frame_count:]
        return (continued,)

class FunPackStoryMemKeyframeExtractor:
    """
    Extracts keyframes from video frames using:
    1. HPSv3 for quality assessment (optional)
    2. CLIP Vision for frame similarity
    3. Adaptive threshold to limit keyframe count
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE",),  # ComfyUI IMAGE format [B, H, W, C]
                "clip_vision": (folder_paths.get_filename_list("clip_vision"),),
                "max_keyframes": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Maximum number of keyframes to extract"
                }),
                "similarity_threshold": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "CLIP similarity threshold (lower = more keyframes)"
                }),
                "use_quality_filter": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use HPSv3 to filter low-quality frames (requires hpsv3 package)"
                }),
                "quality_threshold": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "HPSv3 quality threshold (higher = stricter)"
                }),
            },
            "optional": {
                "memory_frames": ("IMAGE", {
                    "tooltip": "Previous keyframes to compare against (avoid duplicates)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("keyframes", "keyframe_count",)
    FUNCTION = "extract_keyframes"
    CATEGORY = "FunPack"
    DESCRIPTION = "Extract keyframes using CLIP similarity + HPSv3 quality (StoryMem algorithm)"

    def __init__(self):
        self.quality_model = None
        
    def load_clip_model(self, clip_vision_name):
        """Load CLIP Vision model from ComfyUI models/clip_vision folder"""
        clip_path = folder_paths.get_full_path("clip_vision", clip_vision_name)
        clip_vision = comfy.clip_vision.load(clip_path)
        return clip_vision
        
    def load_quality_model(self):
        """Load HPSv3 quality assessment model"""
        if self.quality_model is not None:
            return
            
        try:
            from hpsv3 import HPSv3RewardInferencer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.quality_model = HPSv3RewardInferencer(device=device)
        except ImportError:
            print("WARNING: HPSv3 not installed. Install with: pip install hpsv3")
            print("Quality filtering will be disabled.")
            self.quality_model = None
    
    def smart_resize(self, height: int, width: int) -> tuple:
        """Resize frame to efficient size for processing"""
        factor = IMAGE_FACTOR
        min_pixels = VIDEO_MIN_PIXELS
        max_pixels = 256 * IMAGE_FACTOR * IMAGE_FACTOR
        
        h_bar = max(factor, round(height / factor) * factor)
        w_bar = max(factor, round(width / factor) * factor)
        
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
            
        return max(h_bar, factor), max(w_bar, factor)
    
    def clip_preprocess(self, frame_chw: torch.Tensor, clip_vision) -> torch.Tensor:
        """Preprocess frame for CLIP Vision model"""
        # ComfyUI CLIP Vision expects [B, H, W, C] format in range [0, 1]
        # Convert from [C, H, W] to [1, H, W, C]
        frame = frame_chw.permute(1, 2, 0).unsqueeze(0)
        
        # Ensure [0, 1] range
        if not torch.is_floating_point(frame):
            frame = frame.float()
        if frame.max() > 1.5:
            frame = frame / 255.0
        frame = frame.clamp(0.0, 1.0)
        
        return frame
    
    @torch.no_grad()
    def get_clip_similarity(self, frame1: torch.Tensor, frame2: torch.Tensor, clip_vision) -> float:
    
        # Preprocess frames to [1, H, W, C] format
        x1 = self.clip_preprocess(frame1, clip_vision)
        x2 = self.clip_preprocess(frame2, clip_vision)
    
        # Get CLIP Vision embeddings
        z1_raw = clip_vision.encode_image(x1)
        z2_raw = clip_vision.encode_image(x2)
    
        # Extract the actual embedding tensor from various possible return formats
        def extract_embedding(output):
            # Case 1: Direct tensor (older/basic models)
            if isinstance(output, torch.Tensor):
                return output
        
            # Case 2: ComfyUI's custom Output wrapper (common with projection models)
            if isinstance(output, comfy.clip_vision.Output):  # Import at top if needed: import comfy.clip_vision
                if hasattr(output, 'image_embeds'):
                    return output.image_embeds
                elif hasattr(output, 'pooled_output'):
                    return output.pooled_output
                # Fallback: treat like dict
                try:
                    return output['image_embeds']
                except:
                    pass
        
            # Case 3: Dictionary (some models)
            if isinstance(output, dict):
                if 'image_embeds' in output:
                    return output['image_embeds']
                if 'pooled_output' in output:
                    return output['pooled_output']
                if 'last_hidden_state' in output:
                    return output['last_hidden_state'][:, 0]  # CLS token if sequence
                # Fallback: first tensor value
                for v in output.values():
                    if isinstance(v, torch.Tensor) and v.ndim >= 2:
                        return v
        
            # Case 4: Tuple (rare here, but safe)
            if isinstance(output, (tuple, list)) and len(output) == 1:
                return extract_embedding(output[0])
        
            raise TypeError(f"Unexpected output from encode_image: {type(output)}. "
                            "Supported: tensor, dict, or comfy.clip_vision.Output with 'image_embeds'.")

        z1 = extract_embedding(z1_raw)
        z2 = extract_embedding(z2_raw)
    
        # Final check
        if not (isinstance(z1, torch.Tensor) and isinstance(z2, torch.Tensor)):
            raise RuntimeError(f"Failed to extract tensor embeddings: {type(z1)}, {type(z2)}")
    
        # Normalize and compute cosine similarity
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        similarity = (z1 * z2).sum(dim=-1).item()
    
        return similarity
    
    def is_low_quality(self, frame: torch.Tensor, threshold: float) -> bool:
        """Check if frame quality is below threshold using HPSv3"""
        if self.quality_model is None:
            return False  # Skip quality check if model not available
        
        # Convert to PIL Image
        frame_np = frame.permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np * 255).astype(np.uint8).clip(0, 255)
        pil_image = Image.fromarray(frame_np)
        
        # Get quality score
        try:
            rewards = self.quality_model.reward(image_paths=[pil_image], prompts=[""])
            score = rewards[0][0].item()
            return score < threshold
        except Exception as e:
            print(f"Quality check failed: {e}")
            return False
    
    def extract_keyframe_indices(self, frames: torch.Tensor, threshold: float,
                                  quality_threshold: float, use_quality: bool, clip_vision) -> list:
        """
        Extract keyframe indices using CLIP similarity and quality filtering
        
        Args:
            frames: [N, C, H, W] tensor
            threshold: CLIP similarity threshold
            quality_threshold: HPSv3 quality threshold
            use_quality: Whether to use quality filtering
            clip_vision: ComfyUI CLIP Vision model
            
        Returns:
            List of keyframe indices
        """
        num_frames, _, height, width = frames.shape
        
        # Resize frames for efficient processing
        resized_h, resized_w = self.smart_resize(height, width)
        resized_frames = F.interpolate(
            frames,
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False
        ).float()
        
        # Load quality model if needed
        if use_quality:
            self.load_quality_model()
        
        # Find first high-quality frame
        first_idx = 0
        if use_quality and self.quality_model is not None:
            while first_idx < num_frames:
                if not self.is_low_quality(resized_frames[first_idx], quality_threshold):
                    break
                first_idx += 1
            
            if first_idx >= num_frames:
                return []  # No high-quality frames found
        
        # Initialize keyframes
        keyframe_indices = [first_idx]
        last_keyframe = resized_frames[first_idx]
        
        # Iterate through remaining frames
        pbar = ProgressBar(num_frames - first_idx - 1)
        for i in range(first_idx + 1, num_frames):
            current_frame = resized_frames[i]
            
            # Calculate similarity with last keyframe
            similarity = self.get_clip_similarity(last_keyframe, current_frame, clip_vision)
            
            # Check if frame is different enough and high quality
            is_different = similarity < threshold
            is_quality = True
            if use_quality and self.quality_model is not None:
                is_quality = not self.is_low_quality(current_frame, quality_threshold)
            
            if is_different and is_quality:
                keyframe_indices.append(i)
                last_keyframe = current_frame
            
            pbar.update(1)
        
        return keyframe_indices
    
    def check_memory_duplicates(self, keyframes: torch.Tensor, 
                                memory_frames: torch.Tensor,
                                clip_vision,
                                threshold: float = 0.9) -> list:
        """
        Filter out keyframes that are too similar to memory frames
        
        Returns:
            List of boolean flags (True = keep, False = duplicate)
        """
        keep_flags = []
        
        for keyframe in keyframes:
            is_duplicate = False
            for memory_frame in memory_frames:
                similarity = self.get_clip_similarity(keyframe, memory_frame, clip_vision)
                if similarity > threshold:
                    is_duplicate = True
                    break
            keep_flags.append(not is_duplicate)
        
        return keep_flags
    
    def extract_keyframes(self, frames, clip_vision, max_keyframes, similarity_threshold,
                         use_quality_filter, quality_threshold, memory_frames=None):
        """
        Main extraction function
        
        Args:
            frames: ComfyUI IMAGE format [B, H, W, C] in range [0, 1]
            clip_vision: CLIP Vision model name from dropdown
            max_keyframes: Maximum number of keyframes
            similarity_threshold: Initial CLIP similarity threshold
            use_quality_filter: Whether to use HPSv3 filtering
            quality_threshold: HPSv3 threshold
            memory_frames: Optional previous keyframes to avoid duplicates
            
        Returns:
            (keyframes, keyframe_count)
        """
        # Load CLIP Vision model from ComfyUI models folder
        clip_vision_model = self.load_clip_model(clip_vision)
        
        # Convert ComfyUI format [B, H, W, C] to PyTorch [B, C, H, W]
        frames_tensor = frames.permute(0, 3, 1, 2).contiguous()
        
        # Adaptive threshold loop
        threshold = similarity_threshold
        while True:
            keyframe_indices = self.extract_keyframe_indices(
                frames_tensor,
                threshold,
                quality_threshold,
                use_quality_filter,
                clip_vision_model
            )
            
            # Check if we have too many keyframes
            if len(keyframe_indices) <= max_keyframes:
                break
            
            # Increase threshold to get fewer keyframes
            threshold -= ADAPTIVE_ALPHA
            
            # Safety check
            if threshold < 0.5:
                # Take first N keyframes
                keyframe_indices = keyframe_indices[:max_keyframes]
                break
        
        print(f"Extracted {len(keyframe_indices)} keyframes at threshold {threshold:.3f}")
        
        # Extract keyframes
        if len(keyframe_indices) == 0:
            # Return first frame as fallback
            keyframes_out = frames[:1]
            return (keyframes_out, 1)
        
        keyframes_tensor = frames_tensor[keyframe_indices]
        
        # Check against memory frames to avoid duplicates
        if memory_frames is not None:
            memory_tensor = memory_frames.permute(0, 3, 1, 2).contiguous()
            keep_flags = self.check_memory_duplicates(
                keyframes_tensor,
                memory_tensor,
                clip_vision_model,
                threshold=MIN_FRAME_SIMILARITY
            )
            
            # Filter keyframes
            kept_indices = [i for i, keep in enumerate(keep_flags) if keep]
            if len(kept_indices) > 0:
                keyframes_tensor = keyframes_tensor[kept_indices]
            else:
                # Keep at least one keyframe
                keyframes_tensor = keyframes_tensor[:1]
        
        # Convert back to ComfyUI format [B, H, W, C]
        keyframes_out = keyframes_tensor.permute(0, 2, 3, 1).contiguous()
        
        return (keyframes_out, keyframes_out.shape[0])


class FunPackStoryMemLastFrameExtractor:
    """Extract last frame and last N frames for MI2V/MM2V continuity"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE",),
                "n_frames": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of frames to extract from end (for MM2V)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("last_frame", "motion_frames",)
    FUNCTION = "extract"
    CATEGORY = "FunPack"
    DESCRIPTION = "Extract last frame and last N frames for shot continuity (MI2V/MM2V)"
    
    def extract(self, frames, n_frames):
        """
        Extract last frame and last N frames
        
        Returns:
            (last_frame [1, H, W, C], motion_frames [N, H, W, C])
        """
        last_frame = frames[-1:]
        motion_frames = frames[-n_frames:]
        
        return (last_frame, motion_frames)

# Update NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "FunPackPromptCombiner": FunPackPromptCombiner,
    "FunPackStoryMemKeyframeExtractor": FunPackStoryMemKeyframeExtractor,
    "FunPackStoryMemLastFrameExtractor": FunPackStoryMemLastFrameExtractor,
    "FunPackPromptEnhancer": FunPackPromptEnhancer,
    "FunPackStoryWriter": FunPackStoryWriter,
    "FunPackVideoStitch": FunPackVideoStitch,
    "FunPackContinueVideo": FunPackContinueVideo,
    "FunPackLorebookEnhancer": FunPackLorebookEnhancer,
    "FunPackGemmaEmbeddingRefiner": FunPackGemmaEmbeddingRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FunPackPromptCombiner": "FunPack Prompt Combiner",
    "FunPackStoryMemKeyframeExtractor": "FunPack StoryMem Keyframe Extractor",
    "FunPackStoryMemLastFrameExtractor": "FunPack StoryMem Last Frame Extractor",
    "FunPackPromptEnhancer": "FunPack Prompt Enhancer (Standalone)",
    "FunPackStoryWriter": "FunPack Story Writer",
    "FunPackVideoStitch": "FunPack Video Stitch",
    "FunPackContinueVideo": "FunPack Continue Video",
    "FunPackLorebookEnhancer": "FunPack Lorebook Enhancer",
    "FunPackGemmaEmbeddingRefiner": "FunPack Gemma Embedding Refiner (Self-Refinement)"
}
