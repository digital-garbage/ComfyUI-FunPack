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
    _tokenizer = None

    @classmethod
    def _get_tokenizer(cls):
        if cls._tokenizer is None:
            try:
                cls._tokenizer = AutoTokenizer.from_pretrained(
                    "DreamFast/gemma-3-12b-it-heretic-v2",
                    trust_remote_code=True,
                    use_fast=True
                )
            except Exception as e:
                print(f"[FunPackGemmaEmbeddingRefiner] Tokenizer load failed: {e}")
                cls._tokenizer = None
        return cls._tokenizer

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_conditioning": ("CONDITIONING",),
                "rating": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "slider",
                    "label": "Rating (1=horrific, 10=masterpiece)"
                }),
                "refinement_key": ("STRING", {"default": "my_style_v1", "multiline": False}),
            },
            "optional": {
                "positive_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Positive prompt"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Negative prompt (optional)"
                }),
                "negative_conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
                "reset_session": ("BOOLEAN", {"default": False, "label": "Reset Session (clears ALL history)"}),
                "unlimited_history": ("BOOLEAN", {
                    "default": False,
                    "label": "Unlimited History (never prunes)"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "label": "Exploration Seed"}),
                "feedback_enabled": ("BOOLEAN", {"default": False, "label": "Enable Token Feedback"}),
                "feedback_rating": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 6,
                    "step": 1,
                    "display": "slider",
                    "label": "Token Feedback (1=absent, 5=perfect, 6=overrepresented)"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING", "STRING")
    RETURN_NAMES = ("modified_positive", "modified_negative", "modified_latent", "status", "feedback_question")
    FUNCTION = "refine"
    CATEGORY = "FunPack/Refinement"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def _get_top_tokens(self, token_dict, tokenizer, top_k=10):
        if not token_dict or not tokenizer:
            return "N/A"
        sorted_tokens = sorted(token_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_list = []
        for tid_str, score in sorted_tokens:
            try:
                token_text = tokenizer.decode([int(tid_str)], skip_special_tokens=True).strip()
                if token_text:
                    top_list.append(f"{token_text}({score:.2f})")
            except:
                top_list.append(f"ID{tid_str}({score:.2f})")
        return ", ".join(top_list) if top_list else "None"

    def _is_valuable_token(self, token_text):
        if not token_text:
            return False
        t = token_text.strip()
        if '<' in t or '>' in t or len(t) < 3:
            return False
        t_lower = t.lower()
        stopwords = {
            "the","a","an","and","or","but","with","for","of","in","on","at","to","from","by","is",
            "are","was","were","be","been","being","have","has","had","do","does","did","will","would",
            "her","his","him","she","he","it","they","them","this","that","these","those","i","you",
            "my","your","our","their","me","us"
        }
        if t_lower in stopwords:
            return False
        if t in {",",".","!","?",":",";","-","*","(",")","[","]","{","}","'","\"", "..."} or t.isdigit():
            return False
        if not any(c.isalpha() for c in t):
            return False
        return True

    def refine(self, positive_conditioning, rating: int, refinement_key: str,
               positive_prompt: str = "", negative_prompt: str = "",
               negative_conditioning=None, latent=None,
               reset_session: bool = False, unlimited_history: bool = False,
               seed: int = 0,
               feedback_enabled: bool = False, feedback_rating: int = 3):

        if seed != 0:
            torch.manual_seed(seed)
            random.seed(seed)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        refinements_dir = os.path.join(base_dir, "refinements")
        os.makedirs(refinements_dir, exist_ok=True)

        safe_key = md5(refinement_key.encode("utf-8")).hexdigest()
        json_file = os.path.join(refinements_dir, f"refine_{safe_key}.json")

        if not positive_conditioning or not isinstance(positive_conditioning, list) or len(positive_conditioning) == 0:
            return (positive_conditioning, negative_conditioning, latent, "ERROR: Empty positive CONDITIONING input", "")

        item = positive_conditioning[0]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            raw_positive = item[0]
            positive_meta = item[1] if isinstance(item[1], dict) else {"pooled_output": None}
        else:
            raw_positive = item if isinstance(item, torch.Tensor) else None
            positive_meta = {"pooled_output": None}

        if not isinstance(raw_positive, torch.Tensor):
            return (positive_conditioning, negative_conditioning, latent, "ERROR: No positive embedding tensor found", "")

        prompt_key = positive_prompt
        if negative_prompt:
            prompt_key += f" |||NEG||| {negative_prompt}"

        if reset_session or not os.path.exists(json_file):
            data = {
                "refinement_key": refinement_key,
                "global_adaptive": {
                    "token_importance": {},
                    "infusion_tokens": {},
                    "token_library": {},
                    "exploration_base": 0.08,
                    "momentum": None,
                    "avg_reward_ema": 0.0,
                    "good_ratio": 0.0,
                    "dynamic_sim_threshold": 0.82,
                    "last_feedback_tid": None,
                    "last_feedback_word": None
                },
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
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return (positive_conditioning, negative_conditioning, latent, "✓ New session started – Reference saved", "")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        global_adaptive = data["global_adaptive"]
        prompt_histories = data.get("prompt_histories", {})
        tokenizer = self._get_tokenizer()

        # ====================== FEEDBACK STATE MACHINE ======================
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
                token_id = pending["token_id"]
                tid_str = str(token_id)
                token_importance = global_adaptive["token_importance"]

                old_imp = token_importance.get(tid_str, 1.0)

                if feedback_rating == 6:
                    target_imp = 0.65
                else:
                    target_imp = 0.4 + (feedback_rating / 5.0) * 2.1

                new_imp = 0.65 * old_imp + 0.35 * target_imp
                token_importance[tid_str] = max(0.3, min(2.5, new_imp))

                token_library = global_adaptive.setdefault("token_library", {})
                if tid_str in token_library:
                    if feedback_rating >= 4 and feedback_rating != 6:
                        token_library[tid_str]["high_rated_count"] += 2
                    elif feedback_rating <= 2:
                        token_library[tid_str]["low_rated_count"] += 2

                data["pending_feedback"] = None
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            else:
                feedback_question_output = "Feedback enabled. A question will appear after this generation."

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

        old_reference = serializable_to_tensor(active["reference_embeds"])
        device = old_reference.device
        cur_positive = raw_positive.to(device) if raw_positive.device != device else raw_positive

        full_token_ids = tokenizer.encode(positive_prompt, add_special_tokens=True) if tokenizer and positive_prompt else []

        # ====================== WORD GROUPING ======================
        word_groups = []
        if tokenizer and positive_prompt and full_token_ids:
            try:
                words = positive_prompt.split()
                token_idx = 0
                for word in words:
                    if not word:
                        continue
                    word_tokens = tokenizer.encode(word, add_special_tokens=False)
                    word_len = len(word_tokens)
                    if token_idx + word_len <= len(full_token_ids) and full_token_ids[token_idx:token_idx + word_len] == word_tokens:
                        word_groups.append((token_idx, token_idx + word_len, word, word_tokens))
                        token_idx += word_len
                    else:
                        if token_idx < len(full_token_ids):
                            single_token = full_token_ids[token_idx]
                            decoded = tokenizer.decode([single_token]).strip()
                            word_groups.append((token_idx, token_idx + 1, decoded, [single_token]))
                            token_idx += 1
            except Exception:
                word_groups = [(i, i+1, tokenizer.decode([tid]).strip(), [tid]) for i, tid in enumerate(full_token_ids)]

        # ====================== CROSS-PROMPT TRANSFER ======================
        if is_new_prompt and tokenizer and full_token_ids:
            current_set = set(full_token_ids)
            token_importance = global_adaptive["token_importance"]
            for other_key, other in prompt_histories.items():
                if other_key == prompt_key:
                    continue
                other_tokens = set()
                for entry in other.get("history", []):
                    if "prompt" in entry and entry["prompt"]:
                        try:
                            other_tokens.update(tokenizer.encode(entry["prompt"], add_special_tokens=True)[:60])
                        except:
                            pass
                if not other_tokens:
                    continue
                intersection = len(current_set & other_tokens)
                union = len(current_set | other_tokens)
                sim_weight = intersection / union if union > 0 else 0.0
                if sim_weight < 0.3:
                    continue
                for entry in other.get("history", []):
                    if "prompt" not in entry or not entry["prompt"]:
                        continue
                    try:
                        old_tokens = tokenizer.encode(entry["prompt"], add_special_tokens=True)[:60]
                        matching = current_set & set(old_tokens)
                        r = entry.get("rating", 5)
                        adj = 1.25 if r >= 8 else 0.75 if r <= 3 else None
                        if adj is not None:
                            for tid in matching:
                                tid_str = str(tid)
                                if tid_str in token_importance:
                                    token_importance[tid_str] = max(0.3, min(2.5,
                                        token_importance[tid_str] * (1.0 + (adj - 1.0) * sim_weight)))
                    except:
                        pass

        active["reference_embeds"] = tensor_to_serializable(cur_positive)
        reference = cur_positive.clone()

        # ====================== TOKEN LIBRARY UPDATE ======================
        token_library = global_adaptive.setdefault("token_library", {})
        if tokenizer and full_token_ids:
            valuable_token_ids = [tid for tid in full_token_ids if self._is_valuable_token(tokenizer.decode([tid]))]
            norm_factor = len(valuable_token_ids) if valuable_token_ids else 1

            for tid in full_token_ids:
                tid_str = str(tid)
                if tid_str not in token_library:
                    token_library[tid_str] = {
                        "fqc": 0,
                        "unique_prompts": 0,
                        "high_rated_count": 0,
                        "low_rated_count": 0,
                        "good_pairs": {}
                    }
                lib_entry = token_library[tid_str]
                lib_entry["fqc"] += 1
                if is_new_prompt:
                    lib_entry["unique_prompts"] += 1

            if rating >= 8:
                for i, tid1 in enumerate(full_token_ids):
                    tid1_str = str(tid1)
                    for tid2 in full_token_ids[i+1:]:
                        tid2_str = str(tid2)
                        if tid1_str in token_library and tid2_str in token_library:
                            if tid2_str not in token_library[tid1_str]["good_pairs"]:
                                token_library[tid1_str]["good_pairs"][tid2_str] = 0
                            if tid1_str not in token_library[tid2_str]["good_pairs"]:
                                token_library[tid2_str]["good_pairs"][tid1_str] = 0
                            token_library[tid1_str]["good_pairs"][tid2_str] += 1
                            token_library[tid2_str]["good_pairs"][tid1_str] += 1
                    if tid1_str in token_library:
                        token_library[tid1_str]["high_rated_count"] += 1
            elif rating <= 4:
                for tid in full_token_ids:
                    tid_str = str(tid)
                    if tid_str in token_library:
                        token_library[tid_str]["low_rated_count"] += 1

            for tid_str, entry in token_library.items():
                total_rated = entry["high_rated_count"] + entry["low_rated_count"]
                if total_rated == 0:
                    continue
                appears_in_many = entry["unique_prompts"] >= 5
                balanced = entry["high_rated_count"] > 0 and entry["low_rated_count"] > 0
                mostly_high = entry["high_rated_count"] > entry["low_rated_count"] * 2
                entry["is_general"] = appears_in_many and balanced
                entry["is_specific"] = not entry["is_general"] and mostly_high and entry["unique_prompts"] <= 8
                entry["infusable"] = (entry["is_general"] or entry["is_specific"]) and entry["high_rated_count"] >= 2

        # ====================== CORE REFINEMENT ======================
        ref_mean = old_reference.mean(dim=1)
        cur_mean = cur_positive.mean(dim=1)
        similarity = F.cosine_similarity(ref_mean, cur_mean, dim=-1).mean().item()

        reward = (rating - 5.5) / 4.5
        last_rating = active.get("last_rating", 5)
        last_reward = (last_rating - 5.5) / 4.5

        expl = global_adaptive["exploration_base"]
        momentum = global_adaptive.get("momentum")
        avg_reward_ema = global_adaptive["avg_reward_ema"]
        good_ratio = global_adaptive["good_ratio"]
        sim_threshold = global_adaptive["dynamic_sim_threshold"]
        token_importance = global_adaptive["token_importance"]
        infusion_tokens = global_adaptive["infusion_tokens"]

        avg_reward_ema = 0.85 * avg_reward_ema + 0.15 * reward

        if momentum is None or not isinstance(momentum, dict):
            momentum = torch.zeros_like(reference)
        else:
            momentum = serializable_to_tensor(momentum).to(device)
            if list(momentum.shape) != list(reference.shape):
                momentum = torch.zeros_like(reference)

        new_token_mask = None
        infusion_active = False
        infusion_weights = None

        if tokenizer and positive_prompt:
            try:
                embed_seq_len = cur_positive.shape[1] if cur_positive.dim() == 3 else cur_positive.shape[0] if cur_positive.dim() == 2 else 1
                prev_token_ids = tokenizer.encode(active.get("last_positive_prompt", ""), add_special_tokens=True) if "last_positive_prompt" in active else []

                new_token_mask = torch.ones(embed_seq_len, device=device)
                compare_len = min(len(full_token_ids), len(prev_token_ids), embed_seq_len)
                for i in range(compare_len):
                    if full_token_ids[i] != prev_token_ids[i]:
                        new_token_mask[i] = 0.0

                # ====================== WORD-LEVEL IMPORTANCE UPDATE ======================
                for start, end, full_word, word_token_list in word_groups:
                    is_valuable = self._is_valuable_token(full_word)
                    group_key = str(word_token_list[0])

                    if group_key not in token_importance:
                        token_importance[group_key] = 1.0 if is_valuable else 0.4

                    is_new = any(tid not in prev_token_ids for tid in word_token_list)
                    update_strength = 1.8 if is_new else 1.0
                    if not is_valuable:
                        update_strength *= 0.25

                    fqc = token_library.get(group_key, {}).get("fqc", 0)
                    lr = 0.18 / (1 + 0.08 * (fqc ** 0.5))

                    group_delta = reward * lr * update_strength

                    for tid in word_token_list:
                        tid_str = str(tid)
                        if tid_str in token_importance:
                            token_importance[tid_str] = max(0.3, min(2.5, token_importance[tid_str] + group_delta))
                        else:
                            token_importance[tid_str] = max(0.3, min(2.5, 1.0 + group_delta))

                # Infusion
                if random.random() < 0.18 and token_library:
                    infusion_weights = torch.ones(embed_seq_len, device=device)
                    infusion_any = False
                    for i, tid in enumerate(full_token_ids[:embed_seq_len]):
                        tid_str = str(tid)
                        if tid_str not in token_library or not token_library[tid_str].get("infusable", False):
                            continue
                        token_text = tokenizer.decode([int(tid)], skip_special_tokens=True).strip()
                        if not self._is_valuable_token(token_text):
                            continue
                        entry = token_library[tid_str]
                        if not entry.get("good_pairs"):
                            continue
                        pair_scores = [entry["good_pairs"].get(str(ct), 0) for ct in full_token_ids[:embed_seq_len]]
                        pair_scores = [s for s in pair_scores if s > 0]
                        if pair_scores:
                            avg_pair = sum(pair_scores) / len(pair_scores)
                            if avg_pair > 1.2 or entry.get("is_general", False):
                                if tid_str not in infusion_tokens:
                                    infusion_tokens[tid_str] = 0.0
                                infusion_tokens[tid_str] = min(3.0, infusion_tokens[tid_str] + 0.6)
                                infusion_weights[i] += infusion_tokens[tid_str] * 0.15
                                infusion_any = True
                    if infusion_any:
                        infusion_active = True

            except Exception as e:
                print(f"[FunPackGemmaEmbeddingRefiner] Token comparison failed: {e}")
                new_token_mask = None
                infusion_weights = None

        history = active.get("history", [])
        is_close = (not is_new_prompt) and (similarity >= sim_threshold)

        if is_close and history:
            last_entry = history[-1]
            mod_data = last_entry.get("modified_embeds")
            if mod_data is None:
                prev_modified = torch.zeros_like(old_reference)
            else:
                prev_modified = serializable_to_tensor(mod_data).to(device)
                if list(prev_modified.shape) != list(old_reference.shape):
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
                    mod = serializable_to_tensor(mod_data).to(device)
                    if list(mod.shape) == list(old_reference.shape):
                        good_deltas.append(mod - old_reference)
            if good_deltas:
                new_delta = torch.stack(good_deltas).mean(dim=0) * (0.7 + reward * 0.4)
            else:
                if is_new_prompt and isinstance(momentum, torch.Tensor) and list(momentum.shape) == list(reference.shape):
                    seed_strength = min(0.3, abs(avg_reward_ema) * 0.4)
                    new_delta = momentum * seed_strength + torch.randn_like(reference) * expl * 0.3
                else:
                    new_delta = torch.randn_like(reference) * expl * 0.45

        if token_importance and cur_positive.dim() > 1:
            seq_len = cur_positive.shape[1] if cur_positive.dim() == 3 else cur_positive.shape[0]
            importance_tensor = torch.ones(seq_len, device=device)
            for i, tid in enumerate(full_token_ids[:seq_len]):
                tid_str = str(tid)
                if tid_str in token_importance:
                    importance_tensor[i] = token_importance[tid_str]
            new_delta = new_delta * importance_tensor.unsqueeze(-1)

        if new_token_mask is not None:
            boost_factor = 1.0 + 1.8 * (1.0 - new_token_mask).unsqueeze(-1)
            new_delta = new_delta * boost_factor

        if infusion_active and infusion_weights is not None:
            new_delta = new_delta * infusion_weights.unsqueeze(-1)

        new_positive = reference + new_delta
        new_positive = torch.clamp(new_positive, min=-60.0, max=60.0)
        norm_factor = reference.norm(dim=-1, keepdim=True) + 1e-8
        new_positive = new_positive / (new_positive.norm(dim=-1, keepdim=True) + 1e-8) * norm_factor

        # ====================== NEGATIVE REFINEMENT ======================
        new_negative = None
        negative_meta = {"pooled_output": None}
        if negative_conditioning is not None:
            neg_item = negative_conditioning[0]
            if isinstance(neg_item, (list, tuple)) and len(neg_item) >= 2:
                raw_negative = neg_item[0]
                negative_meta = neg_item[1] if isinstance(neg_item[1], dict) else {"pooled_output": None}
            else:
                raw_negative = neg_item if isinstance(neg_item, torch.Tensor) else None
            if isinstance(raw_negative, torch.Tensor):
                raw_negative = raw_negative.to(device)
                neg_strength = 0.15 if rating <= 4 else 0.05
                neg_delta = torch.randn_like(raw_negative) * neg_strength * (5.5 - rating) / 4.5
                new_negative = raw_negative + neg_delta
                new_negative = torch.clamp(new_negative, min=-60.0, max=60.0)
                norm_factor_neg = raw_negative.norm(dim=-1, keepdim=True) + 1e-8
                new_negative = new_negative / (new_negative.norm(dim=-1, keepdim=True) + 1e-8) * norm_factor_neg

        # ====================== LATENT REFINEMENT ======================
        modified_latent = latent
        latent_info = " | Latent unchanged"
        if latent is not None and "samples" in latent:
            samples = latent["samples"].to(device) if latent["samples"].device != device else latent["samples"]
            current_shape = list(samples.shape)
            good_latents = []
            bad_latents = []
            for entry in history:
                if "latent_samples" in entry:
                    mod_data = entry["latent_samples"]
                    if mod_data is not None:
                        mod = serializable_to_tensor(mod_data).to(device)
                        if list(mod.shape) == current_shape:
                            if entry.get("rating", 5) >= 8:
                                good_latents.append(mod)
                            elif entry.get("rating", 5) <= 3:
                                bad_latents.append(mod)
            if good_latents or bad_latents:
                latent_strength = max(0.0, avg_reward_ema * 0.35)
                latent_delta = torch.zeros_like(samples)
                if good_latents:
                    mean_good = torch.stack(good_latents).mean(dim=0)
                    latent_delta += (mean_good - samples) * latent_strength
                if bad_latents:
                    mean_bad = torch.stack(bad_latents).mean(dim=0)
                    latent_delta -= (mean_bad - samples) * (latent_strength * 0.65)
                mask = torch.abs(samples) < 0.05
                latent_delta = latent_delta * (~mask).float()
                if torch.abs(latent_delta).max() > 1e-4:
                    new_samples = torch.clamp(samples + latent_delta, min=-12.0, max=12.0)
                    modified_latent = {"samples": new_samples.cpu(), "noise_mask": latent.get("noise_mask")}
                    latent_info = f" | Latent ±{len(good_latents)}g/{len(bad_latents)}b"

        # Update momentum and adaptive parameters
        momentum = 0.75 * momentum + 0.25 * (new_delta * reward)
        if avg_reward_ema > 0.3:
            sim_threshold = max(0.75, sim_threshold - 0.002)
        if rating >= 8:
            expl = max(0.015, expl * 0.96)
            good_ratio = 0.9 * good_ratio + 0.1 * 1.0
        else:
            expl = min(0.12, expl * 1.08)
            good_ratio = 0.9 * good_ratio + 0.1 * 0.0

        global_adaptive.update({
            "exploration_base": expl,
            "momentum": tensor_to_serializable(momentum),
            "avg_reward_ema": avg_reward_ema,
            "good_ratio": good_ratio,
            "dynamic_sim_threshold": sim_threshold,
            "token_importance": token_importance,
            "infusion_tokens": infusion_tokens
        })

        # ====================== HISTORY ENTRY ======================
        history_entry = {
            "iteration": len(history) + 1,
            "rating": rating,
            "reward": round(reward, 3),
            "modified_embeds": tensor_to_serializable(new_positive),
            "similarity": round(similarity, 4),
            "prompt": positive_prompt[:180]
        }
        if latent is not None and "samples" in latent:
            history_entry["latent_samples"] = tensor_to_serializable(latent["samples"])

        history.append(history_entry)

        if not unlimited_history and len(history) > 200:
            sorted_hist = sorted(history, key=lambda x: x.get("rating", 0), reverse=True)
            top = sorted_hist[:40]
            recent = history[-120:]
            seen = {e["iteration"] for e in top}
            history = top + [e for e in recent if e["iteration"] not in seen]

        active["history"] = history
        active["last_rating"] = rating
        active["last_positive_prompt"] = positive_prompt

        data["last_prompt_key"] = prompt_key
        data["prompt_histories"] = prompt_histories
        data["global_adaptive"] = global_adaptive

        # ====================== FEEDBACK QUESTION (whole word) ======================
        if feedback_enabled and data.get("pending_feedback") is None and tokenizer and word_groups:
            last_word = global_adaptive.get("last_feedback_word")

            candidates = []
            for start, end, full_word, word_token_list in word_groups:
                if not self._is_valuable_token(full_word) or full_word == last_word:
                    continue
                group_key = str(word_token_list[0])
                imp = token_importance.get(group_key, 1.0)
                freq = token_library.get(group_key, {}).get("fqc", 1)
                score = freq * (1.0 - abs(imp - 1.0))
                candidates.append((full_word, group_key, score, word_token_list[0]))

            if candidates:
                low_imp = [c for c in candidates if c[2] < 1.0]
                if low_imp and random.random() < 0.35:
                    chosen = max(low_imp, key=lambda x: x[2])
                else:
                    chosen = max(candidates, key=lambda x: x[2])

                full_word, group_key, _, best_tid = chosen

                data["pending_feedback"] = {
                    "token_id": int(best_tid),
                    "token_text": full_word,
                    "prompt_key": prompt_key,
                    "iteration": len(history)
                }
                global_adaptive["last_feedback_tid"] = int(best_tid)
                global_adaptive["last_feedback_word"] = full_word

                feedback_question_output = f"Word: '{full_word}' – How well represented? (1=absent, 5=perfect, 6=overrepresented)"
            else:
                feedback_question_output = "Feedback enabled but no suitable word found."

        # ====================== FINAL SAVE ======================
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # ====================== STATUS ======================
        iter_num = len(history)
        total_iterations = sum(len(p.get("history", [])) for p in prompt_histories.values())
        unique_prompts = len(prompt_histories)
        infusion_count = len([v for v in infusion_tokens.values() if v > 0.5])
        trend = "↑" if reward > last_reward else "↓" if reward < last_reward else "→"
        health = "🚀 Strong convergence" if avg_reward_ema > 0.6 else "✅ Learning well" if avg_reward_ema > 0.3 else "⚠️ Still exploring" if avg_reward_ema > -0.2 else "🔄 Heavy correction"
        current_top = self._get_top_tokens(token_importance, tokenizer, 10)
        top_infusion = self._get_top_tokens(infusion_tokens, tokenizer, 8)
        lib_size = len(token_library)
        gen_count = sum(1 for e in token_library.values() if e.get("is_general", False))

        status = (f"Iter {iter_num} | Total iters {total_iterations} | "
                  f"History {len(history)} | Unique prompts {unique_prompts} | "
                  f"Infusion tokens {infusion_count}\n"
                  f"Rating {rating}/10 {trend} | Sim {similarity:.3f} | Reward {reward:+.2f} | "
                  f"EMA {avg_reward_ema:+.2f} | Good ratio {good_ratio:.0%} | Expl {expl:.3f} | {health}\n"
                  f"Current focus: {current_top}\n"
                  f"Top infusion: {top_infusion}\n"
                  f"Library: {lib_size} tokens ({gen_count} general)")

        if is_new_prompt:
            status += "\n✓ New prompt pair → token overlap transfer applied"
        if new_token_mask is not None and (1.0 - new_token_mask).sum() > 0:
            status += "\nNew tokens boosted"
        if infusion_active:
            status += "\n🔀 Token infusion active (delta scaled)"
        if rating <= 4:
            status += "\nLow rating: importance down-weighted + negative strengthened"
        status += latent_info

        modified_positive = [(new_positive, positive_meta)]
        modified_negative = [(new_negative, negative_meta)] if new_negative is not None else negative_conditioning

        return (modified_positive, modified_negative, modified_latent, status, feedback_question_output)
        
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

            # Always append to the end - this is the requested behavior
            if enhanced:
                enhanced += "\n" + content
            else:
                enhanced = content

            source = entry.get("comment") or entry.get("name") or f"uid:{entry.get('uid','?')}" or "unnamed"
            injected.append(f"[{source}] {content}")

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
