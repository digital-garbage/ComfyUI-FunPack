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
                from transformers import AutoTokenizer
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
                "conditioning": ("CONDITIONING",),
                "rating": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "slider",
                    "label": "Rating (1=absolutely horrific, 10=masterpiece)"
                }),
                "refinement_key": ("STRING", {"default": "my_style_v1", "multiline": False}),
            },
            "optional": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Positive prompt (helps detect new tokens & prompt drift)"
                }),
                "latent": ("LATENT",),
                "reset_session": ("BOOLEAN", {"default": False, "label": "Reset Session (clears all history)"}),
                "unlimited_history": ("BOOLEAN", {
                    "default": False,
                    "label": "Unlimited History (for multi-prompt / NSFW training — never prunes)"
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("modified_conditioning", "modified_latent", "status")
    FUNCTION = "refine"
    CATEGORY = "FunPack/Refinement"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def _get_top_tokens(self, token_importance, tokenizer, top_k=10):
        if not token_importance or not tokenizer:
            return "N/A"
        sorted_tokens = sorted(token_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_list = []
        for tid_str, score in sorted_tokens:
            try:
                token_text = tokenizer.decode([int(tid_str)], skip_special_tokens=True).strip()
                if token_text:
                    top_list.append(f"{token_text}({score:.2f})")
            except:
                top_list.append(f"ID{tid_str}({score:.2f})")
        return ", ".join(top_list) if top_list else "None"

    def refine(self, conditioning, rating: int, refinement_key: str, prompt: str = "", latent=None,
               reset_session: bool = False, unlimited_history: bool = False):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        refinements_dir = os.path.join(base_dir, "refinements")
        os.makedirs(refinements_dir, exist_ok=True)
        safe_key = md5(refinement_key.encode("utf-8")).hexdigest()
        json_file = os.path.join(refinements_dir, f"refine_{safe_key}.json")

        if not conditioning or not isinstance(conditioning, list) or len(conditioning) == 0:
            return (conditioning, latent, "ERROR: Empty CONDITIONING input")

        item = conditioning[0]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            raw_embeds = item[0]
            meta_dict = item[1] if isinstance(item[1], dict) else {"pooled_output": None}
        else:
            raw_embeds = item if isinstance(item, torch.Tensor) else None
            meta_dict = {"pooled_output": None}

        if not isinstance(raw_embeds, torch.Tensor):
            return (conditioning, latent, "ERROR: No embedding tensor found")

        # First run or reset
        if reset_session or not os.path.exists(json_file):
            data = {
                "refinement_key": refinement_key,
                "reference_embeds": tensor_to_serializable(raw_embeds),
                "history": [],
                "adaptive": {
                    "exploration_base": 0.08,
                    "momentum": None,
                    "avg_reward_ema": 0.0,
                    "good_ratio": 0.0,
                    "dynamic_sim_threshold": 0.82,
                    "token_importance": {}
                },
                "last_rating": rating,
                "last_prompt": prompt
            }
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return (conditioning, latent, "✓ New session started – Reference saved")

        # Load session
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        reference = serializable_to_tensor(data["reference_embeds"])
        device = reference.device
        cur_embeds = raw_embeds.to(device) if raw_embeds.device != device else raw_embeds

        ref_mean = reference.mean(dim=1)
        cur_mean = cur_embeds.mean(dim=1)
        similarity = torch.nn.functional.cosine_similarity(ref_mean, cur_mean, dim=1).item()

        reward = (rating - 5.5) / 4.5
        last_rating = data.get("last_rating", 5)
        last_reward = (last_rating - 5.5) / 4.5

        adaptive = data["adaptive"]
        expl = adaptive["exploration_base"]
        momentum = adaptive.get("momentum")
        avg_reward_ema = adaptive["avg_reward_ema"]
        good_ratio = adaptive["good_ratio"]
        sim_threshold = adaptive["dynamic_sim_threshold"]
        token_importance = adaptive.get("token_importance", {})

        avg_reward_ema = 0.85 * avg_reward_ema + 0.15 * reward

        if momentum is None or not isinstance(momentum, dict):
            momentum = torch.zeros_like(reference)
        else:
            momentum = serializable_to_tensor(momentum)

        # === SAFE Token handling ===
        tokenizer = self._get_tokenizer()
        new_token_mask = None
        cur_token_ids = []

        if tokenizer and prompt:
            try:
                cur_token_ids = tokenizer.encode(prompt, add_special_tokens=True)
                prev_token_ids = tokenizer.encode(data.get("last_prompt", ""), add_special_tokens=True)

                # Get actual embedding sequence length safely
                if cur_embeds.dim() == 3:           # [batch, seq, hidden]
                    embed_seq_len = cur_embeds.shape[1]
                elif cur_embeds.dim() == 2:         # [seq, hidden]
                    embed_seq_len = cur_embeds.shape[0]
                else:
                    embed_seq_len = 1

                new_token_mask = torch.ones(embed_seq_len, device=device)

                compare_len = min(len(cur_token_ids), len(prev_token_ids), embed_seq_len)

                for i in range(compare_len):
                    if cur_token_ids[i] != prev_token_ids[i]:
                        new_token_mask[i] = 0.0

                # Update per-token importance safely
                for tid in cur_token_ids[:embed_seq_len]:
                    tid_str = str(tid)
                    if tid_str not in token_importance:
                        token_importance[tid_str] = 1.0

                    is_new = tid not in prev_token_ids
                    update_strength = 1.8 if is_new else 1.0
                    token_importance[tid_str] = max(0.3, min(2.5,
                        token_importance[tid_str] + reward * 0.12 * update_strength))

            except Exception as e:
                print(f"[FunPackGemmaEmbeddingRefiner] Token comparison failed: {e}")
                new_token_mask = None

        # === Delta computation ===
        history = data.get("history", [])
        is_close = similarity >= sim_threshold

        if is_close and history:
            last_entry = history[-1]
            prev_modified = serializable_to_tensor(
                last_entry.get("modified_embeds", data["reference_embeds"])
            )
            prev_delta = prev_modified - reference
            multiplier = max(0.05, 1.0 + reward * 1.45)
            if rating == last_rating and rating >= 8:
                multiplier += 0.35
            noise = torch.randn_like(prev_delta) * (expl * (1.0 - avg_reward_ema * 0.7))
            new_delta = (prev_delta * multiplier) + noise + (momentum * 0.6)
        else:
            good_deltas = [serializable_to_tensor(entry["modified_embeds"]) - reference 
                          for entry in history if entry.get("rating", 0) >= 7]
            if good_deltas:
                new_delta = torch.stack(good_deltas).mean(dim=0) * (0.7 + reward * 0.4)
            else:
                new_delta = torch.randn_like(reference) * expl * 0.45

        # Apply token importance — always match current embedding length
        if token_importance and cur_embeds.dim() > 1:
            if cur_embeds.dim() == 3:
                seq_len = cur_embeds.shape[1]
            else:
                seq_len = cur_embeds.shape[0]

            importance_tensor = torch.ones(seq_len, device=device)
            for i, tid in enumerate(cur_token_ids[:seq_len]):
                tid_str = str(tid)
                if tid_str in token_importance:
                    importance_tensor[i] = token_importance[tid_str]

            new_delta = new_delta * importance_tensor.unsqueeze(-1)

        # Apply new token boost safely
        if new_token_mask is not None:
            boost_factor = 1.0 + 1.8 * (1.0 - new_token_mask).unsqueeze(-1)
            new_delta = new_delta * boost_factor

        # Apply & normalize
        new_modified = reference + new_delta
        new_modified = torch.clamp(new_modified, min=-60.0, max=60.0)
        norm_factor = reference.norm(dim=-1, keepdim=True) + 1e-8
        new_modified = new_modified / (new_modified.norm(dim=-1, keepdim=True) + 1e-8) * norm_factor

        # Latent refinement
        modified_latent = latent
        latent_info = ""
        if latent is not None and "samples" in latent:
            samples = latent["samples"].to(device) if latent["samples"].device != device else latent["samples"]
            good_latents = [serializable_to_tensor(entry["latent_samples"]) for entry in history if entry.get("rating", 5) >= 8]
            bad_latents = [serializable_to_tensor(entry["latent_samples"]) for entry in history if entry.get("rating", 5) <= 3]

            latent_strength_auto = max(0.0, avg_reward_ema * 0.35)
            latent_delta = torch.zeros_like(samples)
            if good_latents:
                latent_delta += (torch.stack(good_latents).mean(dim=0) - samples) * latent_strength_auto
            if bad_latents:
                latent_delta -= (torch.stack(bad_latents).mean(dim=0) - samples) * (latent_strength_auto * 0.7)

            if good_latents or bad_latents:
                new_samples = torch.clamp(samples + latent_delta, min=-12.0, max=12.0)
                modified_latent = {"samples": new_samples.cpu(), "noise_mask": latent.get("noise_mask")}
                latent_info = f" | Latent ±{len(good_latents)}g/{len(bad_latents)}b"

        # Update adaptive state
        momentum = 0.75 * momentum + 0.25 * (new_delta * reward)
        if avg_reward_ema > 0.3:
            sim_threshold = max(0.75, sim_threshold - 0.002)

        if rating >= 8:
            expl = max(0.015, expl * 0.96)
            good_ratio = 0.9 * good_ratio + 0.1 * 1.0
        else:
            expl = min(0.12, expl * 1.08)
            good_ratio = 0.9 * good_ratio + 0.1 * 0.0

        adaptive.update({
            "exploration_base": expl,
            "momentum": tensor_to_serializable(momentum),
            "avg_reward_ema": avg_reward_ema,
            "good_ratio": good_ratio,
            "dynamic_sim_threshold": sim_threshold,
            "token_importance": token_importance
        })

        # History
        history_entry = {
            "iteration": len(history) + 1,
            "rating": rating,
            "reward": round(reward, 3),
            "modified_embeds": tensor_to_serializable(new_modified),
            "similarity": round(similarity, 4),
            "prompt": prompt[:180]
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

        data["history"] = history
        data["last_rating"] = rating
        data["last_prompt"] = prompt
        data["adaptive"] = adaptive

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # === Rich Status ===
        iter_num = len(history)
        trend = "↑" if reward > last_reward else "↓" if reward < last_reward else "→"
        health = "🚀 Strong convergence" if avg_reward_ema > 0.6 else \
                 "✅ Learning well" if avg_reward_ema > 0.3 else \
                 "⚠️ Still exploring" if avg_reward_ema > -0.2 else "🔄 Heavy correction"

        current_top = self._get_top_tokens(token_importance, tokenizer, 10)

        status = (
            f"Iter {iter_num} | Rating {rating}/10 {trend} | "
            f"Sim {similarity:.3f} | Reward {reward:+.2f} | EMA {avg_reward_ema:+.2f} | "
            f"Good ratio {good_ratio:.0%} | Expl {expl:.3f} | {health}\n"
            f"Current focus: {current_top}"
        )
        if new_token_mask is not None and (1.0 - new_token_mask).sum() > 0:
            status += "\nNew tokens boosted"
        if latent_info:
            status += latent_info
        if unlimited_history:
            status += " | Unlimited history ON"

        modified_conditioning = [(new_modified, meta_dict)]
        return (modified_conditioning, modified_latent, status)
        
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
