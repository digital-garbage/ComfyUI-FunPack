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

# Constants from StoryMem
IMAGE_FACTOR = 28
VIDEO_MIN_PIXELS = 48 * IMAGE_FACTOR * IMAGE_FACTOR  # 37,632
MIN_FRAME_SIMILARITY = 0.9
MAX_KEYFRAME_NUM = 3
ADAPTIVE_ALPHA = 0.01
HPSV3_QUALITY_THRESHOLD = 3.0

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
                
                if not is_constant:
                    if not self._match_keys(entry.get("key", []), scan_text):
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
                context_history="", scan_depth=4):
        
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

class FunPackStoryMemJSONConverter:
    """
    FunPack StoryMem LoRA JSON Converter - supports both manual input and Qwen-generated output.
    
    Features:
    - Lenient parsing: auto-adjusts mismatched lengths for cut/first_frame_prompt
    - Handles 'cut' as single bool (replicates) or list
    - NO forced first cut=True (user can control via prompt or manual input)
    - Safe defaults on invalid data → no crashes
    - Splits full Qwen story into up to 3 separate scene JSONs
    
    When use_qwen_output=False → uses manual per-scene inputs with the same leniency.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_qwen_output": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Use Qwen Output (ignores manual scenes below)",
                    "label_off": "Use Manual Scene Inputs",
                    "tooltip": "Toggle between Qwen-generated full story JSON or classic manual per-scene setup"
                }),
                "qwen_output": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Connect STRING output from Kijai's WanVideoPromptExtender / Qwen enhancer here.\nDrag wire onto the left side of this box to convert to input socket."
                }),
                "story_name": ("STRING", {
                    "default": "",
                    "tooltip": "Story title (used in all output JSONs)"
                }),
                "story_overview": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Overall story summary (shared across all scenes)"
                }),
                # Scene 1
                "scene1_video_prompts": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "One prompt per line (each = 5-sec clip)"
                }),
                "scene1_first_frame_prompts": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "First frame description per prompt (will auto-adjust to match count)"
                }),
                "scene1_cut_values": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated booleans: true,false,true,... (will auto-adjust to match prompt count)"
                }),
                # Scene 2
                "scene2_video_prompts": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "One prompt per line (each = 5-sec clip)"
                }),
                "scene2_first_frame_prompts": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "First frame description per prompt (will auto-adjust to match count)"
                }),
                "scene2_cut_values": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated booleans: true,false,true,... (will auto-adjust to match prompt count)"
                }),
                # Scene 3
                "scene3_video_prompts": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "One prompt per line (each = 5-sec clip)"
                }),
                "scene3_first_frame_prompts": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "First frame description per prompt (will auto-adjust to match count)"
                }),
                "scene3_cut_values": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated booleans: true,false,true,... (will auto-adjust to match prompt count)"
                }),
            },
            "optional": {
                "qwen_mode_notice": ("STRING", {
                    "multiline": True,
                    "default": "!!! QWEN MODE ACTIVE !!!\nAll manual inputs below are completely ignored\nConnect the Qwen JSON string above.",
                    "tooltip": "Reminder - only visible/meaningful when use_qwen_output = True"
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("json_scene_1", "json_scene_2", "json_scene_3")
    OUTPUT_NODE = True
    FUNCTION = "convert"
    CATEGORY = "FunPack"

    def convert(self,
                use_qwen_output,
                qwen_output,
                story_name, story_overview,
                scene1_video_prompts, scene1_first_frame_prompts, scene1_cut_values,
                scene2_video_prompts, scene2_first_frame_prompts, scene2_cut_values,
                scene3_video_prompts, scene3_first_frame_prompts, scene3_cut_values,
                qwen_mode_notice=""):

        if use_qwen_output:
            if not qwen_output.strip():
                raise ValueError("Qwen mode enabled but 'qwen_output' is empty! Connect Kijai's prompt enhancer output.")

            pbar = ProgressBar(2)

            try:
                full_data = json.loads(qwen_output)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse Qwen output as JSON:\n{str(e)}\n\nMake sure system prompt instructs Qwen to return **only** valid JSON.")

            story_name = full_data.get("story_name", "Generated Story")
            story_overview = full_data.get("story_overview", "Generated overview")
            scenes = full_data.get("scenes", [])

            if not isinstance(scenes, list) or not scenes:
                raise ValueError("Qwen JSON must contain non-empty 'scenes' list.")

            outputs = ["", "", ""]

            for i, scene in enumerate(scenes[:3]):
                if not isinstance(scene, dict):
                    continue

                scene_num = scene.get("scene_num", i+1)
                video_prompts = scene.get("video_prompts", [])
                first_frame_prompt = scene.get("first_frame_prompt", [])
                cut = scene.get("cut", [])

                if not video_prompts:
                    continue

                num_prompts = len(video_prompts)

                # Lenient: adjust first_frame_prompt
                if len(first_frame_prompt) < num_prompts:
                    if first_frame_prompt:
                        first_frame_prompt += [first_frame_prompt[-1]] * (num_prompts - len(first_frame_prompt))
                    else:
                        first_frame_prompt = [""] * num_prompts
                elif len(first_frame_prompt) > num_prompts:
                    first_frame_prompt = first_frame_prompt[:num_prompts]

                # Lenient: handle cut (single bool or mismatched list)
                if isinstance(cut, bool):
                    cut = [cut] * num_prompts
                elif not isinstance(cut, list) or not all(isinstance(c, bool) for c in cut):
                    cut = [False] * num_prompts  # Default to no cuts
                else:
                    if len(cut) < num_prompts:
                        cut += [cut[-1] if cut else False] * (num_prompts - len(cut))
                    elif len(cut) > num_prompts:
                        cut = cut[:num_prompts]

                # NO forced first cut=True anymore

                single_scene_data = {
                    "scene_num": scene_num,
                    "video_prompts": video_prompts,
                    "first_frame_prompt": first_frame_prompt,
                    "cut": cut
                }

                json_data = {
                    "story_name": story_name,
                    "story_overview": story_overview,
                    "scenes": [single_scene_data]
                }

                outputs[i] = json.dumps(json_data, indent=2, ensure_ascii=False)

            pbar.update(2)
            return tuple(outputs)

        else:
            # Manual mode - same leniency
            scenes_input = [
                (scene1_video_prompts, scene1_first_frame_prompts, scene1_cut_values, 1),
                (scene2_video_prompts, scene2_first_frame_prompts, scene2_cut_values, 2),
                (scene3_video_prompts, scene3_first_frame_prompts, scene3_cut_values, 3),
            ]

            outputs = ["", "", ""]

            for i, (video_text, first_text, cuts_text, scene_num) in enumerate(scenes_input):
                if not video_text.strip():
                    continue

                video_prompts = [p.strip() for p in video_text.split("\n") if p.strip()]
                if not video_prompts:
                    continue

                num_prompts = len(video_prompts)

                first_prompts = [f.strip() for f in first_text.split("\n") if f.strip()]

                # Lenient adjust first_prompts
                if len(first_prompts) < num_prompts:
                    if first_prompts:
                        first_prompts += [first_prompts[-1]] * (num_prompts - len(first_prompts))
                    else:
                        first_prompts = [""] * num_prompts
                elif len(first_prompts) > num_prompts:
                    first_prompts = first_prompts[:num_prompts]

                # Parse cuts leniently
                if cuts_text.strip():
                    cuts_str = [c.strip().lower() for c in cuts_text.split(",") if c.strip()]
                    cuts = []
                    for c in cuts_str:
                        if c in ("true", "t", "1", "yes"):
                            cuts.append(True)
                        else:
                            cuts.append(False)  # default for invalid or false
                else:
                    cuts = [False] * num_prompts

                # Lenient adjust cuts
                if len(cuts) < num_prompts:
                    cuts += [cuts[-1] if cuts else False] * (num_prompts - len(cuts))
                elif len(cuts) > num_prompts:
                    cuts = cuts[:num_prompts]

                # NO forced first cut=True

                scene = {
                    "scene_num": scene_num,
                    "video_prompts": video_prompts,
                    "first_frame_prompt": first_prompts,
                    "cut": cuts
                }

                full_json = {
                    "story_name": story_name.strip() or "Manual Story",
                    "story_overview": story_overview.strip() or "Manual overview",
                    "scenes": [scene]
                }

                outputs[i] = json.dumps(full_json, indent=2, ensure_ascii=False)

            return tuple(outputs)
            
class FunPackImg2LatentInterpolation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_count": ("INT", {"default": 25, "min": 1, "max": 125, "step": 4}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("img_batch_for_encode", "img_for_start_image")
    FUNCTION = "process"
    CATEGORY = "FunPack"

    def process(self, images, frame_count):
        device = images.device
        total_input_frames = images.shape[0]
        
        # 1. Take the last frame as our starting point
        last_frame = images[-1:]
        
        # 2. Generate smooth transition from this frame
        interpolated = []
        
        # First frame is exact copy of last input frame (for perfect continuity)
        interpolated.append(last_frame.clone())
        
        # Generate remaining frames with increasing denoise
        for i in range(1, frame_count):
            # Calculate denoise strength (0 at start, 1 at end)
            denoise_strength = i / (frame_count - 1)
            
            # Apply denoising
            noise = torch.randn_like(last_frame)
            blended = (1 - denoise_strength) * last_frame + denoise_strength * noise
            interpolated.append(blended)
        
        # Convert to tensor
        output = torch.cat(interpolated, dim=0)
        
        # Preview is first frame (same as input's last frame)
        preview = interpolated[0].clone()
        
        return (output, preview)

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

            output_text = llm_tokenizer.decode(generated_ids[0][llm_tokens.shape[1]:], skip_special_tokens=True)
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
                "temperature": ("FLOAT", {"min": 0.0, "max": 2.0, "step": 0.01, "default": 0.6}),
                "max_new_tokens": ("INT", {"min": 64, "max": 4096, "step": 64, "default": 512}),
                "repetition_penalty": ("FLOAT", {"min": 0.0, "max": 3.0, "step": 0.01, "default": 1.0}),
                "mode": (["Sequences from story", "Sequences from user prompt"],),
                "recommend_loras": ("BOOLEAN", {"default": True, "label": "Enable LoRA suggestion"}),
                "sanity_check": ("BOOLEAN", {"default": True, "label": "Enable Sanity Check"}),
                "sanity_check_system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Analyze the given sequence and perform a correction, if the sequence does not match the given requirements:\n1. The sequence is related to given user's prompt.\n2. The sequence contains only physically possible actions.\n3. The sequence contains information about characters, their appearances, positioning, actions, camera angle, focus and zoom.\n4. The sequence is fully describing the requested action.\n\nOutput ONLY corrected sequence, or return it unchanged if it matches the requirements. No additional text except for sequence is allowed."
                }), 
            }
        }

    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING",)
    RETURN_NAMES = ("prompt1","prompt2","prompt3","prompt4","prompt5",)
    FUNCTION = "write_story"
    CATEGORY = "FunPack"

    def write_story(self, user_prompt, story_system_prompt, sequence_system_prompt, model_path_type, model_path, llm_safetensors_file, prompt_count, top_p, top_k, temperature, max_new_tokens, repetition_penalty, mode, recommend_loras, sanity_check, sanity_check_system_prompt):
        llm_model = None
        llm_tokenizer = None
        llm_model_device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[FunPackPromptEnhancer] Making initial story...")

        try:
            if model_path_type == "HuggingFace Pretrained":
                print(f"[FunPackStoryWriter] Loading LLM from HuggingFace pretrained: {model_path}")
                llm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                llm_model = AutoModelForCausalLM.from_pretrained(model_path, ignore_mismatched_sizes=True, trust_remote_code=True)
            elif model_path_type == "Local Safetensors":
                print(f"[FunPackStoryWriter] Loading LLM from local safetensors file: {llm_safetensors_file}")
                full_safetensors_path = folder_paths.get_full_path('clip', llm_safetensors_file)
                
                llm_tokenizer = AutoTokenizer.from_pretrained("xtuner/llava-llama-3-8b-v1_1-transformers", trust_remote_code=True)
                
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
                        top_p=top_p, top_k=top_k, temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=llm_tokenizer.pad_token_id,
                        eos_token_id=llm_tokenizer.eos_token_id,
                    )

                story = llm_tokenizer.decode(generated_ids[0][llm_tokens.shape[1]:], skip_special_tokens=True).strip()
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
                messages = [
                    {"role": "system", "content": sequence_system_prompt},
                    {"role": "user", "content": f"""Current sequence number: {seq_idx + 1}\nTotal sequences requested: {prompt_count}\nOriginal user prompt: {user_prompt}n\Generate the next sequence now."""},
                    {"role": "assistant", "content": f"""Previous sequences for continuity:{chr(10).join([f"Sequence {i+1}: {text}" for i, text in enumerate(outputs[:seq_idx])]) if seq_idx > 0 else "This is the first sequence."}"""}
                ]

                llm_tokens = llm_tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt", tokenize=True
                ).to(llm_model_device)

                print(f"[FunPackStoryWriter] Generating sequence {seq_idx + 1}/{prompt_count}...")
                with torch.no_grad():
                    generated_ids = llm_model.generate(
                        **llm_tokens,
                        do_sample=True,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=llm_tokenizer.pad_token_id,
                        eos_token_id=llm_tokenizer.eos_token_id,
                    )

                seq_text = llm_tokenizer.decode(generated_ids[0][llm_tokens.shape[1]:], skip_special_tokens=True).strip()
                
                if sanity_check == False:
                    print(f"[FunPackStoryWriter] Sequence {seq_idx + 1} (sanity check skipped): {seq_text}...")

                # Performing sanity check - comparing sequence text to user's prompt according to rules in sanity check system prompt.
                if sanity_check == True:
                    if mode == "Sequences from story":
                        sanity_messages = [
                            {"role": "system", "content": sanity_check_system_prompt},
                            {"role": "user", "content": f"""Original story: {story}
                            Original system prompt (rules that must have been followed to generate the sequence): {story_system_prompt}
                            Original user prompt: {user_prompt}
                             Previous sequence (for continuity check): {outputs[seq_idx-1] if seq_idx > 0 else "This is the first sequence"}
                             Sequence to validate and correct if needed: {seq_text}"""}
                        ]
                    else:
                        sanity_messages = [
                            {"role": "system", "content": sanity_check_system_prompt},
                            {"role": "user", "content": f"""Original user prompt: {user_prompt}
                            Original system prompt (rules that must have been followed to generate the sequence): {sequence_system_prompt}
                             Previous sequence (for continuity check): {outputs[seq_idx-1] if seq_idx > 0 else "This is the first sequence"}
                             Sequence to validate and correct if needed: {seq_text}"""}
                        ]
                    llm_tokens = llm_tokenizer.apply_chat_template(
                        sanity_messages, add_generation_prompt=True, return_tensors="pt", tokenize=True
                        ).to(llm_model_device)

                    print(f"[FunPackStoryWriter] Performing sanity check on sequence {seq_idx + 1}/{prompt_count}...")
                    with torch.no_grad():
                        generated_ids = llm_model.generate(
                            **llm_tokens,
                            do_sample=False,
                            top_p=top_p,
                            top_k=top_k,
                            temperature=temperature,
                            max_new_tokens=1024,
                            repetition_penalty=1.05,
                            pad_token_id=llm_tokenizer.pad_token_id,
                            eos_token_id=llm_tokenizer.eos_token_id,
                        )

                    seq_text = llm_tokenizer.decode(generated_ids[0][llm_tokens.shape[1]:], skip_special_tokens=True).strip()
                    print(f"[FunPackStoryWriter] Sequence {seq_idx + 1} (sanity check performed): {seq_text}...")
                
                outputs[seq_idx] = seq_text

                # Append generated sequence — this is what chains everything
                messages.append({"role": "assistant", "content": seq_text})

            if recommend_loras == True:
                lora_list = folder_paths.get_filename_list('loras')
                lora_messages = [
                        {"role": "system", "content": "Analyze the given array of sequences and the list of LoRAs (low-ranking adaptations - additions to video generation models that enhance certain action, concept or character) and decide which LoRAs out of user's list are recommended to use for each sequence, based on the content of those sequences and the names of LoRAs. Output only list of recommended LoRAs in the format:\nSequence 1: lora name, lora name\nSequence 2: lora name, lora name\nSequence N: lora name, lora name"},
                        {"role": "user", "content": lora_list}
                    ]
                for seq_idx, seq_output in enumerate(outputs):
                    seq_output = outputs[seq_idx]
                    lora_messages.append({"role": "user", "content": f"This is sequence ID {seq_idx + 1}: {seq_output}"})

                # Generate recommended LoRAs
                llm_tokens = llm_tokenizer.apply_chat_template(
                      lora_messages, add_generation_prompt=True, return_tensors="pt", tokenize=True
                ).to(llm_model_device)

                print("[FunPackStoryWriter] Generating LoRA recommendations...")
                with torch.no_grad():
                    generated_ids = llm_model.generate(
                        **llm_tokens,
                        do_sample=True,
                        top_p=top_p, top_k=top_k, temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=repetition_penalty,
                        pad_token_id=llm_tokenizer.pad_token_id,
                        eos_token_id=llm_tokenizer.eos_token_id,
                    )

                recommended_loras = llm_tokenizer.decode(generated_ids[0][llm_tokens.shape[1]:], skip_special_tokens=True).strip()
                print(f"[FunPackStoryWriter] LLM recommended to use this list of LoRAs: {recommended_loras}")
            
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

import os
import json

class FunPackCreativeTemplate:
    @classmethod
    def INPUT_TYPES(s):
        # Path to templates.json in the same directory as this script
        template_file = os.path.join(os.path.dirname(__file__), "templates.json")
        templates = {}
        if os.path.exists(template_file):
            with open(template_file, 'r') as f:
                try:
                    templates = json.load(f)
                except json.JSONDecodeError:
                    templates = {}
        template_names = sorted(list(templates.keys())) + ["Custom"]
        default_template = template_names[0] if template_names else "Custom"

        return {
            "required": {
                "template_name": (template_names, {"default": default_template}),
                "template_text": ("STRING", {"multiline": True, "default": ""}),
                "replacements": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "save_name": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "replace"
    CATEGORY = "FunPack/Text"
    DESCRIPTION = "Select or create text templates, replace placeholders, and output the result."

    def replace(self, template_name, template_text, replacements, save_name=""):
        # Load templates
        template_file = os.path.join(os.path.dirname(__file__), "templates.json")
        templates = {}
        if os.path.exists(template_file):
            with open(template_file, 'r') as f:
                try:
                    templates = json.load(f)
                except json.JSONDecodeError:
                    templates = {}

        # Get the template string
        if template_name == "Custom":
            if not template_text.strip():
                return ("",)  # Empty if no custom text provided
            template_str = template_text
        else:
            template_str = templates.get(template_name, "")

        # Parse replacements (format: KEY: value per line)
        replace_dict = {}
        for line in replacements.splitlines():
            if ':' in line:
                key, val = line.split(':', 1)
                replace_dict[key.strip()] = val.strip()

        # Perform replacements
        for key, val in replace_dict.items():
            template_str = template_str.replace(f"[{key}]", val)

        # Save new template if save_name provided and using Custom
        if save_name and template_name == "Custom" and template_text.strip():
            templates[save_name] = template_text
            # Create directories if needed (though unlikely)
            os.makedirs(os.path.dirname(template_file), exist_ok=True)
            with open(template_file, 'w') as f:
                json.dump(templates, f, indent=4)

        return (template_str,)


# Update NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "FunPackStoryMemJSONConverter": FunPackStoryMemJSONConverter,
    "FunPackStoryMemKeyframeExtractor": FunPackStoryMemKeyframeExtractor,
    "FunPackStoryMemLastFrameExtractor": FunPackStoryMemLastFrameExtractor,
    "FunPackImg2LatentInterpolation": FunPackImg2LatentInterpolation,
    "FunPackPromptEnhancer": FunPackPromptEnhancer,
    "FunPackStoryWriter": FunPackStoryWriter,
    "FunPackVideoStitch": FunPackVideoStitch,
    "FunPackContinueVideo": FunPackContinueVideo,
    "FunPackCreativeTemplate": FunPackCreativeTemplate,
    "FunPackLorebookEnhancer": FunPackLorebookEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FunPackStoryMemJSONConverter": "FunPack StoryMem JSON Converter",
    "FunPackStoryMemKeyframeExtractor": "FunPack StoryMem Keyframe Extractor",
    "FunPackStoryMemLastFrameExtractor": "FunPack StoryMem Last Frame Extractor",
    "FunPackImg2LatentInterpolation": "FunPack Img2Latent Interpolation",
    "FunPackPromptEnhancer": "FunPack Prompt Enhancer (Standalone)",
    "FunPackStoryWriter": "FunPack Story Writer",
    "FunPackVideoStitch": "FunPack Video Stitch",
    "FunPackContinueVideo": "FunPack Continue Video",
    "FunPackCreativeTemplate": "FunPack Creative Template",
    "FunPackLorebookEnhancer": "FunPack Lorebook Enhancer"
}

























































