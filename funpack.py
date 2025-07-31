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

# Removed the problematic list_clip_files() helper function

class FunPackCLIPLoader:
    class _CLIPAdapter:
        def __init__(self, clip_l_model, llama_te_model, optional_secondary_clip=None, optional_merge_mode="None"):
            self.clip_l_model = clip_l_model
            self.llama_te_model = llama_te_model
            self.optional_secondary_clip = optional_secondary_clip
            self.optional_merge_mode = optional_merge_mode
            
            print(f"[_CLIPAdapter] Initialized with CLIP-L and LLaMA text encoder.")
            if self.optional_secondary_clip:
                print(f"[_CLIPAdapter] Optional secondary CLIP also loaded. Merge mode: {self.optional_merge_mode}")
            else:
                print("[_CLIPAdapter] No optional secondary CLIP provided.")

        def tokenize(self, prompt):
            # Use the CLIP-L model's tokenizer
            return self.clip_l_model.tokenize(prompt)

        def encode_from_tokens(self, tokens, return_pooled=True):
            # 1. Get embeddings from the primary CLIP-L model
            clip_l_cond, clip_l_pooled = self.clip_l_model.encode_from_tokens(tokens, return_pooled=True)

            # 2. Get embeddings from the LLaMA text encoder
            llama_cond, llama_pooled = self.llama_te_model.encode_from_tokens(tokens, return_pooled=True)

            # 3. Handle optional secondary CLIP merge specifically with CLIP-L's output
            final_clip_l_cond = clip_l_cond
            final_clip_l_pooled = clip_l_pooled

            if self.optional_secondary_clip and self.optional_merge_mode != "None":
                print(f"[_CLIPAdapter] Merging optional secondary CLIP embeddings (Mode: {self.optional_merge_mode}).")
                secondary_cond, secondary_pooled = self.optional_secondary_clip.encode_from_tokens(tokens, return_pooled=True)

                if self.optional_merge_mode == "Concatenate Pooled with CLIP-L":
                    final_clip_l_pooled = torch.cat([clip_l_pooled, secondary_pooled], dim=-1)
                    print(f"[_CLIPAdapter] Optional CLIP pooled embeddings concatenated with CLIP-L pooled. New CLIP-L pooled dim: {final_clip_l_pooled.shape[-1]}")
                elif self.optional_merge_mode == "Concatenate Unpooled with CLIP-L":
                    if clip_l_cond.shape[1] != secondary_cond.shape[1]:
                        print("[_CLIPAdapter] WARNING: Unpooled embeddings have different sequence lengths for CLIP-L and optional secondary CLIP. Skipping unpooled concatenation.")
                    else:
                        final_clip_l_cond = torch.cat([clip_l_cond, secondary_cond], dim=-1)
                        print(f"[_CLIPAdapter] Optional CLIP unpooled embeddings concatenated with CLIP-L unpooled. New CLIP-L unpooled dim: {final_clip_l_cond.shape[-1]}")
            
            # 4. Final combination: Concatenate the (potentially merged) CLIP-L output with LLaMA output
            final_cond = torch.cat([final_clip_l_cond, llama_cond], dim=-1)
            final_pooled = torch.cat([final_clip_l_pooled, llama_pooled], dim=-1)

            if return_pooled:
                return final_cond, final_pooled
            else:
                return final_cond

        def encode_token_weights(self, tokens):
            return self.clip_l_model.encode_token_weights(tokens)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_l_model_name": (s.get_filename_list(), {"default": "clip_l.safetensors"}),
                "llama_text_encoder_name": (s.get_filename_list(), {"default": "clip_g.safetensors"}),
                "clip_type": (["HUNYUAN_VIDEO"], {"default": "HUNYUAN_VIDEO"}),
            },
            "optional": {
                "optional_secondary_clip_model_name": (s.get_filename_list(), {"default": "None"}),
                "optional_secondary_clip_type": (["CLIP_G", "CLIP_L", "T5_XXL"], {"default": "CLIP_G"}),
                "optional_merge_mode": (["None", "Concatenate Pooled with CLIP-L", "Concatenate Unpooled with CLIP-L"], {"default": "None"})
            }
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load"
    CATEGORY = "FunPack"

    @classmethod
    def get_filename_list(s):
        return folder_paths.get_filename_list("clip")

    def load(self, clip_l_model_name, llama_text_encoder_name, clip_type,
             optional_secondary_clip_model_name="None", optional_secondary_clip_type="CLIP_G", optional_merge_mode="None"):
        
        # Load CLIP-L model with HUNYUAN_VIDEO type
        clip_l_path = folder_paths.get_full_path('clip', clip_l_model_name)
        loaded_clip_l = sd.load_clip(
            ckpt_paths=[clip_l_path],
            clip_type=sd.CLIPType.HUNYUAN_VIDEO,
            embedding_directory=None,
            model_options={"ignore_mismatched_sizes": True}
        )
        print(f"[FunPackCLIPLoader] Loaded CLIP-L model: {clip_l_model_name}")

        # Load LLaMA text encoder (using CLIP_G as it's part of the Hunyuan architecture)
        llama_te_path = folder_paths.get_full_path('clip', llama_text_encoder_name)
        loaded_llama_te = sd.load_clip(
            ckpt_paths=[llama_te_path],
            clip_type=sd.CLIPType.HUNYUAN_VIDEO,
            embedding_directory=None,
            model_options={"ignore_mismatched_sizes": True}
        )
        print(f"[FunPackCLIPLoader] Loaded LLaMA text encoder: {llama_text_encoder_name}")

        # Load optional secondary CLIP
        loaded_optional_secondary_clip = None
        if optional_secondary_clip_model_name != "None":
            optional_secondary_clip_path = folder_paths.get_full_path('clip', optional_secondary_clip_model_name)
            
            # Map the input type to CLIPType enum
            clip_type_mapping = {
                "CLIP_G": sd.CLIPType.STABLE_CASCADE,
                "CLIP_L": sd.CLIPType.STABLE_DIFFUSION,
                "T5_XXL": sd.CLIPType.SD3
            }
            secondary_clip_type = clip_type_mapping.get(optional_secondary_clip_type, sd.CLIPType.STABLE_DIFFUSION)
            
            loaded_optional_secondary_clip = sd.load_clip(
                ckpt_paths=[optional_secondary_clip_path],
                clip_type=secondary_clip_type,
                embedding_directory=None,
                model_options={"ignore_mismatched_sizes": True}
            )
            print(f"[FunPackCLIPLoader] Loaded optional secondary CLIP model: {optional_secondary_clip_model_name}")
        else:
            print("[FunPackCLIPLoader] No optional secondary CLIP model specified.")

        # Create the adapter
        clip_adapter = FunPackCLIPLoader._CLIPAdapter(
            clip_l_model=loaded_clip_l,
            llama_te_model=loaded_llama_te,
            optional_secondary_clip=loaded_optional_secondary_clip,
            optional_merge_mode=optional_merge_mode
        )
        
        return (clip_adapter,)

# The standalone FunPackPromptEnhancer remains completely unchanged as requested
class FunPackPromptEnhancer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "user_prompt": ("STRING", {"multiline": True, "default": "A photo of a [subject] in a [setting]. [action]."}),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "<|start_header_id|>system<|end_header_id|>\n\nYou are a creative AI assistant tasked with describing videos.\n\nDescribe the video by detailing the following aspects:\n1. The main content and theme of the video.\n2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.\n3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.\n4. background environment, light, style and atmosphere.\n5. camera angles, movements, and transitions used in the video:<|eot_id|>"
                }),
                "model_path_type": (["Local Safetensors", "HuggingFace Pretrained"],),
                "model_path": ("STRING", {"multiline": False, "default": "mlabonne/NeuralLlama-3-8B-Instruct-abliterated"}),
                "llm_safetensors_file": (folder_paths.get_filename_list('clip'),), 
                "top_p": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.05, "default": 0.75}),
                "top_k": ("INT", {"min": 0, "max": 1000, "step": 1, "default": 40}),
                "temperature": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.6}),
                "max_new_tokens": ("INT", {"min": 1, "max": 2048, "step": 64, "default": 1024}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "enhance_prompt"
    CATEGORY = "FunPack"

    def enhance_prompt(self, user_prompt, system_prompt, model_path_type, model_path, llm_safetensors_file, top_p, top_k, temperature, max_new_tokens):
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

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            llm_tokens = llm_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True 
            ).to(llm_model_device)

            print("[FunPackPromptEnhancer] Generating enhanced prompt...")
            with torch.no_grad():
                generated_ids = llm_model.generate(
                    input_ids=llm_tokens,
                    do_sample=True,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
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


# Update NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "FunPackImg2LatentInterpolation": FunPackImg2LatentInterpolation,
    "FunPackCLIPLoader": FunPackCLIPLoader,
    "FunPackPromptEnhancer": FunPackPromptEnhancer,
    "FunPackVideoStitch": FunPackVideoStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FunPackImg2LatentInterpolation": "FunPack Img2Latent Interpolation",
    "FunPackCLIPLoader": "FunPack CLIP Loader",
    "FunPackPromptEnhancer": "FunPack Prompt Enhancer (Standalone)",
    "FunPackVideoStitch": "FunPack Video Stitch"
}