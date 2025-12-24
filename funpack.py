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

# Constants from StoryMem
IMAGE_FACTOR = 28
VIDEO_MIN_PIXELS = 48 * IMAGE_FACTOR * IMAGE_FACTOR  # 37,632
MIN_FRAME_SIMILARITY = 0.9
MAX_KEYFRAME_NUM = 3
ADAPTIVE_ALPHA = 0.01
HPSV3_QUALITY_THRESHOLD = 3.0

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
        z1 = clip_vision.encode_image(x1)
        z2 = clip_vision.encode_image(x2)
    
        # Extract tensor if wrapped in ComfyUI's CLIP_VISION_OUTPUT (single-element tuple)
        if isinstance(z1, (tuple, list)):
            if len(z1) == 1:
                z1 = z1[0]
            else:
                raise ValueError(f"Unexpected CLIP vision output structure: tuple/list of length {len(z1)}")
        if isinstance(z2, (tuple, list)):
            if len(z2) == 1:
                z2 = z2[0]
            else:
                raise ValueError(f"Unexpected CLIP vision output structure: tuple/list of length {len(z2)}")
    
        # Ensure we have actual tensors
        if not isinstance(z1, torch.Tensor) or not isinstance(z2, torch.Tensor):
            raise TypeError("CLIP vision encoding did not return a tensor. "
                            "Ensure 'clip_vision' input is connected to CLIP Vision Loader (not Encode node).")
    
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
    "FunPackStoryMemKeyframeExtractor": FunPackStoryMemKeyframeExtractor,
    "FunPackStoryMemLastFrameExtractor": FunPackStoryMemLastFrameExtractor,
    "FunPackImg2LatentInterpolation": FunPackImg2LatentInterpolation,
    "FunPackPromptEnhancer": FunPackPromptEnhancer,
    "FunPackVideoStitch": FunPackVideoStitch,
    "FunPackContinueVideo": FunPackContinueVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FunPackStoryMemKeyframeExtractor": "FunPack StoryMem Keyframe Extractor",
    "FunPackStoryMemLastFrameExtractor": "FunPack StoryMem Last Frame Extractor",
    "FunPackImg2LatentInterpolation": "FunPack Img2Latent Interpolation",
    "FunPackPromptEnhancer": "FunPack Prompt Enhancer (Standalone)",
    "FunPackVideoStitch": "FunPack Video Stitch",
    "FunPackContinueVideo": "FunPack Continue Video"

}



