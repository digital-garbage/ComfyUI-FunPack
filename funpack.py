"""
olivv's FunPack v1.1.1

Changelog:

v1.1.1 - Added DualCLIP Instruct loader node for experimenting with Instruct models in FramePack.
v1.1.0 - Changed interpolation logic.

Custom ComfyUI node created specifically to imitate FramePack's approach, but using WAN2.1 models.
It takes the last frame of a video, then multiplies it many times,
gradually increasing denoising to the end, somewhat similar to what FramePack does,
but without frame prediction.

"""
import os
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from safetensors.torch import load_file
import comfy.model_management as mm
import comfy.sd as sd
import folder_paths
import nodes

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
    CATEGORY = "WANPack/Video"

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

# Helper to list files in models/clip
def list_clip_files():
    model_dir = os.path.join(mm.model_path, "clip")
    return sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))

class FunPackDualCLIPLoaderInstruct:
    @classmethod
    def INPUT_TYPES(s):
        base = nodes.DualCLIPLoader.INPUT_TYPES()
        return {
            'required': 
                {
                    'clip_model_name': (s.get_filename_list(),),
                    'llama_instruct_model_name': (s.get_filename_list(),),
                    'llama3_model_name':(s.get_filename_list(),),
                    'type': base['required']['type'],
                    'pretrained_path': ("STRING", {"multiline": False, "default": "HiTZ/Latxa-Llama-3.1-8B-Instruct"}),
                    'system_prompt': ("STRING", {
                        "multiline": True,
                        "default": "You are a creative assistant optimized for generating vivid, detailed descriptions for video generation."
                    })
                }
            }
    RETURN_TYPES = 'CLIP',
    RETURN_NAMES = 'clip',
    FUNCTION = "load"
    CATEGORY = "conditioning"
    
    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list('clip')
        return sorted(files)
    
    def load(self, clip_model_name, llama_instruct_model_name, llama3_model_name, type, system_prompt, pretrained_path):
        # Load CLIP model using ComfyUI
        clip_path = folder_paths.get_full_path('clip', clip_model_name)
        def get_clip_type(type):
            clip_type = getattr(sd.CLIPType, type.upper(), sd.CLIPType.HUNYUAN_VIDEO)
            print("Detected clip type:", clip_type)
            return clip_type
        
        # Load LLaMA-3.1-Instruct from weights
        llama_path = folder_paths.get_full_path('clip', llama_instruct_model_name)
        config_source = pretrained_path

        # Load regular LLaMA3 for compatibility
        llama3_path = folder_paths.get_full_path('clip', llama3_model_name)

        # Load tokenizer and config
        tokenizer = AutoTokenizer.from_pretrained(config_source)
        config = LlamaConfig.from_pretrained(config_source)
        
        model = LlamaForCausalLM(config)
        
        try:
            print("Loading Instruct LLaMA from the path:", llama_path)
            state_dict = load_file(llama_path, device="cuda")
            model.load_state_dict(state_dict, strict=False)
            model.eval().to(torch.float16).requires_grad_(False)
            print("Llama Instruct model loaded successfully!")
        except Exception as e:
            print(f"Error loading LLaMA Instruct model: {e}")
            raise

        # Wrap it like a CLIP-compatible text encoder
        class InstructWrapper:
            def __init__(self):
                print("InstructWrapper initialized!")
                self.system_prompt = system_prompt
                print("System prompt is set to:", self.system_prompt)
            def tokenize(self, text):
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ]
                return tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt").to("cuda")

            def encode_from_tokens(self, tokens, return_pooled=True):
                with torch.no_grad():
                    output = model(input_ids=tokens, output_hidden_states=True)
                    hidden = output.hidden_states[-1]  # final layer
                    pooled = hidden[:, -1, :]  # use last token for pooled
                return hidden, pooled

        # Replace text encoder in CLIP model
        clip_model = sd.load_clip(ckpt_paths=[clip_path, llama3_path], embedding_directory=None, clip_type=get_clip_type(type), model_options={})
        clip_model.text = InstructWrapper()
        print("Current TE:", clip_model.text)  # Check if encoder is replaced 
        return (clip_model,)


NODE_CLASS_MAPPINGS = {
    "FunPackImg2LatentInterpolation": FunPackImg2LatentInterpolation,
    "FunPackDualCLIPLoaderInstruct": FunPackDualCLIPLoaderInstruct,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FunPackImg2LatentInterpolation": "FunPack img2latent Interpolation",
    "FunPackDualCLIPLoaderInstruct": "FunPack DualCLIP Instruct Loader"
}