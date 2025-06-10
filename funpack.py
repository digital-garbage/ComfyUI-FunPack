"""
olivv's FunPack v1.2

Changelog:

v1.2 - Major changes. Custom CLIP loader for whatever your needs are.
v1.1.2 - Added "instruct_from_pretrained"
v1.1.1 - Added DualCLIP Instruct loader node for experimenting with Instruct models in FramePack.
v1.1.0 - Changed interpolation logic.

Custom ComfyUI node created specifically to imitate FramePack's approach, but using WAN2.1 models.
It takes the last frame of a video, then multiplies it many times,
gradually increasing denoising to the end, somewhat similar to what FramePack does,
but without frame prediction.

"""
import os
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file, safe_open
import glob
import comfy.model_management as mm
import comfy.sd as sd
import folder_paths
import nodes
import tempfile

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

class FunPackCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        base = nodes.DualCLIPLoader.INPUT_TYPES()
        return {
            'required': {
                'clip_model_name': (s.get_filename_list(),),
                'text_encoder_model_name': (s.get_filename_list(),),
                'llm_vision_model_name': (s.get_filename_list(),),
                'type': base['required']['type'],
                'encoder_pretrained_path': ("STRING", {"multiline": False, "default": "mlabonne/NeuralLlama-3-8B-Instruct-abliterated"}),
                'vision_pretrained_path': ("STRING", {"multiline": False, "default": "huihui-ai/Llama-3.2-11B-Vision-Instruct-abliterated"}),
                'encoder_from_pretrained': ("BOOLEAN", {"default": False}),
                'vision_from_pretrained': ("BOOLEAN", {"default": False}),
                'vision_from_pretrained_comfy': ("BOOLEAN", {"default": False}),
                'load_te': ("BOOLEAN", {"default": True}),
                'patch_vision': ("BOOLEAN", {"default": False}),
                'use_custom_loader': ("BOOLEAN", {"default": False, "tooltip": "Bypass sd.load_clip and use internal loader logic."}),
                'system_prompt': ("STRING", {
                    "multiline": True,
                    "default": "<image>You are an expert visual describer for AI video generation..."
                }),
            }
        }

    RETURN_TYPES = 'CLIP',
    RETURN_NAMES = 'clip',
    FUNCTION = "load"
    CATEGORY = "conditioning"

    @classmethod
    def get_filename_list(s):
        return sorted(folder_paths.get_filename_list('clip'))

    def load(self, clip_model_name, type, text_encoder_model_name, llm_vision_model_name,
             encoder_pretrained_path, vision_pretrained_path, system_prompt,
             encoder_from_pretrained, vision_from_pretrained, vision_from_pretrained_comfy,
             load_te, patch_vision, use_custom_loader):

        clip_path = folder_paths.get_full_path('clip', clip_model_name)
        encoder_path = folder_paths.get_full_path('clip', text_encoder_model_name)
        vision_path = folder_paths.get_full_path('clip', llm_vision_model_name)

        def get_clip_type(type):
            clip_type = getattr(sd.CLIPType, type.upper(), sd.CLIPType.HUNYUAN_VIDEO)
            print("Detected clip type:", clip_type)
            return clip_type

        if use_custom_loader:
            return self.custom_clip_loader(
                clip_path, encoder_path, vision_path, encoder_pretrained_path,
                vision_pretrained_path, system_prompt, encoder_from_pretrained,
                vision_from_pretrained, vision_from_pretrained_comfy, load_te, patch_vision
            )
        else:
            print("Using Comfy's sd.load_clip...")
            clip_model = sd.load_clip(
                ckpt_paths=[clip_path, vision_path],
                embedding_directory=None,
                clip_type=get_clip_type(type),
                model_options={"ignore_mismatched_sizes": True}
            )
            return (clip_model,)

    def custom_clip_loader(self, clip_path, encoder_path, vision_path, encoder_pretrained_path,
                           vision_pretrained_path, system_prompt, encoder_from_pretrained,
                           vision_from_pretrained, vision_from_pretrained_comfy, load_te, patch_vision):

        print("Using custom CLIP loader pipeline...")

        # Load or patch vision model
        if not vision_from_pretrained_comfy:
            pretrained_vision_local_path = snapshot_download(repo_id=vision_pretrained_path)
        else:
            pretrained_vision_local_path = folder_paths.models_dir + "/clip/" + vision_pretrained_path

        pvlp_model = os.path.join(pretrained_vision_local_path, "model.safetensors")

        def model_patch(path):
            def load_safetensors_dict(path):
                print("Loading original vision model to patch...")
                tensors = {}
                with safe_open(path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key)
                return tensors

            def patch_safetensors_for_llava(tensors: dict, model_dim=4096):
                print("Patching for LLaVA compatibility...")
                dummy = lambda shape: torch.zeros(shape, dtype=torch.float16)
                required_keys = [
                    "vision_tower.0.dummy",
                    "mm_projector.0.weight",
                    "mm_projector.0.bias",
                    "model.embed_tokens.weight",
                    "model.norm.weight",
                ]
                for i in range(32):
                    base = f"model.layers.{i}"
                    tensors[f"{base}.self_attn.q_proj.weight"] = dummy((model_dim, model_dim))
                    tensors[f"{base}.self_attn.k_proj.weight"] = dummy((model_dim, model_dim))
                    tensors[f"{base}.self_attn.v_proj.weight"] = dummy((model_dim, model_dim))
                    tensors[f"{base}.self_attn.o_proj.weight"] = dummy((model_dim, model_dim))
                    tensors[f"{base}.mlp.gate_proj.weight"] = dummy((model_dim*4, model_dim))
                    tensors[f"{base}.mlp.up_proj.weight"] = dummy((model_dim*4, model_dim))
                    tensors[f"{base}.mlp.down_proj.weight"] = dummy((model_dim, model_dim*4))
                    tensors[f"{base}.input_layernorm.weight"] = dummy((model_dim,))
                    tensors[f"{base}.post_attention_layernorm.weight"] = dummy((model_dim,))
                for key in required_keys:
                    if key not in tensors:
                        tensors[key] = dummy((model_dim, model_dim))
                return tensors

            def write_temp_safetensors(tensors):
                fd, path = tempfile.mkstemp(suffix=".safetensors")
                os.close(fd)
                save_file(tensors, path)
                return path

            tensors = load_safetensors_dict(path)
            patched = patch_safetensors_for_llava(tensors)
            return write_temp_safetensors(patched)

        if vision_from_pretrained:
            if not os.path.exists(pvlp_model):
                print("Merging vision model shards...")
                model_dir = pretrained_vision_local_path
                shard_paths = [p for p in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))) if "index" not in p]
                combined = {}
                for shard_path in shard_paths:
                    print(f"Loading {shard_path}")
                    shard = load_file(shard_path, device="cpu")
                    combined.update(shard)
                save_file(combined, pvlp_model)
            if patch_vision:
                pvlp_model = model_patch(pvlp_model)
        else:
            print("Using local safetensors file as vision model.")

        # Load or initialize TE
        if load_te:
            try:
                if encoder_from_pretrained:
                    tokenizer = AutoTokenizer.from_pretrained(encoder_pretrained_path, trust_remote_code=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        encoder_pretrained_path,
                        ignore_mismatched_sizes=True,
                        trust_remote_code=True
                    )
                    model = model.to(torch.float16).eval()
                else:
                    tokenizer = AutoTokenizer.from_pretrained(encoder_pretrained_path, trust_remote_code=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        encoder_pretrained_path,
                        trust_remote_code=True,
                        ignore_mismatched_sizes=True
                    )
                    state_dict = load_file(encoder_path)
                    model.load_state_dict(state_dict, strict=False)
                    model = model.to(torch.float16).eval()
                    
            except Exception as e:
                print(f"Error loading text encoder: {e}")
                raise

            class InstructWrapper:
                def __init__(self):
                    self.system_prompt = system_prompt
                    print("System prompt set.")

                def tokenize(self, text):
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": text}
                    ]
                    return tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt")

                def encode_from_tokens(self, tokens, return_pooled=True):
                        with torch.no_grad():
                            out = model(input_ids=tokens, output_hidden_states=True)
                            hidden = out.hidden_states[-1]  # (batch, seq_len, dim)
                            pooled = hidden.mean(dim=1)  # pooled representation
                            if return_pooled:
                                # Return (cond, pooled)
                                return hidden, pooled
                            else:
                                return hidden

            class CustomCLIP:
                def __init__(self, text_encoder):
                    self.text = text_encoder
                def tokenize(self, text):
                    return self.text.tokenize(text)
                def encode_from_tokens(self, tokens, return_pooled=True):
                    return self.text.encode_from_tokens(tokens, return_pooled)

            return (CustomCLIP(InstructWrapper()),)
        else:
            print("TE loading disabled. Returning vision model only (not wrapped).")
            return (pvlp_model,)  # You could wrap this later as needed


NODE_CLASS_MAPPINGS = {
    "FunPackImg2LatentInterpolation": FunPackImg2LatentInterpolation,
    "FunPackCLIPLoader": FunPackCLIPLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FunPackImg2LatentInterpolation": "FunPack img2latent Interpolation",
    "FunPackCLIPLoader": "FunPack CLIP Loader"
}