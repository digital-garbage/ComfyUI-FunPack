"""
olivv's FunPack v1.1.1

Changelog:

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
            'required': 
                {
                    'clip_model_name': (s.get_filename_list(),),
                    'text_encoder_model_name': (s.get_filename_list(),),
                    'llm_vision_model_name':(s.get_filename_list(),),
                    'type': base['required']['type'],
                    'encoder_pretrained_path': ("STRING", {"multiline": False, "default": "mlabonne/NeuralLlama-3-8B-Instruct-abliterated"}),
                    'vision_pretrained_path': ("STRING", {"multiline": False, "default": "qresearch/llama-3.1-8B-vision-378"}),
                    'encoder_from_pretrained': ("BOOLEAN", {"default": False, "tooltip": "Load Instruct model from pretrained_path"}),
                    'vision_from_pretrained': ("BOOLEAN", {"default": False, "tooltip": "Load LLM+vision model from pretrained_path"}),
                    'load_te': ("BOOLEAN", {"default": True, "tooltip": "If off, does not load separate model as text encoder, using only llm_vision_model_name"}),
                    'patch_vision': ("BOOLEAN", {"default": False, "tooltip": "Try to patch vision model for HYV compatibility (experimental)"}),
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
    
    def load(self, clip_model_name, type, text_encoder_model_name, llm_vision_model_name, encoder_pretrained_path, vision_pretrained_path, system_prompt, encoder_from_pretrained=None, vision_from_pretrained=None, load_te=None, patch_vision=None):
        # Load CLIP model using ComfyUI
        clip_path = folder_paths.get_full_path('clip', clip_model_name)
        def get_clip_type(type):
            clip_type = getattr(sd.CLIPType, type.upper(), sd.CLIPType.HUNYUAN_VIDEO)
            print("Detected clip type:", clip_type)
            return clip_type
        
        # Load TE model from weights
        encoder_path = folder_paths.get_full_path('clip', text_encoder_model_name)
        config_source = encoder_pretrained_path
        
        pretrained_vision_local_path = snapshot_download(repo_id=vision_pretrained_path)
        pvlp_model = pretrained_vision_local_path + "/model.safetensors"
        
        print("Loading TE from pretrained is set to", encoder_from_pretrained)
        print("Loading LLM+vision from pretrained is set to", vision_from_pretrained)
        print("Loading custom TE is set to", load_te)

        # Load LLM with vision capabilities (expected llava-llama-3-8b_v1_1)
        if vision_from_pretrained == True:
            print("Loading LLM+vision from", vision_pretrained_path)
            
            # Oho-ho! A funny part! Patching model to force ComfyUI into thinking we are working with LLaVA!
            
            def model_patch(path):
                def load_safetensors_dict(path):
                    print("Loading original LLaMA to patch your custom vision model...")
                    tensors = {}
                    with safe_open(path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            tensors[key] = f.get_tensor(key)
                    return tensors
                    
                def patch_safetensors_for_llava(tensors: dict, model_dim=4096):
                    print("Creating dummy tensors...")
                    dummy_tensor = lambda shape: torch.zeros(shape, dtype=torch.float16)

                    required_keys = [
                        "vision_tower.0.dummy",  # placeholder for vision module
                        "mm_projector.0.weight",  # dummy projector
                        "mm_projector.0.bias",
                        "model.embed_tokens.weight",
                        "model.norm.weight",
                    ]

                    for i in range(32):  # LLaMA-3 32-layer
                        base = f"model.layers.{i}"
                        tensors[f"{base}.self_attn.q_proj.weight"] = dummy_tensor((model_dim, model_dim))
                        tensors[f"{base}.self_attn.k_proj.weight"] = dummy_tensor((model_dim, model_dim))
                        tensors[f"{base}.self_attn.v_proj.weight"] = dummy_tensor((model_dim, model_dim))
                        tensors[f"{base}.self_attn.o_proj.weight"] = dummy_tensor((model_dim, model_dim))
                        tensors[f"{base}.mlp.gate_proj.weight"] = dummy_tensor((model_dim*4, model_dim))
                        tensors[f"{base}.mlp.up_proj.weight"] = dummy_tensor((model_dim*4, model_dim))
                        tensors[f"{base}.mlp.down_proj.weight"] = dummy_tensor((model_dim, model_dim*4))
                        tensors[f"{base}.input_layernorm.weight"] = dummy_tensor((model_dim,))
                        tensors[f"{base}.post_attention_layernorm.weight"] = dummy_tensor((model_dim,))

                    # Add the minimum essential keys if they don't exist
                    for key in required_keys:
                        if key not in tensors:
                            tensors[key] = dummy_tensor((model_dim, model_dim))
                    return tensors
                    
                def write_temp_safetensors(tensors):
                    print("Writing temporary model into memory...")
                    fd, path = tempfile.mkstemp(suffix=".safetensors")
                    os.close(fd)
                    save_file(tensors, path)
                    return path
                    
                vision_weights = load_safetensors_dict(encoder_path)
                patched = patch_safetensors_for_llava(vision_weights)
                vision_path_patched = write_temp_safetensors(patched)
                print("Patching completed successfully.")
                return vision_path_patched
            
            if os.path.exists(pvlp_model):
                print("Model already saved in a single file. Loading from local path...")
                model_dir = snapshot_download(repo_id=vision_pretrained_path)
                vision_path = pretrained_vision_local_path + "/model.safetensors"
                if patch_vision:
                    vision_path = model_patch(vision_path)
                print ("Loading from", vision_path)
            else:
                print("Local model does not exist. Loading, merging and saving it locally..")
                model = AutoModelForCausalLM.from_pretrained(vision_pretrained_path, ignore_mismatched_sizes=True, trust_remote_code=True)
                model_dir = snapshot_download(repo_id=vision_pretrained_path)
                shard_paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
                shard_paths = [p for p in shard_paths if "index" not in p]
                print("Shard files:", shard_paths)
                # Step 3: Load and combine shards
                combined_state_dict = {}
                for shard_path in shard_paths:
                    print(shard_path)
                    # Load shard to CPU to minimize memory usage
                    shard_state_dict = load_file(shard_path, device="cpu")
                    combined_state_dict.update(shard_state_dict)
                    # Free memory
                    del shard_state_dict
                
                output_safetensors = os.path.join(pretrained_vision_local_path, "model.safetensors")
                print("Model output path:", output_safetensors)
                save_file(combined_state_dict, output_safetensors)
                
                print("Shards successfully transformed into single model.")
                vision_path = pretrained_vision_local_path + "/model.safetensors"
                if patch_vision:
                    vision_path = model_patch(vision_path)
        else:
            print("Loading LLM+vision from existing local safetensors file...")
            vision_path = folder_paths.get_full_path('clip', llm_vision_model_name)
        
        if load_te == True:
            try:
                if encoder_from_pretrained == False:
                    print("Loading custom text encoder from the path:", encoder_path)
                    model = LlamaForCausalLM.from_pretrained(config_source, trust_remote_code=True)
                    state_dict = load_file(encoder_path, device="cuda")
                    model.load_state_dict(state_dict, strict=False)
                    model.eval().to(torch.float16).requires_grad_(False)
                    print("Custom text encoder from safetensors file loaded successfully!")
                else:
                    print("Loading custom text encoder from the path:", encoder_pretrained_path)
                    config = AutoConfig.from_pretrained(encoder_pretrained_path, trust_remote_code=True)
                    model = AutoModelForCausalLM.from_pretrained(encoder_pretrained_path, trust_remote_code=True)
                    tokenizer = AutoTokenizer.from_pretrained(encoder_pretrained_path, trust_remote_code=True)
                    model.eval().to(torch.float16).requires_grad_(False)
                    print("Custom text encoder from transformers loaded successfully!")
            except Exception as e:
                print(f"Error loading custom text encoder: {e}")
                raise

        # Wrap it like a CLIP-compatible text encoder
        class InstructWrapper:
            def __init__(self):
                print("TEWrapper initialized!")
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
        clip_model = sd.load_clip(ckpt_paths=[clip_path, vision_path], embedding_directory=None, clip_type=get_clip_type(type), model_options={})
        if load_te == True:
            clip_model.text = InstructWrapper()
        print("Current TE:", clip_model.text)  # Check if encoder is replaced 
        print(dir(clip_model))
        return (clip_model,)


NODE_CLASS_MAPPINGS = {
    "FunPackImg2LatentInterpolation": FunPackImg2LatentInterpolation,
    "FunPackCLIPLoader": FunPackCLIPLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FunPackImg2LatentInterpolation": "FunPack img2latent Interpolation",
    "FunPackCLIPLoader": "FunPack CLIP Loader"
}