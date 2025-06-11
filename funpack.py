"""
olivv's FunPack v1.1.2

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
                    'vision_pretrained_path': ("STRING", {"multiline": False, "default": "huihui-ai/Llama-3.2-11B-Vision-Instruct-abliterated"}),
                    'encoder_from_pretrained': ("BOOLEAN", {"default": False, "tooltip": "Load Instruct model from pretrained_path"}),
                    'vision_from_pretrained': ("BOOLEAN", {"default": False, "tooltip": "Load LLM+vision model from pretrained_path"}),
                    'vision_from_pretrained_comfy': ("BOOLEAN", {"default": False, "tooltip": "Checks for the model in models/LLM instead of HuggingFace"}),
                    'load_te': ("BOOLEAN", {"default": True, "tooltip": "If off, does not load separate model as text encoder, using only llm_vision_model_name"}),
                    'system_prompt': ("STRING", {
                        "multiline": True,
                        "default": "<image>You are an expert visual describer for AI video generation. Your task is to interpret user prompts and transform them into detailed, vivid descriptions optimized for image-to-video synthesis. Ensure your descriptions prioritize visual consistency, dynamic actions, and coherent scene elements to guide the generative model in creating smooth, logical video sequences from an initial image. Do not include conversational filler or explanations; just the descriptive text:<|eot_id|>"
                    }),
                    'top_p': ("FLOAT", {"min": 0.0, "max": 10.0, "step": 0.05, "default": 0.75}),
                    'top_k': ("INT", {"min": 0, "max": 1000, "step": 1, "default": 40}),
                    'temperature': ("FLOAT", {"min": 0.0, "max": 10.0, "step": 0.01, "default": 0.6})
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
    
    def load(self, clip_model_name, type, text_encoder_model_name, llm_vision_model_name, encoder_pretrained_path, vision_pretrained_path, system_prompt, top_p, top_k, temperature, encoder_from_pretrained=None, vision_from_pretrained=None, vision_from_pretrained_comfy=None, load_te=None):
        # Load CLIP model using ComfyUI
        clip_path = folder_paths.get_full_path('clip', clip_model_name)
        def get_clip_type(type):
            clip_type = getattr(sd.CLIPType, type.upper(), sd.CLIPType.HUNYUAN_VIDEO)
            print("Detected clip type:", clip_type)
            return clip_type
        
        # Load TE model from weights
        encoder_path = folder_paths.get_full_path('clip', text_encoder_model_name)
        config_source = encoder_pretrained_path
        
        if not vision_from_pretrained_comfy:
            pretrained_vision_local_path = snapshot_download(repo_id=vision_pretrained_path)
            pvlp_model = pretrained_vision_local_path + "/model.safetensors"
        else:
            pretrained_vision_local_path = folder_paths.models_dir + "/clip/" + vision_pretrained_path
            print(pretrained_vision_local_path)
            pvlp_model = pretrained_vision_local_path + "/model.safetensors"
        
        print("Loading TE from pretrained is set to", encoder_from_pretrained)
        print("Loading LLM+vision from pretrained is set to", vision_from_pretrained)
        print("Loading custom TE is set to", load_te)

        # Load LLM with vision capabilities (expected llava-llama-3-8b_v1_1)
        if vision_from_pretrained:
            print("Loading LLM+vision from", vision_pretrained_path)            
            if os.path.exists(pvlp_model):
                print("Model already saved in a single file. Loading from local path...")
                model_dir = snapshot_download(repo_id=vision_pretrained_path)
                vision_path = pretrained_vision_local_path + "/model.safetensors"
                print ("Loading from", vision_path)
            else:
                print("Local model does not exist. Loading, merging and saving it locally..")
                if not vision_from_pretrained_comfy:
                    model_dir = snapshot_download(repo_id=vision_pretrained_path)
                    model = AutoModelForCausalLM.from_pretrained(vision_pretrained_path, ignore_mismatched_sizes=True, trust_remote_code=True)
                else:
                    model_dir = folder_paths.models_dir + "/clip/" + vision_pretrained_path
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
        else:
            print("Loading LLM+vision from existing local safetensors file...")
            vision_path = folder_paths.get_full_path('clip', llm_vision_model_name)
        
        if load_te:
            try:
                if not encoder_from_pretrained:
                    print("Loading custom text encoder from the path:", encoder_path)
                    config = AutoConfig.from_pretrained(config_source, ignore_mismatched_sizes=True, trust_remote_code=True)
                    model = AutoModelForCausalLM.from_config(config)
                    tokenizer = AutoTokenizer.from_pretrained(config_source, trust_remote_code=True)
                    state_dict = load_file(encoder_path, device="cuda")
                    model.load_state_dict(state_dict, strict=False)
                    model.eval().to(torch.float16).requires_grad_(False)
                    print("Custom text encoder from safetensors file loaded successfully!")
                else:
                    print("Loading custom text encoder from the path:", encoder_pretrained_path)
                    config = AutoConfig.from_pretrained(encoder_pretrained_path, trust_remote_code=True)
                    model = AutoModelForCausalLM.from_pretrained(encoder_pretrained_path, ignore_mismatched_sizes=True, trust_remote_code=True)
                    tokenizer = AutoTokenizer.from_pretrained(encoder_pretrained_path, trust_remote_code=True)
                    model.eval().to(torch.float16).requires_grad_(False)
                    print("Custom text encoder from transformers loaded successfully!")
            except Exception as e:
                print(f"Error loading custom text encoder: {e}")
                raise

        # Wrap it like a CLIP-compatible text encoder
        class InstructWrapper:
            def __init__(self, model, tokenizer, system_prompt, top_p, top_k, temperature):
                print("TEWrapper initialized!")
                self.model = model
                self.tokenizer = tokenizer
                self.system_prompt = system_prompt
                self.top_p = top_p
                self.top_k = top_k
                self.temperature = temperature
                self.max_new_tokens = 128
                print("top_p:", self.top_p)
                print("top_k:", self.top_k)
                print("temperature:", self.temperature)
                print("max new tokens is hardcoded to", self.max_new_tokens)
                print("System prompt is set to:", self.system_prompt)
                
            def tokenize(self, text):
                print("Calling generate() from tokenize()", flush=True)
                assistant_reply = self.generate(text)
                print("Assistant generated:", assistant_reply, flush=True)
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": assistant_reply}
                ]
                return self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt").to("cuda")
                
            def generate(self, text):
                # Prepare chat messages for assistant generation
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ]
                tokens = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,  # Important for generation
                    return_tensors="pt"
                ).to("cuda")

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=tokens,
                        do_sample=True,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        temperature=self.temperature,
                        max_new_tokens=self.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    return output_text

            def encode_from_tokens(self, tokens, return_pooled=True):
                with torch.no_grad():
                    output = self.model(input_ids=tokens, output_hidden_states=True)
                    hidden_states = output.hidden_states[-1]
                    pooled_output = hidden_states.mean(dim=1)
                    # pooled = hidden[:, -1, :]  # use last token for pooled
                return pooled_output

        # Replace text encoder in CLIP model
        clip_model = sd.load_clip(ckpt_paths=[clip_path, vision_path], embedding_directory=None, clip_type=get_clip_type(type), model_options={"ignore_mismatched_sizes": True})
        if load_te == True:
            text_encoder = sd.load_text_encoder_state_dicts(state_dicts = [InstructWrapper(model, tokenizer, system_prompt, top_p, top_k, temperature)], embedding_directory=None, clip_type = get_clip_type(type), model_options={})
            #clip_model.text = InstructWrapper(
            #    model=model,
            #    tokenizer=tokenizer,
            #    system_prompt=system_prompt,
            #    top_p=top_p,
            #    top_k=top_k,
            #    temperature=temperature
            #)
            #print("Current TE:", clip_model.text)  # Check if encoder is replaced
            print("Current TE:", text_encoder)
            return (clip_model, text_encoder,)
        else:
            return (clip_model,)


NODE_CLASS_MAPPINGS = {
    "FunPackImg2LatentInterpolation": FunPackImg2LatentInterpolation,
    "FunPackCLIPLoader": FunPackCLIPLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FunPackImg2LatentInterpolation": "FunPack img2latent Interpolation",
    "FunPackCLIPLoader": "FunPack CLIP Loader"
}