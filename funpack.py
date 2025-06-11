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
                    'llm_vision_model_name':(s.get_filename_list(),), # Retained for local loading
                    'type': base['required']['type'],
                    'encoder_pretrained_path': ("STRING", {"multiline": False, "default": "mlabonne/NeuralLlama-3-8B-Instruct-abliterated"}),
                    'encoder_from_pretrained': ("BOOLEAN", {"default": False, "tooltip": "Load Instruct model from pretrained_path"}),
                    'load_te': ("BOOLEAN", {"default": True, "tooltip": "If off, does not load separate model as text encoder, using only llm_vision_model_name"}),
                    'system_prompt': ("STRING", {
                        "multiline": True,
                        "default": "<image>You are an expert visual describer for AI video generation. Your task is to interpret user prompts and transform them into detailed, vivid descriptions optimized for image-to-video synthesis. Ensure your descriptions prioritize visual consistency, dynamic actions, and coherent scene elements to guide the generative model in creating smooth, logical video sequences from an initial image. Do not include conversational filler or explanations; just the descriptive text:<|eot_id|>"
                    }),
                    'top_p': ("FLOAT", {"min": 0.0, "max": 10.0, "step": 0.05, "default": 0.75}),
                    'top_k': ("INT", {"min": 0, "max": 1000, "step": 1, "default": 40}),
                    'temperature': ("FLOAT", {"min": 0.0, "max": 10.0, "step": 0.01, "default": 0.6}),
                    'generate_assist_prompt': ("BOOLEAN", {"default": False})
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
    
    def load(self, clip_model_name, type, text_encoder_model_name, llm_vision_model_name, encoder_pretrained_path, system_prompt, top_p, top_k, temperature, encoder_from_pretrained=None, load_te=None, generate_assist_prompt=None):
        # Load CLIP model using ComfyUI
        clip_path = folder_paths.get_full_path('clip', clip_model_name)
        def get_clip_type(type):
            clip_type = getattr(sd.CLIPType, type.upper(), sd.CLIPType.HUNYUAN_VIDEO)
            print("Detected clip type:", clip_type)
            return clip_type
        
        # Load TE model from weights
        encoder_path = folder_paths.get_full_path('clip', text_encoder_model_name)
        
        # Simplified LLM+vision model loading: always load from existing local safetensors file
        print("Loading LLM+vision from existing local safetensors file:", llm_vision_model_name)
        vision_path = folder_paths.get_full_path('clip', llm_vision_model_name)
        
        print("Loading TE from pretrained is set to", encoder_from_pretrained)
        print("Loading custom TE is set to", load_te)

        # Load the base CLIP model first
        clip_model = sd.load_clip(ckpt_paths=[clip_path, vision_path], embedding_directory=None, clip_type=get_clip_type(type), model_options={"ignore_mismatched_sizes": True})
        
        if load_te == True:
            # Wrap the original CLIP model with our prompt enhancer
            # Pass loading parameters instead of actual model/tokenizer instances
            wrapped_clip_with_enhancement = PromptEnhancerClipWrapper(
                original_clip_model=clip_model,
                encoder_path=encoder_path,
                encoder_pretrained_path=encoder_pretrained_path,
                encoder_from_pretrained=encoder_from_pretrained,
                system_prompt=system_prompt,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                generate_assist_prompt=generate_assist_prompt
            )
            
            # Return our wrapper instance directly. ComfyUI will then call methods on it.
            return (wrapped_clip_with_enhancement,)
        
        else:
            # If no TE, return the original CLIP model without enhancement
            return (clip_model,)

class PromptEnhancerClipWrapper:
    def __init__(self, original_clip_model, encoder_path, encoder_pretrained_path, encoder_from_pretrained, system_prompt, top_p, top_k, temperature, generate_assist_prompt):
        self.original_clip = original_clip_model # Keep a reference to the actual CLIP object
        
        # Store loading parameters for on-demand loading
        self.encoder_path = encoder_path
        self.encoder_pretrained_path = encoder_pretrained_path
        self.encoder_from_pretrained = encoder_from_pretrained
        
        self.llm_model = None # Will be loaded on demand
        self.llm_tokenizer = None # Will be loaded on demand
        self.llm_model_device = None # Will be set upon loading
        
        self.system_prompt = system_prompt
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.max_new_tokens = 1024 # Setting to 256 to allow full detailed LLM output
        self.assistant_reply = None # For conversational context if generate_assist_prompt is true
        self.generate_assist_prompt = generate_assist_prompt
        
        self.enhanced_prompts_cache = {} # Stores original_user_prompt -> enhanced_text
        self._is_llm_generating = False # Flag to prevent recursive LLM calls if LLM's own tokenization process calls us

        print("[PromptEnhancerClipWrapper] Initialized for on-demand LLM loading/unloading.")
        print(f"top_p: {self.top_p}, top_k: {self.top_k}, temperature: {self.temperature}")
        print(f"System prompt: {self.system_prompt}")
        print(f"LLM max_new_tokens set to: {self.max_new_tokens} for full detail.")

    def _generate_enhanced_prompt(self, user_prompt):
        print("[PromptEnhancerClipWrapper] Calling _generate_enhanced_prompt for LLM inference...")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # If generate_assist_prompt is true and we have a previous assistant reply, add it
        if self.generate_assist_prompt and self.assistant_reply:
            messages.append({"role": "assistant", "content": self.assistant_reply})
        
        # Load LLM model and tokenizer only if not already loaded (e.g., first call in a generation cycle)
        if self.llm_model is None or self.llm_tokenizer is None:
            print("[PromptEnhancerClipWrapper] LLM model and tokenizer not loaded. Loading now...")
            try:
                if not self.encoder_from_pretrained:
                    self.llm_tokenizer = AutoTokenizer.from_pretrained("xtuner/llava-llama-3-8b-v1_1-transformers", trust_remote_code=True)
                    model_base = LlamaForCausalLM.from_pretrained("xtuner/llava-llama-3-8b-v1_1-transformers", trust_remote_code=True)
                    state_dict = load_file(self.encoder_path, device="cpu") # Load to CPU initially to control device
                    model_base.load_state_dict(state_dict, strict=False)
                    self.llm_model = model_base.eval().to(torch.float16).requires_grad_(False) # Convert to float16 on CPU
                    print(f"Custom text encoder from safetensors file loaded successfully to CPU!")
                else:
                    self.llm_tokenizer = AutoTokenizer.from_pretrained(self.encoder_pretrained_path, trust_remote_code=True)
                    self.llm_model = AutoModelForCausalLM.from_pretrained(self.encoder_pretrained_path, ignore_mismatched_sizes=True, trust_remote_code=True)
                    self.llm_model = self.llm_model.eval().to(torch.float16).requires_grad_(False) # Convert to float16 on CPU
                    print(f"Custom text encoder from transformers loaded successfully to CPU!")

            except Exception as e:
                print(f"Error loading LLM model for enhancement: {e}")
                raise

        llm_tokens = self.llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            self._is_llm_generating = True # Set flag before LLM generation
            try:
                generated_ids = self.llm_model.generate(
                    input_ids=llm_tokens,
                    do_sample=True,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.llm_tokenizer.pad_token_id
                )
            finally:
                self._is_llm_generating = False # Always reset flag

            # Decode only the newly generated part
            output_text = self.llm_tokenizer.decode(generated_ids[0][llm_tokens.shape[1]:], skip_special_tokens=True)
            self.assistant_reply = output_text # Store for next turn
        torch.cuda.empty_cache() # Clear GPU cache
        gc.collect() # Force Python garbage collection
        print(f"[PromptEnhancerClipWrapper] LLM model fully unloaded (from GPU and attempting to free RAM).")

        return output_text

    def tokenize(self, prompt):
        print(f"[PromptEnhancerClipWrapper] Received prompt for tokenization: '{prompt}'")

        # If LLM is currently generating, this call is likely from LLM's internal tokenization (e.g. from tokenizer.apply_chat_template).
        # Pass through to original CLIP tokenizer directly.
        if self._is_llm_generating:
            print(f"[PromptEnhancerClipWrapper] LLM generation in progress. Passing directly to original CLIP tokenizer: '{prompt}'")
            return self.original_clip.tokenize(prompt)

        # Check if the prompt is an already enhanced prompt (a value in our cache)
        # This prevents re-enhancing text that was already LLM-generated.
        if prompt in self.enhanced_prompts_cache.values():
            print(f"[PromptEnhancerClipWrapper] Prompt '{prompt}' is an already enhanced prompt. Passing directly to original CLIP tokenizer.")
            return self.original_clip.tokenize(prompt)
        
        # If it's a new, original user prompt (key not in cache)
        if prompt not in self.enhanced_prompts_cache:
            print(f"[PromptEnhancerClipWrapper] Prompt '{prompt}' NOT in cache. Calling LLM for enhancement...")
            enhanced_text = self._generate_enhanced_prompt(prompt)
            self.enhanced_prompts_cache[prompt] = enhanced_text
            print(f"[PromptEnhancerClipWrapper] LLM enhanced and cached for '{prompt}': {enhanced_text}")
        else:
            print(f"[PromptEnhancerClipWrapper] Using cached enhanced prompt for '{prompt}'.")
            enhanced_text = self.enhanced_prompts_cache[prompt]
        
        print(f"[PromptEnhancerClipWrapper] Final prompt being passed to CLIP: {enhanced_text}")
        
        # Pass the enhanced text to the original CLIP's tokenize method.
        # With max_new_tokens at 256, the LLM will generate more detailed text.
        # CLIP's tokenizer will then handle its own internal truncation if the text is still too long,
        # but the truncated result should be of higher quality.
        return self.original_clip.tokenize(enhanced_text)

    # These methods simply delegate to the original CLIP model
    def encode_from_tokens(self, tokens, return_pooled=True):
        return self.original_clip.encode_from_tokens(tokens, return_pooled)

    def encode_token_weights(self, tokens):
        return self.original_clip.encode_token_weights(tokens)


NODE_CLASS_MAPPINGS = {
    "FunPackImg2LatentInterpolation": FunPackImg2LatentInterpolation,
    "FunPackCLIPLoader": FunPackCLIPLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FunPackImg2LatentInterpolation": "FunPack img2latent Interpolation",
    "FunPackCLIPLoader": "FunPack CLIP Loader"
}