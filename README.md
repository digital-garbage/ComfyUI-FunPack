# ComfyUI-FunPack
A set of custom nodes designed for experiments with video diffusion models.
EXPERIMENTAL, and I mean it. Constantly updating, changing, adding and removing, just for sake of making something work.
You have been warned.

**FunPack CLIP Loader**

![image](https://github.com/user-attachments/assets/780de28e-4d69-4048-8dab-e1f80c847eb8)


Update: This node now serves as... I guess, prompt enhancer? It processes user input, adds an enhanced prompt, then does tokenizing using regular CLIP.

Inputs:
- clip_model_name - your CLIP-L model that you usually use with Hunyuan/FramePack (e.g. clip-vit-large-patch14);
- text_encoder_model_name - your instruct (or any other LLM?) model. Expects .safetensors file, if "instruct_from_pretrained" is on - ignores this;
- llm_vision_model_name - your llava-llama-3 model you usually use with Hunyuan/FramePack (e.g. llava-llama-3-8b-v1_1). Also it's possible to load any other LLM, with or without vision capabilities (I guess);
- type - select "hunyuan_video", left for compatibility;
- encoder_pretrained_path - Provide a HuggingFace path for config and tokenizer for your encoder model (or for weights as well, if encoder_from_pretrained=True);
- encoder_from_pretrained - if enabled, loads encoder model weights from encoder_pretrained_path as well, ignoring local "text_encoder_model_name";
- load_te - if enabled, loads your custom text encoder model. If disabled, uses only vision one (e.g. llava-llama-3-8b-v1_1);
- system_prompt - your system prompt that Instruct model is going to be using.
- top_p, top_k, temperature - these are parameters for generating an "assistant prompt";
- generate_assist_prompt - if disabled, bypasses generation of "assistant prompt", if enable - does it with a model that is load as your text_encoder (might be a custom or a standard one).

Please notice: the encoding runs on CPU because I'm stupid and I have two to three extra minutes to wait for it. You might not have them, but that's you.
Also if you have 24GB of VRAM or less, you might go OOM in Comfy - that's expected, just run the sequence again and it will use cached enhanced prompt.

Technically speaking, it's possible to load just any model as text encoder.

Outputs:
Just CLIP. Pass it through your nodes like you will do with regular DualCLIPLoader.

**FunPack img2latent Interpolation**

![image](https://github.com/user-attachments/assets/1f84d00b-e835-4b0a-96da-e8fb9a1c1366)


Inputs:
 - images - a batch of images e.g. from Load Image, Load Video (Upload), Load Video (Path) et cetera.
 - frame_count - same as frame count in HunyuanVideoEmptyLatent or similar. Actually, works better with WAN.

Outputs:
- img_batch_for_encode - latent output with interpolated image, resize if needed and put it into VAE Encode, pass as latent_image into your sampler;
- img_for_start_images - takes exactly the last image from the batch. You can pass it to your CLIP Vision and encoders as start_image. Or end_image. Who am I to tell you.
