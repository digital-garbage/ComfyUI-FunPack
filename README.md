# ComfyUI-FunPack
A set of custom nodes designed for experiments with video diffusion models.

**FunPack DualCLIP Instruct Loader**

![image](https://github.com/user-attachments/assets/de7e08ae-8ee7-4cb4-a7a5-6ced069a786c)


This node is designed specifically for FramePack/HunyuanVideo, aiming to replace text encoder module with LLama-3 Instruct model.
Not entirely, just text encoder. Vision module stays the same, so you'll need original llava-llama-3 as well.

Inputs:
- clip_model_name - your CLIP-L model that you usually use with Hunyuan/FramePack (e.g. clip-vit-large-patch14);
- llama_instruct_model_name - your Llama-3 8B instruct model. Currently does not support loading as from_pretrained, expects .safetensors file;
- llama3_model_name - your llava-llama-3 model you usually use with Hunyuan/FramePack (e.g. llava-llama-3-8b-v1_1)
- type - select "hunyuan_video", left for compatibility;
- pretrained_path - Provide a HuggingFace path for config and tokenizer for your model;
- instruct_from_pretrained - if enabled, loads model weights from pretrained_path as well, ignoring "llama_instruct_model_name";
- system_prompt - your system prompt that Instruct model is going to be using.

Technically speaking, it's possible to load just any model as instruct one. It might not even be an instruct model.

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
