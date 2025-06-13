# ComfyUI-FunPack
A set of custom nodes designed for experiments with video diffusion models.
EXPERIMENTAL, and I mean it. Constantly updating, changing, adding and removing, just for sake of making something work.
You have been warned.

**FunPack Prompt Enhancer (Standalone)**

![image](https://github.com/user-attachments/assets/2768efc5-35f1-4897-811e-dc3b334356d5)

Inputs:
- Prompt - here you add your prompt to be enhanced;
- System prompt - here you pass an instruction on how your LLM should interpret the given prompt. Basic system prompt works fine but you can change it however you like;
- Model path type - choose if you want to load a local .safetensors file from ComfyUI/models/clip or load a HuggingFace pretrained model;
- Model path - path for a model on HuggingFace;
- LLM safetensors path - selector for a local model;
- top_p, top_k, temperature, max_new_tokens - these parameters are responsible for your "enhanced prompt" being creative or strict, short or long. Better watch some video to know how it works.
  For the regular system prompt these settings work really well. If you are not aiming at "extremely creative" or "strictly short" prompts - I won't recommend changing it.

Please notice: after changing a prompt, it's better to call model unloading in Comfy. Believe me.
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

FunPackCLIPLoader node is currently in development and seems like not functioning at all. Please don't use it until otherwise is stated.
