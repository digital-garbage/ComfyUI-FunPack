# ComfyUI-FunPack

A set of nodes designed for experiments with video diffusion models, such as WAN, HunyuanVideo, LTX and others.

All pip packages used by FunPack nodes are Comfy dependencies so no additional installations are required. 
hpsv3 is an _optional_ dependency for Keyframe Extractor node, thus not included into requirements. 
You can install it separately: _pip3 install hpsv3_.

To discover how to work with nodes, please check "docs" folder.

To see the changes between the versions of FunPack, please check CHANGELOG.md.

**FunPack is available on Comfy Registry and offers three different ways to install it:**
1. With comfy-cli: run _comfy node install funpack_ in your Terminal (make sure comfy-cli is installed).
2. With git: run _git clone https://github.com/aimfordeb/ComfyUI-FunPack_ in your Terminal inside your /ComfyUI/custom_nodes folder.
3. With ComfyUI-Manager: _open "Install custom nodes" and search for ComfyUI-FunPack, then click on "Install"._

Additional message: Sorry for testing new features in "main" branch and spamming with commits. It's just inconvenient for me to test on "dev" since I'm not running the code on my own PC locally.

If you have any suggestions or questions, or you have a new feature to add into the node set, feel free to add new topic in Issues or create a pull request.
