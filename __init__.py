# ComfyUI-FunPack/__init__.py

if __package__:
    from .conditioning import (
        FunPackAdvisorLLM,
        FunPackConditioningAdjust,
        FunPackLorebookEnhancer,
        FunPackPromptCombiner,
        FunPackPromptEnhancer,
        FunPackSaveRefinementLatent,
        FunPackStoryWriter,
        FunPackStudio,
        FunPackVideoRefinerV2,
    )
    from .image_processing import (
        FunPackClipVisionOutputCombine,
        FunPackContinueVideo,
        FunPackStoryMemKeyframeExtractor,
        FunPackStoryMemLastFrameExtractor,
        FunPackVideoStitch,
    )
    from .model_management import FunPackApplyLoraWeights, FunPackLoraLoader
    from .samplers import FunPackHybridEuler2SSampler, FunPackDistilledFlowSampler
    from .context_transition import FunPackSceneCutWindows, FunPackSceneNoise
    from .templates import FunPackRefinementKeyLoader, FunPackSceneBuilder
else:
    # Standalone tests may not have the full ComfyUI/CUDA runtime loaded.
    from conditioning import (
        FunPackAdvisorLLM,
        FunPackConditioningAdjust,
        FunPackLorebookEnhancer,
        FunPackPromptCombiner,
        FunPackPromptEnhancer,
        FunPackSaveRefinementLatent,
        FunPackStoryWriter,
        FunPackStudio,
        FunPackVideoRefinerV2,
    )
    try:
        from image_processing import (
            FunPackClipVisionOutputCombine,
            FunPackContinueVideo,
            FunPackStoryMemKeyframeExtractor,
            FunPackStoryMemLastFrameExtractor,
            FunPackVideoStitch,
        )
    except Exception:
        FunPackClipVisionOutputCombine = None
        FunPackContinueVideo = None
        FunPackStoryMemKeyframeExtractor = None
        FunPackStoryMemLastFrameExtractor = None
        FunPackVideoStitch = None
    try:
        from model_management import FunPackApplyLoraWeights, FunPackLoraLoader
    except Exception:
        FunPackApplyLoraWeights = None
        FunPackLoraLoader = None
    try:
        from samplers import FunPackHybridEuler2SSampler, FunPackDistilledFlowSampler
    except Exception:
        FunPackHybridEuler2SSampler = None
        FunPackDistilledFlowSampler = None
    try:
        from context_transition import FunPackSceneCutWindows, FunPackSceneNoise
    except Exception:
        FunPackSceneCutWindows = None
        FunPackSceneNoise = None
    try:
        from templates import FunPackRefinementKeyLoader, FunPackSceneBuilder
    except Exception:
        FunPackRefinementKeyLoader = None
        FunPackSceneBuilder = None

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "FunPackAdvisorLLM": FunPackAdvisorLLM,
    "FunPackConditioningAdjust": FunPackConditioningAdjust,
    "FunPackStudio": FunPackStudio,
    "FunPackPromptCombiner": FunPackPromptCombiner,
    "FunPackStoryMemKeyframeExtractor": FunPackStoryMemKeyframeExtractor,
    "FunPackStoryMemLastFrameExtractor": FunPackStoryMemLastFrameExtractor,
    "FunPackPromptEnhancer": FunPackPromptEnhancer,
    "FunPackStoryWriter": FunPackStoryWriter,
    "FunPackVideoStitch": FunPackVideoStitch,
    "FunPackClipVisionOutputCombine": FunPackClipVisionOutputCombine,
    "FunPackContinueVideo": FunPackContinueVideo,
    "FunPackLorebookEnhancer": FunPackLorebookEnhancer,
    "FunPackVideoRefinerV2": FunPackVideoRefinerV2,
    "FunPackSaveRefinementLatent": FunPackSaveRefinementLatent,
    "FunPackHybridEuler2SSampler": FunPackHybridEuler2SSampler,
    "FunPackDistilledFlowSampler": FunPackDistilledFlowSampler,
    "FunPackSceneCutWindows": FunPackSceneCutWindows,
    "FunPackSceneNoise": FunPackSceneNoise,
    "FunPackApplyLoraWeights": FunPackApplyLoraWeights,
    "FunPackLoraLoader": FunPackLoraLoader,
    "FunPackRefinementKeyLoader": FunPackRefinementKeyLoader,
    "FunPackSceneBuilder": FunPackSceneBuilder,
}
NODE_CLASS_MAPPINGS = {name: cls for name, cls in NODE_CLASS_MAPPINGS.items() if cls is not None}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FunPackAdvisorLLM": "FunPack Advisor LLM",
    "FunPackConditioningAdjust": "FunPack Conditioning Adjust",
    "FunPackStudio": "FunPack Studio",
    "FunPackPromptCombiner": "FunPack Prompt Combiner",
    "FunPackStoryMemKeyframeExtractor": "FunPack StoryMem Keyframe Extractor",
    "FunPackStoryMemLastFrameExtractor": "FunPack StoryMem Last Frame Extractor",
    "FunPackPromptEnhancer": "FunPack Prompt Enhancer (Standalone)",
    "FunPackStoryWriter": "FunPack Story Writer",
    "FunPackVideoStitch": "FunPack Video Stitch",
    "FunPackClipVisionOutputCombine": "FunPack CLIP Vision Output Combine",
    "FunPackContinueVideo": "FunPack Continue Video",
    "FunPackLorebookEnhancer": "FunPack Lorebook Enhancer",
    "FunPackVideoRefinerV2": "FunPack Video Refiner V2",
    "FunPackSaveRefinementLatent": "FunPack Save Refinement Latent",
    "FunPackHybridEuler2SSampler": "FunPack Hybrid Euler 2S Sampler",
    "FunPackDistilledFlowSampler": "FunPack Distilled Flow Sampler",
    "FunPackSceneCutWindows": "FunPack Scene Cut Windows",
    "FunPackSceneNoise": "FunPack Scene Noise",
    "FunPackApplyLoraWeights": "FunPack Apply LoRA Weights",
    "FunPackLoraLoader": "FunPack LoRA Loader",
    "FunPackRefinementKeyLoader": "FunPack Refinement Key Loader",
    "FunPackSceneBuilder": "FunPack Scene Builder",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
