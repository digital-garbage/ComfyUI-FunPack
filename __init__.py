# ComfyUI-FunPack/__init__.py

if __package__:
    # BEFORE anything loads a model. ComfyUI computes its pinned-memory budget when
    # model_management is imported (core, so already done) and commits it as models are
    # staged (custom nodes, so not yet) — this import sits in the only window where the
    # number can still be lowered. No-op unless FUNPACK_PINNED_MEMORY is set.
    try:
        from . import diagnostics as _fp_diagnostics
        _fp_diag_note = _fp_diagnostics.enable()
        if _fp_diag_note:
            print(f"[FunPack] {_fp_diag_note}")
    except Exception as _e:  # noqa: BLE001
        print(f"[FunPack] could not enable faulthandler: {_e}")
    try:
        from . import host_memory as _fp_host_memory
        _fp_mem_note = _fp_host_memory.apply()
        if _fp_mem_note:
            print(f"[FunPack] {_fp_mem_note}")
    except Exception as _e:  # noqa: BLE001
        print(f"[FunPack] could not adjust pinned memory: {_e}")
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
    from .loaders import FunPackCLIPLoader, FunPackDiffusionModelLoader, FunPackVAELoader
    from .model_management import FunPackApplyLoraWeights, FunPackLoraLoader
    from .samplers import FunPackHybridEuler2SSampler, FunPackDistilledFlowSampler, FunPackLTXAVSceneChainSampler
    from .templates import FunPackRefinementKeyLoader
    try:
        from . import batch_training  # noqa: F401  registers /funpack/batch/* routes
    except Exception as _e:
        print(f"[FunPack] batch_training routes unavailable: {_e}")
    try:
        from .movie_editor import server as _movie_editor_server  # noqa: F401  registers /funpack/* routes
    except Exception as _e:
        print(f"[FunPack] FunPack UI routes unavailable: {_e}")
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
        from loaders import FunPackCLIPLoader, FunPackDiffusionModelLoader, FunPackVAELoader
    except Exception:
        FunPackCLIPLoader = None
        FunPackDiffusionModelLoader = None
        FunPackVAELoader = None
    try:
        from samplers import FunPackHybridEuler2SSampler, FunPackDistilledFlowSampler, FunPackLTXAVSceneChainSampler
    except Exception:
        FunPackHybridEuler2SSampler = None
        FunPackDistilledFlowSampler = None
        FunPackLTXAVSceneChainSampler = None
    try:
        from templates import FunPackRefinementKeyLoader
    except Exception:
        FunPackRefinementKeyLoader = None

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

    "FunPackLTXAVSceneChainSampler": FunPackLTXAVSceneChainSampler,
    "FunPackApplyLoraWeights": FunPackApplyLoraWeights,
    "FunPackLoraLoader": FunPackLoraLoader,
    "FunPackDiffusionModelLoader": FunPackDiffusionModelLoader,
    "FunPackCLIPLoader": FunPackCLIPLoader,
    "FunPackVAELoader": FunPackVAELoader,
    "FunPackRefinementKeyLoader": FunPackRefinementKeyLoader,
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

    "FunPackLTXAVSceneChainSampler": "FunPack LTXAV Scene Chain Sampler",
    "FunPackApplyLoraWeights": "FunPack Apply LoRA Weights",
    "FunPackLoraLoader": "FunPack LoRA Loader",
    "FunPackDiffusionModelLoader": "FunPack Diffusion Model Loader",
    "FunPackCLIPLoader": "FunPack CLIP Loader",
    "FunPackVAELoader": "FunPack VAE Loader",
    "FunPackRefinementKeyLoader": "FunPack Refinement Key Loader",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
