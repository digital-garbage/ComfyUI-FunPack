# Compatibility re-export for older imports that expect funpack.py.

try:
    from . import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from .conditioning import (
        FunPackLorebookEnhancer,
        FunPackPromptCombiner,
        FunPackPromptEnhancer,
        FunPackStoryWriter,
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
    from .samplers import FunPackHybridEuler2SSampler, sample_funpack_hybrid_euler_2s
    from .templates import FunPackRefinementKeyLoader, FunPackSceneBuilder
except ImportError:
    from conditioning import (
        FunPackLorebookEnhancer,
        FunPackPromptCombiner,
        FunPackPromptEnhancer,
        FunPackStoryWriter,
        FunPackVideoRefinerV2,
    )
    from image_processing import (
        FunPackClipVisionOutputCombine,
        FunPackContinueVideo,
        FunPackStoryMemKeyframeExtractor,
        FunPackStoryMemLastFrameExtractor,
        FunPackVideoStitch,
    )
    from model_management import FunPackApplyLoraWeights, FunPackLoraLoader
    from samplers import FunPackHybridEuler2SSampler, sample_funpack_hybrid_euler_2s
    from templates import FunPackRefinementKeyLoader, FunPackSceneBuilder

    NODE_CLASS_MAPPINGS = {
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
        "FunPackHybridEuler2SSampler": FunPackHybridEuler2SSampler,
        "FunPackApplyLoraWeights": FunPackApplyLoraWeights,
        "FunPackLoraLoader": FunPackLoraLoader,
        "FunPackRefinementKeyLoader": FunPackRefinementKeyLoader,
        "FunPackSceneBuilder": FunPackSceneBuilder,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
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
        "FunPackHybridEuler2SSampler": "FunPack Hybrid Euler 2S Sampler",
        "FunPackApplyLoraWeights": "FunPack Apply LoRA Weights",
        "FunPackLoraLoader": "FunPack LoRA Loader",
        "FunPackRefinementKeyLoader": "FunPack Refinement Key Loader",
        "FunPackSceneBuilder": "FunPack Scene Builder",
    }

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "FunPackVideoRefinerV2",
    "FunPackHybridEuler2SSampler",
    "FunPackPromptCombiner",
    "FunPackLorebookEnhancer",
    "FunPackPromptEnhancer",
    "FunPackStoryWriter",
    "FunPackVideoStitch",
    "FunPackClipVisionOutputCombine",
    "FunPackContinueVideo",
    "FunPackStoryMemKeyframeExtractor",
    "FunPackStoryMemLastFrameExtractor",
    "FunPackApplyLoraWeights",
    "FunPackLoraLoader",
    "FunPackRefinementKeyLoader",
    "FunPackSceneBuilder",
    "sample_funpack_hybrid_euler_2s",
]
