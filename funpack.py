# Compatibility re-export for older imports that expect funpack.py.

try:
    from . import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from .conditioning import (
        FunPackGemmaEmbeddingRefiner,
        FunPackLorebookEnhancer,
        FunPackPromptCombiner,
        FunPackPromptEnhancer,
        FunPackSaveRefinementLatent,
        FunPackStoryWriter,
        FunPackVideoRefiner,
    )
    from .image_processing import (
        FunPackContinueVideo,
        FunPackStoryMemKeyframeExtractor,
        FunPackStoryMemLastFrameExtractor,
        FunPackVideoStitch,
    )
    from .model_management import FunPackApplyLoraWeights, FunPackLoraLoader
    from .samplers import FunPackHybridEuler2SSampler, sample_funpack_hybrid_euler_2s
except ImportError:
    from conditioning import (
        FunPackGemmaEmbeddingRefiner,
        FunPackLorebookEnhancer,
        FunPackPromptCombiner,
        FunPackPromptEnhancer,
        FunPackSaveRefinementLatent,
        FunPackStoryWriter,
        FunPackVideoRefiner,
    )
    from image_processing import (
        FunPackContinueVideo,
        FunPackStoryMemKeyframeExtractor,
        FunPackStoryMemLastFrameExtractor,
        FunPackVideoStitch,
    )
    from model_management import FunPackApplyLoraWeights, FunPackLoraLoader
    from samplers import FunPackHybridEuler2SSampler, sample_funpack_hybrid_euler_2s

    NODE_CLASS_MAPPINGS = {
        "FunPackPromptCombiner": FunPackPromptCombiner,
        "FunPackStoryMemKeyframeExtractor": FunPackStoryMemKeyframeExtractor,
        "FunPackStoryMemLastFrameExtractor": FunPackStoryMemLastFrameExtractor,
        "FunPackPromptEnhancer": FunPackPromptEnhancer,
        "FunPackStoryWriter": FunPackStoryWriter,
        "FunPackVideoStitch": FunPackVideoStitch,
        "FunPackContinueVideo": FunPackContinueVideo,
        "FunPackLorebookEnhancer": FunPackLorebookEnhancer,
        "FunPackGemmaEmbeddingRefiner": FunPackGemmaEmbeddingRefiner,
        "FunPackVideoRefiner": FunPackVideoRefiner,
        "FunPackSaveRefinementLatent": FunPackSaveRefinementLatent,
        "FunPackHybridEuler2SSampler": FunPackHybridEuler2SSampler,
        "FunPackApplyLoraWeights": FunPackApplyLoraWeights,
        "FunPackLoraLoader": FunPackLoraLoader,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "FunPackPromptCombiner": "FunPack Prompt Combiner",
        "FunPackStoryMemKeyframeExtractor": "FunPack StoryMem Keyframe Extractor",
        "FunPackStoryMemLastFrameExtractor": "FunPack StoryMem Last Frame Extractor",
        "FunPackPromptEnhancer": "FunPack Prompt Enhancer (Standalone)",
        "FunPackStoryWriter": "FunPack Story Writer",
        "FunPackVideoStitch": "FunPack Video Stitch",
        "FunPackContinueVideo": "FunPack Continue Video",
        "FunPackLorebookEnhancer": "FunPack Lorebook Enhancer",
        "FunPackGemmaEmbeddingRefiner": "FunPack Video Refiner (Compatibility)",
        "FunPackVideoRefiner": "FunPack Video Refiner",
        "FunPackSaveRefinementLatent": "FunPack Save Refinement Latent",
        "FunPackHybridEuler2SSampler": "FunPack Hybrid Euler 2S Sampler",
        "FunPackApplyLoraWeights": "FunPack Apply LoRA Weights",
        "FunPackLoraLoader": "FunPack LoRA Loader",
    }

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "FunPackGemmaEmbeddingRefiner",
    "FunPackSaveRefinementLatent",
    "FunPackVideoRefiner",
    "FunPackHybridEuler2SSampler",
    "FunPackPromptCombiner",
    "FunPackLorebookEnhancer",
    "FunPackPromptEnhancer",
    "FunPackStoryWriter",
    "FunPackVideoStitch",
    "FunPackContinueVideo",
    "FunPackStoryMemKeyframeExtractor",
    "FunPackStoryMemLastFrameExtractor",
    "FunPackApplyLoraWeights",
    "FunPackLoraLoader",
    "sample_funpack_hybrid_euler_2s",
]
