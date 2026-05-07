# ComfyUI-FunPack/__init__.py

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
from .samplers import FunPackHybridEuler2SSampler
from .context_transition import FunPackContextTransitionWindows
from .templates import FunPackTemplateManager

WEB_DIRECTORY = "./web"

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
    "FunPackContextTransitionWindows": FunPackContextTransitionWindows,
    "FunPackApplyLoraWeights": FunPackApplyLoraWeights,
    "FunPackLoraLoader": FunPackLoraLoader,
    "FunPackTemplateManager": FunPackTemplateManager,
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
    "FunPackContextTransitionWindows": "FunPack Context Transition Windows",
    "FunPackApplyLoraWeights": "FunPack Apply LoRA Weights",
    "FunPackLoraLoader": "FunPack LoRA Loader",
    "FunPackTemplateManager": "FunPack Template Manager",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
