# FunPack StoryMem Keyframe Extractor

This node extracts representative keyframes from a video batch using CLIP Vision similarity, with optional HPSv3 quality filtering.

## Parameters

**frames**: Input frame batch in ComfyUI `IMAGE` format.

**clip_vision**: CLIP Vision model from your ComfyUI `models/clip_vision` folder.

**max_keyframes**: Maximum number of keyframes to return.

**similarity_threshold**: Similarity cutoff used when deciding whether a frame is different enough from the previous keyframe. Lower values usually keep more frames.

**use_quality_filter**: If enabled, the node also filters frames using HPSv3 quality scoring.

**quality_threshold**: Minimum HPSv3 score accepted when quality filtering is enabled.

**memory_frames**: Optional earlier keyframes to compare against so near-duplicates can be skipped.

## Outputs

**keyframes**: Extracted keyframes as an image batch.

**keyframe_count**: Number of frames returned in the `keyframes` output.

## Purpose

Use this node to distill a longer clip into a few representative frames for StoryMem, continuity, or recap workflows.

## Notes

HPSv3 is optional and is only used when `use_quality_filter` is enabled and the package is installed.
