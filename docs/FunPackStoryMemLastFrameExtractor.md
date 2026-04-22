# FunPack StoryMem Last Frame Extractor

This node extracts the final frame and the final N frames from a video batch for continuity workflows.

## Parameters

**frames**: Input frame batch in ComfyUI `IMAGE` format.

**n_frames**: Number of trailing frames to include in the motion output.

## Outputs

**last_frame**: Final frame from the input batch, returned as a single-frame image batch.

**motion_frames**: Last `n_frames` frames from the input batch.

## Purpose

Use this node when the next shot needs the exact last frame for image-to-video continuity, plus a short tail of recent motion frames for motion-aware continuation workflows.
