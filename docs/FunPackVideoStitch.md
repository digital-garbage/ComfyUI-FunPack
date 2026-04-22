# FunPack Video Stitch

This node blends multiple image-frame batches into one longer sequence by adding configurable transitions between neighboring videos, or by cutting directly when transitions are disabled.

## Parameters

**blend_frames**: Number of frames used for each transition between neighboring videos. Set this to `0` to disable transitions and concatenate clips directly.

**transition_type**: Blend curve used for each transition. Supported values are `linear`, `ease_in`, `ease_out`, `ease_in_out`, and `cosine`.

**video1, video2, video3, video4, video5, video6, video7, video8**: Input frame batches to stitch together. At least 2 connected inputs are required.

## Outputs

**STITCHED**: Single image batch containing the stitched sequence.

## Purpose

Use this node when you have multiple generated clips and want to merge them into one sequence, either with smoother transitions or with direct cuts.

## Notes

Each connected input batch must contain at least as many frames as `blend_frames` when transitions are enabled.
