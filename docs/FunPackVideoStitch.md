# FunPack Video Stitch

This node blends multiple image-frame batches into one longer sequence by crossfading between neighboring videos.

## Parameters

**blend_frames**: Number of frames used for each transition between neighboring videos.

**video1, video2, video3, video4, video5, video6, video7, video8**: Input frame batches to stitch together. At least 2 connected inputs are required.

## Outputs

**STITCHED**: Single image batch containing the stitched sequence with blended transitions.

## Purpose

Use this node when you have multiple generated clips and want to merge them into one smoother sequence instead of cutting abruptly between them.

## Notes

Each connected input batch must contain at least as many frames as `blend_frames`.
