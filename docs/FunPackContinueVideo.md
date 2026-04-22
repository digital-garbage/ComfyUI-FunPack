# FunPack Continue Video

This node extracts the last part of a frame batch so it can be reused as continuation input for another generation step.

## Parameters

**images**: Input image batch or video frame sequence.

**frame_count**: Number of trailing frames to extract from the end of the batch.

## Outputs

**CONTINUED**: The last `frame_count` frames from the input sequence.

## Purpose

Use this node when your next workflow step needs the ending frames of a clip to continue motion or preserve temporal context.
