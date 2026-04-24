# FunPack CLIP Vision Output Combine

This node combines up to four `CLIP_VISION_OUTPUT` inputs into one output.

Use it when a workflow has several CLIP Vision encodes and the next node only accepts one CLIP Vision output.

## Parameters

**clip_vision_output1**: Required CLIP Vision output.

**clip_vision_output2, clip_vision_output3, clip_vision_output4**: Optional extra CLIP Vision outputs.

**method**: How tensor fields are combined:

- `mean`: average value
- `median`: median value
- `maximum`: highest value
- `minimum`: lowest value

All tensor fields being combined must have the same shape. Non-tensor fields are copied from the first input.

## Output

**clip_vision_output**: Combined CLIP Vision output.
