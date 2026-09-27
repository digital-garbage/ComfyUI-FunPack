"""H3's rotary position encoding, undone and redone outside the model.

H3 turns the first `2*half` dims of every head's query by a per-row 2x2 matrix
(split-half: dim i pairs with i+half). A feature that averages queries across
rows, or adds one vector to all of them, has to work where the rows are NOT yet
turned -- otherwise the average cancels and the push points somewhere different
at every position. Mirrors comfy_kitchen's eager `apply_rope_split_half1`.
"""

import torch


def query_rotation(rope_freqs, seq_len):
    """-> rotate(x [..., n, D], rows slice, inverse=False), or None when
    `rope_freqs` is not H3's [1, S, 1, half, 2, 2] table for this sequence."""
    if not torch.is_tensor(rope_freqs) or rope_freqs.ndim != 6 \
            or rope_freqs.shape[1] != seq_len or tuple(rope_freqs.shape[-2:]) != (2, 2):
        return None
    table = rope_freqs[0, :, 0]                               # [S, half, 2, 2]

    def rotate(x, rows, inverse=False):
        t = table[rows]
        half = int(t.shape[-3])
        rot = 2 * half
        pairs = x[..., :rot].reshape(*x.shape[:-1], 2, half).movedim(-2, -1).float()
        mat = t.float().transpose(-1, -2) if inverse else t.float()
        out = (mat @ pairs.unsqueeze(-1)).squeeze(-1)
        out = out.movedim(-1, -2).reshape(*x.shape[:-1], rot).to(x.dtype)
        return torch.cat([out, x[..., rot:]], dim=-1)

    return rotate
