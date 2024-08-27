import numpy as np
import torch
from quant_utils import quantize_weights

# Adapted from https://github.com/IST-DASLab/marlin/blob/master/test.py


def marlin_weight_perm():
    # shape: (1024) for 4 16x16 matrices
    perm = []
    # build permutation for 32 threads
    for i in range(32):
        perm1 = []
        col = i // 4
        # two blocks for each 16x16 matrix
        for block in [0, 1]:
            # each block is 16x8 (row x col)
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        # total 4 16x16 blocks with stride 256
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])
    perm = np.array(perm)
    # permute for fast interleaved_numeric_conversion
    # [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 2, 4, 6, 1, 3, 5, 7]
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm


def marlin_scales_perm():
    # shape: (32) for 4 16x16 matrices
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])

    # shape: (32)
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


# returns the packed weights (k/16, n*16/pack_factor)
def pack_marlin_weights(
    w: torch.Tensor,  # quantized weights, int32 (k, n)
):
    k, n = w.shape
    tile = 16
    # tile the weights to 16x16 blocks
    # [m/16, 16, n/16, 16]
    w = w.reshape(k // tile, tile, n // tile, tile)
    # [m/16, n/16, 16, 16]
    w = w.permute((0, 2, 1, 3))
    # [m/16, n*16]
    w = w.reshape((k // tile, n * tile))

    # permute the weights
    perm = marlin_weight_perm()
    res = w.reshape((-1, perm.numel()))[:, perm].reshape(w.shape)
    # [m/16, n*16/8]
    q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
    res = res.cpu().numpy().astype(np.uint32)
    # pack 8 bytes into 32-bit integers
    for i in range(8):
        q |= res[:, i::8] << 4 * i
    return torch.from_numpy(q.astype(np.int32))


# permute the scales
def permute_marlin_scales(s, groupsize=-1):
    _, n = s.shape
    perm, perm_single = marlin_scales_perm()
    if groupsize != -1:
        s = s.reshape((-1, len(perm)))[:, perm]
    else:
        s = s.reshape((-1, len(perm_single)))[:, perm_single]
    return s.reshape((-1, n))


def gen_marlin_weights(k, n, groupsize=-1, device="cpu"):
    # generate random weights
    w = torch.randn((k, n), dtype=torch.half, device=device)

    # quantize weights
    w_ref, q_w, s, _, _ = quantize_weights(w, num_bits=4, group_size=groupsize)

    # pack weights to marlin format
    q_w = pack_marlin_weights(q_w).to(device)
    # permute scales
    s = permute_marlin_scales(s, groupsize).to(device)
    return w_ref, q_w, s


if __name__ == "__main__":
    w, q, s = gen_marlin_weights(16, 64)
    perm, perm_single = marlin_scales_perm()
