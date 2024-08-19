import numpy as np
import torch

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


def pack_marlin_weights(w):
    m, n = w.shape
    tile = 16
    # tile the weights to 16x16 blocks
    # [m/16, 16, n/16, 16]
    w = w.reshape(m // tile, tile, n // tile, tile)
    # [m/16, n/16, 16, 16]
    w = w.permute((0, 2, 1, 3))
    # [m/16, n*16]
    w = w.reshape((m // tile, n * tile))

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


def permute_marlin_scales(s, groupsize=-1):
    _, n = s.shape
    perm, perm_single = marlin_scales_perm()
    if groupsize != -1:
        s = s.reshape((-1, len(perm)))[:, perm]
    else:
        s = s.reshape((-1, len(perm_single)))[:, perm_single]
    return s.reshape((-1, n))


def gen_marlin_weights(m, n, groupsize=-1, device="cpu"):
    num_bits = 4
    max_q = 2**num_bits - 1
    w = torch.randn((m, n), dtype=torch.half, device=device)
    if groupsize != -1:
        # (m, n) -> (m/groupsize, groupsize, n)
        w = w.reshape((-1, groupsize, n))
        # (groupsize, m/groupsize, n)
        w = w.permute(1, 0, 2)
        # (groupsize, m /groupsize * n)
        w = w.reshape((groupsize, -1))

    # max value of each column
    s = torch.max(torch.abs(w), dim=0, keepdim=True)[0]
    s *= 2 / max_q

    zero = (max_q + 1) // 2
    w = torch.round(w / s).int()
    w += zero
    # values are clampped to [0, 15] for 4 bits
    w = torch.clamp(w, 0, max_q)
    # weights = (qweights - qzeros) * scales
    w_ref = (w - zero).half() * s
    if groupsize != -1:

        def reshape(w):
            # (groupsize, m/groupsize, n)
            w = w.reshape((groupsize, -1, n))
            # (m/groupsize, groupsize, n)
            w = w.permute(1, 0, 2)
            # (m, n)
            w = w.reshape((m, n)).contiguous()
            return w

        w_ref = reshape(w_ref)
        w = reshape(w)

    s = s.reshape((-1, n)).contiguous()

    # pack weights for marlin
    q = pack_marlin_weights(w).to(device)
    s = permute_marlin_scales(s, groupsize).to(device)
    return w_ref, q, s


if __name__ == "__main__":
    # w, q, s = gen_marlin_weights(16, 64)
    perm, perm_single = marlin_scales_perm()
    print(perm)
    print(perm_single)
