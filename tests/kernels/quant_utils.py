import numpy as np
import torch


def permute_rows(
    q_w: torch.Tensor,  # quantized weights, int32 (k, n)
    w_ref: torch.Tensor,  # dequantized weights, float16 (k, n)
    g_idx: torch.Tensor,  # group indices (k,)
    perm: torch.Tensor,  # permutation row indices (k,)
):
    assert q_w.shape == w_ref.shape
    assert g_idx.size(0) == q_w.size(0)

    w_ref = w_ref[perm, :].contiguous()
    q_w = q_w[perm, :].contiguous()
    g_idx = g_idx[perm].contiguous()

    return (w_ref, q_w, g_idx)


# sort rows by group index
def sort_rows(
    q_w: torch.Tensor,  # quantized weights, int32 (k, ...)
    g_idx: torch.Tensor,  # group indices (k,)
):
    assert q_w.size(0) == g_idx.size(0)

    perm = torch.argsort(g_idx).to(torch.int32)
    q_w = q_w[perm, :].contiguous()
    g_idx = g_idx[perm].contiguous()
    return (q_w, g_idx, perm)


def quantize_weights(
    w: torch.Tensor,  # unquantized weights, float (k, n)
    num_bits: int,  # number of bits
    group_size: int = -1,  # group size
    act_order: bool = False,  # whether to permute rows
):
    k, n = w.shape

    assert w.is_floating_point(), "w must be float"

    max_q = 2**num_bits - 1

    if group_size != -1:
        # (m, n) -> (m/group_size, group_size, n)
        w = w.reshape((-1, group_size, n))
        # (group_size, m/group_size, n)
        w = w.permute(1, 0, 2)
        # (group_size, m /group_size * n)
        w = w.reshape((group_size, -1))

    # max value of each column
    s = torch.max(torch.abs(w), dim=0, keepdim=True)[0]
    # 2 for symmetric quantization
    s *= 2 / max_q

    q_zero = (max_q + 1) // 2

    # Quantize
    q_w = torch.round(w / s).int()
    q_w += q_zero
    # values are clampped to [0, max_q]
    q_w = torch.clamp(q_w, 0, max_q)

    # Compute ref (dequantized): weights = (qweights - qzeros) * scales
    w_ref = (q_w - q_zero).to(w.dtype) * s

    # Restore original shapes
    if group_size != -1:

        def reshape_w(w):
            w = w.reshape((group_size, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, n)).contiguous()
            return w

        q_w = reshape_w(q_w)
        w_ref = reshape_w(w_ref)

    s = s.reshape((-1, n)).contiguous().to(w.device)

    if act_order:
        if group_size == -1:
            group_size = k
        assert k % group_size == 0
        # permute rows to simulate act_order
        g_idx = torch.arange(k, dtype=torch.int32, device=w.device)
        g_idx //= group_size
        perm = torch.randperm(k)
        w_ref, q_w, g_idx = permute_rows(q_w, w_ref, g_idx, perm)
    else:
        g_idx = torch.empty(0, dtype=torch.int, device=w.device)
        perm = torch.empty(0, dtype=torch.int, device=w.device)

    return (w_ref, q_w, s, g_idx, perm)


# returns the packed weights (k/pack_factor, n)
def pack_rows(
    q_w: torch.Tensor,  # quantized weights, int32 (k, n)
    num_bits: int,  # number of bits
):
    k, n = q_w.shape
    pack_factor = 32 // num_bits
    assert k % pack_factor == 0

    # use numpy for bit manipulation
    w = q_w.cpu().numpy().astype(np.uint32)
    p_w = np.zeros((k // pack_factor, n), dtype=np.uint32)
    for i in range(pack_factor):
        p_w |= w[i::pack_factor, :] << num_bits * i
    # convert back to torch tensor
    return torch.from_numpy(p_w.astype(np.int32)).to(q_w.device)


def unpack_rows(
    packed_q_w: torch.Tensor,  # packed quantized weights, int32 (k // pack_factor, n)
    num_bits: int,  # number of bits
):
    pack_factor = 32 // num_bits
    k, n = packed_q_w.shape
    k *= pack_factor

    # use numpy for bit manipulation
    p_w = packed_q_w.cpu().numpy().astype(np.uint32)
    w = np.zeros((k, n), dtype=np.uint32)

    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        w[i::pack_factor, :] = p_w & mask
        p_w >>= num_bits

    # convert back to torch tensor
    return torch.from_numpy(w.astype(np.int32)).to(packed_q_w.device)


def pack_cols(
    q_w: torch.Tensor,  # quantized weights, int32 (k, n)
    num_bits: int,  # number of bits
):
    k, n = q_w.shape
    pack_factor = 32 // num_bits
    assert n % pack_factor == 0

    # use numpy for bit manipulation
    w = q_w.cpu().numpy().astype(np.uint32)
    p_w = np.zeros((k, n // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        p_w |= w[:, i::pack_factor] << num_bits * i
    # convert back to torch tensor
    return torch.from_numpy(p_w.astype(np.int32)).to(q_w.device)


def unpack_cols(
    packed_q_w: torch.Tensor,  # packed quantized weights, int32 (k, n // pack_factor)
    num_bits: int,  # number of bits
):
    pack_factor = 32 // num_bits
    k, n = packed_q_w.shape
    n *= pack_factor

    # use numpy for bit manipulation
    p_w = packed_q_w.cpu().numpy().astype(np.uint32)
    w = np.zeros((k, n), dtype=np.uint32)

    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        w[:, i::pack_factor] = p_w & mask
        p_w >>= num_bits

    # convert back to torch tensor
    return torch.from_numpy(w.astype(np.int32)).to(packed_q_w.device)


# permute for fast interleaved_numeric_conversion
def fast_conversion_interleave(num_bits):
    if num_bits == 4:
        # [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 2, 4, 6, 1, 3, 5, 7]
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        # [0, 1, 2, 3] -> [0, 2, 1, 3]
        interleave = np.array([0, 2, 1, 3])
    else:
        raise NotImplementedError(f"num_bits={num_bits} not implemented")
    return interleave


def pack_gptq_weights(q_w: torch.Tensor, num_bits: int):
    return pack_rows(q_w, num_bits=num_bits)


def pack_awq_weights(q_w: torch.Tensor, num_bits: int):
    interleave = fast_conversion_interleave(num_bits)
    q_w = q_w.reshape(-1, len(interleave))[:, interleave].reshape(q_w.shape)
    return pack_cols(q_w, num_bits=num_bits)


#################### Helper functions for Marlin ####################
def marlin_weight_perm(num_bits: int):
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
    interleave = fast_conversion_interleave(num_bits)

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    # return as torch tensor
    return torch.from_numpy(perm)


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
    q_w: torch.Tensor,  # quantized weights, int32 (k, n)
    num_bits: int = 4,  # number of bits
    tile: int = 16,  # marlin tile size
):
    device = q_w.device
    k, n = q_w.shape
    assert k % tile == 0
    assert n % tile == 0

    # tile the weights to 16x16 blocks
    # [m/16, 16, n/16, 16]
    q_w = q_w.reshape(k // tile, tile, n // tile, tile)
    # [m/16, n/16, 16, 16]
    q_w = q_w.permute((0, 2, 1, 3))
    # [m/16, n*16]
    q_w = q_w.reshape((k // tile, n * tile))

    # permute the weights
    perm = marlin_weight_perm(num_bits).to(device)
    # [m/16, n*16]
    res = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    # pack the weights using numpy for bit manipulation
    pack_factor = 32 // num_bits
    # [m/16, n*16/pack_factor]
    packed = np.zeros((res.shape[0], res.shape[1] // pack_factor), dtype=np.uint32)
    res = res.cpu().numpy().astype(np.uint32)
    # pack bytes into 32-bit integers
    for i in range(pack_factor):
        packed |= res[:, i::pack_factor] << num_bits * i

    # convert back to torch tensor
    return torch.from_numpy(packed.astype(np.int32)).to(device)


# permute the scales
def permute_marlin_scales(
    s: torch.Tensor,  # scales, float32 (k, n)
):
    n_groups, n = s.shape
    perm, perm_single = marlin_scales_perm()
    if n_groups == 1:
        s = s.reshape((-1, len(perm_single)))[:, perm_single]
    else:
        s = s.reshape((-1, len(perm)))[:, perm]
    return s.reshape((-1, n)).contiguous()


if __name__ == "__main__":
    # test quantize_weights
    num_bits = 4
    w = torch.randn((16, 32), dtype=torch.float32)
    w_ref, q_w, s, g_idx, perm = quantize_weights(
        w, num_bits=num_bits, group_size=8, act_order=True
    )

    q_zero = 2 ** (num_bits - 1)
    # weights = (qweights - qzeros) * scales
    w = (q_w - q_zero) * s[g_idx, :]
    assert torch.equal(w, w_ref)

    # test pack_rows
    p_w = pack_rows(q_w, num_bits=4)
    unpacked_w = unpack_rows(p_w, num_bits=4)
    assert torch.equal(q_w, unpacked_w)

    # test pack_cols
    p_w = pack_cols(q_w, num_bits=4)
    unpacked_w = unpack_cols(p_w, num_bits=4)
    assert torch.equal(q_w, unpacked_w)
