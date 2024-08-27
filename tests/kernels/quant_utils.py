from typing import List

import numpy
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

    s = s.reshape((-1, n)).contiguous()

    if act_order and group_size != -1:
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


def pack_rows(
    q_w: torch.Tensor,  # quantized weights, int32 (k, n)
    num_bits: int,  # number of bits
):
    k, n = q_w.shape
    pack_factor = 32 // num_bits
    assert k % pack_factor == 0

    # use numpy for bit manipulation
    w = q_w.cpu().numpy().astype(numpy.uint32)
    p_w = numpy.zeros((k // pack_factor, n), dtype=numpy.uint32)
    for i in range(pack_factor):
        p_w |= w[i::pack_factor, :] << num_bits * i
    # convert back to torch tensor
    return torch.from_numpy(p_w.astype(numpy.int32)).to(q_w.device)


def unpack_rows(
    packed_q_w: torch.Tensor,  # packed quantized weights, int32 (k // pack_factor, n)
    num_bits: int,  # number of bits
):
    pack_factor = 32 // num_bits
    k, n = packed_q_w.shape
    k *= pack_factor

    # use numpy for bit manipulation
    p_w = packed_q_w.cpu().numpy().astype(numpy.uint32)
    w = numpy.zeros((k, n), dtype=numpy.uint32)

    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        w[i::pack_factor, :] = p_w & mask
        p_w >>= num_bits

    # convert back to torch tensor
    return torch.from_numpy(w.astype(numpy.int32)).to(packed_q_w.device)


def pack_cols(
    q_w: torch.Tensor,  # quantized weights, int32 (k, n)
    num_bits: int,  # number of bits
):
    k, n = q_w.shape
    pack_factor = 32 // num_bits
    assert n % pack_factor == 0

    # use numpy for bit manipulation
    w = q_w.cpu().numpy().astype(numpy.uint32)
    p_w = numpy.zeros((k, n // pack_factor), dtype=numpy.uint32)
    for i in range(pack_factor):
        p_w |= w[:, i::pack_factor] << num_bits * i
    # convert back to torch tensor
    return torch.from_numpy(p_w.astype(numpy.int32)).to(q_w.device)


def unpack_cols(
    packed_q_w: torch.Tensor,  # packed quantized weights, int32 (k, n // pack_factor)
    num_bits: int,  # number of bits
):
    pack_factor = 32 // num_bits
    k, n = packed_q_w.shape
    n *= pack_factor

    # use numpy for bit manipulation
    p_w = packed_q_w.cpu().numpy().astype(numpy.uint32)
    w = numpy.zeros((k, n), dtype=numpy.uint32)

    mask = (1 << num_bits) - 1
    for i in range(pack_factor):
        w[:, i::pack_factor] = p_w & mask
        p_w >>= num_bits

    # convert back to torch tensor
    return torch.from_numpy(w.astype(numpy.int32)).to(packed_q_w.device)


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
    assert torch.allclose(w, w_ref)

    # test pack_rows
    p_w = pack_rows(q_w, num_bits=4)
    unpacked_w = unpack_rows(p_w, num_bits=4)
    assert torch.equal(q_w, unpacked_w)

    # test pack_cols
    p_w = pack_cols(q_w, num_bits=4)
    unpacked_w = unpack_cols(p_w, num_bits=4)
    assert torch.equal(q_w, unpacked_w)
