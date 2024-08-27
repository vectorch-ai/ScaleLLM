import sys

import pytest
import torch
from quant_utils import (pack_awq_weights, pack_gptq_weights,
                         pack_marlin_weights, quantize_weights, sort_rows)

import scalellm._C.kernels as kernels  # type: ignore


@pytest.mark.parametrize("k", [128, 256])
@pytest.mark.parametrize("n", [64, 128, 256])
@pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
@pytest.mark.parametrize("act_order", [False, True])
def test_gptq_repack(k, n, num_bits, group_size, act_order):
    # generate random weights
    w = torch.randn((k, n), dtype=torch.half, device="cuda")

    # quantize weights
    w_ref, q_w, s, g_idx, perm = quantize_weights(
        w, num_bits=num_bits, group_size=group_size, act_order=act_order
    )

    # pack weights to gptq format then repack to marlin format
    # (k/pack_factor, n)
    gptq_q_w = pack_gptq_weights(q_w, num_bits=num_bits)
    if act_order:
        q_w, g_idx, perm = sort_rows(q_w, g_idx)
    else:
        perm = torch.empty(0, dtype=torch.int32, device="cuda")

    # (k/tile, n*tile/pack_factor)
    pack_factor = 32 // num_bits
    marlin_out = torch.empty(
        k // 16, n * 16 // pack_factor, dtype=torch.int32, device="cuda"
    )
    kernels.marlin_gptq_repack(
        gptq_q_w,
        perm=perm,
        out=marlin_out,
        num_bits=num_bits,
    )
    torch.cuda.synchronize()

    # pack weights to marlin format
    # (k/16, n*16/pack_factor)
    marlin_q_w = pack_marlin_weights(q_w, num_bits=num_bits)
    assert torch.equal(marlin_q_w, marlin_out)


@pytest.mark.parametrize("k", [128, 256])
@pytest.mark.parametrize("n", [64, 128, 256])
@pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
def test_awq_repack(k, n, num_bits, group_size):
    # generate random weights
    w = torch.randn((k, n), dtype=torch.half, device="cuda")

    # quantize weights
    w_ref, q_w, s, g_idx, perm = quantize_weights(
        w, num_bits=num_bits, group_size=group_size, act_order=False
    )

    # pack weights to gptq format then repack to marlin format
    # (k/pack_factor, n)
    awq_q_w = pack_awq_weights(q_w, num_bits=num_bits)

    # (k/tile, n*tile/pack_factor)
    pack_factor = 32 // num_bits
    marlin_out = torch.empty(
        k // 16, n * 16 // pack_factor, dtype=torch.int32, device="cuda"
    )
    kernels.marlin_awq_repack(
        awq_q_w,
        out=marlin_out,
        num_bits=num_bits,
    )
    torch.cuda.synchronize()

    # pack weights to marlin format
    # (k/16, n*16/pack_factor)
    marlin_q_w = pack_marlin_weights(q_w, num_bits=num_bits)
    assert torch.equal(marlin_q_w, marlin_out)


if __name__ == "__main__":
    pytest.main(sys.argv)
