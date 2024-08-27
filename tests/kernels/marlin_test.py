import sys

import pytest
import torch
from quant_utils import (pack_marlin_weights, permute_marlin_scales,
                         quantize_weights)

import scalellm._C.kernels as kernels  # type: ignore


@pytest.mark.parametrize("m", [16, 32])
@pytest.mark.parametrize("n", [512])
@pytest.mark.parametrize("k", [64, 128, 192])
@pytest.mark.parametrize("num_bits", [4])
@pytest.mark.parametrize("group_size", [-1])
@pytest.mark.parametrize("device", ["cuda"])
def test_fp16_int4_gemm(m, n, k, num_bits, group_size, device):
    thread_k = 64
    thread_n = 256
    A = torch.randn((m, k), dtype=torch.half, device=device)
    workspace = torch.zeros(n // 128 * 16, device=device)

    # generate random weights
    w = torch.randn((k, n), dtype=torch.half, device=device)

    # quantize weights
    B_ref, B, s, _, _ = quantize_weights(w, num_bits=num_bits, group_size=group_size)

    # pack weights to marlin format
    B = pack_marlin_weights(B, num_bits=num_bits)
    # permute scales
    s = permute_marlin_scales(s, group_size)

    # B_ref, B, s = gen_marlin_weights(k, n, num_bits=num_bits, device=device)
    C = torch.zeros((m, n), dtype=torch.half, device=device)

    kernels.fp16_int4_gemm_marlin(
        A=A, B=B, C=C, s=s, workspace=workspace, thread_k=thread_k, thread_n=thread_n
    )
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B_ref)
    assert torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref)) < 0.001


if __name__ == "__main__":
    pytest.main(sys.argv)
