import sys

import pytest
import torch
from quant_utils import (pack_marlin_weights, permute_marlin_scales,
                         quantize_weights, sort_rows)

import scalellm._C.kernels as kernels  # type: ignore


@pytest.mark.skip(reason="Only works for Ampere")
@pytest.mark.parametrize("m", [16, 32])
@pytest.mark.parametrize("n", [512])
@pytest.mark.parametrize("k", [64, 128, 192])
@pytest.mark.parametrize("num_bits", [4])
@pytest.mark.parametrize("group_size", [-1])
@pytest.mark.parametrize("device", ["cuda"])
def test_marlin_fp16_int4_gemm(m, n, k, num_bits, group_size, device):
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
    s = permute_marlin_scales(s)

    # B_ref, B, s = gen_marlin_weights(k, n, num_bits=num_bits, device=device)
    C = torch.zeros((m, n), dtype=torch.half, device=device)

    kernels.marlin_fp16_int4_gemm(
        A=A, B=B, C=C, s=s, workspace=workspace, thread_k=thread_k, thread_n=thread_n
    )
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B_ref)
    max_diff = torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref))
    assert max_diff < 0.001


@pytest.mark.parametrize("m", [16, 32, 64])
@pytest.mark.parametrize("n", [64, 128, 256, 512])
@pytest.mark.parametrize("k", [128, 256])
@pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("is_k_full", [False, True])
@pytest.mark.parametrize("use_fp32_reduce", [False, True])
def test_marlin_gemm(
    m, n, k, num_bits, group_size, act_order, is_k_full, use_fp32_reduce
):
    if act_order and (group_size == -1 or group_size == k):
        # act_order=True requires group_size < k
        return

    # generate random inputs and weights
    a = torch.randn((m, k), dtype=torch.half, device="cuda")
    w = torch.randn((k, n), dtype=torch.half, device="cuda")

    # quantize weights
    w_ref, q_w, s, g_idx, _ = quantize_weights(
        w, num_bits=num_bits, group_size=group_size, act_order=act_order
    )

    # permute weights based on group index in ascending order
    if act_order:
        q_w, g_idx, perm = sort_rows(q_w, g_idx)
    else:
        perm = torch.empty(0, dtype=torch.int32, device="cuda")

    # pack weights to marlin format: (k/16, n*16/pack_factor)
    marlin_q_w = pack_marlin_weights(q_w, num_bits=num_bits)
    # permute scales: (n_group, n)
    marlin_s = permute_marlin_scales(s)
    marlin_zp = torch.empty(0, dtype=torch.int32, device="cuda")

    workspace = torch.zeros(n // 64 * 16, dtype=torch.int32, device="cuda")
    output = torch.empty((m, n), dtype=torch.half, device="cuda")

    kernels.marlin_gemm(
        A=a,
        B=marlin_q_w,
        C=output,
        scales=marlin_s,
        zeros=marlin_zp,
        g_idx=g_idx,
        perm=perm,
        workspace=workspace,
        num_bits=num_bits,
        is_k_full=is_k_full,
        has_zp=False,  # TODO: test with zero point
        use_fp32_reduce=use_fp32_reduce,
    )
    torch.cuda.synchronize()

    output_ref = torch.matmul(a, w_ref)

    max_diff = torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref)
    )
    assert max_diff < 0.001


if __name__ == "__main__":
    pytest.main(sys.argv)
    # test_marlin_gemm(64, 64, 128, 4, 128, False, False, False)
