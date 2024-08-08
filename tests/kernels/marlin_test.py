import sys

import pytest
import torch
from marlin_utils import gen_marlin_weights

import scalellm._C.kernels as kernels


@pytest.mark.parametrize("m", [16, 32])
@pytest.mark.parametrize("n", [512])
@pytest.mark.parametrize("k", [64, 128, 192])
def test_fp16_int4_gemm(m, n, k):
    thread_k = 64
    thread_n = 256
    A = torch.randn((m, k), dtype=torch.half, device="cuda")
    workspace = torch.zeros(n // 128 * 16, device="cuda")
    B_ref, B, s = gen_marlin_weights(k, n, device="cuda")
    C = torch.zeros((m, n), dtype=torch.half, device="cuda")
    B_ref = B_ref.to("cuda")
    B = B.to("cuda")
    s = s.to("cuda")

    kernels.fp16_int4_gemm_marlin(
        A=A, B=B, C=C, s=s, workspace=workspace, thread_k=thread_k, thread_n=thread_n
    )
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B_ref)
    assert torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref)) < 0.001


if __name__ == "__main__":
    pytest.main(sys.argv)
