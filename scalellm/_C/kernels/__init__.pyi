import torch

# Defined in csrc/kernels.cpp
def add_test(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...

# marlin gemm kernel
def marlin_fp16_int4_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    s: torch.Tensor,
    workspace: torch.Tensor,
    thread_k: int = -1,
    thread_n: int = -1,
    thread_m: int = -1,
    num_threads: int = 8,
) -> None: ...
