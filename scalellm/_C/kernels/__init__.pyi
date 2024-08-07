import torch

# Defined in csrc/kernels.cpp
def add_test(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
