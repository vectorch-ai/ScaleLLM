#!/usr/bin/env python3

# This file is run to generate the kernel instantiations for the attention kernels
import shutil
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DTYPE_MAP = {
    "fp16": "cute::half_t",
    "bf16": "cute::bfloat16_t",
}

# TODO: add support for mixed precision kernels
KV_DTYPE_MAP = {
}

HEAD_DIMENSIONS = [64, 96, 128, 256]

PAGEDKV_KERNEL_IMPL_TEMPLATE = """
#include "attention_launch_sm80.cuh" // IWYU pragma: keep

namespace llm {{

using Params = PagedKVAttentionParams;
template void run_attention_kernel_sm80<{DTYPE}, {KV_DTYPE}, {HEAD_DIM}, Params>(
    Params& params, cudaStream_t stream);

}}  // namespace llm
"""

@dataclass
class Kernel:
    dtype: str
    kv_dtype: str
    head_dim: int

    @property
    def template(self) -> str:
        return PAGEDKV_KERNEL_IMPL_TEMPLATE.format(
            DTYPE=DTYPE_MAP[self.dtype], KV_DTYPE=DTYPE_MAP[self.kv_dtype], HEAD_DIM=self.head_dim
        )

    @property
    def filename(self) -> str:
        if self.dtype == self.kv_dtype:
            return f"attention_{self.dtype}_hd{self.head_dim}_sm80.cu"
        # include the kv dtype in the filename
        return f"attention_{self.dtype}_{self.kv_dtype}_hd{self.head_dim}_sm80.cu"


def get_all_kernels() -> Iterator[Kernel]:
    # fp16 and bf16 kernels
    for dtype, head_dim in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS):
        yield Kernel(dtype=dtype, kv_dtype=dtype, head_dim=head_dim)
    
    # mixed precision kernels
    for dtype, kv_dtype, head_dim in itertools.product(DTYPE_MAP.keys(), KV_DTYPE_MAP.keys(), HEAD_DIMENSIONS):
        yield Kernel(dtype=dtype, kv_dtype=kv_dtype, head_dim=head_dim)


if __name__ == "__main__":
    output_dir = Path.cwd() / "generated"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # written to several files to speed up compilation
    for kernel in get_all_kernels():
        (output_dir / kernel.filename).write_text(kernel.template)