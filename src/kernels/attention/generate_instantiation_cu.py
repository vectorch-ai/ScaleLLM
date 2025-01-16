#!/usr/bin/env python3

# This file is run to generate the kernel instantiations for the attention kernels
# They are written to several files in order to speed up compilation
import shutil
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List

DTYPE_MAP = {
    "fp16": "cute::half_t",
    "bf16": "cute::bfloat16_t",
}

SM = [80]  # Sm80 kernels support up to
HEAD_DIMENSIONS = [64, 128, 256]
PAGEDKV_KERNEL_IMPL_TEMPLATE = """
#include "attention_launch_sm80.cuh" // IWYU pragma: keep

namespace llm {{

using Params = PagedKVAttentionParams;
template void run_attention_kernel_sm80<{DTYPE}, {HEAD_DIM}, Params>(
    const Params& params, cudaStream_t stream);

}}  // namespace llm
"""


@dataclass
class Kernel:
    sm: int
    dtype: str
    head_dim: int

    @property
    def template(self) -> str:
        return PAGEDKV_KERNEL_IMPL_TEMPLATE.format(
            DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim
        )

    @property
    def filename(self) -> str:
        return f"attention_{self.dtype}_hdim{self.head_dim}_sm{self.sm}.cu"


def get_all_kernels() -> List[Kernel]:
    for dtype, head_dim, sm in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS, SM):
        yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim)


def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
    (autogen_dir / kernel.filename).write_text(kernel.template)


if __name__ == "__main__":
    output_dir = Path.cwd() / "generated"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for kernel in get_all_kernels():
        write_kernel(kernel, output_dir)