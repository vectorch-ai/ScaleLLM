#!/usr/bin/env python3

# This file is run to generate the kernel instantiations for the attention kernels
import itertools
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# map from python to c++ types
DTYPE_MAP = {
    "fp16": "cute::half_t",
    "bf16": "cute::bfloat16_t",
}

BOOL_MAP = {
    False: "false",
    True: "true",
}


SM80_MHA_KERNEL_TEMPLATE = """
#include "device/sm80_mha_launch.cuh"   // IWYU pragma: export
#include "mha_params.h"                 // IWYU pragma: export

namespace llm {{

using Params = MHAPagedKVParams;

template void sm80_launch_mha_kernel</*DTYPE=*/{DTYPE},
                                     /*HEAD_DIM=*/{HEAD_DIM},
                                     /*EVEN_K=*/{EVEN_K},
                                     /*ALIBI=*/{ALIBI},
                                     /*SOFT_CAP=*/{SOFT_CAP},
                                     /*LOCAL=*/{LOCAL},
                                     Params>(const Params& params,
                                             cudaStream_t stream);
}}  // namespace llm
"""

SM120_MHA_KERNEL_TEMPLATE = """
#include "device/sm120_fmha_launch.cuh" // IWYU pragma: export
#include "mha_params.h"                 // IWYU pragma: export

namespace llm {{

using Params = MHAPagedKVParams;

template void sm120_launch_mha_kernel</*DTYPE=*/{DTYPE},
                                     /*HEAD_DIM=*/{HEAD_DIM},
                                     /*EVEN_K=*/{EVEN_K},
                                     /*ALIBI=*/{ALIBI},
                                     /*SOFT_CAP=*/{SOFT_CAP},
                                     /*LOCAL=*/{LOCAL},
                                     Params>(const Params& params,
                                             cudaStream_t stream);
}}  // namespace llm
"""

MLA_KERNEL_TEMPLATE = """
#include "device/sm80_mla_launch.cuh"   // IWYU pragma: export
#include "mla_params.h"                 // IWYU pragma: export

namespace llm {{

using Params = MLAPagedKVParams;

template void sm80_launch_mla_kernel</*DTYPE=*/{DTYPE},
                                     /*HEAD_DIM=*/{HEAD_DIM},
                                     /*ROPE_HEAD_DIM=*/{ROPE_HEAD_DIM},
                                     Params>(const Params& params,
                                             cudaStream_t stream);
}}  // namespace llm
"""


@dataclass
class SM80MHAKernel:
    dtype: str
    head_dim: int
    even_k: bool
    alibi: bool
    soft_cap: bool
    local: bool

    @property
    def template(self) -> str:
        return SM80_MHA_KERNEL_TEMPLATE.format(
            DTYPE=DTYPE_MAP[self.dtype],
            HEAD_DIM=self.head_dim,
            EVEN_K=BOOL_MAP[self.even_k],
            ALIBI=BOOL_MAP[self.alibi],
            SOFT_CAP=BOOL_MAP[self.soft_cap],
            LOCAL=BOOL_MAP[self.local],
        )

    @property
    def filename(self) -> str:
        def to_str(val: bool) -> str:
            return "1" if val else "0"

        return f"sm80_mha_{self.dtype}_hd{self.head_dim}_ek{to_str(self.even_k)}_al{to_str(self.alibi)}_sc{to_str(self.soft_cap)}_lc{to_str(self.local)}.cu"

@dataclass
class SM120MHAKernel:
    dtype: str
    head_dim: int
    even_k: bool
    alibi: bool
    soft_cap: bool
    local: bool

    @property
    def template(self) -> str:
        return SM120_MHA_KERNEL_TEMPLATE.format(
            DTYPE=DTYPE_MAP[self.dtype],
            HEAD_DIM=self.head_dim,
            EVEN_K=BOOL_MAP[self.even_k],
            ALIBI=BOOL_MAP[self.alibi],
            SOFT_CAP=BOOL_MAP[self.soft_cap],
            LOCAL=BOOL_MAP[self.local],
        )

    @property
    def filename(self) -> str:
        def to_str(val: bool) -> str:
            return "1" if val else "0"

        return f"sm120_fmha_{self.dtype}_hd{self.head_dim}_ek{to_str(self.even_k)}_al{to_str(self.alibi)}_sc{to_str(self.soft_cap)}_lc{to_str(self.local)}.cu"


@dataclass
class MLAKernel:
    dtype: str
    head_dim: int
    rope_head_dim: int

    @property
    def template(self) -> str:
        return MLA_KERNEL_TEMPLATE.format(
            DTYPE=DTYPE_MAP[self.dtype],
            HEAD_DIM=self.head_dim,
            ROPE_HEAD_DIM=self.rope_head_dim,
        )

    @property
    def filename(self) -> str:
        return f"sm80_mla_{self.dtype}_hd{self.head_dim}_rhd{self.rope_head_dim}.cu"


def gen_sm80_mha_kernels() -> Iterator[SM80MHAKernel]:
    # mha kernel instantiations
    for (
        dtype,
        head_dim,
        even_k,
        alibi,
        soft_cap,
        local,
    ) in itertools.product(
        ["fp16", "bf16"],  # dtype
        [64, 96, 128, 256],  # head_dim
        [False, True],  # even_k
        [False, True],  # alibi
        [False, True],  # soft_cap
        [False, True],  # local
    ):
        yield SM80MHAKernel(
            dtype=dtype,
            head_dim=head_dim,
            even_k=even_k,
            alibi=alibi,
            soft_cap=soft_cap,
            local=local,
        )

def gen_sm120_fmha_kernels() -> Iterator[SM120MHAKernel]:
    # mha kernel instantiations
    for (
        dtype,
        head_dim,
        even_k,
        alibi,
        soft_cap,
        local,
    ) in itertools.product(
        ["fp16"],  # dtype
        [64],  # head_dim
        [False, True],  # even_k
        [False, True],  # alibi
        [False, True],  # soft_cap
        [False, True],  # local
    ):
        yield SM120MHAKernel(
            dtype=dtype,
            head_dim=head_dim,
            even_k=even_k,
            alibi=alibi,
            soft_cap=soft_cap,
            local=local,
        )


def gen_mla_kernels() -> Iterator[MLAKernel]:
    # TODO: choose BLK_M, BLK_N, BLK_K, STAGES based on compute capability
    # mla kernel instantiations
    for dtype, head_dim, rope_head_dim in itertools.product(
        ["fp16", "bf16"],  # dtype
        [512],  # head_dim
        [64],  # rope_head_dim
    ):
        yield MLAKernel(
            dtype=dtype,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
        )


if __name__ == "__main__":
    output_dir = Path.cwd() / "gensrc"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # written to several files to speed up compilation
    for kernel in gen_sm80_mha_kernels():
        (output_dir / kernel.filename).write_text(kernel.template)

    for kernel in gen_sm120_fmha_kernels():
        (output_dir / kernel.filename).write_text(kernel.template)

    for kernel in gen_mla_kernels():
        (output_dir / kernel.filename).write_text(kernel.template)
