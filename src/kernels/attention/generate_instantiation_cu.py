#!/usr/bin/env python3

# This file is run to generate the kernel instantiations for the attention kernels
import shutil
import itertools
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


MHA_KERNEL_TEMPLATE = """
#include "mha_kernel_sm80.cuh"  // IWYU pragma: export
#include "mha_params.h"         // IWYU pragma: export
#include "mha_traits_sm80.h"    // IWYU pragma: export

namespace llm {{

using Traits = MHATraitsSM80<{DTYPE}, {HEAD_DIM}, {BLK_M}, {BLK_N}, {BLK_K}>;
using Params = MHAPagedKVParams;

template void launch_mha_kernel_sm80<Traits,
                                     Params,
                                     /*EVEN_K=*/{EVEN_K},
                                     /*ALIBI=*/{ALIBI},
                                     /*SOFT_CAP=*/{SOFT_CAP},
                                     /*LOCAL=*/{LOCAL}>(const Params& params, 
                                                        cudaStream_t stream);
}}  // namespace llm
"""


@dataclass
class MHAKernel:
    dtype: str
    head_dim: int
    blk_m: int
    blk_n: int
    blk_k: int
    even_k: bool
    alibi: bool
    soft_cap: bool
    local: bool

    @property
    def template(self) -> str:
        assert self.head_dim % self.blk_k == 0

        return MHA_KERNEL_TEMPLATE.format(
            DTYPE=DTYPE_MAP[self.dtype],
            HEAD_DIM=self.head_dim,
            BLK_M=self.blk_m,
            BLK_N=self.blk_n,
            BLK_K=self.blk_k,
            EVEN_K=BOOL_MAP[self.even_k],
            ALIBI=BOOL_MAP[self.alibi],
            SOFT_CAP=BOOL_MAP[self.soft_cap],
            LOCAL=BOOL_MAP[self.local],
        )

    @property
    def filename(self) -> str:
        def to_str(val: bool) -> str:
            return "1" if val else "0"

        return f"mha_{self.dtype}_hd{self.head_dim}_m{self.blk_m}_n{self.blk_n}_k{self.blk_k}_ek{to_str(self.even_k)}_al{to_str(self.alibi)}_sc{to_str(self.soft_cap)}_lc{to_str(self.local)}.cu"


def gen_all_kernels() -> Iterator[MHAKernel]:
    # mha kernel instantiations
    for (
        dtype,
        head_dim,
        blk_m,
        blk_n,
        blk_k,
        even_k,
        alibi,
        soft_cap,
        local,
    ) in itertools.product(
        ["fp16", "bf16"],  # dtype
        [64, 96, 128, 256],  # head_dim
        [64],  # blk_m
        [64],  # blk_n
        [32, 64],  # blk_k
        [False, True],  # even_k
        [False, True],  # alibi
        [False, True],  # soft_cap
        [False, True],  # local
    ):
        # skip invalid configurations
        if head_dim % blk_k != 0:
            continue
        yield MHAKernel(
            dtype=dtype,
            head_dim=head_dim,
            blk_m=blk_m,
            blk_n=blk_n,
            blk_k=blk_k,
            even_k=even_k,
            alibi=alibi,
            soft_cap=soft_cap,
            local=local,
        )


if __name__ == "__main__":
    output_dir = Path.cwd() / "generated"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # written to several files to speed up compilation
    for kernel in gen_all_kernels():
        (output_dir / kernel.filename).write_text(kernel.template)
