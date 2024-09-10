#!/usr/bin/env python3

# This script generates the instantiation files for the flashinfer kernels

import shutil
from itertools import product
from dataclasses import dataclass
from pathlib import Path

MaskModes = {
    "MaskMode::kNone": 0,
    "MaskMode::kCausal": 1,
    "MaskMode::kCustom": 2,
}

LogitsPostHooks = {
    "LogitsPostHook::kNone": 0,
    "LogitsPostHook::kSoftCap": 1,
}

WarpLayouts = {
    "WarpLayout::k4x1x2": 0,
    "WarpLayout::k4x1x1": 1,
    "WarpLayout::k1x4x1": 2,
}

PosEncodingModes = {
    "PosEncodingMode::kNone": 0,
    "PosEncodingMode::kRoPELlama": 1,
    "PosEncodingMode::kALiBi": 2,
}

DTypes = {
    "f16": "half",
    "bf16": "nv_bfloat16",
    "e4m3": "__nv_fp8_e4m3",
    "e5m2": "__nv_fp8_e5m2",
}

IDTypes = {
    "i32": "int32_t",
}

Bools = {
    False: "0",
    True: "1",
}

KERNEL_IMPL_TEMPLATE = """
template cudaError_t mha_varlen_dispatch<{WARP_LAYOUT}, 
                                         {HEAD_DIM}, 
                                         {LOGITS_HOOK}, 
                                         {POS_ENCODING_MODE}, 
                                         {QK_FP16_REDUCTION}, 
                                         {MASK_MODE}, 
                                         {QDType}, 
                                         {KVDType}, 
                                         {DType}, 
                                         {IDType}>(
    {QDType}* q, {IDType}* request_indices, {IDType}* q_tile_indices, {IDType}* kv_tile_indices,
    {IDType}* q_indptr, {IDType}* kv_indptr,
    paged_kv_t<{KVDType}, {IDType}> paged_kv, uint8_t* custom_mask,
    {IDType}* qk_indptr, {IDType}* o_indptr, {DType}* o, {DType}* tmp_v, float* tmp_s, float* lse,
    {IDType}* merge_indptr, bool* block_valid_mask, {IDType}* kv_chunk_size_ptr, uint32_t max_num_rows,
    uint32_t num_qo_heads, uint32_t num_kv_heads, uint32_t padded_batch_size, int32_t window_left,
    float logits_soft_cap, float sm_scale, float* alibi_slopes, cudaStream_t stream);
"""

FILE_TEMPLATE = """#include "attention_kernel.h"

namespace flashinfer {{

{INSTANTIATIONS}

}} // namespace flashinfer
"""


@dataclass
class Kernel:
    warp_layouts: list[str]
    head_dim: int
    logits_hook: str
    pos_encoding_mode: str
    qk_fp16_reduction: bool
    mask_mode: str
    qdtype: str
    kvdtype: str
    idtype: str

    @property
    def template(self) -> str:
        insts = "\n".join(
            [
                KERNEL_IMPL_TEMPLATE.format(
                    WARP_LAYOUT=warp_layout,
                    HEAD_DIM=self.head_dim,
                    LOGITS_HOOK=self.logits_hook,
                    POS_ENCODING_MODE=self.pos_encoding_mode,
                    QK_FP16_REDUCTION="true" if self.qk_fp16_reduction else "false",
                    MASK_MODE=self.mask_mode,
                    QDType=DTypes[self.qdtype],
                    KVDType=DTypes[self.kvdtype],
                    DType=DTypes[self.qdtype],
                    IDType=IDTypes[self.idtype],
                )
                for warp_layout in self.warp_layouts
            ]
        )
        return FILE_TEMPLATE.format(INSTANTIATIONS=insts)

    @property
    def filename(self) -> str:
        return f"mha_h{self.head_dim}_l{LogitsPostHooks[self.logits_hook]}_p{PosEncodingModes[self.pos_encoding_mode]}_r{Bools[self.qk_fp16_reduction]}_m{MaskModes[self.mask_mode]}_{self.qdtype}_{self.kvdtype}_{self.idtype}.cu"


def all_kernels(
    warp_layouts,
    head_dims,
    logits_hooks,
    pos_encoding_modes,
    qk_fp16_reduction_options,
    mask_modes,
    dtypes,
    fp8_dtypes,
    idtypes,
):
    for (
        head_dim,
        logits_hook,
        pos_encoding_mode,
        qk_fp16_reduction,
        mask_mode,
        idtype,
    ) in product(
        head_dims,
        logits_hooks,
        pos_encoding_modes,
        qk_fp16_reduction_options,
        mask_modes,
        idtypes,
    ):
        for qdtype, kvdtype in list(zip(dtypes, dtypes)) + list(
            product(dtypes, fp8_dtypes)
        ):
            yield Kernel(
                warp_layouts=warp_layouts,
                head_dim=head_dim,
                logits_hook=logits_hook,
                pos_encoding_mode=pos_encoding_mode,
                qk_fp16_reduction=qk_fp16_reduction,
                mask_mode=mask_mode,
                qdtype=qdtype,
                kvdtype=kvdtype,
                idtype=idtype,
            )


if __name__ == "__main__":
    output_dir = Path.cwd() / "generated"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for kernel in all_kernels(
        warp_layouts=["WarpLayout::k4x1x2", "WarpLayout::k4x1x1", "WarpLayout::k1x4x1"],
        head_dims=[64, 128, 256],
        logits_hooks=["LogitsPostHook::kNone", "LogitsPostHook::kSoftCap"],
        pos_encoding_modes=["PosEncodingMode::kNone", "PosEncodingMode::kALiBi"],
        qk_fp16_reduction_options=[False],
        mask_modes=["MaskMode::kCausal"],
        dtypes=["f16", "bf16"],
        fp8_dtypes=["e4m3", "e5m2"],
        idtypes=["i32"],
    ):
        (output_dir / kernel.filename).write_text(kernel.template)
