#!/usr/bin/env python3

# Adapted from https://github.com/flashinfer-ai/flashinfer/
# This script generates the instantiation files for the flashinfer kernels

import shutil
import pathlib
import itertools

kv_layout_literal = {
    0: "QKVLayout::kNHD",
    1: "QKVLayout::kHND",
}

pos_encoding_mode_literal = {
    0: "PosEncodingMode::kNone",
    1: "PosEncodingMode::kRoPELlama",
    2: "PosEncodingMode::kALiBi",
}

dtype_literal = {
    "f16": "half",
    "bf16": "nv_bfloat16",
    "e4m3": "__nv_fp8_e4m3",
    "e5m2": "__nv_fp8_e5m2",
}

idtype_literal = {
    "i32": "int32_t",
    "u32": "uint32_t",
    "i64": "int64_t",
    "u64": "uint64_t",
}

root = pathlib.Path.cwd()

def get_single_prefill_inst_str(
    group_size,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    allow_fp16_qk_reduction,
    causal,
    dtype_in,
    dtype_out,
):

    content = """#include <flashinfer/attention_impl.cuh>

namespace flashinfer {{

template cudaError_t SinglePrefillWithKVCacheDispatched<{group_size}, {head_dim}, {kv_layout}, {pos_encoding_mode}, {allow_fp16_qk_reduction}, {causal}, {dtype_in}, {dtype_out}>(
    {dtype_in}* q, {dtype_in}* k, {dtype_in}* v, {dtype_out}* o,
    float* tmp, float* lse, uint32_t num_kv_heads, uint32_t qo_len, uint32_t kv_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

}}
    """.format(
        kv_layout=kv_layout_literal[int(kv_layout)],
        group_size=group_size,
        head_dim=head_dim,
        pos_encoding_mode=pos_encoding_mode_literal[int(pos_encoding_mode)],
        allow_fp16_qk_reduction=allow_fp16_qk_reduction,
        causal=causal,
        dtype_in=dtype_literal[dtype_in],
        dtype_out=dtype_literal[dtype_out],
    )
    return content


def get_batch_paged_prefill_inst_str(
    group_size,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    allow_fp16_qk_reduction,
    causal,
    dtype_in,
    dtype_out,
    idtype,
    page_size_choices=[1, 8, 16, 32],
):
    num_frags_x_choices = [1, 2]
    insts = "\n".join(
        [
            """template cudaError_t BatchPrefillWithPagedKVCacheDispatched<page_storage, {kv_layout}, {num_frags_x}, {page_size}, {group_size}, {head_dim}, {pos_encoding_mode}, {allow_fp16_qk_reduction}, {causal}, {dtype_in}, {dtype_out}, {idtype}>(
    {dtype_in}* q, {idtype}* request_indices, {idtype}* tile_indices,
    {idtype}* qo_indptr, {idtype}* q_offset,
    paged_kv_t<page_storage, {kv_layout}, {dtype_in}, {idtype}> paged_kv,
    {dtype_out}* o, float* tmp, float* lse,
    uint32_t num_qo_tiles,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);
    """.format(
                kv_layout=kv_layout_literal[int(kv_layout)],
                num_frags_x=num_frags_x,
                page_size=page_size,
                group_size=group_size,
                head_dim=head_dim,
                pos_encoding_mode=pos_encoding_mode_literal[int(pos_encoding_mode)],
                allow_fp16_qk_reduction=allow_fp16_qk_reduction,
                causal=causal,
                dtype_in=dtype_literal[dtype_in],
                dtype_out=dtype_literal[dtype_out],
                idtype=idtype_literal[idtype],
            )
            for num_frags_x, page_size in itertools.product(
                num_frags_x_choices,
                page_size_choices,
            )
        ]
    )

    content = f"""#include <flashinfer/attention_impl.cuh>

namespace flashinfer {{

constexpr PageStorage page_storage = PageStorage::kIndices;

{insts}

}}"""
    return content


def get_single_decode_inst_str(
    group_size, head_dim, kv_layout, pos_encoding_mode, dtype_in, dtype_out
):
    content = """#include <flashinfer/attention_impl.cuh>

namespace flashinfer {{

template cudaError_t SingleDecodeWithKVCacheDispatched<{group_size}, {head_dim}, {kv_layout}, {pos_encoding_mode}, {dtype_in}, {dtype_out}>(
    {dtype_in}* q, {dtype_in}* k, {dtype_in}* v, {dtype_out}* o,
    {dtype_out}* tmp, uint32_t num_kv_heads, uint32_t seq_len,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

}}
    """.format(
        kv_layout=kv_layout_literal[int(kv_layout)],
        group_size=group_size,
        head_dim=head_dim,
        pos_encoding_mode=pos_encoding_mode_literal[int(pos_encoding_mode)],
        dtype_in=dtype_literal[dtype_in],
        dtype_out=dtype_literal[dtype_out],
    )
    return content


def get_batch_paged_decode_inst_str(
    group_size, head_dim, kv_layout, pos_encoding_mode, dtype_in, dtype_out, idtype
):
    content = """#include <flashinfer/attention_impl.cuh>

namespace flashinfer {{

constexpr PageStorage page_storage = PageStorage::kIndices;

template cudaError_t BatchDecodeWithPagedKVCacheDispatched<{group_size}, {head_dim}, page_storage, {kv_layout}, {pos_encoding_mode}, {dtype_in}, {dtype_out}, {idtype}>(
    {dtype_in}* q, {idtype}* q_offset,
    paged_kv_t<page_storage, {kv_layout}, {dtype_in}, {idtype}> paged_kv,
    kv_partition_info_t<{idtype}> kv_partition_info,
    {dtype_out}* o, {dtype_out}* tmp, float* lse,
    float sm_scale, float rope_scale,
    float rope_theta, cudaStream_t stream);

}}
    """.format(
        kv_layout=kv_layout_literal[int(kv_layout)],
        group_size=group_size,
        head_dim=head_dim,
        pos_encoding_mode=pos_encoding_mode_literal[int(pos_encoding_mode)],
        dtype_in=dtype_literal[dtype_in],
        dtype_out=dtype_literal[dtype_out],
        idtype=idtype_literal[idtype],
    )
    return content


def write_if_different(path: pathlib.Path, content: str) -> None:
    if path.exists():
        with open(path, "r") as f:
            if f.read() == content:
                return
    with open(path, "w") as f:
        f.write(content)

def generate_instantiation_cu(group_sizes, 
                              head_dims, 
                              enable_bf16, 
                              enable_fp8,
                              causal_options, 
                              allow_fp16_qk_reduction_options, 
                              kv_layouts, 
                              pos_encoding_modes):
    prefix = "generated"
    shutil.rmtree(root / prefix, ignore_errors=True)
    (root / prefix).mkdir(parents=True, exist_ok=True)

    # dispatch.inc
    path = root / prefix / "dispatch.inc"
    if not path.exists():
        with open(root / prefix / "dispatch.inc", "w") as f:
            f.write("#define _DISPATCH_CASES_group_size(...)      \\\n")
            for x in group_sizes:
                f.write(f"  _DISPATCH_CASE({x}, GROUP_SIZE, __VA_ARGS__) \\\n")
            f.write("// EOL\n")

            f.write("#define _DISPATCH_CASES_head_dim(...)        \\\n")
            for x in head_dims:
                f.write(f"  _DISPATCH_CASE({x}, HEAD_DIM, __VA_ARGS__) \\\n")
            f.write("// EOL\n")
            f.write("\n")

    idtypes = ["i32"]
    prefill_dtypes = ["f16"]
    if enable_bf16:
        prefill_dtypes.append("bf16")
    decode_dtypes = ["f16", "bf16"]
    fp8_dtypes = []
    if enable_fp8:
        fp8_dtypes = ["e4m3", "e5m2"]

    files = []
    # single decode files
    for (
        group_size,
        head_dim,
        kv_layout,
        pos_encoding_mode,
    ) in itertools.product(
        group_sizes,
        head_dims,
        kv_layouts,
        pos_encoding_modes,
    ):
        for dtype in decode_dtypes:
            fname = f"single_decode_group_{group_size}_head_{head_dim}_layout_{kv_layout}_posenc_{pos_encoding_mode}_dtypein_{dtype}_dtypeout_{dtype}.cu"
            files.append(prefix + "/" + fname)
            content = get_single_decode_inst_str(
                group_size,
                head_dim,
                kv_layout,
                pos_encoding_mode,
                dtype,
                dtype,
            )
            write_if_different(root / prefix / fname, content)

        for dtype_in in fp8_dtypes:
            dtype_out = "f16"
            fname = f"single_decode_group_{group_size}_head_{head_dim}_layout_{kv_layout}_posenc_{pos_encoding_mode}_dtypein_{dtype_in}_dtypeout_{dtype_out}.cu"
            files.append(prefix + "/" + fname)
            content = get_single_decode_inst_str(
                group_size,
                head_dim,
                kv_layout,
                pos_encoding_mode,
                dtype_in,
                dtype_out,
            )
            write_if_different(root / prefix / fname, content)

    # batch decode files
    for (
        group_size,
        head_dim,
        kv_layout,
        pos_encoding_mode,
    ) in itertools.product(
        group_sizes,
        head_dims,
        kv_layouts,
        pos_encoding_modes,
    ):
        for idtype in idtypes:
            for dtype in decode_dtypes:
                fname = f"batch_paged_decode_group_{group_size}_head_{head_dim}_layout_{kv_layout}_posenc_{pos_encoding_mode}_dtypein_{dtype}_dtypeout_{dtype}_idtype_{idtype}.cu"
                files.append(prefix + "/" + fname)
                content = get_batch_paged_decode_inst_str(
                    group_size,
                    head_dim,
                    kv_layout,
                    pos_encoding_mode,
                    dtype,
                    dtype,
                    idtype,
                )
                write_if_different(root / prefix / fname, content)

            for dtype_in in fp8_dtypes:
                dtype_out = "f16"
                fname = f"batch_paged_decode_group_{group_size}_head_{head_dim}_layout_{kv_layout}_posenc_{pos_encoding_mode}_dtypein_{dtype_in}_dtypeout_{dtype_out}_idtype_{idtype}.cu"
                files.append(prefix + "/" + fname)
                content = get_batch_paged_decode_inst_str(
                    group_size,
                    head_dim,
                    kv_layout,
                    pos_encoding_mode,
                    dtype_in,
                    dtype_out,
                    idtype,
                )
                write_if_different(root / prefix / fname, content)

    # single prefill files
    for (
        group_size,
        head_dim,
        kv_layout,
        pos_encoding_mode,
        allow_fp16_qk_reduction,
        causal,
    ) in itertools.product(
        group_sizes,
        head_dims,
        kv_layouts,
        pos_encoding_modes,
        allow_fp16_qk_reduction_options,
        causal_options,
    ):
        for dtype in prefill_dtypes:
            fname = f"single_prefill_group_{group_size}_head_{head_dim}_layout_{kv_layout}_posenc_{pos_encoding_mode}_fp16qkred_{allow_fp16_qk_reduction}_causal_{causal}_dtypein_{dtype}_dtypeout_{dtype}.cu"
            files.append(prefix + "/" + fname)
            content = get_single_prefill_inst_str(
                group_size,
                head_dim,
                kv_layout,
                pos_encoding_mode,
                allow_fp16_qk_reduction,
                causal,
                dtype,
                dtype,
            )
            write_if_different(root / prefix / fname, content)

    # batch paged prefill files
    for (
        group_size,
        head_dim,
        kv_layout,
        pos_encoding_mode,
        allow_fp16_qk_reduction,
        causal,
        idtype,
    ) in itertools.product(
        group_sizes,
        head_dims,
        kv_layouts,
        pos_encoding_modes,
        allow_fp16_qk_reduction_options,
        causal_options,
        idtypes,
    ):
        for dtype in prefill_dtypes:
            fname = f"batch_paged_prefill_group_{group_size}_head_{head_dim}_layout_{kv_layout}_posenc_{pos_encoding_mode}_fp16qkred_{allow_fp16_qk_reduction}_causal_{causal}_dtypein_{dtype}_dtypeout_{dtype}_idtype_{idtype}.cu"
            files.append(prefix + "/" + fname)
            content = get_batch_paged_prefill_inst_str(
                group_size,
                head_dim,
                kv_layout,
                pos_encoding_mode,
                allow_fp16_qk_reduction,
                causal,
                dtype,
                dtype,
                idtype,
                page_size_choices=[1, 16, 32],
            )
            write_if_different(root / prefix / fname, content)

if __name__ == "__main__":
    generate_instantiation_cu(group_sizes=[1, 4, 8], 
                              head_dims=[64, 128, 256],
                              enable_bf16=True,
                              enable_fp8=False,
                              causal_options = [1],
                              allow_fp16_qk_reduction_options=[1],
                              kv_layouts=[0], 
                              pos_encoding_modes=[0, 1, 2])
