#!/usr/bin/env python3

# Adapted from https://github.com/flashinfer-ai/flashinfer/
# This script generates the instantiation files for the flashinfer kernels

import shutil
import pathlib
import itertools

root = pathlib.Path.cwd()

def generate_instantiation_cu(group_sizes, head_dims, enable_bf16, causal_options, allow_fp16_qk_reduction_options, layout_options, rotary_mode_options):
    prefix = "generated"
    shutil.rmtree(root / prefix, ignore_errors=True)
    (root / prefix).mkdir(parents=True, exist_ok=True)
    dtypes = {"fp16": "nv_half"}
    if enable_bf16:
        dtypes["bf16"] = "nv_bfloat16"

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

    for (
        group_size,
        head_dim,
        dtype,
        causal,
        allow_fp16_qk_reduction,
        layout,
        rotary_mode,
    ) in itertools.product(
        group_sizes,
        head_dims,
        dtypes,
        causal_options,
        allow_fp16_qk_reduction_options,
        layout_options,
        rotary_mode_options,
    ):
        # paged batch prefill
        fname = f"paged_batch_prefill_group{group_size}_head{head_dim}_causal{causal}_fp16qk{allow_fp16_qk_reduction}_layout{layout}_rotary{rotary_mode}_{dtype}.cu"
        if not (root / prefix / fname).exists():
            with open(root / prefix / fname, "w") as f:
                f.write('#include "../flashinfer_decl.h"\n\n')
                f.write(f"#include <flashinfer.cuh>\n\n")
                f.write(f"using namespace flashinfer;\n\n")
                f.write(
                    "INST_BatchPrefillPagedWrapper({}, {}, {}, {}, {}, {}, {})\n".format(
                        dtypes[dtype],
                        group_size,
                        head_dim,
                        str(causal).lower(),
                        str(allow_fp16_qk_reduction).lower(),
                        "QKVLayout::k" + layout,
                        "RotaryMode::k" + rotary_mode,
                    )
                )

        # ragged batch prefill
        # fname = f"ragged_batch_prefill_group{group_size}_head{head_dim}_causal{causal}_fp16qk{allow_fp16_qk_reduction}_layout{layout}_rotary{rotary_mode}_{dtype}.cu"
        # if not (root / prefix / fname).exists():
        #     with open(root / prefix / fname, "w") as f:
        #         f.write('#include "../flashinfer_decl.h"\n\n')
        #         f.write(f"#include <flashinfer.cuh>\n\n")
        #         f.write(f"using namespace flashinfer;\n\n")
        #         f.write(
        #             "INST_BatchPrefillRaggedWrapper({}, {}, {}, {}, {}, {}, {})\n".format(
        #                 dtypes[dtype],
        #                 group_size,
        #                 head_dim,
        #                 str(causal).lower(),
        #                 str(allow_fp16_qk_reduction).lower(),
        #                 "QKVLayout::k" + layout,
        #                 "RotaryMode::k" + rotary_mode,
        #             )
        #         )

        # single prefill
        fname = f"single_prefill_group{group_size}_head{head_dim}_causal{causal}_fp16qk{allow_fp16_qk_reduction}_layout{layout}_rotary{rotary_mode}_{dtype}.cu"
        if not (root / prefix / fname).exists():
            with open(root / prefix / fname, "w") as f:
                f.write('#include "../flashinfer_decl.h"\n\n')
                f.write(f"#include <flashinfer.cuh>\n\n")
                f.write(f"using namespace flashinfer;\n\n")
                f.write(
                    "INST_SinglePrefill({}, {}, {}, {}, {}, {}, {})\n".format(
                        dtypes[dtype],
                        group_size,
                        head_dim,
                        str(causal).lower(),
                        str(allow_fp16_qk_reduction).lower(),
                        "QKVLayout::k" + layout,
                        "RotaryMode::k" + rotary_mode,
                    )
                )

if __name__ == "__main__":
    generate_instantiation_cu(group_sizes=[1, 4, 8], 
                              head_dims=[64, 128, 256],
                              enable_bf16=True,
                              causal_options = [True],
                              allow_fp16_qk_reduction_options = [True],
                              layout_options = ["NHD"],
                              rotary_mode_options = ["None", "Llama"])