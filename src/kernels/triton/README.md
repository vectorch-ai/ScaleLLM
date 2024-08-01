Here's the polished README:

# OpenAI Triton Kernels

This directory contains kernels written in Triton. These kernels can be compiled into self-contained C source code that embeds the 'cubin' data using the Triton ahead-of-time compiler.

## Building Triton Kernels

For example, you can run the following command to compile the `vector_add` kernel for the `float16` data type:

```bash
# Set up environment variables for Triton
export TRITON_ROOT=$(pip show triton | grep Location | cut -d' ' -f2)
export COMPILE=${TRITON_ROOT}/triton/tools/compile.py
export LINK=${TRITON_ROOT}/triton/tools/link.py

# Remove previous generated files
rm -rf aot

# Kernel for data type=float16, BLOCK_SIZE=16
mkdir -p aot/fp16
# compile the kernel and generate headers and 'cubin' files
python ${COMPILE} kernel.py \
    --kernel-name add_kernel \
    --out-path aot/fp16/add_kernel_fp16 \
    --out-name add_kernel_fp16 \
    --signature "*fp16, *fp16, *fp16, i32, 16" \
    --grid "(n_elements + 15) / 16, 1, 1"
# Link generated headers and create dispatchers.
python ${LINK} aot/fp16/*.h --out aot/add_kernel_fp16
```

The generated files will be located in the `aot` directory. You should find three functions:
* `load_vector_add_{fp16|fp32}` to load the GPU kernel
* `unload_vector_add_{fp16|fp32}` to unload the GPU kernel
* `vector_add_{fp16|fp32}` to launch the kernel