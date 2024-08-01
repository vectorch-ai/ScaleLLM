# OpenAI Triton Kernels

This directory contains kernels written in Triton. These kernels can be compiled into self-contained C source code that embeds the 'cubin' data using the Triton ahead-of-time compiler.

## How to compile Triton Kernels

For example, you can run the following command to compile 'add_kernel' in example folder for the `float16` data type:

```bash
# Kernel for data type=float16, BLOCK_SIZE=16
mkdir -p aot/fp16

# Compile the kernel and generate headers and 'cubin' files
python -m triton.tools.compile kernel.py \
    --kernel-name add_kernel \
    --out-path aot/fp16/add_kernel_fp16 \
    --out-name add_kernel_fp16 \
    --signature "*fp16, *fp16, *fp16, i32, 16" \
    --grid "(n_elements + 15) / 16, 1, 1"

# Link generated headers and create dispatchers.
python -m triton.tools.link aot/fp16/*.h --out aot/add_kernel_fp16
```

The generated files will be located in the `aot` directory. You should find three functions:
* `load_add_kernel_{fp16|fp32}` to load the GPU kernel
* `unload_add_kernel_{fp16|fp32}` to unload the GPU kernel
* `add_kernel_{fp16|fp32}` to launch the kernel