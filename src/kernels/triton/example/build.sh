set -e

# Remove previous generated files
rm -rf aot

# Kernel for data type=float16, BLOCK_SIZE=16
mkdir -p aot/fp16
# compile the kernel and generate headers and 'cubin' files
python ../tools/compile.py kernel.py \
    --kernel-name add_kernel \
    --out-path aot/fp16/add_kernel_fp16 \
    --out-name add_kernel_fp16 \
    --signature "*fp16, *fp16, *fp16, i32, 16" \
    --grid "(n_elements + 15) / 16, 1, 1" \
    --target "cuda:80"
# Link generated headers and create dispatchers.
python -m triton.tools.link aot/fp16/*.h --out aot/add_kernel_fp16

# Kernel for data type=float32, BLOCK_SIZE=16
mkdir -p aot/fp32
python ../tools/compile.py kernel.py \
    --kernel-name add_kernel \
    --out-path aot/fp32/add_kernel_fp32 \
    --out-name add_kernel_fp32 \
    --signature "*fp32, *fp32, *fp32, i32, 16" \
    --grid "(n_elements + 15) / 16, 1, 1" \
    --target "cuda:80"
python -m triton.tools.link aot/fp32/*.h --out aot/add_kernel_fp32
