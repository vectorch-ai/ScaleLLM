set -e

# Remove previous generated files
rm -rf aot
mkdir -p aot

# Kernel for data type=float16, BLOCK_SIZE=16
# compile the kernel and generate headers and 'cubin' files
python ../tools/compile.py kernel.py \
    --kernel-name add_kernel \
    --out-path aot/add_kernel_fp16_sm80 \
    --out-name add_kernel_fp16_sm80 \
    --signature "*fp16, *fp16, *fp16, i32, 16" \
    --grid "(n_elements + 15) / 16, 1, 1" \
    --target "cuda:80"
# Link generated headers and create dispatchers.
python ../tools/link.py aot/add_kernel_fp16_sm80*.cuh --out aot/add_kernel_fp16_sm80

# Kernel for data type=float32, BLOCK_SIZE=16
python ../tools/compile.py kernel.py \
    --kernel-name add_kernel \
    --out-path aot/add_kernel_fp32_sm80 \
    --out-name add_kernel_fp32_sm80 \
    --signature "*fp32, *fp32, *fp32, i32, 16" \
    --grid "(n_elements + 15) / 16, 1, 1" \
    --target "cuda:80"
python ../tools/link.py aot/add_kernel_fp32_sm80*.cuh --out aot/add_kernel_fp32_sm80
