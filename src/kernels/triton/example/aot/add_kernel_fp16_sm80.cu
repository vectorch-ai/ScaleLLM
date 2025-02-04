/* clang-format off */
#include <cuda.h>
#include <stdint.h>
#include <assert.h>

// launcher for: add_kernel_fp16_sm80_16_warps1xstages3
CUresult add_kernel_fp16_sm80_6a55b24f_0123(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);

CUresult add_kernel_fp16_sm80_16_warps1xstages3(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements){
if (1)
    return add_kernel_fp16_sm80_6a55b24f_0123(stream, x_ptr, y_ptr, output_ptr, n_elements);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: add_kernel_fp16_sm80_16_warps1xstages3
void load_add_kernel_fp16_sm80_6a55b24f_0123();
void load_add_kernel_fp16_sm80_16_warps1xstages3() {
  load_add_kernel_fp16_sm80_6a55b24f_0123();
}

// unload for: add_kernel_fp16_sm80_16_warps1xstages3
void unload_add_kernel_fp16_sm80_6a55b24f_0123();
void unload_add_kernel_fp16_sm80_16_warps1xstages3() {
  unload_add_kernel_fp16_sm80_6a55b24f_0123();
}

typedef CUresult (*kernel_func_t)(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements);
kernel_func_t add_kernel_fp16_sm80_kernels[] = {
  add_kernel_fp16_sm80_16_warps1xstages3,
};

int add_kernel_fp16_sm80_get_num_algos(void){
  return (int)(sizeof(add_kernel_fp16_sm80_kernels) / sizeof(add_kernel_fp16_sm80_kernels[0]));
}

CUresult add_kernel_fp16_sm80(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements, int algo_id){
  assert (algo_id < (int)sizeof(add_kernel_fp16_sm80_kernels));
  return add_kernel_fp16_sm80_kernels[algo_id](stream, x_ptr, y_ptr, output_ptr, n_elements);
}

void load_add_kernel_fp16_sm80(void){
  load_add_kernel_fp16_sm80_16_warps1xstages3();
}

void unload_add_kernel_fp16_sm80(void){
  unload_add_kernel_fp16_sm80_16_warps1xstages3();
}


CUresult add_kernel_fp16_sm80_default(CUstream stream, CUdeviceptr x_ptr, CUdeviceptr y_ptr, CUdeviceptr output_ptr, int32_t n_elements){
  return add_kernel_fp16_sm80(stream, x_ptr, y_ptr, output_ptr, n_elements, 0);
}
